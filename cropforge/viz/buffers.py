"""
cropforge/viz/buffers.py
========================
Parquet-to-binary converter for the Three.js instanced plant renderer.

PRD References:
    Section 7.3  — Binary Float32Array buffers; per-timestep field state
                   packed in the order Three.js InstancedMesh expects.
                   Conversion done once at startup, held in memory.
    PRD v0.2.0   — Multi-field support: each field has its own buffer store,
                   served via ?field= query parameter.

Binary frame layout (per plant, 14 float32 = 56 bytes):
    [0]  x              — column index (grid position)
    [1]  y              — height_cm scaled to scene units (0.0 on dead plants)
    [2]  z              — row index (grid position)
    [3]  half_h         — half-height (y offset so cylinder sits on ground)
    [4]  radius         — LAI proxy (clamped to reasonable cylinder radius)
    [5]  r              — colour red   channel [0..1]
    [6]  g              — colour green channel [0..1]
    [7]  b              — colour blue  channel [0..1]
    [8]  alive          — 1.0 = alive, 0.0 = dead (as float for buffer alignment)
    [9]  model_index    — integer key into model_index_map (0 = cylinder fallback)
    [10] stage_progress — fractional progress within current pheno stage [0.0, 1.0]
    [11] morph_weight   — stage morph interpolation weight [0.0, 1.0]
    [12] stress_ks      — water stress coefficient for wilt deformation [0.0, 1.0]
    [13] disease_severity — disease necrosis shader severity [0.0, 1.0]

Total buffer per day  = n_plants × 14 × 4 bytes
Total buffer all days = n_days   × n_plants × 14 × 4 bytes

Multi-field API (v0.2.0):
    FIELD_STORES[field_name] → BufferStore instance per field.
    BUFFER_STORE              → legacy alias to first field (backward compat).

    GET /api/buffer/meta?field=<name>  → metadata for that field
    GET /api/buffer?day=<d>&field=<n>  → binary frame for that field/day

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import colorsys
import json
import logging
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Floats per plant in the binary frame (see layout above)
BUFFER_FIELDS = [
    "x",
    "y",
    "z",
    "scale_y",
    "radius",
    "r",
    "g",
    "b",
    "alive",
    "model_index",
    "stage_progress",
    "morph_weight",
    "stress_ks",
    "disease_severity",
]
FLOATS_PER_PLANT = len(BUFFER_FIELDS)
BYTES_PER_PLANT  = FLOATS_PER_PLANT * 4      # float32 = 4 bytes

# PRD Section 7.3: dead plant colour = #8B6914 (dried brown)
_DEAD_R = 0x8B / 255.0
_DEAD_G = 0x69 / 255.0
_DEAD_B = 0x14 / 255.0

# Scene scale: 1 simulation cm → this many Three.js world units
_CM_TO_WORLD = 0.015

# Grid spacing between plants (Three.js world units)
_GRID_SPACING = 1.0

# Maximum radius (LAI proxy) in world units
_MAX_RADIUS = 0.4
_MIN_RADIUS = 0.05


# ---------------------------------------------------------------------------
# Colour mapping helpers
# ---------------------------------------------------------------------------

def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    """Convert HSL (all in [0, 1]) to RGB (all in [0, 1])."""
    return colorsys.hls_to_rgb(h, l, s)


def _clamp01(value: object, default: float = 0.0) -> float:
    """Return *value* as a finite float clamped into [0, 1]."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if not np.isfinite(v):
        v = default
    return max(0.0, min(1.0, v))


def _parse_custom_json(value: object) -> dict:
    """Parse logger ``custom_json`` defensively for optional viz fields."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _value_to_rgb(
    value: float,
    vmin: float,
    vmax: float,
    variable: str,
) -> Tuple[float, float, float]:
    """Map a scalar value to an RGB triple using a variable-specific gradient.

    Gradients (PRD Section 7.3 — HSL gradient across field):
        biomass_g    : yellow (hue 60) → vivid green (hue 120)
        lai          : hue 80 → 130 (yellow-green → green)
        height_cm    : hue 200 → 240 (sky-blue → deep blue)
        stress_index : green (hue 120) → red (hue 0) [high stress = red]
        default      : hue 120 (green gradient lightness)
    """
    if vmax <= vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))

    if variable == "biomass_g":
        h = 0.16 + t * (0.33 - 0.16)   # 60° → 120° (yellow → green)
        s, l = 0.80, 0.35 + t * 0.20
    elif variable == "lai":
        h = 0.22 + t * (0.36 - 0.22)
        s, l = 0.75, 0.30 + t * 0.25
    elif variable == "height_cm":
        h = 0.56 + t * (0.67 - 0.56)   # sky-blue → deep blue
        s, l = 0.80, 0.35 + t * 0.20
    elif variable == "stress_index":
        h = 0.33 * (1.0 - t)            # green → red (high stress = red)
        s, l = 0.85, 0.40
    else:
        h, s, l = 0.33, 0.70, 0.35 + t * 0.30

    return _hsl_to_rgb(h, s, l)


# ---------------------------------------------------------------------------
# Pre-computation: build the full binary payload at startup
# ---------------------------------------------------------------------------

class BufferStore:
    """Holds pre-packed binary frames for every simulation day of ONE field.

    Attributes
    ----------
    field_name : str
    n_plants : int
    n_days   : int
    rows, cols : int   — field grid dimensions
    _frames  : Dict[int, bytes]   — keyed by simulation day (1-indexed)
    _meta    : dict
    """

    def __init__(self, field_name: str = "") -> None:
        self.field_name: str   = field_name
        self.n_plants:   int   = 0
        self.n_days:     int   = 0
        self.rows:       int   = 0
        self.cols:       int   = 0
        self._frames:    Dict[int, bytes] = {}
        self._meta:      Dict = {}
        self._ready:     bool = False
        self._model_index_map: Dict[str, int] = {}

    def build(
        self,
        plants_df: pd.DataFrame,
        variable: str = "biomass_g",
    ) -> None:
        """Pre-pack all simulation days into binary frames.

        Parameters
        ----------
        plants_df:
            Plant DataFrame for THIS FIELD ONLY (already filtered).
        variable:
            The colour-mapped variable (default ``biomass_g``).
        """
        if plants_df is None or plants_df.empty:
            logger.warning(
                "BufferStore.build() called with empty plants DataFrame "
                "(field=%s).", self.field_name
            )
            return

        # Normalise types for safe arithmetic and dict-key lookup
        plants_df = plants_df.copy()
        plants_df["day"] = plants_df["day"].astype(int)
        plants_df["row"] = plants_df["row"].astype(int)
        plants_df["col"] = plants_df["col"].astype(int)

        days = sorted(plants_df["day"].unique())
        self.n_days = len(days)

        # Infer grid dimensions from the data
        self.rows = int(plants_df["row"].max()) + 1
        self.cols = int(plants_df["col"].max()) + 1
        self.n_plants = self.rows * self.cols

        # Pre-compute global min/max for the colour variable for stable mapping
        if variable in plants_df.columns:
            vmin = float(plants_df[variable].min())
            vmax = float(plants_df[variable].max())
        else:
            vmin, vmax = 0.0, 1.0

        # Build model_index_map: model_id string → unique int index (0 = no model / cylinder).
        # ponytail: sort for determinism; 0 is reserved for empty string.
        if "model_id" in plants_df.columns:
            unique_ids = sorted(set(plants_df["model_id"].dropna().unique()) - {""})
            model_index_map = {uri: (i + 1) for i, uri in enumerate(unique_ids)}
        else:
            model_index_map = {}
        self._model_index_map = model_index_map  # kept for _pack_frame lookup

        logger.info(
            "BufferStore.build(): field=%s variable=%s, vmin=%.3f, vmax=%.3f, "
            "days=%d, plants/day=%d",
            self.field_name, variable, vmin, vmax, self.n_days, self.n_plants,
        )

        # Build a frame per day
        for day in days:
            day_df = plants_df[plants_df["day"] == day].copy()
            frame = self._pack_frame(day_df, variable, vmin, vmax)
            self._frames[int(day)] = frame

        self._meta = {
            "field_name":        self.field_name,
            "n_plants":          self.n_plants,
            "n_days":            self.n_days,
            "rows":              self.rows,
            "cols":              self.cols,
            "days":              [int(d) for d in days],
            "variable":          variable,
            "vmin":              vmin,
            "vmax":              vmax,
            "bytes_per_plant":   BYTES_PER_PLANT,
            "floats_per_plant":  FLOATS_PER_PLANT,
            "buffer_fields":     BUFFER_FIELDS,
            "grid_spacing":      _GRID_SPACING,
            "cm_to_world":       _CM_TO_WORLD,
            # v0.9.0 Phase 2: model_id → int index (0 = cylinder fallback)
            "model_index_map":   model_index_map,
        }
        self._ready = True
        logger.info(
            "BufferStore ready: field=%s %d days × %d plants = %d KB total",
            self.field_name, self.n_days, self.n_plants,
            sum(len(v) for v in self._frames.values()) // 1024,
        )

    def _pack_frame(
        self,
        day_df: pd.DataFrame,
        variable: str,
        vmin: float,
        vmax: float,
    ) -> bytes:
        """Pack one day's plant data into a flat bytes array.

        Plants are ordered row-major (row 0 col 0, row 0 col 1, …) so that
        Three.js can index them by ``instanceId = row * cols + col``.

        Uses numpy arrays for safe, type-stable lookups — avoids pandas
        MultiIndex int32/int64 key ambiguity on Python 3.14.
        """
        # Cast row/col to plain int to avoid any numpy int type issues
        day_df = day_df.copy()
        day_df["row"] = day_df["row"].astype(int)
        day_df["col"] = day_df["col"].astype(int)

        # Build a lookup dict: (row, col) → row-dict for O(1) access
        records = {
            (int(rec["row"]), int(rec["col"])): rec
            for rec in day_df.to_dict("records")
        }

        buf = bytearray(self.n_plants * BYTES_PER_PLANT)
        view = memoryview(buf).cast("f")   # float32 view

        idx = 0
        for row_idx in range(self.rows):
            for col_idx in range(self.cols):
                rec = records.get((row_idx, col_idx))

                if rec is not None:
                    alive    = bool(rec["alive"])
                    height_w = float(rec["height_cm"]) * _CM_TO_WORLD if alive else 0.0
                    lai_v    = float(rec.get("lai", 0.05) or 0.05)
                    radius   = max(_MIN_RADIUS, min(_MAX_RADIUS, lai_v * 0.25))
                    half_h   = height_w / 2.0

                    if alive:
                        val = float(rec.get(variable, 0.0) or 0.0)
                        r_c, g_c, b_c = _value_to_rgb(val, vmin, vmax, variable)
                    else:
                        r_c, g_c, b_c = _DEAD_R, _DEAD_G, _DEAD_B
                        height_w = 0.0
                        half_h   = 0.0
                else:
                    alive    = False
                    height_w = 0.0
                    half_h   = 0.0
                    radius   = _MIN_RADIUS
                    r_c, g_c, b_c = _DEAD_R, _DEAD_G, _DEAD_B

                base = idx * FLOATS_PER_PLANT
                view[base + 0] = col_idx * _GRID_SPACING   # x = col
                view[base + 1] = half_h                     # y = half-height
                view[base + 2] = row_idx * _GRID_SPACING    # z = row
                view[base + 3] = height_w                   # scale Y
                view[base + 4] = radius                     # radius
                view[base + 5] = r_c                        # R
                view[base + 6] = g_c                        # G
                view[base + 7] = b_c                        # B
                view[base + 8] = 1.0 if alive else 0.0      # alive flag
                # v0.9.0 Phase 2 — model_index and stage_progress
                stage_progress = _clamp01(
                    rec.get("stage_progress", 0.0) if rec is not None else 0.0
                )
                custom = _parse_custom_json(
                    rec.get("custom_json", "") if rec is not None else ""
                )
                morph_weight = _clamp01(custom.get("morph_weight", stage_progress))
                stress_ks = _clamp01(
                    custom.get("water_stress_ks", custom.get("stress_ks", 1.0)),
                    default=1.0,
                )
                disease_severity = _clamp01(
                    rec.get(
                        "disease_severity",
                        custom.get("disease_stress", custom.get("disease_severity", 0.0)),
                    ) if rec is not None else 0.0
                )

                view[base + 9]  = float(self._model_index_map.get(
                    rec.get("model_id", "") if rec is not None else "", 0
                ))                                           # model_index (0=cylinder)
                view[base + 10] = stage_progress             # stage_progress [0,1]
                view[base + 11] = morph_weight               # morph_weight [0,1]
                view[base + 12] = stress_ks                  # stress_ks [0,1]
                view[base + 13] = disease_severity           # disease_severity [0,1]
                idx += 1

        return bytes(buf)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_frame(self, day: int) -> Optional[bytes]:
        """Return the packed binary frame for *day*, or None if not found."""
        return self._frames.get(day)

    def rebuild(self, plants_df: pd.DataFrame, variable: str) -> None:
        """Rebuild all frames with a new colour variable."""
        self._frames.clear()
        self._ready = False
        self.build(plants_df, variable)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def meta(self) -> dict:
        return self._meta


# ---------------------------------------------------------------------------
# Multi-field store registry (v0.2.0)
# ---------------------------------------------------------------------------

class FieldBufferRegistry:
    """Registry of one BufferStore per field name.

    Populated at server startup by ``build_all(plants_df)``.
    Queried per request by ``get(field_name)``.
    """

    def __init__(self) -> None:
        self._stores: Dict[str, BufferStore] = {}
        self._field_order: List[str] = []   # ordered as first-seen

    def build_all(
        self,
        plants_df: pd.DataFrame,
        variable: str = "biomass_g",
    ) -> None:
        """Build one BufferStore per unique field_name in *plants_df*."""
        if plants_df is None or plants_df.empty:
            logger.warning("FieldBufferRegistry.build_all() called with empty DataFrame.")
            return

        field_names = sorted(plants_df["field_name"].unique())
        self._field_order = field_names

        for fn in field_names:
            field_df = plants_df[plants_df["field_name"] == fn].copy()
            store = BufferStore(field_name=fn)
            store.build(field_df, variable=variable)
            self._stores[fn] = store
            logger.info("FieldBufferRegistry: built store for field '%s'", fn)

    def get(self, field_name: Optional[str]) -> Optional[BufferStore]:
        """Return the BufferStore for *field_name*, or the first store if None/blank."""
        if not field_name and self._field_order:
            field_name = self._field_order[0]
        return self._stores.get(field_name)

    def rebuild_all(self, plants_df: pd.DataFrame, variable: str) -> None:
        """Rebuild all field stores with a new colour variable."""
        self._stores.clear()
        self._field_order = []
        self.build_all(plants_df, variable=variable)

    def rebuild_field(
        self,
        field_name: Optional[str],
        plants_df: pd.DataFrame,
        variable: str,
    ) -> Optional[BufferStore]:
        """Rebuild the store for a single field."""
        if not field_name and self._field_order:
            field_name = self._field_order[0]
        if field_name is None:
            return None
        field_df = plants_df[plants_df["field_name"] == field_name].copy()
        store = self._stores.get(field_name, BufferStore(field_name=field_name))
        store.rebuild(field_df, variable)
        self._stores[field_name] = store
        return store

    @property
    def field_names(self) -> List[str]:
        return list(self._field_order)

    @property
    def is_ready(self) -> bool:
        return bool(self._stores) and all(s.is_ready for s in self._stores.values())

    def __bool__(self) -> bool:
        return bool(self._stores)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

# v0.2.0 multi-field registry (primary)
FIELD_REGISTRY = FieldBufferRegistry()

# Legacy v0.1.0 alias — points to the first field's store after build_all()
# Preserved for any code that imports BUFFER_STORE directly.
class _LegacyBufferStoreProxy:
    """Proxy that forwards attribute access to the first field's store."""

    def __getattr__(self, name: str):
        store = FIELD_REGISTRY.get(None)
        if store is None:
            raise AttributeError(
                f"BUFFER_STORE: no field stores built yet. "
                f"Call FIELD_REGISTRY.build_all(plants_df) first."
            )
        return getattr(store, name)

    def build(self, plants_df, variable="biomass_g"):
        """Legacy build() — routes to FIELD_REGISTRY.build_all()."""
        FIELD_REGISTRY.build_all(plants_df, variable=variable)

    def rebuild(self, plants_df, variable):
        """Legacy rebuild() — routes to FIELD_REGISTRY.rebuild_all()."""
        FIELD_REGISTRY.rebuild_all(plants_df, variable)


BUFFER_STORE = _LegacyBufferStoreProxy()
