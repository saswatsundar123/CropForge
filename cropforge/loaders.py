"""
cropforge/loaders.py
====================
CSV data loaders for Weather and Soil input files.

PRD References:
    Section 8.1 — Weather Data (CSV), flexible column mapping, SI units
    Section 8.2 — Soil Profile (CSV), apply="uniform" | "spatial"
    Section 11  — Wind input in m/s internally (SI)

Design decisions:
  - Column names are flexible via keyword arguments so researchers are not
    forced to rename their CSV files (PRD Section 8.1).
  - All meteorological values are normalised to SI units inside the loader
    before being stored. A ``wind_unit`` conversion argument handles knots
    or km/h → m/s (PRD Section 11).
  - If two fields receive the same file path the Weather object is the same
    Python object in memory; callers who want deduplication should pass the
    same instance rather than calling from_csv twice (PRD Section 8.1).
  - Soil.from_csv with apply="uniform" broadcasts one layer-set across every
    (row, col) cell.  apply="spatial" expects the CSV to be pre-ordered or
    indexed by (row, col, layer) (PRD Section 8.2).

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cropforge.state import EnvironmentState, SoilVoxelState


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

_WIND_UNIT_FACTORS: Dict[str, float] = {
    "m/s": 1.0,
    "ms": 1.0,       # shorthand accepted
    "kmh": 1.0 / 3.6,
    "km/h": 1.0 / 3.6,
    "knots": 0.514444,
    "kt": 0.514444,
}


def _wind_to_ms(value: float, unit: str) -> float:
    """Convert wind speed *value* from *unit* to m/s."""
    factor = _WIND_UNIT_FACTORS.get(unit.lower().strip())
    if factor is None:
        raise ValueError(
            f"Unknown wind unit {unit!r}. "
            f"Accepted: {list(_WIND_UNIT_FACTORS)}."
        )
    return value * factor


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

class Weather:
    """Parsed daily weather time-series attached to a Field.

    Do not instantiate directly — use :meth:`from_csv`.

    The internal store is a pandas DataFrame indexed by **simulation day**
    (1-indexed integer matching ``EnvironmentState.day``).

    The ``get_day(day)`` method returns a ready-to-use :class:`EnvironmentState`.
    """

    def __init__(self, df: pd.DataFrame, source_path: str = "<unknown>") -> None:
        self._df = df          # indexed by simulation day (1 … N)
        self._source = source_path
        self._n_days = len(df)

    # ------------------------------------------------------------------
    # Factory method (PRD Section 8.1)
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        # Column mapping — default names match the PRD example CSV header
        date_col: str = "date",
        tmax_col: str = "tmax_c",
        tmin_col: str = "tmin_c",
        tmean_col: Optional[str] = None,       # computed from tmax/tmin if absent
        radiation_col: str = "radiation_mj",
        rainfall_col: str = "rainfall_mm",
        humidity_col: str = "humidity_pct",
        wind_col: str = "wind_ms",
        et0_col: Optional[str] = None,         # optional; defaults to 0.0 if absent
        co2_col: Optional[str] = None,         # optional; defaults to 415.0
        # Unit conversion
        wind_unit: str = "m/s",
        # Simulation start (day of year of the first row)
        start_doy: Optional[int] = None,
    ) -> "Weather":
        """Load daily weather data from a CSV file.

        Parameters
        ----------
        path:
            Path to the CSV file.
        date_col, tmax_col, tmin_col, radiation_col, rainfall_col,
        humidity_col, wind_col:
            Column names in the CSV.  Defaults match the PRD example header.
        tmean_col:
            Name of a pre-computed mean temperature column.  If ``None``,
            ``temp_mean_c = (tmax + tmin) / 2`` is computed by the loader.
        et0_col:
            Name of a reference ET₀ column.  If ``None`` the column defaults
            to 0.0 — researchers can supply ET₀ in their step function or
            provide it here.
        co2_col:
            Optional CO₂ column (ppm).  Defaults to 415.0 if absent.
        wind_unit:
            Unit of the raw wind column.  Accepted: ``"m/s"`` (default),
            ``"km/h"``, ``"knots"``.  Values are converted to m/s internally
            (PRD Section 11).
        start_doy:
            Day-of-year of the first CSV row.  If ``None`` the loader reads
            the date column to compute DOY; if the date column is absent it
            defaults to DOY 1.

        Returns
        -------
        Weather
            A ``Weather`` instance ready to be passed to ``Field.set_weather()``.

        Examples
        --------
        >>> w = Weather.from_csv(
        ...     "weather.csv",
        ...     date_col="date",
        ...     tmax_col="tmax_c",
        ...     radiation_col="radiation_mj",
        ... )
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Weather CSV not found: {path}")

        # Skip comment lines (lines beginning with '#')
        raw = pd.read_csv(p, comment="#")

        # ---- Validate required columns --------------------------------
        required = {tmax_col, tmin_col, radiation_col, rainfall_col,
                    humidity_col, wind_col}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(
                f"Weather CSV {path!r} is missing required columns: {sorted(missing)}. "
                f"Available columns: {list(raw.columns)}."
            )

        n = len(raw)
        if n == 0:
            raise ValueError(f"Weather CSV {path!r} contains no data rows.")

        # ---- DOY -------------------------------------------------------
        if date_col in raw.columns:
            try:
                dates = pd.to_datetime(raw[date_col], errors="coerce")
                doys = dates.dt.day_of_year.tolist()
            except Exception:
                doys = list(range(1, n + 1))
        elif start_doy is not None:
            doys = [((start_doy - 1 + i) % 365) + 1 for i in range(n)]
        else:
            doys = list(range(1, n + 1))

        # ---- Build internal DataFrame ---------------------------------
        tmax = raw[tmax_col].astype(float).tolist()
        tmin = raw[tmin_col].astype(float).tolist()

        if tmean_col and tmean_col in raw.columns:
            tmean = raw[tmean_col].astype(float).tolist()
        else:
            tmean = [(mx + mn) / 2.0 for mx, mn in zip(tmax, tmin)]

        radiation = raw[radiation_col].astype(float).tolist()
        rainfall = raw[rainfall_col].astype(float).tolist()
        humidity = raw[humidity_col].astype(float).tolist()

        # Wind: convert to m/s (PRD Section 11)
        wind_raw = raw[wind_col].astype(float).tolist()
        wind_ms = [_wind_to_ms(v, wind_unit) for v in wind_raw]

        et0: List[float]
        if et0_col and et0_col in raw.columns:
            et0 = raw[et0_col].astype(float).tolist()
        else:
            et0 = [0.0] * n

        co2: List[float]
        if co2_col and co2_col in raw.columns:
            co2 = raw[co2_col].astype(float).tolist()
        else:
            co2 = [415.0] * n

        data = {
            "doy":            doys,
            "temp_max_c":     tmax,
            "temp_min_c":     tmin,
            "temp_mean_c":    tmean,
            "radiation_mj_m2": radiation,
            "rainfall_mm":    rainfall,
            "humidity_pct":   humidity,
            "wind_speed_ms":  wind_ms,
            "et0_mm":         et0,
            "co2_ppm":        co2,
        }

        df = pd.DataFrame(data, index=range(1, n + 1))  # 1-indexed day
        return cls(df, source_path=str(p.resolve()))

    # ------------------------------------------------------------------
    # Runtime access
    # ------------------------------------------------------------------

    def get_day(self, day: int) -> EnvironmentState:
        """Return the :class:`EnvironmentState` for simulation *day*.

        Parameters
        ----------
        day:
            Simulation day (1-indexed).  Days beyond the length of the CSV
            wrap cyclically, so a 90-day weather file can drive a longer
            simulation (useful for perennial crops or multi-season runs).
        """
        # Cyclic wrap for simulations longer than the weather record
        effective = ((day - 1) % self._n_days) + 1
        row = self._df.loc[effective]
        return EnvironmentState(
            day=day,
            doy=int(row["doy"]),
            temp_max_c=float(row["temp_max_c"]),
            temp_min_c=float(row["temp_min_c"]),
            temp_mean_c=float(row["temp_mean_c"]),
            radiation_mj_m2=float(row["radiation_mj_m2"]),
            rainfall_mm=float(row["rainfall_mm"]),
            humidity_pct=float(row["humidity_pct"]),
            wind_speed_ms=float(row["wind_speed_ms"]),
            et0_mm=float(row["et0_mm"]),
            co2_ppm=float(row["co2_ppm"]),
        )

    @property
    def n_days(self) -> int:
        """Number of weather records loaded."""
        return self._n_days

    def __repr__(self) -> str:
        return f"Weather(n_days={self._n_days}, source={self._source!r})"


# ---------------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------------

class Soil:
    """Parsed soil profile attached to a Field.

    Do not instantiate directly — use :meth:`from_csv`.

    Internally stores a list of layer-dictionaries (one dict per layer) that
    describe the uniform profile, or a full spatial grid for apply="spatial".

    The ``build_grid(rows, cols)`` method is called by :meth:`Field._init_field_state`
    to construct the ``soil: List[List[List[SoilVoxelState]]]`` grid.
    """

    def __init__(
        self,
        layers: List[Dict[str, Any]],
        apply: str = "uniform",
        spatial_grid: Optional[List[Dict[str, Any]]] = None,
        rows: int = 0,
        cols: int = 0,
    ) -> None:
        self._layers = layers         # uniform profile (list of layer dicts)
        self._apply = apply
        self._spatial_grid = spatial_grid or []   # flat list for spatial mode
        self._rows = rows
        self._cols = cols

    # ------------------------------------------------------------------
    # Factory method (PRD Section 8.2)
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str,
        apply: str = "uniform",
        rows: int = 0,
        cols: int = 0,
    ) -> "Soil":
        """Load a soil profile from a CSV file.

        Parameters
        ----------
        path:
            Path to the CSV file.  Expected columns (PRD Section 8.2):
            ``layer, depth_top_cm, depth_bottom_cm, moisture_pct, n_kg_ha,
            bulk_density, pen_resistance_mpa``.
        apply:
            ``"uniform"`` — the single profile is applied identically to all
            grid cells.  The CSV must have one row per soil layer.
            ``"spatial"`` — the CSV has one row per (row, col, layer) triplet,
            with additional ``row`` and ``col`` columns.  The grid dimensions
            must be supplied via the *rows* and *cols* arguments.
        rows, cols:
            Field grid dimensions.  Required when ``apply="spatial"``.

        Returns
        -------
        Soil
            Ready to pass to ``Field.set_soil()``.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If required columns are absent or apply mode is invalid.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Soil CSV not found: {path}")

        raw = pd.read_csv(p, comment="#")

        # ---- Required base columns ------------------------------------
        required_base = {
            "layer", "depth_top_cm", "depth_bottom_cm",
            "moisture_pct", "n_kg_ha", "bulk_density", "pen_resistance_mpa",
        }
        missing = required_base - set(raw.columns)
        if missing:
            raise ValueError(
                f"Soil CSV {path!r} missing columns: {sorted(missing)}. "
                f"Available: {list(raw.columns)}."
            )

        if apply == "uniform":
            layers = []
            for _, row in raw.iterrows():
                layers.append({
                    "layer":                int(row["layer"]),
                    "depth_top_cm":         float(row["depth_top_cm"]),
                    "depth_bottom_cm":      float(row["depth_bottom_cm"]),
                    "moisture_pct":         float(row["moisture_pct"]),
                    "nitrogen_kg_ha":       float(row["n_kg_ha"]),
                    "bulk_density":         float(row["bulk_density"]),
                    "penetration_resistance": float(row["pen_resistance_mpa"]),
                })
            return cls(layers=layers, apply="uniform")

        elif apply == "spatial":
            for col_name in ("row", "col"):
                if col_name not in raw.columns:
                    raise ValueError(
                        f"Soil CSV {path!r} with apply='spatial' requires a "
                        f"'{col_name}' column."
                    )
            if rows <= 0 or cols <= 0:
                raise ValueError(
                    "apply='spatial' requires rows > 0 and cols > 0 to be supplied."
                )
            spatial = []
            for _, row in raw.iterrows():
                spatial.append({
                    "row":                  int(row["row"]),
                    "col":                  int(row["col"]),
                    "layer":                int(row["layer"]),
                    "depth_top_cm":         float(row["depth_top_cm"]),
                    "depth_bottom_cm":      float(row["depth_bottom_cm"]),
                    "moisture_pct":         float(row["moisture_pct"]),
                    "nitrogen_kg_ha":       float(row["n_kg_ha"]),
                    "bulk_density":         float(row["bulk_density"]),
                    "penetration_resistance": float(row["pen_resistance_mpa"]),
                })
            return cls(
                layers=[],
                apply="spatial",
                spatial_grid=spatial,
                rows=rows,
                cols=cols,
            )
        else:
            raise ValueError(
                f"Soil.from_csv apply must be 'uniform' or 'spatial', got {apply!r}."
            )

    # ------------------------------------------------------------------
    # Grid builder — called by Field._init_field_state()
    # ------------------------------------------------------------------

    def build_grid(
        self, rows: int, cols: int
    ) -> List[List[List[SoilVoxelState]]]:
        """Construct the ``soil[row][col][layer]`` state grid.

        Parameters
        ----------
        rows, cols:
            Field grid dimensions.

        Returns
        -------
        List[List[List[SoilVoxelState]]]
            Indexed as ``[row][col][layer]``.
        """
        if self._apply == "uniform":
            return self._build_uniform(rows, cols)
        return self._build_spatial(rows, cols)

    def _build_uniform(
        self, rows: int, cols: int
    ) -> List[List[List[SoilVoxelState]]]:
        return [
            [
                [
                    SoilVoxelState(
                        row=r,
                        col=c,
                        layer=layer["layer"],
                        depth_top_cm=layer["depth_top_cm"],
                        depth_bottom_cm=layer["depth_bottom_cm"],
                        moisture_pct=layer["moisture_pct"],
                        nitrogen_kg_ha=layer["nitrogen_kg_ha"],
                        bulk_density=layer["bulk_density"],
                        penetration_resistance=layer["penetration_resistance"],
                    )
                    for layer in self._layers
                ]
                for c in range(cols)
            ]
            for r in range(rows)
        ]

    def _build_spatial(
        self, rows: int, cols: int
    ) -> List[List[List[SoilVoxelState]]]:
        # Pre-sort by (row, col, layer) and index
        from collections import defaultdict
        index: Dict[tuple, List[Dict]] = defaultdict(list)
        for entry in self._spatial_grid:
            index[(entry["row"], entry["col"])].append(entry)

        grid = []
        for r in range(rows):
            row_list = []
            for c in range(cols):
                layers_for_cell = sorted(
                    index.get((r, c), []), key=lambda x: x["layer"]
                )
                if not layers_for_cell:
                    # Fallback: 1 default layer if this cell has no data
                    layers_for_cell = [{
                        "layer": 0, "depth_top_cm": 0.0,
                        "depth_bottom_cm": 20.0, "moisture_pct": 0.0,
                        "nitrogen_kg_ha": 0.0, "bulk_density": 1.3,
                        "penetration_resistance": 0.5,
                    }]
                row_list.append([
                    SoilVoxelState(
                        row=r, col=c,
                        layer=lyr["layer"],
                        depth_top_cm=lyr["depth_top_cm"],
                        depth_bottom_cm=lyr["depth_bottom_cm"],
                        moisture_pct=lyr["moisture_pct"],
                        nitrogen_kg_ha=lyr["nitrogen_kg_ha"],
                        bulk_density=lyr["bulk_density"],
                        penetration_resistance=lyr["penetration_resistance"],
                    )
                    for lyr in layers_for_cell
                ])
            grid.append(row_list)
        return grid

    @property
    def n_layers(self) -> int:
        """Number of layers in the uniform profile (0 for spatial mode)."""
        return len(self._layers)

    def __repr__(self) -> str:
        return f"Soil(apply={self._apply!r}, n_layers={self.n_layers})"
