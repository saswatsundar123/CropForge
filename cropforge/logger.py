"""
cropforge/logger.py
===================
Parquet state logger — serialises the simulation state at every timestep.

PRD References:
    Section 16.1 — Schema (plant, soil, environment tables — flat / denormalised)
    Section 16.2 — Partitioning (field_name → day), Snappy compression,
                   row group size = one day's plant records
    Section 16.3 — custom dicts serialised as JSON strings in custom_json
    Section 16.4 — Schema stability guarantee; cropforge_version in file metadata

Design:
    The logger accumulates rows for each table in plain Python lists during the
    run (O(field_size × soil_layers) memory per timestep — lists are flushed
    in partitioned batches).  At the end of the run (or on crash flush) each
    batch is converted to a pyarrow Table with the exact schema from Section 16
    and written as a partitioned Parquet dataset under
    ``cropforge_output/<session_name>/``.

    Partitioning is done via pyarrow's ``write_to_dataset`` with
    ``partition_cols=["field_name", "day"]`` so the frontend can read a single
    day/field slice without scanning the whole file (PRD Section 16.2).

    Compression = Snappy (PRD Section 16.2).

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from cropforge.farm import Farm, Field
    from cropforge.state import EnvironmentState, FieldState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen Parquet schemas (PRD Section 16.1)
# ---------------------------------------------------------------------------
# FLOAT  → pa.float32()     (PRD says FLOAT)
# STRING → pa.string()
# INT32  → pa.int32()
# INT16  → pa.int16()
# INT8   → pa.int8()
# BOOLEAN → pa.bool_()

PLANT_SCHEMA = pa.schema([
    pa.field("day",                 pa.int32()),
    pa.field("field_name",          pa.string()),
    pa.field("plant_id",            pa.string()),
    pa.field("row",                 pa.int16()),
    pa.field("col",                 pa.int16()),
    pa.field("age_days",            pa.int16()),
    pa.field("lai",                 pa.float32()),
    pa.field("biomass_g",           pa.float32()),
    pa.field("height_cm",           pa.float32()),
    pa.field("root_depth_cm",       pa.float32()),
    pa.field("stress_index",        pa.float32()),
    pa.field("alive",               pa.bool_()),
    pa.field("phenological_stage",  pa.string()),
    pa.field("custom_json",         pa.string()),
])

SOIL_SCHEMA = pa.schema([
    pa.field("day",                     pa.int32()),
    pa.field("field_name",              pa.string()),
    pa.field("row",                     pa.int16()),
    pa.field("col",                     pa.int16()),
    pa.field("layer",                   pa.int8()),
    pa.field("depth_top_cm",            pa.float32()),
    pa.field("depth_bottom_cm",         pa.float32()),
    pa.field("moisture_pct",            pa.float32()),
    pa.field("nitrogen_kg_ha",          pa.float32()),
    pa.field("bulk_density",            pa.float32()),
    pa.field("penetration_resistance",  pa.float32()),
    pa.field("custom_json",             pa.string()),
])

ENV_SCHEMA = pa.schema([
    pa.field("day",             pa.int32()),
    pa.field("field_name",      pa.string()),
    pa.field("season",          pa.int32()),    # v0.4.0 -- multi-season tracking
    pa.field("doy",             pa.int16()),
    pa.field("temp_max_c",      pa.float32()),
    pa.field("temp_min_c",      pa.float32()),
    pa.field("temp_mean_c",     pa.float32()),
    pa.field("radiation_mj_m2", pa.float32()),
    pa.field("rainfall_mm",     pa.float32()),
    pa.field("et0_mm",          pa.float32()),
    pa.field("wind_speed_ms",   pa.float32()),
    pa.field("humidity_pct",    pa.float32()),
    pa.field("co2_ppm",         pa.float32()),
    pa.field("events_fired",    pa.string()),   # JSON array of strings
    pa.field("custom_json",     pa.string()),
])


# ---------------------------------------------------------------------------
# StateLogger
# ---------------------------------------------------------------------------

class StateLogger:
    """Accumulates per-timestep state and writes it to a partitioned Parquet dataset.

    Parameters
    ----------
    session_name:
        Human-readable identifier for this run, used as the output directory
        name (e.g. ``"wheat_basic_20260624T183000"``).
    output_root:
        Root directory under which the session subdirectory is created.
        Defaults to ``./cropforge_output``.

    Usage
    -----
    The logger is created by ``Farm.run()`` and driven by ``_execute_run``::

        log = StateLogger(session_name="wheat_basic_20260624")
        log.record(field, state, env)   # called every day for every field
        log.flush()                     # called at the end of the run

    A partial flush is triggered by the error handler so that completed
    timesteps survive a crash (PRD Section 6.4, rule 2).
    """

    def __init__(
        self,
        session_name: str,
        output_root: str = "cropforge_output",
        cropforge_version: str = "0.1.0",
    ) -> None:
        self.session_name = session_name
        self.output_dir = Path(output_root) / session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._version = cropforge_version

        # Accumulator buffers — one list per table
        self._plant_rows:   List[Dict[str, Any]] = []
        self._soil_rows:    List[Dict[str, Any]] = []
        self._env_rows:     List[Dict[str, Any]] = []

        # Track whether anything has been written yet (for partial flush logic)
        self._flushed_days: int = 0

        logger.info("StateLogger initialised -> %s", self.output_dir)

    # ------------------------------------------------------------------
    # Accumulate
    # ------------------------------------------------------------------

    def record(
        self,
        field: "Field",
        state: "FieldState",
        env: "EnvironmentState",
    ) -> None:
        """Append one day's state for one field to the in-memory buffers.

        Called by the engine after all step functions for a given
        (day, field) have completed and plant ages have been incremented.
        """
        day = state.day
        field_name = field.name

        # ---- Plant rows (one per plant) -------------------------------
        for plant in state.plants:
            self._plant_rows.append({
                "day":                day,
                "field_name":         field_name,
                "plant_id":           plant.plant_id,
                "row":                plant.row,
                "col":                plant.col,
                "age_days":           plant.age_days,
                "lai":                plant.lai,
                "biomass_g":          plant.biomass_g,
                "height_cm":          plant.height_cm,
                "root_depth_cm":      plant.root_depth_cm,
                "stress_index":       plant.stress_index,
                "alive":              plant.alive,
                "phenological_stage": plant.phenological_stage,
                "custom_json":        json.dumps(plant.custom),
            })

        # ---- Soil rows (one per voxel = per row × col × layer) --------
        for row_list in state.soil:
            for col_list in row_list:
                for voxel in col_list:
                    self._soil_rows.append({
                        "day":                    day,
                        "field_name":             field_name,
                        "row":                    voxel.row,
                        "col":                    voxel.col,
                        "layer":                  voxel.layer,
                        "depth_top_cm":           voxel.depth_top_cm,
                        "depth_bottom_cm":        voxel.depth_bottom_cm,
                        "moisture_pct":           voxel.moisture_pct,
                        "nitrogen_kg_ha":         voxel.nitrogen_kg_ha,
                        "bulk_density":           voxel.bulk_density,
                        "penetration_resistance": voxel.penetration_resistance,
                        "custom_json":            json.dumps(voxel.custom),
                    })

        # ---- Environment row (one per field per day) ------------------
        self._env_rows.append({
            "day":             day,
            "field_name":      field_name,
            "season":          getattr(env, "season", 1),  # v0.4.0
            "doy":             env.doy,
            "temp_max_c":      env.temp_max_c,
            "temp_min_c":      env.temp_min_c,
            "temp_mean_c":     env.temp_mean_c,
            "radiation_mj_m2": env.radiation_mj_m2,
            "rainfall_mm":     env.rainfall_mm,
            "et0_mm":          env.et0_mm,
            "wind_speed_ms":   env.wind_speed_ms,
            "humidity_pct":    env.humidity_pct,
            "co2_ppm":         env.co2_ppm,
            "events_fired":    json.dumps(state.events_fired),
            "custom_json":     json.dumps(env.custom),
        })

    # ------------------------------------------------------------------
    # Write to Parquet
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all accumulated rows to the partitioned Parquet dataset.

        PRD Section 16.2:
          - Partitioned by ``field_name``, then by ``day``.
          - Snappy compression.
          - Row group size = one day's plant records per group.

        PRD Section 16.4:
          - ``cropforge_version`` stored in file-level Parquet metadata.

        Safe to call multiple times (idempotent for the same data).
        If called on an empty buffer, it creates the directory tree but
        writes no Parquet files.
        """
        if not self._plant_rows:
            logger.warning("StateLogger.flush() called with no recorded data.")
            return

        file_meta = {
            "cropforge_version": self._version,
            "session_name":      self.session_name,
            "flushed_at":        datetime.now(timezone.utc).isoformat(),
        }

        self._write_table(
            rows=self._plant_rows,
            schema=PLANT_SCHEMA,
            subdir="plants",
            file_meta=file_meta,
        )
        self._write_table(
            rows=self._soil_rows,
            schema=SOIL_SCHEMA,
            subdir="soil",
            file_meta=file_meta,
        )
        self._write_table(
            rows=self._env_rows,
            schema=ENV_SCHEMA,
            subdir="environment",
            file_meta=file_meta,
        )

        self._flushed_days = len(set(r["day"] for r in self._env_rows))
        logger.info(
            "StateLogger flushed %d plant-rows, %d soil-rows, %d env-rows -> %s",
            len(self._plant_rows),
            len(self._soil_rows),
            len(self._env_rows),
            self.output_dir,
        )

    def _write_table(
        self,
        rows: List[Dict[str, Any]],
        schema: pa.Schema,
        subdir: str,
        file_meta: Dict[str, str],
    ) -> None:
        """Convert *rows* to a pyarrow Table and write a partitioned dataset."""
        if not rows:
            return

        # Build columns dict — cast each column to its declared type
        cols: Dict[str, Any] = {field.name: [] for field in schema}
        for row in rows:
            for field in schema:
                cols[field.name].append(row[field.name])

        arrays = []
        for field in schema:
            raw_array = pa.array(cols[field.name], type=field.type)
            arrays.append(raw_array)

        table = pa.table(
            {field.name: arrays[i] for i, field in enumerate(schema)},
            schema=schema,
        )

        # Attach file-level metadata (PRD Section 16.4)
        existing_meta = table.schema.metadata or {}
        merged_meta = {
            **{k.encode(): v.encode() for k, v in file_meta.items()},
            **existing_meta,
        }
        table = table.replace_schema_metadata(merged_meta)

        dest = self.output_dir / subdir
        dest.mkdir(parents=True, exist_ok=True)

        # Write partitioned by field_name → day (PRD Section 16.2).
        # 'season' is stored as a plain data column (not a partition key) so it
        # remains readable in the returned DataFrame without path-decoding.
        # Season 1 and Season 2 rows are distinguished by the continuous day
        # numbers (Season 2 starts at _day_offset + 1), so there is no collision.
        pq.write_to_dataset(
            table,
            root_path=str(dest),
            partition_cols=["field_name", "day"],
            compression="snappy",
            existing_data_behavior="overwrite_or_ignore",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> str:
        """Absolute path to the session output directory."""
        return str(self.output_dir.resolve())

    def __repr__(self) -> str:
        return (
            f"StateLogger(session={self.session_name!r}, "
            f"plant_rows={len(self._plant_rows)}, "
            f"soil_rows={len(self._soil_rows)}, "
            f"env_rows={len(self._env_rows)})"
        )
