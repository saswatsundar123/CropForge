"""
tests/test_logger_observables.py
=================================
Integration tests confirming that v0.7.0 Phase 6 observables
(surface_runoff_mm_today, cumulative_erosion_index) are correctly
written to the Parquet soil table when the erosion engine is enabled.

Covered:
    Schema: the two new columns exist in SOIL_SCHEMA
    Record: record() extracts both columns from voxel/state data
    End-to-end: farm.run() with erosion=True writes non-zero
                cumulative_erosion_index for sloped cells by day 5
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

from cropforge import Farm, Field, Crop
from cropforge.land_prep import ContourBund
from cropforge.logger import SOIL_SCHEMA, StateLogger
from cropforge.loaders import Weather
from cropforge.state import EnvironmentState, FieldState, PlantState, SoilVoxelState


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from test_logger.py)
# ---------------------------------------------------------------------------

def _read_parquet(path: Path) -> pd.DataFrame:
    # partitioning="hive" restores field_name and day as proper columns
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()


def _make_weather(days: int, rainfall_mm: float = 30.0) -> Weather:
    rows = [
        {
            "day": d, "doy": d,
            "temp_max_c": 32.0, "temp_min_c": 20.0, "temp_mean_c": 26.0,
            "radiation_mj_m2": 20.0, "rainfall_mm": rainfall_mm,
            "et0_mm": 6.0, "wind_speed_ms": 2.0, "humidity_pct": 70.0,
            "co2_ppm": 415.0,
        }
        for d in range(1, days + 1)
    ]
    return Weather(pd.DataFrame(rows).set_index("day"))


def _steep_elevation() -> np.ndarray:
    """2×4 grid: columns 1.0 → 0.67 → 0.33 → 0.0."""
    return np.array([
        [1.0, 0.67, 0.33, 0.0],
        [1.0, 0.67, 0.33, 0.0],
    ])


def _make_voxel(row: int, col: int, runoff: float = 0.0) -> SoilVoxelState:
    v = SoilVoxelState(
        row=row, col=col, layer=0,
        depth_top_cm=0.0, depth_bottom_cm=20.0,
        moisture_pct=25.0, nitrogen_kg_ha=30.0,
        bulk_density=1.3, penetration_resistance=0.8,
    )
    if runoff:
        v.custom["surface_runoff_mm_today"] = runoff
    return v


# ---------------------------------------------------------------------------
# Unit: schema contains the new columns
# ---------------------------------------------------------------------------

class TestSoilSchemaObservables:
    def test_surface_runoff_column_in_schema(self):
        names = [f.name for f in SOIL_SCHEMA]
        assert "surface_runoff_mm_today" in names

    def test_cumulative_erosion_column_in_schema(self):
        names = [f.name for f in SOIL_SCHEMA]
        assert "cumulative_erosion_index" in names

    def test_surface_runoff_is_float32(self):
        import pyarrow as pa
        field = next(f for f in SOIL_SCHEMA if f.name == "surface_runoff_mm_today")
        assert field.type == pa.float32()

    def test_cumulative_erosion_is_float32(self):
        import pyarrow as pa
        field = next(f for f in SOIL_SCHEMA if f.name == "cumulative_erosion_index")
        assert field.type == pa.float32()


# ---------------------------------------------------------------------------
# Unit: record() extracts observables from voxel.custom and state.custom
# ---------------------------------------------------------------------------

class TestRecordObservables:
    def _make_state_with_erosion(self, rows: int = 2, cols: int = 2) -> FieldState:
        plants = [
            PlantState(plant_id=f"r{r:02d}c{c:02d}", row=r, col=c)
            for r in range(rows) for c in range(cols)
        ]
        soil = [
            [[_make_voxel(r, c, runoff=5.0)] for c in range(cols)]
            for r in range(rows)
        ]
        state = FieldState(
            day=1, plants=plants, soil=soil,
            elevation_grid=np.zeros((rows, cols)),
            events_fired=[],
        )
        # Simulate erosion accumulator (matches make_erosion_hook structure)
        state.custom["cumulative_erosion_index_grid"] = [
            [float(r * cols + c + 1) for c in range(cols)]
            for r in range(rows)
        ]
        return state

    def test_surface_runoff_written_to_soil_row(self, tmp_path):
        log = StateLogger(session_name="obs_runoff", output_root=str(tmp_path))
        field = Field(name="F", rows=2, cols=2)
        state = self._make_state_with_erosion()
        env = EnvironmentState(
            day=1, doy=1, temp_max_c=30.0, temp_min_c=20.0, temp_mean_c=25.0,
            radiation_mj_m2=18.0, rainfall_mm=30.0, et0_mm=5.0,
            wind_speed_ms=2.0, humidity_pct=65.0,
        )
        log.record(field, state, env)
        row = log._soil_rows[0]
        assert row["surface_runoff_mm_today"] == pytest.approx(5.0)

    def test_cumulative_erosion_written_to_soil_row(self, tmp_path):
        log = StateLogger(session_name="obs_cum", output_root=str(tmp_path))
        field = Field(name="F", rows=2, cols=2)
        state = self._make_state_with_erosion()
        env = EnvironmentState(
            day=1, doy=1, temp_max_c=30.0, temp_min_c=20.0, temp_mean_c=25.0,
            radiation_mj_m2=18.0, rainfall_mm=30.0, et0_mm=5.0,
            wind_speed_ms=2.0, humidity_pct=65.0,
        )
        log.record(field, state, env)
        # Voxel at row=0, col=0 → grid[0][0] = 1.0
        vox00 = next(r for r in log._soil_rows if r["row"] == 0 and r["col"] == 0)
        assert vox00["cumulative_erosion_index"] == pytest.approx(1.0)
        # Voxel at row=1, col=1 → grid[1][1] = 1*2+1+1 = 4.0
        vox11 = next(r for r in log._soil_rows if r["row"] == 1 and r["col"] == 1)
        assert vox11["cumulative_erosion_index"] == pytest.approx(4.0)

    def test_no_erosion_grid_defaults_to_zero(self, tmp_path):
        log = StateLogger(session_name="obs_noerode", output_root=str(tmp_path))
        field = Field(name="F", rows=1, cols=1)
        plants = [PlantState(plant_id="r00c00", row=0, col=0)]
        soil = [[[_make_voxel(0, 0)]]]
        state = FieldState(day=1, plants=plants, soil=soil,
                           elevation_grid=np.zeros((1, 1)), events_fired=[])
        # No cumulative_erosion_index_grid in state.custom
        env = EnvironmentState(
            day=1, doy=1, temp_max_c=30.0, temp_min_c=20.0, temp_mean_c=25.0,
            radiation_mj_m2=18.0, rainfall_mm=0.0, et0_mm=3.0,
            wind_speed_ms=2.0, humidity_pct=60.0,
        )
        log.record(field, state, env)
        row = log._soil_rows[0]
        assert row["cumulative_erosion_index"] == 0.0
        assert row["surface_runoff_mm_today"] == 0.0


# ---------------------------------------------------------------------------
# Integration: farm.run() with erosion=True writes parquet with observables
# ---------------------------------------------------------------------------

class TestFarmRunErosionParquet:
    """End-to-end: run 5 days with erosion=True on a sloped field, read back
    the soil parquet, and confirm cumulative_erosion_index > 0 on sloped cells."""

    def _run_farm(self) -> str:
        """Returns the session log_path (to the output directory)."""
        farm = Farm(name="ErosionParquetTest", location=(28.6, 77.2))
        field = Field("SlopedField", rows=2, cols=4)
        field.set_elevation(_steep_elevation())
        field.set_crop(Crop(species="Zea mays", variety="ParquetTest"))
        field.set_weather(_make_weather(days=5, rainfall_mm=30.0))
        farm.add_field(field)
        farm.use_physics(erosion=True)
        farm.run(days=5)
        return farm._last_log_path

    def test_soil_parquet_has_surface_runoff_column(self):
        log_path = self._run_farm()
        df = _read_parquet(Path(log_path) / "soil")
        assert "surface_runoff_mm_today" in df.columns, (
            "surface_runoff_mm_today column missing from soil parquet"
        )

    def test_soil_parquet_has_cumulative_erosion_column(self):
        log_path = self._run_farm()
        df = _read_parquet(Path(log_path) / "soil")
        assert "cumulative_erosion_index" in df.columns, (
            "cumulative_erosion_index column missing from soil parquet"
        )

    def test_cumulative_erosion_nonzero_on_sloped_cells_by_day5(self):
        """Sloped cells (col 0,1,2 have slope > 0) must accumulate erosion."""
        log_path = self._run_farm()
        df = _read_parquet(Path(log_path) / "soil")
        day5 = df[(df["day"] == 5) & (df["layer"] == 0)]
        # Columns 0-2 have positive slope; col 3 is lowest point (slope = 0)
        sloped = day5[day5["col"].isin([0, 1, 2])]
        assert (sloped["cumulative_erosion_index"] > 0.0).any(), (
            f"Expected non-zero cumulative erosion on sloped cells by day 5; "
            f"got: {sloped['cumulative_erosion_index'].describe()}"
        )

    def test_flat_cell_has_zero_erosion(self):
        """The lowest-elevation cell (col 3, slope=0) must have zero erosion."""
        log_path = self._run_farm()
        df = _read_parquet(Path(log_path) / "soil")
        day5 = df[(df["day"] == 5) & (df["layer"] == 0)]
        flat_cells = day5[day5["col"] == 3]["cumulative_erosion_index"]
        assert (flat_cells == 0.0).all(), (
            f"Expected zero erosion on flat cell (col=3); got: {flat_cells.values}"
        )

    def test_no_erosion_run_writes_zero_values(self):
        """farm.run() WITHOUT erosion=True must still write 0.0 (backward compat)."""
        farm = Farm(name="NoErosionParquet", location=(28.6, 77.2))
        field = Field("FlatField", rows=2, cols=2)
        field.set_crop(Crop(species="Triticum aestivum", variety="Compat"))
        field.set_weather(_make_weather(days=3, rainfall_mm=5.0))
        farm.add_field(field)
        # No use_physics(erosion=True) — legacy mode
        farm.run(days=3)
        log_path = farm._last_log_path
        df = _read_parquet(Path(log_path) / "soil")
        assert "cumulative_erosion_index" in df.columns
        assert (df["cumulative_erosion_index"] == 0.0).all(), (
            "Legacy run (no erosion engine) must write 0.0 for cumulative_erosion_index"
        )
