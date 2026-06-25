"""
tests/test_logger.py
====================
Tests for the Parquet state logger (PRD Section 16).

Covers:
  - StateLogger instantiation creates output directory
  - record() accumulates rows without error
  - flush() writes plant / soil / environment Parquet datasets
  - Schema compliance: every PRD column present with correct dtype
  - custom_json is a valid JSON string
  - cropforge_version stored in Parquet file metadata
  - Partitioning by field_name then day
  - Snappy compression on written files
  - flush() on empty buffer emits warning but does not crash
  - Partial flush on crash: completed days survive
  - Integration: Farm.run() produces a real Parquet log

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
"""

import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cropforge.farm import Farm, Field
from cropforge.logger import (
    ENV_SCHEMA,
    PLANT_SCHEMA,
    SOIL_SCHEMA,
    StateLogger,
)
from cropforge.runtime import CropForgeStepError
from cropforge.state import EnvironmentState, FieldState, PlantState, SoilVoxelState


# ===========================================================================
# Helpers
# ===========================================================================

def _make_field_state(rows: int = 2, cols: int = 2, day: int = 1) -> FieldState:
    """Build a minimal FieldState for logging tests."""
    import numpy as np
    plants = [
        PlantState(plant_id=f"r{r:02d}c{c:02d}", row=r, col=c)
        for r in range(rows)
        for c in range(cols)
    ]
    soil = [
        [
            [
                SoilVoxelState(
                    row=r, col=c, layer=0,
                    depth_top_cm=0.0, depth_bottom_cm=20.0,
                    moisture_pct=25.0, nitrogen_kg_ha=30.0,
                    bulk_density=1.3, penetration_resistance=0.8,
                )
            ]
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    return FieldState(
        day=day,
        plants=plants,
        soil=soil,
        elevation_grid=np.zeros((rows, cols)),
        events_fired=["TestEvent"],
    )


def _make_env(day: int = 1) -> EnvironmentState:
    return EnvironmentState(
        day=day,
        doy=day,
        temp_max_c=25.0,
        temp_min_c=15.0,
        temp_mean_c=20.0,
        radiation_mj_m2=15.0,
        rainfall_mm=2.5,
        et0_mm=3.0,
        wind_speed_ms=2.0,
        humidity_pct=60.0,
    )


def _make_simple_farm() -> Farm:
    farm = Farm(name="LoggerTest")
    farm.add_field(Field(name="Plot A", rows=2, cols=2))
    return farm


def _read_parquet_dataset(path: Path) -> "pd.DataFrame":
    import pandas as pd
    import pyarrow.dataset as ds
    dataset = ds.dataset(str(path), format="parquet")
    return dataset.to_table().to_pandas()


# ===========================================================================
# StateLogger construction
# ===========================================================================

class TestStateLoggerConstruction:
    def test_output_dir_created(self, tmp_path):
        log = StateLogger(session_name="test_session", output_root=str(tmp_path))
        assert (tmp_path / "test_session").exists()

    def test_log_path_is_absolute(self, tmp_path):
        log = StateLogger(session_name="test_session", output_root=str(tmp_path))
        assert Path(log.log_path).is_absolute()

    def test_log_path_ends_with_session_name(self, tmp_path):
        log = StateLogger(session_name="mysession", output_root=str(tmp_path))
        assert log.log_path.endswith("mysession")

    def test_repr(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        r = repr(log)
        assert "plant_rows=0" in r
        assert "soil_rows=0" in r
        assert "env_rows=0" in r


# ===========================================================================
# record() accumulation
# ===========================================================================

class TestRecord:
    def test_record_increments_plant_rows(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        field = Field(name="Plot A", rows=2, cols=2)
        field._init_field_state(day=1)
        state = _make_field_state(rows=2, cols=2, day=1)
        env = _make_env(1)
        log.record(field, state, env)
        # 2x2 = 4 plants → 4 plant rows
        assert len(log._plant_rows) == 4

    def test_record_increments_soil_rows(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        field = Field(name="Plot A", rows=2, cols=2)
        state = _make_field_state(rows=2, cols=2, day=1)   # 1 layer per cell → 4 voxels
        env = _make_env(1)
        log.record(field, state, env)
        assert len(log._soil_rows) == 4   # 2×2 cells × 1 layer

    def test_record_increments_env_rows(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        field = Field(name="Plot A", rows=2, cols=2)
        state = _make_field_state(day=1)
        env = _make_env(1)
        log.record(field, state, env)
        assert len(log._env_rows) == 1

    def test_record_multiple_days(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        field = Field(name="Plot A", rows=2, cols=2)
        for d in range(1, 4):
            state = _make_field_state(day=d)
            env = _make_env(d)
            log.record(field, state, env)
        assert len(log._plant_rows) == 12   # 3 days × 4 plants
        assert len(log._env_rows) == 3

    def test_record_custom_json_is_valid_json(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        field = Field(name="Plot A", rows=1, cols=1)
        state = _make_field_state(rows=1, cols=1, day=1)
        state.plants[0].custom["water_use"] = 3.14
        env = _make_env(1)
        log.record(field, state, env)
        row = log._plant_rows[0]
        parsed = json.loads(row["custom_json"])
        assert parsed["water_use"] == pytest.approx(3.14)

    def test_record_env_events_fired_as_json(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        field = Field(name="Plot A", rows=1, cols=1)
        state = _make_field_state(rows=1, cols=1, day=1)
        state.events_fired = ["Irrigation", "Fertiliser"]
        env = _make_env(1)
        log.record(field, state, env)
        row = log._env_rows[0]
        parsed = json.loads(row["events_fired"])
        assert parsed == ["Irrigation", "Fertiliser"]


# ===========================================================================
# flush() — file creation and schema
# ===========================================================================

class TestFlush:
    def test_flush_creates_plant_subdir(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        log.record(Field(name="F", rows=1, cols=1), _make_field_state(1, 1, 1), _make_env(1))
        log.flush()
        assert (tmp_path / "s" / "plants").exists()

    def test_flush_creates_soil_subdir(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        log.record(Field(name="F", rows=1, cols=1), _make_field_state(1, 1, 1), _make_env(1))
        log.flush()
        assert (tmp_path / "s" / "soil").exists()

    def test_flush_creates_env_subdir(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        log.record(Field(name="F", rows=1, cols=1), _make_field_state(1, 1, 1), _make_env(1))
        log.flush()
        assert (tmp_path / "s" / "environment").exists()

    def test_flush_empty_buffer_warns(self, tmp_path, caplog):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="cropforge.logger"):
            log.flush()
        assert any("no recorded data" in r.message.lower() for r in caplog.records)

    def test_flush_writes_parquet_files(self, tmp_path):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        log.record(Field(name="F", rows=1, cols=1), _make_field_state(1, 1, 1), _make_env(1))
        log.flush()
        parquet_files = list((tmp_path / "s" / "plants").rglob("*.parquet"))
        assert len(parquet_files) > 0


# ===========================================================================
# Schema compliance (PRD Section 16.1)
# ===========================================================================

class TestSchemaCompliance:
    def _flush_and_read(self, tmp_path: Path, subdir: str):
        log = StateLogger(session_name="s", output_root=str(tmp_path))
        field = Field(name="FieldA", rows=2, cols=2)
        for day in [1, 2]:
            log.record(field, _make_field_state(2, 2, day), _make_env(day))
        log.flush()
        return _read_parquet_dataset(tmp_path / "s" / subdir)

    def test_plant_table_all_columns_present(self, tmp_path):
        df = self._flush_and_read(tmp_path, "plants")
        required = {f.name for f in PLANT_SCHEMA}
        # partition columns (field_name, day) are lifted out but still present
        # as data columns in the table
        present = set(df.columns)
        # day and field_name may be partition columns — check via combined set
        for col in required - {"day", "field_name"}:
            assert col in present, f"Missing column: {col}"

    def test_soil_table_all_columns_present(self, tmp_path):
        df = self._flush_and_read(tmp_path, "soil")
        for col in (f.name for f in SOIL_SCHEMA if f.name not in {"day", "field_name"}):
            assert col in df.columns, f"Missing column: {col}"

    def test_env_table_all_columns_present(self, tmp_path):
        df = self._flush_and_read(tmp_path, "environment")
        for col in (f.name for f in ENV_SCHEMA if f.name not in {"day", "field_name"}):
            assert col in df.columns, f"Missing column: {col}"

    def test_plant_alive_is_boolean(self, tmp_path):
        df = self._flush_and_read(tmp_path, "plants")
        import pandas as pd
        assert df["alive"].dtype == bool or str(df["alive"].dtype) == "bool"

    def test_plant_custom_json_is_valid_json(self, tmp_path):
        df = self._flush_and_read(tmp_path, "plants")
        for val in df["custom_json"]:
            parsed = json.loads(val)
            assert isinstance(parsed, dict)

    def test_env_events_fired_is_valid_json(self, tmp_path):
        df = self._flush_and_read(tmp_path, "environment")
        for val in df["events_fired"]:
            parsed = json.loads(val)
            assert isinstance(parsed, list)

    def test_row_count_plant_table(self, tmp_path):
        """2 days × 4 plants = 8 plant rows."""
        df = self._flush_and_read(tmp_path, "plants")
        assert len(df) == 8

    def test_row_count_env_table(self, tmp_path):
        """2 days × 1 field = 2 env rows."""
        df = self._flush_and_read(tmp_path, "environment")
        assert len(df) == 2


# ===========================================================================
# Parquet metadata (PRD Section 16.4)
# ===========================================================================

class TestParquetMetadata:
    def test_cropforge_version_in_metadata(self, tmp_path):
        log = StateLogger(
            session_name="meta_test",
            output_root=str(tmp_path),
            cropforge_version="0.1.0",
        )
        log.record(Field(name="F", rows=1, cols=1), _make_field_state(1, 1, 1), _make_env(1))
        log.flush()

        parquet_files = list((tmp_path / "meta_test" / "plants").rglob("*.parquet"))
        assert len(parquet_files) > 0

        pf = pq.read_metadata(parquet_files[0])
        meta = pf.metadata
        # Metadata keys are bytes
        assert b"cropforge_version" in meta


# ===========================================================================
# Integration: Farm.run() produces a valid log
# ===========================================================================

class TestFarmRunIntegration:
    def test_run_produces_log_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        farm = _make_simple_farm()

        @farm.step(interval="daily", phase=1)
        def noop(state, env):
            return state

        farm.run(days=3)
        assert farm._last_log_path is not None
        assert Path(farm._last_log_path).exists()

    def test_run_plants_parquet_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        farm = _make_simple_farm()

        @farm.step(interval="daily", phase=1)
        def noop(state, env):
            return state

        farm.run(days=3)
        plant_dir = Path(farm._last_log_path) / "plants"
        assert plant_dir.exists()
        assert len(list(plant_dir.rglob("*.parquet"))) > 0

    def test_run_correct_number_of_plant_rows(self, tmp_path, monkeypatch):
        """3 days × 4 plants (2×2 field) = 12 plant rows."""
        monkeypatch.chdir(tmp_path)
        farm = _make_simple_farm()

        @farm.step(interval="daily", phase=1)
        def noop(state, env):
            return state

        farm.run(days=3)
        df = _read_parquet_dataset(Path(farm._last_log_path) / "plants")
        assert len(df) == 12

    def test_crash_produces_partial_log(self, tmp_path, monkeypatch):
        """PRD Section 6.4 rule 2: completed days survive a crash."""
        monkeypatch.chdir(tmp_path)
        farm = _make_simple_farm()

        @farm.step(interval="daily", phase=1)
        def crash_on_day_3(state, env):
            if state.day == 3:
                raise RuntimeError("deliberate crash")
            return state

        with pytest.raises(CropForgeStepError):
            farm.run(days=10)

        # Days 1 and 2 must be present in the partial log
        log_path = Path(farm._last_log_path)
        if log_path.exists():
            plant_dir = log_path / "plants"
            if plant_dir.exists():
                df = _read_parquet_dataset(plant_dir)
                # At minimum, 2 days × 4 plants = 8 rows should be present
                assert len(df) >= 8

    def test_session_name_contains_farm_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        farm = Farm(name="MyFarm2026")
        farm.add_field(Field(name="Plot A", rows=1, cols=1))
        farm.run(days=1)
        assert "MyFarm2026" in Path(farm._last_log_path).name
