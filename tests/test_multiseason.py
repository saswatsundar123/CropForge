"""
tests/test_multiseason.py
=========================
Multi-Season Carry-Over tests — PRD v0.4.0 Section 7.

Tests PRD §7.6 acceptance criteria:
  1. farm.save_state() writes valid JSON with correct structure
  2. farm.load_state() restores SoilState exactly — all fields, all cells, all layers
  3. THE CRUCIBLE: Season 2 day-1 SoilState matches Season 1 final-day SoilState to 6 dp
  4. PlantState resets to initial defaults after prepare_next_season()
  5. Season counter increments correctly, day_offset is continuous
  6. load_state() with wrong cropforge_version raises CropForgeStateError
  7. load_state() with wrong field_name raises CropForgeStateError
  8. save_state() / load_state() round-trip: float values preserved to 6 decimal places

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import cropforge
from cropforge import Farm, Field, CropForgeStateError
from cropforge.state import SoilVoxelState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_farm(
    farm_name: str = "TestFarm",
    field_name: str = "PlotA",
    rows: int = 2,
    cols: int = 2,
    moisture_pct: float = 28.4,
    nitrogen_kg_ha: float = 142.1,
):
    """Build a minimal Farm+Field with controlled soil state for testing."""
    farm = Farm(name=farm_name)
    field = Field(name=field_name, rows=rows, cols=cols)
    farm.add_field(field)

    @farm.step(interval="daily")
    def _noop(state, env):
        pass

    return farm, field


# ---------------------------------------------------------------------------
# Test 1: save_state() writes valid JSON with correct structure
# ---------------------------------------------------------------------------

class TestSaveState:

    def test_save_creates_file(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=3)
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)
        assert Path(out).exists()

    def test_save_writes_valid_json(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=3)
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)

    def test_save_contains_required_keys(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=3)
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        for key in ("cropforge_version", "season", "final_day", "fields"):
            assert key in data

    def test_save_version_matches_running_version(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=3)
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["cropforge_version"] == cropforge.__version__

    def test_save_field_name_correct(self, tmp_path):
        farm, field = _make_farm(field_name="Plot99")
        farm.run(days=3)
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        names = [f["field_name"] for f in data["fields"]]
        assert "Plot99" in names

    def test_save_soil_voxels_present(self, tmp_path):
        farm, field = _make_farm(rows=2, cols=2)
        farm.run(days=3)
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data["fields"][0]["soil"]) >= 4

    def test_save_voxel_has_required_fields(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=3)
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        voxel = data["fields"][0]["soil"][0]
        for key in ("row", "col", "layer", "moisture_pct", "nitrogen_kg_ha",
                    "bulk_density", "penetration_resistance"):
            assert key in voxel


# ---------------------------------------------------------------------------
# Test 2: load_state() restores SoilState exactly
# ---------------------------------------------------------------------------

class TestLoadState:

    def test_load_restores_moisture_exactly(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=5)
        end_moisture = field._field_state.soil[0][0][0].moisture_pct
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)

        farm2, field2 = _make_farm(farm_name="F2")
        farm2.load_state(out)
        restored = field2._field_state.soil[0][0][0].moisture_pct
        assert restored == pytest.approx(end_moisture, abs=1e-9)

    def test_load_restores_nitrogen_exactly(self, tmp_path):
        farm, field = _make_farm(nitrogen_kg_ha=142.1)
        farm.run(days=5)
        end_n = field._field_state.soil[0][0][0].nitrogen_kg_ha
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)

        farm2, field2 = _make_farm(farm_name="F2")
        farm2.load_state(out)
        assert field2._field_state.soil[0][0][0].nitrogen_kg_ha == pytest.approx(end_n, abs=1e-9)

    def test_load_restores_all_cells(self, tmp_path):
        farm, field = _make_farm(rows=2, cols=2)
        farm.run(days=3)
        out = str(tmp_path / "state.cfstate")
        farm.save_state(out)

        farm2, field2 = _make_farm(farm_name="F2", rows=2, cols=2)
        farm2.load_state(out)

        for r in range(2):
            for c in range(2):
                orig = field._field_state.soil[r][c][0]
                rest = field2._field_state.soil[r][c][0]
                assert rest.moisture_pct == pytest.approx(orig.moisture_pct, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 3: THE CRUCIBLE
# ---------------------------------------------------------------------------

class TestMultiSeasonCrucible:
    """
    PRD v0.4.0 SUCCESS CRITERION:
    Season 2 day-1 SoilState matches Season 1 final-day SoilState to 6 decimal places.
    """

    def test_soil_moisture_preserved_to_6_decimal_places(self, tmp_path):
        """
        CORE CRUCIBLE:
        1. Run Season 1 for 10 days.
        2. Save state.
        3. prepare_next_season().
        4. load_state().
        5. Before Season 2 runs: moisture == Season 1 final moisture to 6 dp.
        """
        farm, field = _make_farm(moisture_pct=28.456789)
        farm.run(days=10)

        s1_moisture = field._field_state.soil[0][0][0].moisture_pct
        s1_nitrogen = field._field_state.soil[0][0][0].nitrogen_kg_ha

        out = str(tmp_path / "s1.cfstate")
        farm.save_state(out)
        farm.prepare_next_season()
        farm.load_state(out)

        s2_moisture = field._field_state.soil[0][0][0].moisture_pct
        s2_nitrogen = field._field_state.soil[0][0][0].nitrogen_kg_ha

        assert round(s2_moisture, 6) == round(s1_moisture, 6), (
            f"CRUCIBLE FAILED: s2 moisture {s2_moisture:.8f} != s1 {s1_moisture:.8f}"
        )
        assert round(s2_nitrogen, 6) == round(s1_nitrogen, 6), (
            f"CRUCIBLE FAILED: s2 nitrogen {s2_nitrogen:.8f} != s1 {s1_nitrogen:.8f}"
        )

    def test_prepare_next_season_preserves_soil_in_memory(self):
        """Without save/load: prepare_next_season() must not alter soil."""
        farm, field = _make_farm(moisture_pct=35.7)
        farm.run(days=5)

        s1_moisture = field._field_state.soil[0][0][0].moisture_pct
        s1_obj_id   = id(field._field_state.soil[0][0][0])

        farm.prepare_next_season()

        assert field._field_state.soil[0][0][0].moisture_pct == s1_moisture
        assert id(field._field_state.soil[0][0][0]) == s1_obj_id

    def test_all_voxels_preserved_across_season_boundary(self, tmp_path):
        farm, field = _make_farm(rows=2, cols=2, moisture_pct=28.5)
        farm.run(days=5)

        s1_moistures = {
            (v.row, v.col, v.layer): v.moisture_pct
            for row_list in field._field_state.soil
            for col_list in row_list
            for v in col_list
        }

        out = str(tmp_path / "s1.cfstate")
        farm.save_state(out)
        farm.prepare_next_season()
        farm.load_state(out)

        for row_list in field._field_state.soil:
            for col_list in row_list:
                for voxel in col_list:
                    key = (voxel.row, voxel.col, voxel.layer)
                    assert round(voxel.moisture_pct, 6) == round(s1_moistures[key], 6)


# ---------------------------------------------------------------------------
# Test 4: PlantState resets
# ---------------------------------------------------------------------------

class TestPlantStateReset:

    def test_biomass_lai_age_stress_reset(self):
        farm, field = _make_farm()
        farm.run(days=5)
        p = field._field_state.plants[0]
        p.biomass_g = 250.0
        p.lai = 2.5
        p.age_days = 30
        p.stress_index = 0.4

        farm.prepare_next_season()

        p2 = field._field_state.plants[0]
        assert p2.biomass_g == 0.0
        assert p2.lai == 0.0
        assert p2.age_days == 0
        assert p2.stress_index == 0.0

    def test_all_plants_alive_after_reset(self):
        farm, field = _make_farm()
        farm.run(days=3)
        for p in field._field_state.plants:
            p.alive = False
        farm.prepare_next_season()
        for p in field._field_state.plants:
            assert p.alive is True

    def test_plant_count_unchanged(self):
        farm, field = _make_farm(rows=3, cols=3)
        farm.run(days=3)
        n = len(field._field_state.plants)
        farm.prepare_next_season()
        assert len(field._field_state.plants) == n == 9

    def test_soil_not_touched_when_plants_reset(self):
        farm, field = _make_farm(moisture_pct=33.3, nitrogen_kg_ha=88.8)
        farm.run(days=5)
        m = field._field_state.soil[0][0][0].moisture_pct
        n = field._field_state.soil[0][0][0].nitrogen_kg_ha
        farm.prepare_next_season()
        assert field._field_state.soil[0][0][0].moisture_pct == m
        assert field._field_state.soil[0][0][0].nitrogen_kg_ha == n


# ---------------------------------------------------------------------------
# Test 5: Season counter and day offset
# ---------------------------------------------------------------------------

class TestSeasonCounter:

    def test_initial_season_is_1(self):
        farm = Farm(name="F")
        assert farm._current_season == 1

    def test_prepare_increments_season(self):
        farm, field = _make_farm()
        farm.run(days=3)
        farm.prepare_next_season()
        assert farm._current_season == 2

    def test_multiple_seasons(self):
        farm, field = _make_farm()
        farm.run(days=3)
        farm.prepare_next_season()
        farm.run(days=3)
        farm.prepare_next_season()
        assert farm._current_season == 3

    def test_day_offset_tracks_total_days(self):
        farm, field = _make_farm()
        farm.run(days=10)
        assert farm._day_offset == 10
        farm.prepare_next_season()
        farm.run(days=7)
        assert farm._day_offset == 17


# ---------------------------------------------------------------------------
# Test 6 & 7: CropForgeStateError
# ---------------------------------------------------------------------------

class TestCropForgeStateError:

    def test_wrong_version_raises(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=3)
        out = str(tmp_path / "s.cfstate")
        farm.save_state(out)

        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        data["cropforge_version"] = "0.0.1"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

        farm2, field2 = _make_farm(farm_name="F2")
        with pytest.raises(CropForgeStateError) as exc_info:
            farm2.load_state(out)
        assert "0.0.1" in str(exc_info.value)

    def test_wrong_field_name_raises(self, tmp_path):
        farm, field = _make_farm(field_name="Plot_A")
        farm.run(days=3)
        out = str(tmp_path / "s.cfstate")
        farm.save_state(out)

        farm2, field2 = _make_farm(farm_name="F2", field_name="Plot_B")
        with pytest.raises(CropForgeStateError):
            farm2.load_state(out)

    def test_state_error_has_path_attribute(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=3)
        out = str(tmp_path / "s.cfstate")
        farm.save_state(out)

        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        data["cropforge_version"] = "99.99.99"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

        farm2, field2 = _make_farm(farm_name="F2")
        with pytest.raises(CropForgeStateError) as exc_info:
            farm2.load_state(out)
        assert exc_info.value.path == out

    def test_state_error_is_importable(self):
        from cropforge import CropForgeStateError as CSE
        assert issubclass(CSE, ValueError)


# ---------------------------------------------------------------------------
# Test 8: Float precision round-trip
# ---------------------------------------------------------------------------

class TestRoundTripPrecision:

    def test_moisture_6dp(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=1)
        voxel = field._field_state.soil[0][0][0]
        precise = 28.456789123456
        voxel.moisture_pct = precise
        out = str(tmp_path / "p.cfstate")
        farm.save_state(out)

        farm2, field2 = _make_farm(farm_name="F2")
        farm2.load_state(out)
        restored = field2._field_state.soil[0][0][0].moisture_pct
        assert round(restored, 6) == round(precise, 6)

    def test_nitrogen_6dp(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=1)
        precise = 142.987654321
        field._field_state.soil[0][0][0].nitrogen_kg_ha = precise
        out = str(tmp_path / "n.cfstate")
        farm.save_state(out)

        farm2, field2 = _make_farm(farm_name="F2")
        farm2.load_state(out)
        restored = field2._field_state.soil[0][0][0].nitrogen_kg_ha
        assert round(restored, 6) == round(precise, 6)

    def test_bulk_density_and_pen_resistance_6dp(self, tmp_path):
        farm, field = _make_farm()
        farm.run(days=1)
        voxel = field._field_state.soil[0][0][0]
        voxel.bulk_density = 1.234567
        voxel.penetration_resistance = 0.987654
        out = str(tmp_path / "bd.cfstate")
        farm.save_state(out)

        farm2, field2 = _make_farm(farm_name="F2")
        farm2.load_state(out)
        v2 = field2._field_state.soil[0][0][0]
        assert round(v2.bulk_density, 6)           == round(voxel.bulk_density, 6)
        assert round(v2.penetration_resistance, 6) == round(voxel.penetration_resistance, 6)
