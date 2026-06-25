"""
tests/test_farm.py
==================
Tests for Field and Farm domain classes (PRD Section 6.1).

Covers:
  - Crop construction and validation
  - Field construction, dimension validation, elevation profiles
  - Field.set_crop / set_weather / set_soil / set_elevation
  - Farm construction
  - Farm.add_field (happy path + duplicate-name guard)
  - @farm.step decorator registration
  - Phase validation rules
  - farm.fields property

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
"""

import numpy as np
import pytest

from cropforge.crop import Crop
from cropforge.farm import Farm, Field


# ===========================================================================
# Crop
# ===========================================================================

class TestCrop:
    def test_minimal_construction(self):
        c = Crop(species="wheat")
        assert c.species == "wheat"
        assert c.variety == "generic"
        assert c.sowing_doy == 1

    def test_all_fields(self):
        c = Crop(species="maize", variety="custom", sowing_doy=120)
        assert c.species == "maize"
        assert c.variety == "custom"
        assert c.sowing_doy == 120

    def test_custom_dict_isolated(self):
        c1 = Crop(species="wheat")
        c2 = Crop(species="maize")
        c1.custom["base_temp"] = 0.0
        assert "base_temp" not in c2.custom

    def test_empty_species_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Crop(species="")

    def test_whitespace_species_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Crop(species="   ")

    def test_sowing_doy_lower_bound(self):
        c = Crop(species="wheat", sowing_doy=1)
        assert c.sowing_doy == 1

    def test_sowing_doy_upper_bound(self):
        c = Crop(species="wheat", sowing_doy=366)
        assert c.sowing_doy == 366

    def test_sowing_doy_out_of_range_low(self):
        with pytest.raises(ValueError, match="sowing_doy"):
            Crop(species="wheat", sowing_doy=0)

    def test_sowing_doy_out_of_range_high(self):
        with pytest.raises(ValueError, match="sowing_doy"):
            Crop(species="wheat", sowing_doy=367)

    def test_repr(self):
        c = Crop(species="wheat", variety="HD-2967", sowing_doy=290)
        r = repr(c)
        assert "wheat" in r
        assert "HD-2967" in r
        assert "290" in r


# ===========================================================================
# Field
# ===========================================================================

class TestField:
    def test_minimal_construction(self):
        f = Field(name="Plot A", rows=20, cols=30)
        assert f.name == "Plot A"
        assert f.rows == 20
        assert f.cols == 30
        assert f.area_ha == 1.0

    def test_elevation_grid_shape(self):
        f = Field(name="X", rows=5, cols=8)
        assert f.elevation_grid.shape == (5, 8)

    def test_elevation_grid_flat_default(self):
        f = Field(name="X", rows=3, cols=4)
        assert np.all(f.elevation_grid == 0.0)

    def test_elevation_profile_slope_1pct_N(self):
        f = Field(name="X", rows=5, cols=3, elevation_profile="slope_1pct_N")
        # row 0 = 0.0 m, row 1 = 0.01 m, row 4 = 0.04 m
        assert f.elevation_grid[0, 0] == pytest.approx(0.0)
        assert f.elevation_grid[1, 0] == pytest.approx(0.01)
        assert f.elevation_grid[4, 0] == pytest.approx(0.04)

    def test_elevation_profile_slope_2pct_N(self):
        f = Field(name="X", rows=4, cols=2, elevation_profile="slope_2pct_N")
        assert f.elevation_grid[3, 0] == pytest.approx(0.06)

    def test_elevation_profile_numpy_array(self):
        dem = np.ones((4, 5)) * 0.5
        f = Field(name="X", rows=4, cols=5, elevation_profile=dem)
        assert f.elevation_grid[2, 3] == pytest.approx(0.5)

    def test_elevation_profile_numpy_wrong_shape_raises(self):
        dem = np.ones((3, 3))
        with pytest.raises(ValueError, match="shape"):
            Field(name="X", rows=4, cols=5, elevation_profile=dem)

    def test_elevation_profile_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown elevation_profile"):
            Field(name="X", rows=3, cols=3, elevation_profile="magic_slope")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Field(name="", rows=5, cols=5)

    def test_zero_rows_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            Field(name="X", rows=0, cols=5)

    def test_zero_cols_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            Field(name="X", rows=5, cols=0)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Field(name="X", rows=5, cols=5, area_ha=-1.0)

    def test_set_crop(self):
        f = Field(name="X", rows=5, cols=5)
        crop = Crop(species="wheat")
        f.set_crop(crop)
        assert f.crop is crop

    def test_set_crop_wrong_type_raises(self):
        f = Field(name="X", rows=5, cols=5)
        with pytest.raises(TypeError, match="Crop instance"):
            f.set_crop("wheat")  # type: ignore[arg-type]

    def test_set_weather(self):
        f = Field(name="X", rows=5, cols=5)
        sentinel = object()
        f.set_weather(sentinel)
        assert f.weather is sentinel

    def test_set_soil(self):
        f = Field(name="X", rows=5, cols=5)
        sentinel = object()
        f.set_soil(sentinel)
        assert f.soil_profile is sentinel

    def test_set_elevation(self):
        f = Field(name="X", rows=4, cols=4)
        dem = np.full((4, 4), 0.1)
        f.set_elevation(dem)
        assert f.elevation_grid[0, 0] == pytest.approx(0.1)

    def test_set_elevation_wrong_shape_raises(self):
        f = Field(name="X", rows=4, cols=4)
        with pytest.raises(ValueError, match="shape"):
            f.set_elevation(np.zeros((3, 4)))

    def test_set_elevation_non_array_raises(self):
        f = Field(name="X", rows=4, cols=4)
        with pytest.raises(TypeError, match="ndarray"):
            f.set_elevation([[0, 1], [2, 3]])  # type: ignore[arg-type]

    def test_elevation_from_csv_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Field.elevation_from_csv("some_file.csv")

    def test_initial_field_state_plant_count(self):
        f = Field(name="X", rows=10, cols=15)
        state = f._init_field_state(day=1)
        assert len(state.plants) == 150

    def test_initial_field_state_plant_ids(self):
        f = Field(name="X", rows=3, cols=4)
        state = f._init_field_state(day=1)
        ids = [p.plant_id for p in state.plants]
        assert "r00c00" in ids
        assert "r02c03" in ids

    def test_initial_field_state_all_alive(self):
        f = Field(name="X", rows=3, cols=3)
        state = f._init_field_state(day=1)
        assert all(p.alive for p in state.plants)

    def test_initial_soil_default_layer(self):
        f = Field(name="X", rows=2, cols=2)
        state = f._init_field_state(day=1)
        # Default: 1 layer per cell
        assert len(state.soil[0][0]) == 1
        topsoil = state.soil[0][0][0]
        assert topsoil.layer == 0
        assert topsoil.depth_top_cm == 0.0
        assert topsoil.depth_bottom_cm == 20.0


# ===========================================================================
# Farm
# ===========================================================================

class TestFarm:
    def test_minimal_construction(self):
        farm = Farm(name="Trial 2026-A")
        assert farm.name == "Trial 2026-A"
        assert farm.location == (0.0, 0.0)

    def test_location(self):
        farm = Farm(name="Test", location=(23.4, 85.3))
        assert farm.location == (23.4, 85.3)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Farm(name="")

    def test_add_field(self):
        farm = Farm(name="F")
        field = Field(name="Plot A", rows=5, cols=5)
        farm.add_field(field)
        assert len(farm.fields) == 1
        assert farm.fields[0].name == "Plot A"

    def test_add_multiple_fields(self):
        farm = Farm(name="F")
        farm.add_field(Field(name="Plot A", rows=5, cols=5))
        farm.add_field(Field(name="Plot B", rows=5, cols=5))
        assert len(farm.fields) == 2

    def test_duplicate_field_name_raises(self):
        farm = Farm(name="F")
        farm.add_field(Field(name="Plot A", rows=5, cols=5))
        with pytest.raises(ValueError, match="already exists"):
            farm.add_field(Field(name="Plot A", rows=10, cols=10))

    def test_add_field_wrong_type_raises(self):
        farm = Farm(name="F")
        with pytest.raises(TypeError, match="Field instance"):
            farm.add_field("not a field")  # type: ignore[arg-type]

    def test_fields_property_is_copy(self):
        """Mutating the returned list must not affect the farm's internal list."""
        farm = Farm(name="F")
        farm.add_field(Field(name="Plot A", rows=5, cols=5))
        fields_copy = farm.fields
        fields_copy.clear()
        assert len(farm.fields) == 1

    def test_repr(self):
        farm = Farm(name="Trial", location=(1.0, 2.0))
        farm.add_field(Field(name="Plot A", rows=5, cols=5))
        r = repr(farm)
        assert "Trial" in r
        assert "Plot A" in r

    def test_add_event(self):
        """add_event accepts any object without error (full Event in Phase 1)."""
        farm = Farm(name="F")
        sentinel = object()
        farm.add_event(sentinel)
        assert sentinel in farm._events

    def test_visualize_raises_when_no_log(self):
        """farm.visualize() must raise CropForgeVisualizeError if no log exists."""
        from cropforge.runtime import CropForgeVisualizeError
        farm = Farm(name="F")
        with pytest.raises(CropForgeVisualizeError, match="No valid simulation log"):
            farm.visualize()


# ===========================================================================
# @farm.step decorator
# ===========================================================================

class TestStepDecorator:
    def _make_farm(self):
        return Farm(name="TestFarm")

    def test_register_single_step(self):
        farm = self._make_farm()

        @farm.step(interval="daily", phase=1)
        def my_step(state, env):
            return state

        assert len(farm._step_registry) == 1
        assert farm._step_registry[0][0] == 1
        assert farm._step_registry[0][1] is my_step

    def test_decorator_returns_original_function(self):
        farm = self._make_farm()

        @farm.step(interval="daily", phase=1)
        def my_step(state, env):
            return state

        # The decorator must not wrap; my_step should still be callable directly
        assert callable(my_step)
        assert my_step.__name__ == "my_step"

    def test_phase_default_is_zero(self):
        farm = self._make_farm()

        @farm.step(interval="daily")
        def unphased_step(state, env):
            return state

        assert farm._step_registry[0][0] == 0

    def test_negative_phase_raises(self):
        farm = self._make_farm()
        with pytest.raises(ValueError, match="non-negative integer"):
            @farm.step(interval="daily", phase=-1)
            def bad_step(state, env):
                return state

    def test_non_integer_phase_raises(self):
        farm = self._make_farm()
        with pytest.raises(ValueError, match="non-negative integer"):
            @farm.step(interval="daily", phase=1.5)  # type: ignore[arg-type]
            def bad_step(state, env):
                return state

    def test_invalid_interval_raises(self):
        farm = self._make_farm()
        with pytest.raises(ValueError, match="interval"):
            @farm.step(interval="weekly", phase=1)
            def bad_step(state, env):
                return state

    def test_multiple_steps_registered(self):
        farm = self._make_farm()

        @farm.step(interval="daily", phase=2)
        def step_a(state, env): return state

        @farm.step(interval="daily", phase=1)
        def step_b(state, env): return state

        @farm.step(interval="daily", phase=3)
        def step_c(state, env): return state

        assert len(farm._step_registry) == 3

    def test_sorted_steps_ascending_phase(self):
        farm = self._make_farm()

        @farm.step(interval="daily", phase=3)
        def step_c(state, env): return state

        @farm.step(interval="daily", phase=1)
        def step_a(state, env): return state

        @farm.step(interval="daily", phase=2)
        def step_b(state, env): return state

        sorted_steps = farm._sorted_steps()
        phases = [p for p, _ in sorted_steps]
        assert phases == sorted(phases)
        assert phases == [1, 2, 3]

    def test_sorted_steps_non_contiguous_phases(self):
        """PRD Section 6.2: phase=1 and phase=10 are valid; 1 runs first."""
        farm = self._make_farm()

        @farm.step(interval="daily", phase=10)
        def late(state, env): return state

        @farm.step(interval="daily", phase=1)
        def early(state, env): return state

        sorted_steps = farm._sorted_steps()
        assert sorted_steps[0][0] == 1
        assert sorted_steps[1][0] == 10

    def test_phase_conflict_emits_warning(self, caplog):
        """PRD Section 6.2: duplicate phase values → warning at run start."""
        import logging
        farm = self._make_farm()

        @farm.step(interval="daily", phase=1)
        def step_a(state, env): return state

        @farm.step(interval="daily", phase=1)
        def step_b(state, env): return state

        with caplog.at_level(logging.WARNING, logger="cropforge.farm"):
            farm._sorted_steps()

        assert any("phase conflict" in record.message.lower() for record in caplog.records)
        assert any("phase=1" in record.message for record in caplog.records)

    def test_unphased_conflict_emits_warning(self, caplog):
        """PRD Section 6.2: multiple unphased steps (phase=0) → warning."""
        import logging
        farm = self._make_farm()

        @farm.step(interval="daily")
        def step_a(state, env): return state

        @farm.step(interval="daily")
        def step_b(state, env): return state

        with caplog.at_level(logging.WARNING, logger="cropforge.farm"):
            farm._sorted_steps()

        assert any("phase conflict" in record.message.lower() for record in caplog.records)

    def test_no_conflict_no_warning_when_unique_phases(self, caplog):
        """Unique phase values must NOT produce a warning."""
        import logging
        farm = self._make_farm()

        @farm.step(interval="daily", phase=1)
        def step_a(state, env): return state

        @farm.step(interval="daily", phase=2)
        def step_b(state, env): return state

        with caplog.at_level(logging.WARNING, logger="cropforge.farm"):
            farm._sorted_steps()

        conflict_warnings = [
            r for r in caplog.records
            if "phase conflict" in r.message.lower()
        ]
        assert conflict_warnings == []
