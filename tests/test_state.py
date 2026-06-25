"""
tests/test_state.py
===================
Robust tests for the CropForge SimulationState schema (PRD Section 5).

Each dataclass is tested for:
  - Required-field construction
  - All default values
  - The ``custom`` dict extensibility hook
  - Type integrity (no accidental coercions)
  - Mutation (researchers mutate state in step functions; this must work)
  - Edge cases: empty grids, zero-size arrays, boundary numeric values

Sections map directly to PRD subsections:
  - 5.1  PlantState
  - 5.2  SoilVoxelState
  - 5.3  FieldState
  - 5.4  EnvironmentState
"""

import math

import numpy as np
import pytest

from cropforge.state import (
    EnvironmentState,
    FieldState,
    PlantState,
    SoilVoxelState,
)


# ===========================================================================
# 5.1  PlantState
# ===========================================================================

class TestPlantState:
    """Tests for PlantState (PRD Section 5.1)."""

    def test_construction_with_required_fields(self):
        """PlantState accepts plant_id, row, col as positional arguments."""
        plant = PlantState(plant_id="r00c00", row=0, col=0)
        assert plant.plant_id == "r00c00"
        assert plant.row == 0
        assert plant.col == 0

    def test_default_age_days(self):
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.age_days == 0

    def test_default_lai(self):
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.lai == 0.0

    def test_default_biomass_g(self):
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.biomass_g == 0.0

    def test_default_height_cm(self):
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.height_cm == 0.0

    def test_default_root_depth_cm(self):
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.root_depth_cm == 0.0

    def test_default_stress_index(self):
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.stress_index == 0.0

    def test_default_alive(self):
        """Plants are alive at initialisation."""
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.alive is True

    def test_default_phenological_stage(self):
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.phenological_stage == "germination"

    def test_default_custom_is_empty_dict(self):
        plant = PlantState(plant_id="r01c01", row=1, col=1)
        assert plant.custom == {}
        assert isinstance(plant.custom, dict)

    def test_custom_dict_is_independent_per_instance(self):
        """Each PlantState must have its OWN custom dict (not shared)."""
        p1 = PlantState(plant_id="r00c00", row=0, col=0)
        p2 = PlantState(plant_id="r00c01", row=0, col=1)
        p1.custom["water_uptake"] = 1.5
        assert "water_uptake" not in p2.custom

    def test_custom_stores_arbitrary_types(self):
        """custom dict must accept any Python value (PRD Section 5 Design Rule)."""
        plant = PlantState(plant_id="r02c03", row=2, col=3)
        plant.custom["flag"] = True
        plant.custom["history"] = [1.0, 2.0, 3.0]
        plant.custom["nested"] = {"key": "value"}
        plant.custom["count"] = 42
        assert plant.custom["flag"] is True
        assert plant.custom["history"] == [1.0, 2.0, 3.0]
        assert plant.custom["nested"]["key"] == "value"

    def test_explicit_construction_all_fields(self):
        """All fields can be supplied explicitly."""
        plant = PlantState(
            plant_id="r04c07",
            row=4,
            col=7,
            age_days=30,
            lai=2.5,
            biomass_g=150.0,
            height_cm=45.0,
            root_depth_cm=25.0,
            stress_index=0.1,
            alive=True,
            phenological_stage="vegetative",
            custom={"n_uptake_kg": 0.5},
        )
        assert plant.plant_id == "r04c07"
        assert plant.age_days == 30
        assert plant.lai == 2.5
        assert plant.biomass_g == 150.0
        assert plant.height_cm == 45.0
        assert plant.root_depth_cm == 25.0
        assert plant.stress_index == 0.1
        assert plant.alive is True
        assert plant.phenological_stage == "vegetative"
        assert plant.custom["n_uptake_kg"] == 0.5

    def test_mutation_of_fields(self):
        """Step functions mutate plant state in-place; this must work."""
        plant = PlantState(plant_id="r00c00", row=0, col=0)
        plant.lai += 0.3
        plant.biomass_g += 10.0
        plant.age_days += 1
        plant.stress_index += 0.05
        assert math.isclose(plant.lai, 0.3)
        assert math.isclose(plant.biomass_g, 10.0)
        assert plant.age_days == 1
        assert math.isclose(plant.stress_index, 0.05)

    def test_alive_can_be_set_false(self):
        """PRD Section 5.1: alive=False when stress_index >= 1.0."""
        plant = PlantState(plant_id="r00c00", row=0, col=0)
        plant.stress_index = 1.0
        plant.alive = False
        assert plant.alive is False

    def test_plant_id_format(self):
        """plant_id is a free-form string; PRD example is 'r04c07'."""
        plant = PlantState(plant_id="r39c59", row=39, col=59)
        assert plant.plant_id == "r39c59"

    def test_stress_index_boundary_values(self):
        plant = PlantState(plant_id="r00c00", row=0, col=0)
        plant.stress_index = 0.0
        assert plant.stress_index == 0.0
        plant.stress_index = 1.0
        assert plant.stress_index == 1.0

    def test_phenological_stage_mutation(self):
        plant = PlantState(plant_id="r00c00", row=0, col=0)
        for stage in ("vegetative", "flowering", "grain_fill", "maturity"):
            plant.phenological_stage = stage
            assert plant.phenological_stage == stage


# ===========================================================================
# 5.2  SoilVoxelState
# ===========================================================================

class TestSoilVoxelState:
    """Tests for SoilVoxelState (PRD Section 5.2)."""

    def _make_voxel(self, **kwargs) -> SoilVoxelState:
        """Helper: create a SoilVoxelState with sensible defaults."""
        defaults = dict(
            row=0, col=0, layer=0,
            depth_top_cm=0.0, depth_bottom_cm=20.0,
            moisture_pct=30.0, nitrogen_kg_ha=50.0,
            bulk_density=1.3, penetration_resistance=0.5,
        )
        defaults.update(kwargs)
        return SoilVoxelState(**defaults)

    def test_construction_required_fields(self):
        voxel = self._make_voxel()
        assert voxel.row == 0
        assert voxel.col == 0
        assert voxel.layer == 0

    def test_no_accidental_defaults_on_required_fields(self):
        """SoilVoxelState has NO optional fields except custom."""
        with pytest.raises(TypeError):
            # Missing required positional fields must raise TypeError
            SoilVoxelState()  # type: ignore[call-arg]

    def test_default_custom_is_empty_dict(self):
        voxel = self._make_voxel()
        assert voxel.custom == {}
        assert isinstance(voxel.custom, dict)

    def test_custom_dict_is_independent_per_instance(self):
        v1 = self._make_voxel(row=0, col=0, layer=0)
        v2 = self._make_voxel(row=0, col=0, layer=1)
        v1.custom["fc_pct"] = 40.0
        assert "fc_pct" not in v2.custom

    def test_layer_zero_is_topsoil(self):
        voxel = self._make_voxel(layer=0, depth_top_cm=0.0, depth_bottom_cm=20.0)
        assert voxel.layer == 0
        assert voxel.depth_top_cm == 0.0

    def test_multiple_layers_depth_continuity(self):
        """Depths should be contiguous across layers (validation is caller's job)."""
        layers = [
            self._make_voxel(layer=0, depth_top_cm=0.0,  depth_bottom_cm=20.0),
            self._make_voxel(layer=1, depth_top_cm=20.0, depth_bottom_cm=40.0),
            self._make_voxel(layer=2, depth_top_cm=40.0, depth_bottom_cm=60.0),
        ]
        for i in range(1, len(layers)):
            assert layers[i].depth_top_cm == layers[i - 1].depth_bottom_cm

    def test_penetration_resistance_field_exists(self):
        """PRD explicitly includes penetration_resistance for hard-pan modelling."""
        voxel = self._make_voxel(penetration_resistance=3.5)
        assert voxel.penetration_resistance == 3.5

    def test_mutation_of_moisture(self):
        voxel = self._make_voxel(moisture_pct=30.0)
        voxel.moisture_pct -= 5.0
        assert math.isclose(voxel.moisture_pct, 25.0)

    def test_mutation_of_nitrogen(self):
        voxel = self._make_voxel(nitrogen_kg_ha=50.0)
        voxel.nitrogen_kg_ha += 40.0  # fertiliser application
        assert math.isclose(voxel.nitrogen_kg_ha, 90.0)

    def test_moisture_can_reach_zero(self):
        """Moisture may be driven to zero (permanent wilting)."""
        voxel = self._make_voxel(moisture_pct=5.0)
        voxel.moisture_pct = max(0.0, voxel.moisture_pct - 10.0)
        assert voxel.moisture_pct == 0.0

    def test_custom_stores_pwp_and_fc(self):
        """Researchers commonly store PWP and FC in custom."""
        voxel = self._make_voxel()
        voxel.custom["pwp_pct"] = 12.0
        voxel.custom["fc_pct"] = 36.0
        assert voxel.custom["pwp_pct"] == 12.0
        assert voxel.custom["fc_pct"] == 36.0

    def test_all_numeric_fields_are_float(self):
        voxel = self._make_voxel(
            depth_top_cm=0.0, depth_bottom_cm=20.0,
            moisture_pct=30.0, nitrogen_kg_ha=50.0,
            bulk_density=1.3, penetration_resistance=0.5,
        )
        assert isinstance(voxel.depth_top_cm, float)
        assert isinstance(voxel.depth_bottom_cm, float)
        assert isinstance(voxel.moisture_pct, float)
        assert isinstance(voxel.nitrogen_kg_ha, float)
        assert isinstance(voxel.bulk_density, float)
        assert isinstance(voxel.penetration_resistance, float)


# ===========================================================================
# 5.3  FieldState
# ===========================================================================

class TestFieldState:
    """Tests for FieldState (PRD Section 5.3)."""

    def _make_plant_grid(self, n_rows: int, n_cols: int) -> list[PlantState]:
        return [
            PlantState(plant_id=f"r{r:02d}c{c:02d}", row=r, col=c)
            for r in range(n_rows)
            for c in range(n_cols)
        ]

    def _make_soil_grid(
        self, n_rows: int, n_cols: int, n_layers: int
    ) -> list[list[list[SoilVoxelState]]]:
        return [
            [
                [
                    SoilVoxelState(
                        row=r, col=c, layer=lyr,
                        depth_top_cm=lyr * 20.0,
                        depth_bottom_cm=(lyr + 1) * 20.0,
                        moisture_pct=30.0, nitrogen_kg_ha=50.0,
                        bulk_density=1.3, penetration_resistance=0.5,
                    )
                    for lyr in range(n_layers)
                ]
                for c in range(n_cols)
            ]
            for r in range(n_rows)
        ]

    def test_construction_minimal(self):
        n_rows, n_cols, n_layers = 2, 3, 1
        field = FieldState(
            day=1,
            plants=self._make_plant_grid(n_rows, n_cols),
            soil=self._make_soil_grid(n_rows, n_cols, n_layers),
            elevation_grid=np.zeros((n_rows, n_cols)),
            events_fired=[],
        )
        assert field.day == 1
        assert len(field.plants) == n_rows * n_cols
        assert len(field.events_fired) == 0

    def test_soil_indexing_row_col_layer(self):
        """soil[row][col][layer] indexing must work as per PRD Section 5.3."""
        n_rows, n_cols, n_layers = 3, 4, 2
        field = FieldState(
            day=5,
            plants=self._make_plant_grid(n_rows, n_cols),
            soil=self._make_soil_grid(n_rows, n_cols, n_layers),
            elevation_grid=np.zeros((n_rows, n_cols)),
            events_fired=[],
        )
        topsoil_r2_c3 = field.soil[2][3][0]
        assert topsoil_r2_c3.row == 2
        assert topsoil_r2_c3.col == 3
        assert topsoil_r2_c3.layer == 0

    def test_elevation_grid_is_numpy_array(self):
        """PRD Section 5.3: elevation_grid must be a np.ndarray."""
        n_rows, n_cols = 4, 5
        elev = np.zeros((n_rows, n_cols))
        field = FieldState(
            day=1,
            plants=self._make_plant_grid(n_rows, n_cols),
            soil=self._make_soil_grid(n_rows, n_cols, 1),
            elevation_grid=elev,
            events_fired=[],
        )
        assert isinstance(field.elevation_grid, np.ndarray)
        assert field.elevation_grid.shape == (n_rows, n_cols)

    def test_elevation_grid_slope(self):
        """Elevation values are relative metres; typical slope is small."""
        n_rows, n_cols = 5, 5
        elev = np.zeros((n_rows, n_cols))
        for r in range(n_rows):
            elev[r, :] = r * 0.005  # 0.5% slope, per PRD Section 8.3 example
        field = FieldState(
            day=1,
            plants=self._make_plant_grid(n_rows, n_cols),
            soil=self._make_soil_grid(n_rows, n_cols, 1),
            elevation_grid=elev,
            events_fired=[],
        )
        assert math.isclose(field.elevation_grid[4, 0], 0.02)

    def test_events_fired_populated(self):
        n_rows, n_cols = 2, 2
        field = FieldState(
            day=15,
            plants=self._make_plant_grid(n_rows, n_cols),
            soil=self._make_soil_grid(n_rows, n_cols, 1),
            elevation_grid=np.zeros((n_rows, n_cols)),
            events_fired=["irrigation", "fertiliser"],
        )
        assert "irrigation" in field.events_fired
        assert "fertiliser" in field.events_fired

    def test_default_custom_is_empty_dict(self):
        n_rows, n_cols = 2, 2
        field = FieldState(
            day=1,
            plants=self._make_plant_grid(n_rows, n_cols),
            soil=self._make_soil_grid(n_rows, n_cols, 1),
            elevation_grid=np.zeros((n_rows, n_cols)),
            events_fired=[],
        )
        assert field.custom == {}
        assert isinstance(field.custom, dict)

    def test_day_counter_mutation(self):
        """The engine increments day between timesteps."""
        n_rows, n_cols = 2, 2
        field = FieldState(
            day=1,
            plants=self._make_plant_grid(n_rows, n_cols),
            soil=self._make_soil_grid(n_rows, n_cols, 1),
            elevation_grid=np.zeros((n_rows, n_cols)),
            events_fired=[],
        )
        field.day = 2
        assert field.day == 2

    def test_plant_count_matches_grid(self):
        """Number of plants must equal n_rows * n_cols."""
        n_rows, n_cols = 20, 30
        plants = self._make_plant_grid(n_rows, n_cols)
        field = FieldState(
            day=1,
            plants=plants,
            soil=self._make_soil_grid(n_rows, n_cols, 3),
            elevation_grid=np.zeros((n_rows, n_cols)),
            events_fired=[],
        )
        assert len(field.plants) == 600  # PRD Section 10.1 wheat_basic size

    def test_plants_are_alive_at_day_one(self):
        n_rows, n_cols = 5, 5
        field = FieldState(
            day=1,
            plants=self._make_plant_grid(n_rows, n_cols),
            soil=self._make_soil_grid(n_rows, n_cols, 1),
            elevation_grid=np.zeros((n_rows, n_cols)),
            events_fired=[],
        )
        assert all(p.alive for p in field.plants)

    def test_three_soil_layers_per_cell(self):
        """wheat_basic.py uses 3 soil layers (PRD Section 10.1)."""
        n_rows, n_cols, n_layers = 2, 2, 3
        soil = self._make_soil_grid(n_rows, n_cols, n_layers)
        assert len(soil[0][0]) == 3
        assert soil[0][0][0].layer == 0
        assert soil[0][0][2].layer == 2


# ===========================================================================
# 5.4  EnvironmentState
# ===========================================================================

class TestEnvironmentState:
    """Tests for EnvironmentState (PRD Section 5.4)."""

    def _make_env(self, **kwargs) -> EnvironmentState:
        """Helper: minimal valid EnvironmentState with sensible defaults."""
        defaults = dict(
            day=1, doy=120,
            temp_max_c=28.0, temp_min_c=15.0, temp_mean_c=21.5,
            radiation_mj_m2=18.0,
            rainfall_mm=0.0,
            et0_mm=5.0,
            wind_speed_ms=2.5,
            humidity_pct=65.0,
        )
        defaults.update(kwargs)
        return EnvironmentState(**defaults)

    def test_construction_required_fields(self):
        env = self._make_env()
        assert env.day == 1
        assert env.doy == 120

    def test_no_defaults_on_required_fields(self):
        """All fields except co2_ppm and custom are required."""
        with pytest.raises(TypeError):
            EnvironmentState()  # type: ignore[call-arg]

    def test_default_co2_ppm(self):
        """PRD Section 5.4: co2_ppm defaults to 415.0."""
        env = self._make_env()
        assert env.co2_ppm == 415.0

    def test_custom_co2_for_elevated_scenario(self):
        """Researchers studying elevated CO₂ supply a custom value."""
        env = self._make_env(co2_ppm=600.0)
        assert env.co2_ppm == 600.0

    def test_default_custom_is_empty_dict(self):
        env = self._make_env()
        assert env.custom == {}
        assert isinstance(env.custom, dict)

    def test_custom_dict_is_independent_per_instance(self):
        e1 = self._make_env(day=1)
        e2 = self._make_env(day=2)
        e1.custom["vpd_kpa"] = 1.2
        assert "vpd_kpa" not in e2.custom

    def test_wind_speed_is_si_ms(self):
        """PRD Section 11: wind is always stored in m/s internally."""
        env = self._make_env(wind_speed_ms=3.0)
        assert env.wind_speed_ms == 3.0

    def test_temp_mean_c_explicit_supply(self):
        """temp_mean_c may be supplied directly from CSV."""
        env = self._make_env(temp_max_c=30.0, temp_min_c=20.0, temp_mean_c=25.0)
        assert env.temp_mean_c == 25.0

    def test_temp_mean_derived_convention(self):
        """(tmax + tmin) / 2 is a common loader convention."""
        tmax, tmin = 32.0, 18.0
        env = self._make_env(
            temp_max_c=tmax, temp_min_c=tmin,
            temp_mean_c=(tmax + tmin) / 2
        )
        assert math.isclose(env.temp_mean_c, 25.0)

    def test_doy_range(self):
        """DOY must be between 1 and 366 inclusive (validation is engine's job)."""
        for doy in (1, 90, 180, 270, 365, 366):
            env = self._make_env(doy=doy)
            assert env.doy == doy

    def test_radiation_field_name(self):
        """PRD uses radiation_mj_m2 (not radiation_mj or rad)."""
        env = self._make_env(radiation_mj_m2=22.5)
        assert env.radiation_mj_m2 == 22.5

    def test_rainfall_zero_on_dry_day(self):
        env = self._make_env(rainfall_mm=0.0)
        assert env.rainfall_mm == 0.0

    def test_rainfall_positive_on_wet_day(self):
        env = self._make_env(rainfall_mm=45.0)
        assert env.rainfall_mm == 45.0

    def test_custom_stores_vpd_and_par(self):
        """Researchers frequently add vapour pressure deficit and PAR to custom."""
        env = self._make_env()
        env.custom["vpd_kpa"] = 1.8
        env.custom["par_mol_m2"] = 38.0
        assert env.custom["vpd_kpa"] == 1.8
        assert env.custom["par_mol_m2"] == 38.0

    def test_humidity_range(self):
        for h in (0.0, 50.0, 100.0):
            env = self._make_env(humidity_pct=h)
            assert env.humidity_pct == h


# ===========================================================================
# Cross-class integration tests
# ===========================================================================

class TestStateIntegration:
    """Verify that the dataclasses compose correctly as the PRD intends."""

    def test_plant_soil_linkage_via_row_col(self):
        """A plant at (row=2, col=3) must have a matching soil voxel."""
        plant = PlantState(plant_id="r02c03", row=2, col=3)
        voxel = SoilVoxelState(
            row=2, col=3, layer=0,
            depth_top_cm=0.0, depth_bottom_cm=20.0,
            moisture_pct=28.0, nitrogen_kg_ha=45.0,
            bulk_density=1.25, penetration_resistance=0.8,
        )
        assert plant.row == voxel.row
        assert plant.col == voxel.col

    def test_fieldstate_soil_access_from_plant(self):
        """Step functions access soil via state.soil[plant.row][plant.col][0]."""
        n_rows, n_cols, n_layers = 3, 3, 2
        plants = [
            PlantState(plant_id=f"r{r:02d}c{c:02d}", row=r, col=c)
            for r in range(n_rows)
            for c in range(n_cols)
        ]
        soil = [
            [
                [
                    SoilVoxelState(
                        row=r, col=c, layer=lyr,
                        depth_top_cm=lyr * 20.0,
                        depth_bottom_cm=(lyr + 1) * 20.0,
                        moisture_pct=30.0, nitrogen_kg_ha=50.0,
                        bulk_density=1.3, penetration_resistance=0.5,
                    )
                    for lyr in range(n_layers)
                ]
                for c in range(n_cols)
            ]
            for r in range(n_rows)
        ]
        field = FieldState(
            day=1,
            plants=plants,
            soil=soil,
            elevation_grid=np.zeros((n_rows, n_cols)),
            events_fired=[],
        )
        # Simulate what a step function does
        for plant in field.plants:
            topsoil = field.soil[plant.row][plant.col][0]
            assert topsoil.row == plant.row
            assert topsoil.col == plant.col
            assert topsoil.layer == 0

    def test_environment_and_field_share_same_day(self):
        """Engine passes matching day to both FieldState and EnvironmentState."""
        day = 45
        env = EnvironmentState(
            day=day, doy=165,
            temp_max_c=33.0, temp_min_c=22.0, temp_mean_c=27.5,
            radiation_mj_m2=20.0, rainfall_mm=0.0, et0_mm=6.0,
            wind_speed_ms=3.0, humidity_pct=55.0,
        )
        field = FieldState(
            day=day,
            plants=[PlantState(plant_id="r00c00", row=0, col=0)],
            soil=[[[
                SoilVoxelState(
                    row=0, col=0, layer=0,
                    depth_top_cm=0.0, depth_bottom_cm=20.0,
                    moisture_pct=30.0, nitrogen_kg_ha=50.0,
                    bulk_density=1.3, penetration_resistance=0.5,
                )
            ]]],
            elevation_grid=np.zeros((1, 1)),
            events_fired=[],
        )
        assert field.day == env.day

    def test_custom_dict_does_not_mutate_class_default(self):
        """The field(default_factory=dict) must prevent the mutable-default bug."""
        p1 = PlantState(plant_id="r00c00", row=0, col=0)
        p1.custom["key"] = "value"
        p2 = PlantState(plant_id="r00c01", row=0, col=1)
        # p2 must NOT see p1's mutation
        assert p2.custom == {}

        v1 = SoilVoxelState(
            row=0, col=0, layer=0,
            depth_top_cm=0.0, depth_bottom_cm=20.0,
            moisture_pct=30.0, nitrogen_kg_ha=50.0,
            bulk_density=1.3, penetration_resistance=0.5,
        )
        v1.custom["key"] = "value"
        v2 = SoilVoxelState(
            row=0, col=0, layer=1,
            depth_top_cm=20.0, depth_bottom_cm=40.0,
            moisture_pct=25.0, nitrogen_kg_ha=40.0,
            bulk_density=1.35, penetration_resistance=0.6,
        )
        assert v2.custom == {}

        e1 = EnvironmentState(
            day=1, doy=1,
            temp_max_c=20.0, temp_min_c=10.0, temp_mean_c=15.0,
            radiation_mj_m2=12.0, rainfall_mm=0.0, et0_mm=3.0,
            wind_speed_ms=1.5, humidity_pct=70.0,
        )
        e1.custom["key"] = "value"
        e2 = EnvironmentState(
            day=2, doy=2,
            temp_max_c=22.0, temp_min_c=12.0, temp_mean_c=17.0,
            radiation_mj_m2=14.0, rainfall_mm=5.0, et0_mm=3.5,
            wind_speed_ms=2.0, humidity_pct=75.0,
        )
        assert e2.custom == {}
