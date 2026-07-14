import numpy as np
import pytest

from cropforge import Crop, Farm, Field
from cropforge.plugins import StandardWheat
from cropforge.state import PlantingConfig
from cropforge.terrain import Terrain


class _WarmWeather:
    def get_day(self, day: int):
        from cropforge.state import EnvironmentState

        return EnvironmentState(
            day=day,
            doy=day,
            temp_max_c=30.0,
            temp_min_c=20.0,
            temp_mean_c=25.0,
            radiation_mj_m2=20.0,
            rainfall_mm=0.0,
            et0_mm=4.0,
            wind_speed_ms=1.0,
            humidity_pct=55.0,
        )


def test_planting_config_computes_plants_per_m2():
    cfg = PlantingConfig(row_spacing_m=0.20, plant_spacing_m=0.08)
    assert cfg.plants_per_m2 == pytest.approx(62.5)


def test_field_set_crop_accepts_sowing_density():
    farm = Farm("density-metadata")
    field = Field("Plot", rows=1, cols=1)
    field.set_crop(Crop(species="wheat"), sowing_density_plants_per_m2=250.0)
    farm.add_field(field)

    field._init_field_state(day=1)

    assert field._field_state.planting_config.plants_per_m2 == pytest.approx(250.0)
    assert field._field_state.plants[0].sowing_density_plants_per_m2 == pytest.approx(250.0)


def test_crop_specific_default_density():
    wheat = Field("Wheat", rows=1, cols=1)
    wheat.set_crop(Crop(species="Triticum aestivum"))
    maize = Field("Maize", rows=1, cols=1)
    maize.set_crop(Crop(species="Zea mays"))

    assert wheat._resolve_planting_config().plants_per_m2 == pytest.approx(250.0)
    assert maize._resolve_planting_config().plants_per_m2 == pytest.approx(8.0)


def test_yield_summary_correct_scaling_from_grain_biomass():
    farm = Farm("known-yield")
    field = Field("Plot", rows=2, cols=2)
    field.set_crop(Crop(species="wheat"), sowing_density_plants_per_m2=62.5)
    farm.add_field(field)
    field._init_field_state(day=1)

    for plant in field._field_state.plants:
        plant.biomass_g = 5.0
        plant.custom["grain_biomass_g"] = 2.0

    summary = farm.yield_summary()

    assert summary["total_yield_kg"] == pytest.approx(0.5)
    assert summary["yield_kg_per_ha"] == pytest.approx(1250.0)
    assert summary["yield_t_per_ha"] == pytest.approx(1.25)


def test_yield_summary_uses_resolution_m_squared_for_sub_metre_cells():
    farm = Farm("submetre-yield")
    field = Field("Plot", rows=10, cols=10)
    field.set_terrain(Terrain.from_array(np.zeros((10, 10)), resolution_m=0.5))
    field.set_crop(Crop(species="wheat"), sowing_density_plants_per_m2=250.0)
    farm.add_field(field)
    field._init_field_state(day=1)

    for plant in field._field_state.plants:
        plant.custom["grain_biomass_g"] = 2.0

    summary = farm.yield_summary()

    assert field.field_area_m2 == pytest.approx(25.0)
    assert summary["total_yield_kg"] == pytest.approx(12.5)
    assert summary["yield_kg_per_ha"] == pytest.approx(5000.0)


def test_yield_summary_falls_back_to_biomass_without_grain():
    farm = Farm("biomass-fallback")
    field = Field("Plot", rows=1, cols=1)
    field.set_crop(Crop(species="generic"), sowing_density_plants_per_m2=10.0)
    farm.add_field(field)
    field._init_field_state(day=1)
    field._field_state.plants[0].biomass_g = 100.0

    summary = farm.yield_summary()

    assert summary["total_yield_kg"] == pytest.approx(1.0)
    assert summary["yield_kg_per_ha"] == pytest.approx(10000.0)


def test_crucible_wheat_density_yield_summary_matches_grain_sum():
    farm = Farm("density-crucible")
    field = Field("Plot", rows=10, cols=10)
    field.set_terrain(Terrain.from_array(np.zeros((10, 10)), resolution_m=1.0))
    field.set_crop(Crop(species="wheat"), sowing_density_plants_per_m2=250.0)
    field.set_weather(_WarmWeather())
    field.use_plugin(StandardWheat)
    farm.add_field(field)
    farm.use_physics(radiation=True)

    farm.run(days=90)

    grain_g = sum(
        plant.custom.get("grain_biomass_g", 0.0)
        for plant in field._field_state.plants
    )
    assert grain_g > 0.0

    area_m2 = 10 * 10 * (1.0 ** 2)
    expected_total_kg = grain_g * 250.0 / 1000.0
    expected_kg_per_ha = expected_total_kg / (area_m2 / 10000.0)

    summary = farm.yield_summary()

    assert summary["total_yield_kg"] == pytest.approx(expected_total_kg)
    assert summary["yield_kg_per_ha"] == pytest.approx(expected_kg_per_ha)
    assert summary["yield_t_per_ha"] == pytest.approx(expected_kg_per_ha / 1000.0)
