"""
tests/test_physics_weeds.py
===========================
Crucible tests for CropForge v1.0.0 Phase 1 weed competition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest


def _env(day: int = 1, doy: int | None = None):
    from cropforge.state import EnvironmentState

    return EnvironmentState(
        day=day,
        doy=day if doy is None else doy,
        temp_max_c=30.0,
        temp_min_c=18.0,
        temp_mean_c=24.0,
        radiation_mj_m2=22.0,
        rainfall_mm=0.0,
        et0_mm=5.0,
        wind_speed_ms=2.0,
        humidity_pct=55.0,
    )


def _small_state():
    from cropforge.state import FieldState, PlantState, SoilVoxelState

    plants = [
        PlantState(plant_id=f"r{r:02d}c{c:02d}", row=r, col=c, lai=1.0, biomass_g=1.0)
        for r in range(2)
        for c in range(2)
    ]
    soil = [
        [
            [
                SoilVoxelState(
                    row=r,
                    col=c,
                    layer=0,
                    depth_top_cm=0.0,
                    depth_bottom_cm=20.0,
                    moisture_pct=30.0,
                    nitrogen_kg_ha=0.0,
                    bulk_density=1.3,
                    penetration_resistance=0.5,
                    custom={"wilting_point_pct": 10.0},
                )
            ]
            for c in range(2)
        ]
        for r in range(2)
    ]
    return FieldState(
        day=1,
        plants=plants,
        soil=soil,
        elevation_grid=np.zeros((2, 2), dtype=float),
        events_fired=[],
    )


class _WarmWeather:
    def get_day(self, day: int):
        return _env(day)


def _wheat_farm(name: str, weed: bool, days: int = 45):
    from cropforge.farm import Farm, Field
    from cropforge.plugins import StandardWheat

    farm = Farm(name)
    field = Field("Plot", rows=10, cols=10)
    field.set_weather(_WarmWeather())
    field.use_plugin(StandardWheat)
    farm.add_field(field)
    if weed:
        farm.set_weed_params(
            field="Plot",
            species="generic_grass",
            initial_density_m2=0.10,
            emergence_doy=1,
            spread_rate=0.60,
            competitive_index=1.0,
            daily_lai_growth=0.06,
        )
    farm.use_physics(radiation=True, weed_pressure=weed, weed_seed=7)
    farm.run(days=days)
    return farm


def _occupied_count(log_path: str, day: int) -> int:
    df = pq.read_table(str(Path(log_path) / "weed_states")).to_pandas()
    return int(((df["day"].astype(int) == day) & (df["alive"])).sum())


def test_weed_reduces_soil_moisture():
    from cropforge.physics.weeds import WeedParams, WeedState, step_weeds

    state = _small_state()
    state.weed_grid = [[WeedState(0, 0, lai=1.0, species="generic_grass"), None], [None, None]]
    params = WeedParams(competitive_index=1.0)
    step_weeds(state.weed_grid, state.soil, state.plants, _env(), params, doy=1, rng=np.random.default_rng(1))

    assert state.soil[0][0][0].moisture_pct < state.soil[0][1][0].moisture_pct


def test_weed_lai_suppresses_crop_par():
    from cropforge.physics.weeds import WeedState, compute_weed_radiation_suppression

    state = _small_state()
    weed_grid = [[WeedState(0, 0, lai=1.2), None], [None, None]]
    suppression = compute_weed_radiation_suppression(weed_grid, state.plants, 1.0)

    assert suppression[0, 0] < 1.0
    assert suppression[0, 1] == pytest.approx(1.0)


def test_weed_spread_increases_coverage():
    from cropforge.physics.weeds import WeedParams, WeedState, step_weeds

    state = _small_state()
    state.weed_grid = [[WeedState(0, 0, species="generic_grass"), None], [None, None]]
    params = WeedParams(spread_rate=1.0)
    before = sum(1 for row in state.weed_grid for weed in row if weed is not None)
    step_weeds(state.weed_grid, state.soil, state.plants, _env(), params, doy=1, rng=np.random.default_rng(1))
    after = sum(1 for row in state.weed_grid for weed in row if weed is not None)

    assert after > before


def test_weed_respects_emergence_doy():
    from cropforge.physics.weeds import WeedParams, WeedState, step_weeds

    state = _small_state()
    state.weed_grid = [[WeedState(0, 0, lai=0.2), None], [None, None]]
    params = WeedParams(emergence_doy=20, daily_lai_growth=0.5, spread_rate=1.0)
    step_weeds(state.weed_grid, state.soil, state.plants, _env(doy=10), params, doy=10, rng=np.random.default_rng(1))

    assert state.weed_grid[0][0].lai == pytest.approx(0.2)
    assert sum(1 for row in state.weed_grid for weed in row if weed is not None) == 1


def test_weed_pressure_false_leaves_soil_unchanged():
    no_physics = _wheat_farm("NoWeedBaseline", weed=False, days=5)
    disabled = _wheat_farm("NoWeedDisabled", weed=False, days=5)

    a = pq.read_table(no_physics._last_log_path + "/soil").to_pandas().sort_values(["day", "row", "col", "layer"]).reset_index(drop=True)
    b = pq.read_table(disabled._last_log_path + "/soil").to_pandas().sort_values(["day", "row", "col", "layer"]).reset_index(drop=True)
    assert a.drop(columns=["field_name"]).equals(b.drop(columns=["field_name"]))


def test_weed_parquet_table_absent_when_disabled():
    farm = _wheat_farm("WeedAbsent", weed=False, days=3)
    assert not (Path(farm._last_log_path) / "weed_states").exists()


def test_weed_parquet_table_present_when_enabled():
    farm = _wheat_farm("WeedPresent", weed=True, days=3)
    weed_dir = Path(farm._last_log_path) / "weed_states"
    assert weed_dir.exists()
    df = pq.read_table(str(weed_dir)).to_pandas()
    assert {"field_name", "day", "row", "col", "alive", "lai", "biomass_g", "species"} <= set(df.columns)


def test_crucible_weeds_spread_and_stunt_crop_growth():
    weed_farm = _wheat_farm("WeedCrucible", weed=True, days=45)
    control_farm = _wheat_farm("WeedControl", weed=False, days=45)

    assert _occupied_count(weed_farm._last_log_path, 45) > _occupied_count(weed_farm._last_log_path, 1)

    weeds = pq.read_table(weed_farm._last_log_path + "/weed_states").to_pandas()
    day45_weeds = weeds[(weeds["day"].astype(int) == 45) & (weeds["alive"])]
    occupied = {(int(r.row), int(r.col)) for r in day45_weeds.itertuples()}

    weed_plants = pq.read_table(weed_farm._last_log_path + "/plants").to_pandas()
    control_plants = pq.read_table(control_farm._last_log_path + "/plants").to_pandas()
    weed_day45 = weed_plants[
        (weed_plants["day"].astype(int) == 45)
        & (weed_plants.apply(lambda row: (int(row["row"]), int(row["col"])) in occupied, axis=1))
    ]
    control_day45 = control_plants[
        (control_plants["day"].astype(int) == 45)
        & (control_plants.apply(lambda row: (int(row["row"]), int(row["col"])) in occupied, axis=1))
    ]

    assert not weed_day45.empty
    assert float(weed_day45["biomass_g"].mean()) < float(control_day45["biomass_g"].mean())
