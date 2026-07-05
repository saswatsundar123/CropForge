"""
tests/test_physics_clods.py
============================
Crucible tests for Phase 4 clod dynamics and infiltration coupling
(CropForge PRD v0.7.0 §7.4).

Coverage:
    Unit:
        calculate_roughness_decay -- temporal decay math
        PHASE_CLOD_ENGINE registration
        make_clod_dynamics_hook not registered without opt-in
    Crucible 1 (Temporal Decay):
        10 days of 20mm rain → roughness measurably lower by day 10
        Surface runoff on day 10 > day 1 (slope + decayed roughness)
    Crucible 2 (Spatial Runoff):
        Steep cell gets strictly higher runoff than gentle cell same-day
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropforge import Farm, Field, Crop
from cropforge.loaders import Weather
from cropforge.land_prep import ConventionalTill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weather(days: int, rainfall_mm: float = 20.0) -> Weather:
    rows = [
        {
            "day": d, "doy": d,
            "temp_max_c": 30.0, "temp_min_c": 18.0, "temp_mean_c": 24.0,
            "radiation_mj_m2": 18.0, "rainfall_mm": rainfall_mm,
            "et0_mm": 5.0, "wind_speed_ms": 2.0, "humidity_pct": 65.0,
            "co2_ppm": 415.0,
        }
        for d in range(1, days + 1)
    ]
    return Weather(pd.DataFrame(rows).set_index("day"))


def _make_field(rows: int, cols: int, elevation: np.ndarray | None = None) -> Field:
    field = Field("F1", rows=rows, cols=cols)
    if elevation is not None:
        field.set_elevation(elevation)
    field.set_crop(Crop(species="Zea mays", variety="TestMaize"))
    return field


def _apply_conv_till(field: Field) -> Field:
    """Set surface_roughness_index = 0.8 via ConventionalTill on all cells."""
    field.set_land_prep(ConventionalTill())
    return field


# ---------------------------------------------------------------------------
# Unit: calculate_roughness_decay
# ---------------------------------------------------------------------------

class TestRoughnessDecayMath:
    def test_no_rain_no_decay(self):
        from cropforge.physics.soil import calculate_roughness_decay
        assert calculate_roughness_decay(0.8, 0.0) == 0.8

    def test_at_floor_no_decay(self):
        from cropforge.physics.soil import calculate_roughness_decay
        assert calculate_roughness_decay(0.1, 20.0) == 0.1

    def test_decay_reduces_roughness(self):
        from cropforge.physics.soil import calculate_roughness_decay
        result = calculate_roughness_decay(0.8, 20.0)
        assert result < 0.8
        assert result >= 0.1

    def test_heavier_rain_decays_more(self):
        from cropforge.physics.soil import calculate_roughness_decay
        light = calculate_roughness_decay(0.8, 5.0)
        heavy = calculate_roughness_decay(0.8, 40.0)
        assert heavy < light

    def test_ten_days_decay(self):
        from cropforge.physics.soil import calculate_roughness_decay
        roughness = 0.8
        for _ in range(10):
            roughness = calculate_roughness_decay(roughness, 20.0)
        assert roughness < 0.8
        assert roughness >= 0.1

    def test_floor_respected(self):
        from cropforge.physics.soil import calculate_roughness_decay
        roughness = 0.8
        for _ in range(200):
            roughness = calculate_roughness_decay(roughness, 100.0)
        assert roughness == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit: hook registration
# ---------------------------------------------------------------------------

class TestClodRegistration:
    def test_clod_not_registered_by_default(self):
        from cropforge.physics.builtin_hooks import PHASE_CLOD_ENGINE
        farm = Farm(name="NoClodsDefault", location=(28.6, 77.2))
        farm.use_physics()
        phases = [phase for phase, _ in farm._physics_registry]
        assert PHASE_CLOD_ENGINE not in phases

    def test_clod_registered_when_opted_in(self):
        from cropforge.physics.builtin_hooks import PHASE_CLOD_ENGINE
        farm = Farm(name="ClodsOpted", location=(28.6, 77.2))
        farm.use_physics(et0=True, water_balance=True, clod_dynamics=True)
        phases = [phase for phase, _ in farm._physics_registry]
        assert PHASE_CLOD_ENGINE in phases

    def test_clod_requires_water_balance(self):
        from cropforge.runtime import CropForgeConfigError
        farm = Farm(name="ClodsNoWater", location=(28.6, 77.2))
        with pytest.raises(CropForgeConfigError, match="water_balance=True"):
            farm.use_physics(clod_dynamics=True)

    def test_clod_hook_name(self):
        from cropforge.physics.builtin_hooks import PHASE_CLOD_ENGINE
        farm = Farm(name="ClodHookName", location=(28.6, 77.2))
        farm.use_physics(et0=True, water_balance=True, clod_dynamics=True)
        clod_fns = [fn for phase, fn in farm._physics_registry
                    if phase == PHASE_CLOD_ENGINE]
        assert len(clod_fns) == 1
        assert clod_fns[0].__name__ == "_clod_step"


# ---------------------------------------------------------------------------
# Crucible Test 1: Temporal Decay
# ---------------------------------------------------------------------------

class TestCrucible1TemporalDecay:
    """
    Crucible: 10 consecutive days of 20mm rain on a freshly-tilled field
    (roughness_index = 0.8 from ConventionalTill).

    Assertions:
        1. Day-10 roughness is measurably LOWER than Day-1 roughness.
        2. Day-10 surface_runoff_mm_today is measurably HIGHER than Day-1
           (slope-driven direct runoff increases as clods break down).
    """

    def _build_sim(self):
        """1x2 field: cell [0,0] elevated 1m above cell [0,1] → slope present."""
        elevation = np.array([[1.0, 0.0]])

        farm = Farm(name="Crucible1", location=(28.6, 77.2))
        field = _make_field(rows=1, cols=2, elevation=elevation)
        field.set_weather(_make_weather(days=10, rainfall_mm=20.0))
        _apply_conv_till(field)
        farm.add_field(field)
        farm.use_physics(et0=True, water_balance=True, clod_dynamics=True)

        roughness_by_day = []
        runoff_by_day = []

        @farm.step(phase=1)
        def _capture(state, env):
            roughness_by_day.append(state.soil[0][0][0].surface_roughness_index)
            runoff_by_day.append(
                state.soil[0][0][0].custom.get("surface_runoff_mm_today", 0.0)
            )
            return state

        return farm, field, roughness_by_day, runoff_by_day

    def test_roughness_decays_over_10_days(self):
        """surface_roughness_index must be lower on Day 10 than Day 1."""
        farm, _, roughness_by_day, _ = self._build_sim()
        farm.run(days=10)

        assert len(roughness_by_day) == 10
        day1 = roughness_by_day[0]
        day10 = roughness_by_day[-1]
        assert day10 < day1, (
            f"Expected roughness to decay: day1={day1:.4f}, day10={day10:.4f}"
        )

    def test_runoff_increases_as_clods_break_down(self):
        """slope-driven direct runoff must increase as roughness decays."""
        farm, _, _, runoff_by_day = self._build_sim()
        farm.run(days=10)

        assert len(runoff_by_day) == 10
        day1 = runoff_by_day[0]
        day10 = runoff_by_day[-1]
        assert day10 > day1, (
            f"Expected runoff to grow as clods break: day1={day1:.4f}, day10={day10:.4f}"
        )


# ---------------------------------------------------------------------------
# Crucible Test 2: Spatial Slope-Driven Runoff
# ---------------------------------------------------------------------------

class TestCrucible2SpatialRunoff:
    """
    Crucible: 1x3 grid with ONE steep cell (elevation 2m) and two flat cells
    (elevation 0m). Identical rainfall applied. After one day, the steep
    cell must have STRICTLY HIGHER surface_runoff_mm_today than the flat cell.
    """

    def test_steep_cell_higher_runoff_than_flat(self):
        """Steep cell (slope_frac=1.0) generates more runoff than flat cell (slope_frac=0)."""
        # elevation: [2.0m, 0.0m, 0.0m]
        # _compute_slope_normalized:
        #   [0][0]: max_drop = 2m (drops to [0][1])
        #   [0][1]: max_drop = 0  (no lower neighbour below it)
        #   [0][2]: max_drop = 0
        # After normalisation: [1.0, 0.0, 0.0]
        elevation = np.array([[2.0, 0.0, 0.0]])

        farm = Farm(name="Crucible2", location=(28.6, 77.2))
        field = _make_field(rows=1, cols=3, elevation=elevation)
        field.set_weather(_make_weather(days=1, rainfall_mm=20.0))
        _apply_conv_till(field)  # roughness=0.8 on all cells
        farm.add_field(field)
        farm.use_physics(et0=True, water_balance=True, clod_dynamics=True)

        farm.run(days=1)

        steep_runoff = field._field_state.soil[0][0][0].custom.get(
            "surface_runoff_mm_today", 0.0
        )
        flat_runoff = field._field_state.soil[0][1][0].custom.get(
            "surface_runoff_mm_today", 0.0
        )

        assert steep_runoff > flat_runoff, (
            f"Steep cell must have higher runoff: steep={steep_runoff:.4f} mm, "
            f"flat={flat_runoff:.4f} mm"
        )

    def test_slope_fractions_are_zero_for_flat_field(self):
        """_compute_slope_normalized returns all-zeros for a flat elevation grid."""
        from cropforge.physics.builtin_hooks import _compute_slope_normalized
        flat = np.zeros((3, 3))
        slopes = _compute_slope_normalized(flat)
        for row in slopes:
            for val in row:
                assert val == 0.0
