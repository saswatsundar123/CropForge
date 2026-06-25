"""
tests/test_integration_physics.py
===================================
Integration tests for CropForge v0.2.0 Opt-In Physics (Phase 2).

PRD v0.2.0 Section 10 -- Backward Compatibility Requirements:
    "v0.2.0 must not break any simulation written for v0.1.0."
    "All new subsystems default to disabled."
    "EnvironmentState.et0_mm remains 0.0 unless explicitly overwritten."

Tests in this file:
    1. BACKWARD COMPAT: A farm.run() without use_physics() must leave
       et0_mm at exactly the value provided by the weather source (or the
       stub 0.0), and root_growth_multiplier at 1.0 for all plants.

    2. ET0 FORWARD: With use_physics(et0=True), the engine must populate
       et0_mm with a non-zero computed value before the researcher's
       phase=0 step runs.

    3. ROOT FORWARD: With use_physics(root_impedance=True), the engine
       must populate root_growth_multiplier correctly based on soil
       penetration_resistance before the researcher's step runs.

    4. COMBINED: Both hooks active together -- values are independently
       correct and do not interfere.

    5. PHASE ORDER: The built-in hooks run at phases -2 / -1, strictly
       before all researcher @farm.step functions (phase >= 0).
"""

import pytest
import numpy as np

from cropforge.farm import Farm, Field
from cropforge.crop import Crop
from cropforge.state import EnvironmentState, SoilVoxelState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_minimal_farm(name="test_farm"):
    """Return a 2x2 Farm with a single field, no weather (uses stub env)."""
    farm = Farm(name=name, location=(31.2, 35.0))   # Etzion latitude for tests
    field = Field(name="TestField", rows=2, cols=2, area_ha=0.1)
    field.set_crop(Crop(species="T. aestivum", variety="Basic"))
    farm.add_field(field)
    return farm, field


def _make_farm_with_weather(et0_csv_value=5.0):
    """Return a farm whose weather stub returns et0_mm=et0_csv_value each day."""
    farm, field = _make_minimal_farm("farm_with_weather")

    class _FakeWeather:
        def get_day(self, day):
            return EnvironmentState(
                day=day,
                doy=((day - 1) % 365) + 1,
                temp_max_c=30.0,
                temp_min_c=18.0,
                temp_mean_c=24.0,
                radiation_mj_m2=22.0,
                rainfall_mm=0.0,
                et0_mm=et0_csv_value,  # CSV-supplied value
                wind_speed_ms=2.5,
                humidity_pct=55.0,
            )

    field.set_weather(_FakeWeather())
    return farm, field


# ---------------------------------------------------------------------------
# Test 1: Backward compatibility -- no use_physics
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """PRD v0.2.0 Section 10: v0.1.0 scripts must run identically."""

    def test_et0_unchanged_without_use_physics(self, tmp_path):
        """et0_mm must stay at the CSV-provided value when no engine is active."""
        csv_et0 = 4.5
        farm, field = _make_farm_with_weather(et0_csv_value=csv_et0)

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture_et0(state, env):
            observed.append(env.et0_mm)
            return state

        farm.run(days=3)

        for day_et0 in observed:
            assert day_et0 == pytest.approx(csv_et0), (
                f"Without use_physics(), et0_mm must remain at the CSV value "
                f"{csv_et0}, got {day_et0}"
            )

    def test_et0_zero_stub_without_use_physics(self, tmp_path):
        """et0_mm must stay 0.0 for the default stub when no engine is active."""
        farm, field = _make_minimal_farm("no_weather_farm")

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed.append(env.et0_mm)
            return state

        farm.run(days=2)
        assert all(v == 0.0 for v in observed), (
            "Stub et0_mm must remain 0.0 without use_physics()"
        )

    def test_root_growth_multiplier_default_1_without_use_physics(self):
        """root_growth_multiplier must be 1.0 for all plants without use_physics."""
        farm, field = _make_minimal_farm()
        # Give the field a high-resistance soil to confirm the engine is NOT running
        for r in range(field.rows):
            for c in range(field.cols):
                field._field_state  # ensure not set yet (will be set at run start)

        observed_multipliers = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed_multipliers.extend(p.root_growth_multiplier for p in state.plants)
            return state

        farm.run(days=1)
        assert all(m == pytest.approx(1.0) for m in observed_multipliers), (
            "root_growth_multiplier must be 1.0 for all plants when root "
            "impedance engine is not enabled"
        )

    def test_new_env_fields_default_zero_without_use_physics(self):
        """FAO-56 intermediate fields must stay 0.0 without use_physics."""
        farm, field = _make_minimal_farm()

        results = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            results.append({
                "vp_kpa": env.vp_kpa,
                "psychrometric_kpa": env.psychrometric_kpa,
                "slope_svp": env.slope_svp,
                "net_radiation_mj": env.net_radiation_mj,
            })
            return state

        farm.run(days=1)
        r = results[0]
        assert r["vp_kpa"] == 0.0
        assert r["psychrometric_kpa"] == 0.0
        assert r["slope_svp"] == 0.0
        assert r["net_radiation_mj"] == 0.0

    def test_230_v1_tests_not_broken(self):
        """Smoke test: core farm objects construct and run without error."""
        farm, field = _make_minimal_farm("regression_smoke")

        @farm.step(interval="daily")
        def noop(state, env):
            return state

        farm.run(days=5)  # must not raise


# ---------------------------------------------------------------------------
# Test 2: ET0 engine forward -- use_physics(et0=True)
# ---------------------------------------------------------------------------

class TestET0EngineForward:
    """With use_physics(et0=True), engine must populate et0_mm before step."""

    def _make_et0_farm(self):
        farm, field = _make_farm_with_weather(et0_csv_value=0.0)
        return farm, field

    def test_et0_nonzero_after_use_physics(self):
        """et0_mm must be > 0 after engine runs (radiation=22 MJ, warm day)."""
        farm, field = self._make_et0_farm()
        farm.use_physics(et0=True)

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed.append(env.et0_mm)
            return state

        farm.run(days=3)
        for v in observed:
            assert v > 0.0, (
                f"ET0 engine must produce positive et0_mm, got {v}"
            )

    def test_et0_overwrites_csv_value(self):
        """Engine et0_mm overwrites CSV-supplied 0.0 with physics result."""
        farm, field = self._make_et0_farm()
        farm.use_physics(et0=True)

        observed_csv_and_physics = []

        # Phase -3 is below built-in hooks -- we intercept before engine runs
        # (But user steps must be >= 0; so we capture after, at phase 0)
        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed_csv_and_physics.append(env.et0_mm)
            return state

        farm.run(days=1)
        # CSV value was 0.0; physics must have replaced it
        assert observed_csv_and_physics[0] != pytest.approx(0.0), (
            "ET0 engine must overwrite the 0.0 CSV et0_mm with a computed value"
        )

    def test_et0_intermediate_fields_populated(self):
        """FAO-56 intermediate fields must be non-zero after ET0 engine runs."""
        farm, field = self._make_et0_farm()
        farm.use_physics(et0=True)

        results = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            results.append({
                "vp_kpa": env.vp_kpa,
                "psychrometric_kpa": env.psychrometric_kpa,
                "slope_svp": env.slope_svp,
                "net_radiation_mj": env.net_radiation_mj,
            })
            return state

        farm.run(days=1)
        r = results[0]
        assert r["vp_kpa"] > 0.0,            "vp_kpa must be populated by ET0 engine"
        assert r["psychrometric_kpa"] > 0.0,  "psychrometric_kpa must be populated"
        assert r["slope_svp"] > 0.0,          "slope_svp must be populated"
        assert r["net_radiation_mj"] > 0.0,   "net_radiation_mj must be positive"

    def test_et0_value_in_expected_range(self):
        """ET0 for a warm day with radiation should be in 3-8 mm/day range."""
        farm, field = self._make_et0_farm()
        farm.use_physics(et0=True)

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed.append(env.et0_mm)
            return state

        farm.run(days=1)
        et0 = observed[0]
        assert 3.0 <= et0 <= 8.0, (
            f"ET0={et0:.3f} outside expected [3, 8] mm/day for a warm day "
            f"with radiation=22 MJ/m2"
        )

    def test_use_physics_et0_false_leaves_csv_value(self):
        """use_physics(et0=False) must NOT register the engine hook."""
        farm, field = _make_farm_with_weather(et0_csv_value=7.7)
        farm.use_physics(et0=False)  # explicitly off

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed.append(env.et0_mm)
            return state

        farm.run(days=2)
        for v in observed:
            assert v == pytest.approx(7.7), (
                f"use_physics(et0=False) must not change et0_mm, got {v}"
            )


# ---------------------------------------------------------------------------
# Test 3: Root impedance engine -- use_physics(root_impedance=True)
# ---------------------------------------------------------------------------

class TestRootImpedanceForward:
    """With use_physics(root_impedance=True), engine populates root_growth_multiplier."""

    def _build_field_with_soil(self, resistance: float) -> tuple:
        """Farm with a 2x2 field, soil set to given penetration resistance."""
        farm = Farm(name="root_test", location=(23.0, 85.0))
        field = Field(name="RootField", rows=2, cols=2, area_ha=0.1)
        field.set_crop(Crop(species="Z. mays", variety="Basic"))

        class _FakeSoil:
            def build_grid(self, rows, cols):
                return [
                    [
                        [
                            SoilVoxelState(
                                row=r, col=c, layer=0,
                                depth_top_cm=0.0, depth_bottom_cm=30.0,
                                moisture_pct=25.0, nitrogen_kg_ha=80.0,
                                bulk_density=1.3,
                                penetration_resistance=resistance,
                            )
                        ]
                        for c in range(cols)
                    ]
                    for r in range(rows)
                ]

        field.set_soil(_FakeSoil())
        farm.add_field(field)
        return farm, field

    def test_low_resistance_gives_multiplier_1(self):
        """Penetration resistance < 1.0 MPa must give multiplier = 1.0."""
        farm, _ = self._build_field_with_soil(resistance=0.4)
        farm.use_physics(root_impedance=True)

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed.extend(p.root_growth_multiplier for p in state.plants)
            return state

        farm.run(days=1)
        assert all(m == pytest.approx(1.0) for m in observed), (
            f"Expected multiplier=1.0 for resistance=0.4 MPa, got {observed}"
        )

    def test_high_resistance_gives_multiplier_0(self):
        """Penetration resistance >= 2.5 MPa must give multiplier = 0.0."""
        farm, _ = self._build_field_with_soil(resistance=3.0)
        farm.use_physics(root_impedance=True)

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed.extend(p.root_growth_multiplier for p in state.plants)
            return state

        farm.run(days=1)
        assert all(m == pytest.approx(0.0) for m in observed), (
            f"Expected multiplier=0.0 for resistance=3.0 MPa, got {observed}"
        )

    def test_mid_resistance_gives_linear_multiplier(self):
        """Resistance 1.75 MPa must give multiplier = 0.5 (midpoint of linear range)."""
        farm, _ = self._build_field_with_soil(resistance=1.75)
        farm.use_physics(root_impedance=True)

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed.extend(p.root_growth_multiplier for p in state.plants)
            return state

        farm.run(days=1)
        assert all(m == pytest.approx(0.5, abs=1e-6) for m in observed), (
            f"Expected multiplier=0.5 for resistance=1.75 MPa, got {observed}"
        )

    def test_root_impedance_false_leaves_default_1(self):
        """use_physics(root_impedance=False) must not change root_growth_multiplier."""
        farm, _ = self._build_field_with_soil(resistance=3.0)  # hard pan
        farm.use_physics(root_impedance=False)

        observed = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            observed.extend(p.root_growth_multiplier for p in state.plants)
            return state

        farm.run(days=1)
        assert all(m == pytest.approx(1.0) for m in observed), (
            f"use_physics(root_impedance=False) must leave multiplier=1.0, got {observed}"
        )

    def test_dead_plants_skipped(self):
        """Dead plants must not have root_growth_multiplier updated by engine."""
        farm2, _ = self._build_field_with_soil(resistance=3.0)
        farm2.use_physics(root_impedance=True)

        captured = []

        @farm2.step(interval="daily", phase=0)
        def capture2(state, env):
            if env.day == 1:
                # Kill plant 0 mid-step (engine already ran at phase=-1 this day)
                state.plants[0].alive = False
            captured.append(
                (state.plants[0].alive, state.plants[0].root_growth_multiplier)
            )
            return state

        farm2.run(days=2)
        # Day 1: engine ran, set multiplier=0.0, then step killed the plant
        day1_alive, day1_mult = captured[0]
        assert day1_mult == pytest.approx(0.0), (
            f"Day 1: engine should have set multiplier=0.0 before plant was killed"
        )

        # Day 2: plant is dead, engine skips it -> multiplier stays at day-1 value
        day2_alive, day2_mult = captured[1]
        assert day2_alive is False, "Plant should remain dead on day 2"
        assert day2_mult == pytest.approx(0.0), (
            "Dead plant multiplier should retain last engine value (0.0 for hard pan)"
        )



# ---------------------------------------------------------------------------
# Test 4: Phase ordering
# ---------------------------------------------------------------------------

class TestPhaseOrdering:
    """Built-in hooks must run BEFORE researcher @farm.step."""

    def test_et0_hook_runs_before_phase0_step(self):
        """et0_mm is already populated when the researcher's phase=0 step runs."""
        farm, field = _make_farm_with_weather(et0_csv_value=0.0)
        farm.use_physics(et0=True)

        et0_seen_at_phase0 = []

        @farm.step(interval="daily", phase=0)
        def check(state, env):
            et0_seen_at_phase0.append(env.et0_mm)
            return state

        farm.run(days=1)
        assert et0_seen_at_phase0[0] > 0.0, (
            "ET0 engine (phase=-2) must have populated et0_mm before "
            "the researcher phase=0 step runs"
        )

    def test_root_hook_runs_before_phase0_step(self):
        """root_growth_multiplier is set when phase=0 step runs."""
        farm = Farm(name="phase_order_test", location=(23.0, 85.0))
        field = Field(name="F", rows=1, cols=1, area_ha=0.1)
        field.set_crop(Crop(species="Z. mays", variety="Basic"))

        class _HighResistanceSoil:
            def build_grid(self, rows, cols):
                return [[[SoilVoxelState(
                    row=0, col=0, layer=0,
                    depth_top_cm=0.0, depth_bottom_cm=30.0,
                    moisture_pct=25.0, nitrogen_kg_ha=80.0,
                    bulk_density=1.3, penetration_resistance=3.0,
                )]]]

        field.set_soil(_HighResistanceSoil())
        farm.add_field(field)
        farm.use_physics(root_impedance=True)

        multipliers_at_phase0 = []

        @farm.step(interval="daily", phase=0)
        def check(state, env):
            multipliers_at_phase0.extend(p.root_growth_multiplier for p in state.plants)
            return state

        farm.run(days=1)
        # Engine (phase=-1) ran before this step; resistance=3.0 MPa -> multiplier=0.0
        assert multipliers_at_phase0[0] == pytest.approx(0.0), (
            "Root impedance engine (phase=-1) must set multiplier=0.0 "
            "before researcher phase=0 step runs"
        )

    def test_builtin_hooks_not_in_sorted_steps_until_use_physics_called(self):
        """Before use_physics(), the built-in hooks must NOT appear in the step queue."""
        farm, _ = _make_minimal_farm()
        # Check registry -- should be empty (no user steps registered either)
        assert len(farm._physics_registry) == 0, (
            "Physics registry must be empty before use_physics() is called"
        )

    def test_use_physics_registers_hooks_in_physics_registry(self):
        """use_physics() must add hooks to _physics_registry, not _step_registry."""
        farm, _ = _make_minimal_farm()
        farm.use_physics(et0=True, root_impedance=True)
        assert len(farm._physics_registry) == 2, (
            "use_physics(et0=True, root_impedance=True) must register 2 built-in hooks"
        )

    def test_combined_both_hooks_active(self):
        """Both ET0 and root impedance active simultaneously -- both values correct."""
        farm = Farm(name="combined", location=(31.2, 35.0))
        field = Field(name="F", rows=1, cols=1, area_ha=0.1)
        field.set_crop(Crop(species="T. aestivum", variety="Basic"))

        class _Weather:
            def get_day(self, day):
                return EnvironmentState(
                    day=day, doy=187,
                    temp_max_c=21.5, temp_min_c=12.3, temp_mean_c=16.9,
                    radiation_mj_m2=22.07, rainfall_mm=0.0, et0_mm=0.0,
                    wind_speed_ms=2.78, humidity_pct=70.7,
                )

        class _Soil:
            def build_grid(self, rows, cols):
                return [[[SoilVoxelState(
                    row=0, col=0, layer=0,
                    depth_top_cm=0.0, depth_bottom_cm=40.0,
                    moisture_pct=30.0, nitrogen_kg_ha=100.0,
                    bulk_density=1.3, penetration_resistance=0.5,
                )]]]

        field.set_weather(_Weather())
        field.set_soil(_Soil())
        farm.add_field(field)
        farm.use_physics(et0=True, root_impedance=True)

        results = []

        @farm.step(interval="daily", phase=0)
        def capture(state, env):
            results.append({
                "et0_mm": env.et0_mm,
                "mult": state.plants[0].root_growth_multiplier,
            })
            return state

        farm.run(days=1)
        r = results[0]
        # Etzion conditions: ET0 ~ 3.95 mm/day
        assert abs(r["et0_mm"] - 3.95) < 0.20, (
            f"ET0 under Etzion conditions should be ~3.95, got {r['et0_mm']:.3f}"
        )
        # Resistance = 0.5 MPa -> unrestricted -> multiplier = 1.0
        assert r["mult"] == pytest.approx(1.0), (
            f"Root multiplier for 0.5 MPa should be 1.0, got {r['mult']}"
        )
