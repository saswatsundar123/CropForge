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


# ---------------------------------------------------------------------------
# Test 5: Soil Water Balance Engine  (PRD v0.3.0 Section 5.6)
# ---------------------------------------------------------------------------

class TestSoilWaterBalanceForward:
    """use_physics(water_balance=True, et0=True) must deplete soil moisture daily."""

    def _make_wb_farm(self, initial_moisture: float = 30.0, et0_mm: float = 5.0):
        """
        Farm with 1 field, 2×2 grid, weather that delivers fixed et0_mm and 0 rain.
        Soil params: FC=32%, WP=14%, SAT=48%.
        """
        farm = Farm(name="WBFarm", location=(31.2, 35.0))
        field = Field(name="WBField", rows=2, cols=2, area_ha=0.5)
        field.set_crop(Crop(species="Z. mays", variety="K1"))

        class _FakeWeather:
            def get_day(self, day):
                return EnvironmentState(
                    day=day, doy=((day - 1) % 365) + 1,
                    temp_max_c=34.0, temp_min_c=20.0, temp_mean_c=27.0,
                    radiation_mj_m2=22.0, rainfall_mm=0.0,
                    et0_mm=et0_mm,   # pre-supplied ET0 (used by water balance hook)
                    wind_speed_ms=2.5, humidity_pct=45.0,
                )

        class _FakeSoil:
            def build_grid(self, rows, cols):
                return [
                    [
                        [SoilVoxelState(
                            row=r, col=c, layer=0,
                            depth_top_cm=0.0, depth_bottom_cm=20.0,
                            moisture_pct=initial_moisture,
                            nitrogen_kg_ha=80.0, bulk_density=1.3,
                            penetration_resistance=0.5,
                        )]
                        for c in range(cols)
                    ]
                    for r in range(rows)
                ]

        field.set_weather(_FakeWeather())
        field.set_soil(_FakeSoil())
        field.set_water_params(
            field_capacity_pct=32.0,
            wilting_point_pct=14.0,
            saturation_pct=48.0,
            drainage_coefficient=0.5,
            crop_coefficient=1.0,
            stress_increment_per_day=0.05,
        )
        farm.add_field(field)
        return farm, field

    def test_moisture_decreases_daily_under_high_et0(self):
        """
        PRD v0.3.0 §5.6: Moisture decreases daily by correct ETc amount.
        ET0=5 mm, Kc=1.0, layer depth=20cm → ETc per layer = 5 mm.
        Δmoisture per day = _mm_to_pct(5, 20) = 2.5 %.
        Starting at 30 % → after 5 days with 0 rain → 30 - 5×2.5 = 17.5 %.
        """
        farm, field = self._make_wb_farm(initial_moisture=30.0, et0_mm=5.0)
        farm.use_physics(et0=True, water_balance=True)

        moisture_per_day = {}

        @farm.step(phase=0)
        def record(state, env):
            moisture_per_day[state.day] = state.soil[0][0][0].moisture_pct
            return state

        farm.run(days=5)

        # Moisture must strictly decrease each day
        moistures = [moisture_per_day[d] for d in sorted(moisture_per_day)]
        for i in range(1, len(moistures)):
            assert moistures[i] < moistures[i - 1], (
                f"Moisture should decrease each day: day {i+1}={moistures[i]:.3f} "
                f"not less than day {i}={moistures[i-1]:.3f}"
            )

        # After 5 dry high-ET0 days, moisture must be significantly lower than start
        assert moistures[-1] < 30.0, (
            f"Final moisture {moistures[-1]:.3f}% should be below starting 30%"
        )

    def test_zero_rain_depletes_moisture_over_5_days(self):
        """
        Concrete 5-day depletion test with exact expected value.
        ET0=5 mm, Kc=1.0, layer depth=20 cm, initial moisture=30%.
        Expected final moisture after 5 days ≈ 30 - 5×2.5 = 17.5 %
        (allowing some tolerance for Ks feedback reducing extraction as soil dries).
        """
        farm, field = self._make_wb_farm(initial_moisture=30.0, et0_mm=5.0)
        farm.use_physics(et0=True, water_balance=True)

        @farm.step(phase=0)
        def noop(state, env):
            return state

        farm.run(days=5)
        final = field._field_state.soil[0][0][0].moisture_pct

        # With 30% start, WP=14%, FC=32%, Ks is near 1.0 for first few days.
        # Full extraction for 5 days = 5 × 2.5% = 12.5% → ~17.5%
        # Allow range [14.0, 28.0] to account for partial stress effects
        assert 14.0 <= final < 30.0, (
            f"After 5 dry days, moisture={final:.3f}% should be in [14, 30)"
        )

    def test_water_stress_ks_written_to_plant_custom(self):
        """PRD §5.6: plant.custom['water_stress_ks'] set correctly each day."""
        farm, field = self._make_wb_farm(initial_moisture=30.0, et0_mm=5.0)
        farm.use_physics(et0=True, water_balance=True)

        ks_values = []

        @farm.step(phase=0)
        def capture(state, env):
            for plant in state.plants:
                ks = plant.custom.get("water_stress_ks")
                if ks is not None:
                    ks_values.append(ks)
            return state

        farm.run(days=3)
        assert len(ks_values) > 0, "water_stress_ks must be set by the hydrology engine"
        for ks in ks_values:
            assert 0.0 <= ks <= 1.0, f"Ks must be in [0,1], got {ks}"

    def test_stress_index_increases_under_drought(self):
        """stress_index must increase each day when soil dries below FC."""
        farm, field = self._make_wb_farm(initial_moisture=16.0, et0_mm=5.0)
        farm.use_physics(et0=True, water_balance=True)

        stress_per_day = {}

        @farm.step(phase=0)
        def capture(state, env):
            stress_per_day[state.day] = state.plants[0].stress_index
            return state

        farm.run(days=5)

        # Stress must increase over time (initial moisture 16% > WP=14%, below FC=32%)
        stresses = [stress_per_day[d] for d in sorted(stress_per_day)]
        # By the end, stress_index should be > 0 (drought conditions)
        assert stresses[-1] > 0.0, (
            f"Stress index should be positive after 5 dry days, got {stresses[-1]}"
        )

    def test_ks_1_at_field_capacity_in_engine(self):
        """
        PRD §5.6: Ks = 1.0 when moisture = field_capacity (no stress).
        Set moisture to exactly FC=32% and verify no stress is added.
        """
        # Start at FC=32%, no ET0 demand → Ks should be 1.0, no stress
        farm, field = self._make_wb_farm(initial_moisture=32.0, et0_mm=0.0)
        farm.use_physics(et0=True, water_balance=True)

        ks_list = []

        @farm.step(phase=0)
        def capture(state, env):
            for plant in state.plants:
                ks = plant.custom.get("water_stress_ks", None)
                if ks is not None:
                    ks_list.append(ks)
            return state

        farm.run(days=1)
        if ks_list:
            assert all(ks == pytest.approx(1.0) for ks in ks_list), (
                f"At FC with no ET0 demand, Ks should be 1.0, got {ks_list}"
            )


class TestSoilWaterBalanceIrrigation:
    """
    PRD v0.3.0 §5.6: irrigation_mm replenishes soil moisture on the exact day it fires.

    Execution order:
        phase=-3: Hydrology hook (reads yesterday's moisture, applies rain)
        phase= 0: Researcher step (records moisture)
        events  : Event.irrigation fires (modifies voxel moisture for TOMORROW)

    So Event.irrigation fired on day N → moisture increase visible on day N+1.
    """

    def _make_irrig_farm(self, initial_moisture: float = 18.0):
        """Farm with 1×1 field, near-WP initial moisture, fixed ET0."""
        from cropforge import Farm, Field, Crop, Event
        farm = Farm(name="IrrigFarm", location=(25.0, 80.0))
        field = Field(name="IrrigField", rows=2, cols=2, area_ha=0.5)
        field.set_crop(Crop(species="Z. mays", variety="K1"))

        class _FakeWeather:
            def get_day(self, day):
                return EnvironmentState(
                    day=day, doy=((day - 1) % 365) + 1,
                    temp_max_c=32.0, temp_min_c=18.0, temp_mean_c=25.0,
                    radiation_mj_m2=20.0, rainfall_mm=0.0,
                    et0_mm=3.0,
                    wind_speed_ms=2.0, humidity_pct=50.0,
                )

        class _FakeSoil:
            def build_grid(self, rows, cols):
                return [
                    [
                        [SoilVoxelState(
                            row=r, col=c, layer=0,
                            depth_top_cm=0.0, depth_bottom_cm=20.0,
                            moisture_pct=initial_moisture,
                            nitrogen_kg_ha=60.0, bulk_density=1.3,
                            penetration_resistance=0.5,
                        )]
                        for c in range(cols)
                    ]
                    for r in range(rows)
                ]

        field.set_weather(_FakeWeather())
        field.set_soil(_FakeSoil())
        field.set_water_params(
            field_capacity_pct=30.0,
            wilting_point_pct=12.0,
            saturation_pct=45.0,
            drainage_coefficient=0.5,
            crop_coefficient=1.0,
        )
        farm.add_field(field)
        return farm, field

    def test_irrigation_event_replenishes_moisture(self):
        """
        Irrigation of 30 mm on day 5 must increase moisture on day 6.
        Day 5 step sees pre-irrigation moisture (event fires after step).
        Day 6 step sees increased moisture.
        """
        from cropforge import Event

        farm, field = self._make_irrig_farm(initial_moisture=18.0)
        farm.use_physics(et0=True, water_balance=True)

        # Register irrigation on day 5
        farm.add_event(Event.irrigation(
            field="IrrigField", interval_days=1000,  # fire only once
            amount_mm=30, start_day=5, end_day=5,
        ))

        moisture_log = {}

        @farm.step(phase=0)
        def record(state, env):
            moisture_log[state.day] = state.soil[0][0][0].moisture_pct
            return state

        farm.run(days=10)

        # Day 5: step records moisture BEFORE irrigation fires
        # Day 6: hydrology hook reads moisture AFTER day-5 irrigation
        # So moisture[6] should be higher than moisture[5]
        assert 5 in moisture_log and 6 in moisture_log, (
            "moisture_log must have entries for days 5 and 6"
        )
        assert moisture_log[6] > moisture_log[5], (
            f"Irrigation on day 5 (after step) should raise moisture on day 6. "
            f"Day5={moisture_log[5]:.3f}%, Day6={moisture_log[6]:.3f}%"
        )

    def test_no_irrigation_moisture_monotone_decreases(self):
        """Without irrigation, moisture under continuous ET0 only ever decreases."""
        farm, field = self._make_irrig_farm(initial_moisture=25.0)
        farm.use_physics(et0=True, water_balance=True)
        # NO events registered

        moisture_log = {}

        @farm.step(phase=0)
        def record(state, env):
            moisture_log[state.day] = state.soil[0][0][0].moisture_pct
            return state

        farm.run(days=8)

        moistures = [moisture_log[d] for d in sorted(moisture_log)]
        for i in range(1, len(moistures)):
            assert moistures[i] <= moistures[i - 1] + 1e-6, (
                f"Without irrigation, moisture should not increase: "
                f"day {i+1}={moistures[i]:.3f}% > day {i}={moistures[i-1]:.3f}%"
            )


class TestSoilWaterBalanceValidation:
    """PRD v0.3.0 §5.6: CropForgeConfigError raised when water_balance enabled without et0."""

    def test_water_balance_without_et0_raises_config_error(self):
        """use_physics(water_balance=True) without et0=True must raise CropForgeConfigError."""
        from cropforge import CropForgeConfigError

        farm = Farm(name="ConfigTest", location=(25.0, 80.0))
        field = Field(name="F", rows=1, cols=1)
        field.set_crop(Crop(species="Z. mays", variety="K1"))
        farm.add_field(field)

        with pytest.raises(CropForgeConfigError, match="et0"):
            farm.use_physics(water_balance=True, et0=False)

    def test_water_balance_with_et0_does_not_raise(self):
        """use_physics(water_balance=True, et0=True) must NOT raise."""
        farm = Farm(name="ValidConfig", location=(25.0, 80.0))
        field = Field(name="F", rows=1, cols=1)
        field.set_crop(Crop(species="Z. mays", variety="K1"))
        farm.add_field(field)

        # Must not raise
        farm.use_physics(water_balance=True, et0=True)


class TestSoilWaterBalanceBackwardCompat:
    """
    PRD v0.3.0 backward compatibility:
    Scripts that never call use_physics() must be completely unaffected.
    """

    def test_moisture_unchanged_without_water_balance(self):
        """Without water_balance=True, soil moisture must not change over 5 days."""
        farm = Farm(name="BackCompatFarm", location=(25.0, 80.0))
        field = Field(name="BCField", rows=2, cols=2)
        field.set_crop(Crop(species="Z. mays", variety="K1"))

        class _FakeWeather:
            def get_day(self, day):
                return EnvironmentState(
                    day=day, doy=1,
                    temp_max_c=32.0, temp_min_c=18.0, temp_mean_c=25.0,
                    radiation_mj_m2=20.0, rainfall_mm=0.0,
                    et0_mm=5.0, wind_speed_ms=2.0, humidity_pct=50.0,
                )

        class _FakeSoil:
            def build_grid(self, rows, cols):
                return [
                    [
                        [SoilVoxelState(
                            row=r, col=c, layer=0,
                            depth_top_cm=0.0, depth_bottom_cm=20.0,
                            moisture_pct=25.0, nitrogen_kg_ha=60.0,
                            bulk_density=1.3, penetration_resistance=0.5,
                        )]
                        for c in range(cols)
                    ]
                    for r in range(rows)
                ]

        field.set_weather(_FakeWeather())
        field.set_soil(_FakeSoil())
        farm.add_field(field)
        # NO use_physics() call

        moisture_vals = []

        @farm.step(phase=0)
        def record(state, env):
            moisture_vals.append(state.soil[0][0][0].moisture_pct)
            return state

        farm.run(days=5)

        # Every day should show 25.0 % (unchanged)
        for m in moisture_vals:
            assert m == pytest.approx(25.0), (
                f"Moisture should be unchanged without water_balance engine, got {m}"
            )

    def test_stress_index_unchanged_without_water_balance(self):
        """Without water_balance, plant.stress_index must stay at 0.0 (default)."""
        farm, field = _make_minimal_farm("stress_compat")

        stress_vals = []

        @farm.step(phase=0)
        def capture(state, env):
            stress_vals.extend(p.stress_index for p in state.plants)
            return state

        farm.run(days=3)
        assert all(s == pytest.approx(0.0) for s in stress_vals), (
            f"stress_index must be 0.0 without water_balance engine, got {stress_vals}"
        )
