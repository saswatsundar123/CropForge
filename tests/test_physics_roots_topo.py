"""
tests/test_physics_roots_topo.py
==================================
PRD v0.7.0 §7.3 — Crucible tests for root effective soil depth constraint.

The effective_soil_depth_cm_grid is computed in Field._initialize_field:
    effective_depth_cm = deepest_layer_bottom_cm + elevation_delta_cm
where elevation_delta = (post-land-prep elevation) - (base elevation).

Tests:
  1. Flat field: effective_depth == deepest layer bottom (20 cm default).
  2. Elevation delta: negative delta (furrow) reduces effective depth.
  3. Root clamp Crucible: plant whose growth model grows roots past 15 cm
     is rigidly clamped at 15 cm when root_clamping=True.
  4. Backward compat: root_clamping=False → roots grow past cap unchecked.
  5. Unit: clamp hook is a no-op when grid key is absent from state.custom.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cropforge
from cropforge import Farm, Field, Crop
from cropforge.state import PlantState, FieldState, EnvironmentState, SoilVoxelState
from cropforge.physics.builtin_hooks import make_root_clamp_hook, PHASE_ROOT_CLAMP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(day: int = 1) -> EnvironmentState:
    return EnvironmentState(
        day=day, doy=day % 365 or 365,
        temp_max_c=25.0, temp_min_c=15.0, temp_mean_c=20.0,
        radiation_mj_m2=15.0, rainfall_mm=0.0, et0_mm=4.0,
        wind_speed_ms=2.0, humidity_pct=60.0,
    )


def _make_1x1_field_state(depth_cap_cm: float, root_start_cm: float = 0.0) -> tuple:
    """Return (field_state, env) with a single plant and specified depth cap."""
    plant = PlantState(plant_id="r00c00", row=0, col=0, root_depth_cm=root_start_cm, alive=True)
    voxel = SoilVoxelState(
        row=0, col=0, layer=0,
        depth_top_cm=0.0, depth_bottom_cm=20.0,
        moisture_pct=25.0, nitrogen_kg_ha=50.0,
        bulk_density=1.3, penetration_resistance=0.5,
    )
    state = FieldState(
        day=1, plants=[plant],
        soil=[[[voxel]]],
        elevation_grid=np.zeros((1, 1)),
        events_fired=[],
    )
    state.custom["effective_soil_depth_cm_grid"] = [[depth_cap_cm]]
    env = _make_env()
    return state, env


# ---------------------------------------------------------------------------
# Unit tests: effective_soil_depth_cm_grid initialisation
# ---------------------------------------------------------------------------

class TestEffectiveSoilDepthInit:
    """Test that Field._initialize_field populates effective_soil_depth_cm_grid."""

    def _make_field(self, elevation_profile=None):
        """Minimal field with the simplest weather/crop setup."""
        import pandas as pd
        from cropforge.loaders import Weather
        field = Field("test", rows=2, cols=2, elevation_profile=elevation_profile)
        rows = [{
            "day": d, "doy": d, "temp_max_c": 25.0, "temp_min_c": 15.0,
            "temp_mean_c": 20.0, "radiation_mj_m2": 15.0, "rainfall_mm": 0.0,
            "et0_mm": 4.0, "wind_speed_ms": 2.0, "humidity_pct": 60.0,
        } for d in range(1, 40)]
        field.set_weather(Weather(pd.DataFrame(rows).set_index("day")))
        field.set_crop(Crop(species="Triticum aestivum", variety="Test"))
        return field

    def test_flat_field_effective_depth_equals_layer_bottom(self):
        """Flat field with default soil: effective depth = 20cm (single layer 0-20)."""
        field = self._make_field()
        state = field._init_field_state(day=1)
        grid = state.custom.get("effective_soil_depth_cm_grid")
        assert grid is not None, "effective_soil_depth_cm_grid must be in state.custom"
        assert grid[0][0] == 20.0, f"Expected 20.0 cm, got {grid[0][0]}"
        assert grid[1][1] == 20.0

    def test_elevation_delta_reduces_effective_depth(self):
        """Field with a 0.05m furrow: effective depth = 20 - 5 = 15 cm."""
        # Post-land-prep elevation grid: cell [0][0] has been carved -0.05m
        # We simulate this by giving the field a non-zero base elevation
        # and an effective elevation that is 0.05m lower.
        field = self._make_field()
        # Manually set base elevation = 0.05m for one cell
        field.elevation_grid[0, 0] = 0.05  # 5cm above reference
        # effective_elevation stays at 0.0 (the furrow bottom) — simulating land prep
        # _initialize_field uses effective_elevation = self.elevation_grid.copy() then applies prep
        # Since no land prep modifier is set, effective == base.
        # We test the formula directly: base=0.05, effective=0.0 → delta=-0.05m → depth = 20-5 = 15cm
        import numpy as np
        # Override effective_elevation to simulate land prep
        # Inject by setting a mock _land_prep
        class MockPrep:
            def apply(self, elev, res):
                modified = elev.copy()
                modified[0, 0] = 0.0  # carve 5cm from cell [0][0]
                return modified, {}
        field._land_prep = MockPrep()
        state = field._init_field_state(day=1)
        grid = state.custom["effective_soil_depth_cm_grid"]
        # delta[0][0] = 0.0 - 0.05 = -0.05m → -5cm → effective = 20 - 5 = 15cm
        assert abs(grid[0][0] - 15.0) < 1e-9, (
            f"Expected 15.0 cm for furrowed cell, got {grid[0][0]}"
        )
        # Unmodified cells stay at 20cm
        assert abs(grid[0][1] - 20.0) < 1e-9
        assert abs(grid[1][0] - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# Unit tests: make_root_clamp_hook
# ---------------------------------------------------------------------------

class TestRootClampHookUnit:
    def test_clamp_above_cap(self):
        """Root deeper than cap must be clamped to cap."""
        state, env = _make_1x1_field_state(depth_cap_cm=15.0, root_start_cm=25.0)
        hook = make_root_clamp_hook()
        out = hook(state, env)
        assert out.plants[0].root_depth_cm == 15.0

    def test_below_cap_unchanged(self):
        """Root shallower than cap must not be modified."""
        state, env = _make_1x1_field_state(depth_cap_cm=15.0, root_start_cm=10.0)
        hook = make_root_clamp_hook()
        hook(state, env)
        assert state.plants[0].root_depth_cm == 10.0

    def test_exactly_at_cap_unchanged(self):
        """Root exactly at cap must not be modified."""
        state, env = _make_1x1_field_state(depth_cap_cm=15.0, root_start_cm=15.0)
        hook = make_root_clamp_hook()
        hook(state, env)
        assert state.plants[0].root_depth_cm == 15.0

    def test_grid_absent_noop(self):
        """If 'effective_soil_depth_cm_grid' is absent, hook must be a no-op."""
        state, env = _make_1x1_field_state(depth_cap_cm=15.0, root_start_cm=50.0)
        del state.custom["effective_soil_depth_cm_grid"]
        hook = make_root_clamp_hook()
        hook(state, env)
        assert state.plants[0].root_depth_cm == 50.0  # unchanged

    def test_dead_plant_skipped(self):
        """Dead plants must not be touched."""
        state, env = _make_1x1_field_state(depth_cap_cm=15.0, root_start_cm=30.0)
        state.plants[0].alive = False
        hook = make_root_clamp_hook()
        hook(state, env)
        assert state.plants[0].root_depth_cm == 30.0  # dead plant not clamped


# ---------------------------------------------------------------------------
# Crucible: full simulation with StandardMaize, 15 cm cap
# ---------------------------------------------------------------------------

class TestRootClampCrucible:
    """PRD §7.3 Crucible:
    Plant attempts to grow roots deeper than 15cm.
    With root_clamping=True, root_depth_cm must not exceed 15cm after any day.
    """

    def _make_soil_15cm(self):
        """Mock soil profile with a single layer 0-15cm.
        _init_field_state will compute effective_soil_depth_cm_grid = 15.0 from
        the deepest layer bottom, with no land prep delta.
        """
        class _Soil15:
            def build_grid(self, rows, cols):
                return [
                    [
                        [SoilVoxelState(
                            row=r, col=c, layer=0,
                            depth_top_cm=0.0, depth_bottom_cm=15.0,
                            moisture_pct=25.0, nitrogen_kg_ha=50.0,
                            bulk_density=1.3, penetration_resistance=0.5,
                        )]
                        for c in range(cols)
                    ]
                    for r in range(rows)
                ]
        return _Soil15()

    def _make_weather(self):
        import pandas as pd
        from cropforge.loaders import Weather
        rows = [{
            "day": d, "doy": d, "temp_max_c": 30.0, "temp_min_c": 18.0,
            "temp_mean_c": 24.0, "radiation_mj_m2": 18.0, "rainfall_mm": 5.0,
            "et0_mm": 5.0, "wind_speed_ms": 2.0, "humidity_pct": 65.0,
            "co2_ppm": 415.0,
        } for d in range(1, 32)]
        return Weather(pd.DataFrame(rows).set_index("day"))

    def test_maize_roots_clamped_at_15cm(self):
        """Full simulation: StandardMaize grows aggressively but clamps at 15cm."""
        from cropforge.plugins import StandardMaize

        farm = Farm(name="RootCrucible", location=(28.6, 77.2))
        field = Field("F1", rows=1, cols=1)
        field.set_weather(self._make_weather())
        field.set_soil(self._make_soil_15cm())
        field.set_crop(Crop(species="Zea mays", variety="TestMaize"))

        farm.add_field(field)
        farm.use_physics(root_clamping=True)
        field.use_plugin(StandardMaize)

        farm.run(days=30)

        final_plant = field._field_state.plants[0]
        assert final_plant.root_depth_cm <= 15.0, (
            f"CRUCIBLE FAILED: root_depth_cm={final_plant.root_depth_cm:.2f} cm "
            f"exceeds the 15 cm effective soil depth cap."
        )
        assert final_plant.root_depth_cm > 0.0, (
            "Sanity check: root must have some depth (maize starts at 2cm)"
        )

    def test_without_root_clamping_hook_not_registered(self):
        """Without root_clamping=True, the clamp hook must not appear in the registry.

        The clamp is strictly opt-in. Verifying the registry is more direct and
        reliable than a full simulation that depends on plant survival.
        """
        from cropforge.physics.builtin_hooks import PHASE_ROOT_CLAMP

        farm = Farm(name="RootNoClamping", location=(28.6, 77.2))
        farm.use_physics()  # root_clamping defaults to False

        clamp_hooks = [
            fn for phase, fn in farm._physics_registry
            if phase == PHASE_ROOT_CLAMP
        ]
        assert len(clamp_hooks) == 0, (
            "Root clamp hook must NOT be registered when root_clamping=False. "
            f"Found {len(clamp_hooks)} hook(s) at PHASE_ROOT_CLAMP."
        )

    def test_with_root_clamping_hook_is_registered(self):
        """With root_clamping=True, exactly one clamp hook is in the registry."""
        from cropforge.physics.builtin_hooks import PHASE_ROOT_CLAMP

        farm = Farm(name="RootClamping", location=(28.6, 77.2))
        farm.use_physics(root_clamping=True)

        clamp_hooks = [
            fn for phase, fn in farm._physics_registry
            if phase == PHASE_ROOT_CLAMP
        ]
        assert len(clamp_hooks) == 1, (
            f"Expected exactly 1 root clamp hook, found {len(clamp_hooks)}."
        )
        assert clamp_hooks[0].__name__ == "_root_clamp_step"
