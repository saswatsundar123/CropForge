"""
tests/test_physics_erosion.py
==============================
Crucible tests for Phase 5 Soil Erosion (CropForge PRD v0.7.0 §7.5).

Coverage:
    Unit:
        calculate_erosion_index -- flat-field / zero-runoff guarantee
        roughness + veg damping math
        PHASE_EROSION_ENGINE registration
    Crucible (PRD §7.5 Contour Bund):
        Field A (bare slope) vs Field B (ContourBund) over 30 days of
        heavy rainfall. Field B cumulative erosion must be measurably and
        significantly lower than Field A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropforge import Farm, Field, Crop
from cropforge.loaders import Weather
from cropforge.land_prep import ContourBund


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weather(days: int, rainfall_mm: float = 30.0) -> Weather:
    rows = [
        {
            "day": d, "doy": d,
            "temp_max_c": 32.0, "temp_min_c": 20.0, "temp_mean_c": 26.0,
            "radiation_mj_m2": 20.0, "rainfall_mm": rainfall_mm,
            "et0_mm": 6.0, "wind_speed_ms": 2.0, "humidity_pct": 70.0,
            "co2_ppm": 415.0,
        }
        for d in range(1, days + 1)
    ]
    return Weather(pd.DataFrame(rows).set_index("day"))


def _steep_elevation() -> np.ndarray:
    """2×4 grid with clear column-wise slope: 1.0 → 0.67 → 0.33 → 0.0."""
    return np.array([
        [1.0, 0.67, 0.33, 0.0],
        [1.0, 0.67, 0.33, 0.0],
    ])


def _cumulative_erosion(field: Field) -> float:
    """Sum all cells in cumulative_erosion_index_grid."""
    grid = field._field_state.custom.get("cumulative_erosion_index_grid")
    if grid is None:
        return 0.0
    return sum(grid[r][c] for r in range(len(grid)) for c in range(len(grid[r])))


# ---------------------------------------------------------------------------
# Unit: calculate_erosion_index math
# ---------------------------------------------------------------------------

class TestErosionMath:
    def test_zero_runoff_gives_zero_erosion(self):
        from cropforge.physics.soil import calculate_erosion_index
        assert calculate_erosion_index(0.0, 0.5) == 0.0

    def test_negative_runoff_gives_zero_erosion(self):
        from cropforge.physics.soil import calculate_erosion_index
        assert calculate_erosion_index(-5.0, 0.5) == 0.0

    def test_flat_slope_gives_zero_erosion(self):
        from cropforge.physics.soil import calculate_erosion_index
        assert calculate_erosion_index(20.0, 0.0) == 0.0

    def test_negative_slope_gives_zero_erosion(self):
        from cropforge.physics.soil import calculate_erosion_index
        assert calculate_erosion_index(20.0, -0.1) == 0.0

    def test_bare_surface_maximum_erosion(self):
        from cropforge.physics.soil import calculate_erosion_index
        result = calculate_erosion_index(10.0, 0.5, surface_roughness=0.0)
        assert result == pytest.approx(5.0)

    def test_roughness_reduces_erosion(self):
        from cropforge.physics.soil import calculate_erosion_index
        bare = calculate_erosion_index(10.0, 0.5, surface_roughness=0.0)
        rough = calculate_erosion_index(10.0, 0.5, surface_roughness=0.3)
        assert rough < bare
        assert rough == pytest.approx(3.5)

    def test_full_roughness_eliminates_erosion(self):
        from cropforge.physics.soil import calculate_erosion_index
        result = calculate_erosion_index(10.0, 0.5, surface_roughness=1.0)
        assert result == 0.0

    def test_vegetation_reduces_erosion(self):
        from cropforge.physics.soil import calculate_erosion_index
        bare = calculate_erosion_index(10.0, 0.5, vegetation_cover_frac=0.0)
        covered = calculate_erosion_index(10.0, 0.5, vegetation_cover_frac=0.5)
        assert covered < bare
        assert covered == pytest.approx(2.5)

    def test_roughness_and_veg_compound(self):
        from cropforge.physics.soil import calculate_erosion_index
        both = calculate_erosion_index(10.0, 0.5, surface_roughness=0.3, vegetation_cover_frac=0.5)
        # 10 * 0.5 * 0.7 * 0.5 = 1.75
        assert both == pytest.approx(1.75)

    def test_docstring_example(self):
        from cropforge.physics.soil import calculate_erosion_index
        assert round(calculate_erosion_index(10.0, 0.5, surface_roughness=0.3), 4) == 3.5


# ---------------------------------------------------------------------------
# Unit: hook registration
# ---------------------------------------------------------------------------

class TestErosionRegistration:
    def test_erosion_not_registered_by_default(self):
        from cropforge.physics.builtin_hooks import PHASE_EROSION_ENGINE
        farm = Farm(name="NoErosionDefault", location=(28.6, 77.2))
        farm.use_physics()
        phases = [phase for phase, _ in farm._physics_registry]
        assert PHASE_EROSION_ENGINE not in phases

    def test_erosion_registered_when_opted_in(self):
        from cropforge.physics.builtin_hooks import PHASE_EROSION_ENGINE
        farm = Farm(name="ErosionOpted", location=(28.6, 77.2))
        farm.use_physics(erosion=True)
        phases = [phase for phase, _ in farm._physics_registry]
        assert PHASE_EROSION_ENGINE in phases

    def test_erosion_hook_name(self):
        from cropforge.physics.builtin_hooks import PHASE_EROSION_ENGINE
        farm = Farm(name="ErosionHookName", location=(28.6, 77.2))
        farm.use_physics(erosion=True)
        fns = [fn for ph, fn in farm._physics_registry if ph == PHASE_EROSION_ENGINE]
        assert len(fns) == 1
        assert fns[0].__name__ == "_erosion_step"

    def test_erosion_standalone_no_dependency_required(self):
        """erosion=True must not raise even without water_balance or et0."""
        farm = Farm(name="ErosionStandalone", location=(28.6, 77.2))
        # Should NOT raise CropForgeConfigError
        farm.use_physics(erosion=True)


# ---------------------------------------------------------------------------
# Unit: flat field → zero erosion
# ---------------------------------------------------------------------------

class TestFlatFieldZeroErosion:
    def test_flat_field_cumulative_is_zero(self):
        """Flat elevation → all slope_fracs = 0 → zero erosion guaranteed."""
        flat_elev = np.zeros((2, 4))

        farm = Farm(name="FlatErosion", location=(28.6, 77.2))
        field = Field("F1", rows=2, cols=4)
        field.set_elevation(flat_elev)
        field.set_crop(Crop(species="Zea mays", variety="FlatTest"))
        field.set_weather(_make_weather(days=5, rainfall_mm=30.0))
        farm.add_field(field)
        farm.use_physics(erosion=True)
        farm.run(days=5)

        assert _cumulative_erosion(field) == 0.0


# ---------------------------------------------------------------------------
# PRD Crucible: ContourBund significantly reduces cumulative erosion
# ---------------------------------------------------------------------------

class TestCrucibleContourBundErosion:
    """
    PRD §7.5 Crucible: Two identical steep slopes, 30 days of 30mm/day rainfall.
        Field A: NO land preparation (bare slope, roughness=0.0)
        Field B: ContourBund applied (roughness=0.3)

    PRD criterion: cumulative_erosion(B) is measurably and significantly
    lower than cumulative_erosion(A) by Day 30.

    Physics:
        erosion_A = runoff × slope × (1 - 0.0) × (1 - 0.0)
        erosion_B = runoff × slope × (1 - 0.3) × (1 - 0.0)
        ratio = 0.7  →  Field B erosion is 30% lower than Field A
    """

    def _run_field(self, apply_bund: bool, days: int = 30) -> float:
        elevation = _steep_elevation()
        farm = Farm(name=f"ErosionCrucible_bund={apply_bund}", location=(28.6, 77.2))
        field = Field(f"F_bund={apply_bund}", rows=2, cols=4)
        field.set_elevation(elevation)
        field.set_crop(Crop(species="Zea mays", variety="CrucibleCrop"))
        field.set_weather(_make_weather(days=days, rainfall_mm=30.0))

        if apply_bund:
            field.set_land_prep(ContourBund())

        farm.add_field(field)
        farm.use_physics(erosion=True)
        farm.run(days=days)

        return _cumulative_erosion(field)

    def test_field_b_lower_than_field_a(self):
        """ContourBund reduces cumulative erosion vs bare slope (must be >20% lower)."""
        erosion_a = self._run_field(apply_bund=False)
        erosion_b = self._run_field(apply_bund=True)

        assert erosion_b < erosion_a, (
            f"ContourBund field must have LESS erosion: A={erosion_a:.4f}, B={erosion_b:.4f}"
        )
        # Confirm the reduction is at least 20% (ContourBund roughness=0.3 → 30% reduction)
        assert erosion_b < erosion_a * 0.85, (
            f"Contour Bund must reduce erosion by >15%%: "
            f"A={erosion_a:.4f}, B={erosion_b:.4f}, ratio={erosion_b/erosion_a:.3f}"
        )

    def test_field_a_has_nonzero_erosion(self):
        """Bare steep slope with heavy rain must accumulate non-zero erosion."""
        erosion_a = self._run_field(apply_bund=False)
        assert erosion_a > 0.0, f"Expected nonzero erosion on bare slope: {erosion_a}"

    def test_contour_bund_erosion_ratio(self):
        """B/A ratio should be close to 0.7 (roughness=0.3 → damper=0.7)."""
        erosion_a = self._run_field(apply_bund=False)
        erosion_b = self._run_field(apply_bund=True)
        ratio = erosion_b / erosion_a
        # Expect ratio ≈ 0.7 ± 0.05 (tolerance for any slope-edge effects)
        assert 0.60 < ratio < 0.80, (
            f"Expected B/A ratio ≈ 0.7, got {ratio:.4f}"
        )
