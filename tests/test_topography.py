"""
tests/test_topography.py
========================
Tests for the Terrain class (PRD v0.6.0 Phase 1).

Covers:
  - Determinism: same seed → identical grid (multiple runs)
  - Flat default: Field without set_terrain() → FieldState.terrain is None,
    elevation_grid is all-zero
  - from_array wrapping + shape
  - from_generator: valid callable and shape mismatch error
  - All four procedural generators: correct shapes and physical properties
  - slope_grid / aspect_grid computed and correct shape
  - set_terrain() API: Field.elevation_grid updated, FieldState.terrain set
  - Legacy set_elevation() still works and now sets FieldState.terrain
"""

import math

import numpy as np
import pytest

from cropforge import Field, Terrain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_field(rows=4, cols=5):
    return Field(name="TestField", rows=rows, cols=cols, area_ha=0.1)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_undulating(self):
        a = Terrain.procedural(rows=10, cols=12, generator="undulating", seed=42)
        b = Terrain.procedural(rows=10, cols=12, generator="undulating", seed=42)
        np.testing.assert_array_equal(a.elevation_grid, b.elevation_grid)

    def test_same_seed_multiple_calls(self):
        grids = [
            Terrain.procedural(rows=8, cols=8, generator="undulating", seed=7).elevation_grid
            for _ in range(5)
        ]
        for g in grids[1:]:
            np.testing.assert_array_equal(grids[0], g)

    def test_different_seeds_differ(self):
        a = Terrain.procedural(rows=10, cols=10, generator="undulating", seed=1)
        b = Terrain.procedural(rows=10, cols=10, generator="undulating", seed=2)
        assert not np.array_equal(a.elevation_grid, b.elevation_grid)

    def test_slope_deterministic(self):
        a = Terrain.procedural(rows=6, cols=6, generator="slope", grade_pct=3.0)
        b = Terrain.procedural(rows=6, cols=6, generator="slope", grade_pct=3.0)
        np.testing.assert_array_equal(a.elevation_grid, b.elevation_grid)


# ---------------------------------------------------------------------------
# Flat default (backward compatibility)
# ---------------------------------------------------------------------------

class TestFlatDefault:
    def test_field_terrain_none_by_default(self):
        field = _flat_field()
        state = field._init_field_state()
        assert state.terrain is None

    def test_field_elevation_grid_flat_by_default(self):
        field = _flat_field(rows=3, cols=4)
        np.testing.assert_array_equal(
            field.elevation_grid,
            np.zeros((3, 4)),
        )

    def test_field_state_elevation_grid_flat_by_default(self):
        field = _flat_field(rows=2, cols=3)
        state = field._init_field_state()
        np.testing.assert_array_equal(
            state.elevation_grid,
            np.zeros((2, 3)),
        )


# ---------------------------------------------------------------------------
# from_array
# ---------------------------------------------------------------------------

class TestFromArray:
    def test_wraps_ndarray(self):
        arr = np.arange(12, dtype=float).reshape(3, 4)
        t = Terrain.from_array(arr, resolution_m=2.0)
        np.testing.assert_array_equal(t.elevation_grid, arr)
        assert t.resolution_m == 2.0
        assert t.source == "array"

    def test_shape_preserved(self):
        arr = np.zeros((5, 7))
        t = Terrain.from_array(arr)
        assert t.elevation_grid.shape == (5, 7)


# ---------------------------------------------------------------------------
# from_generator
# ---------------------------------------------------------------------------

class TestFromGenerator:
    def test_valid_callable(self):
        def my_gen(rows, cols):
            return np.ones((rows, cols)) * 3.14

        t = Terrain.from_generator(my_gen, rows=4, cols=6)
        assert t.elevation_grid.shape == (4, 6)
        assert np.allclose(t.elevation_grid, 3.14)
        assert t.source == "generator"

    def test_shape_mismatch_raises(self):
        def bad_gen(rows, cols):
            return np.zeros((rows + 1, cols))  # wrong shape

        with pytest.raises(ValueError, match="shape"):
            Terrain.from_generator(bad_gen, rows=4, cols=4)


# ---------------------------------------------------------------------------
# Procedural generators — physical properties
# ---------------------------------------------------------------------------

class TestProceduralGenerators:
    def test_slope_linear_gradient(self):
        t = Terrain.procedural(rows=5, cols=5, generator="slope", grade_pct=10.0, direction_deg=0.0)
        # direction_deg=0 → slope along row axis, cos(0)=1
        # Each row step = 1 cell * (10/100) = 0.1 m
        diffs = np.diff(t.elevation_grid, axis=0)
        assert np.allclose(diffs, 0.1, atol=1e-10)

    def test_undulating_shape_and_amplitude(self):
        amp = 3.0
        t = Terrain.procedural(rows=20, cols=20, generator="undulating", amplitude_m=amp, seed=42)
        assert t.elevation_grid.shape == (20, 20)
        # Elevation range should be within ±amp of base_elevation_m
        assert t.elevation_grid.min() >= -amp - 1e-6
        assert t.elevation_grid.max() <= amp + 1e-6

    def test_bowl_deepest_at_centre(self):
        rows, cols = 21, 21
        depth = 4.0
        t = Terrain.procedural(rows=rows, cols=cols, generator="bowl", depth_m=depth)
        centre_val = t.elevation_grid[rows // 2, cols // 2]
        corner_val = t.elevation_grid[0, 0]
        # Centre is lower than corner (depression)
        assert centre_val < corner_val

    def test_ridge_highest_at_centre(self):
        rows, cols = 21, 21
        t = Terrain.procedural(
            rows=rows, cols=cols,
            generator="ridge",
            amplitude_m=5.0,
            width_m=8.0,
            orientation_deg=90.0,
        )
        mid_col = cols // 2
        # Centre column rows should have highest values (ridge at centre column when orientation=90)
        # Actually for orientation_deg=90, ridge crosses column-wise — check centre row
        mid_row = rows // 2
        assert t.elevation_grid[mid_row, mid_col] > t.elevation_grid[0, 0]

    def test_unknown_generator_raises(self):
        with pytest.raises(ValueError, match="Unknown generator"):
            Terrain.procedural(rows=4, cols=4, generator="nonexistent")

    def test_base_elevation_applied(self):
        base = 250.0
        t = Terrain.procedural(rows=5, cols=5, generator="slope", grade_pct=0.0, base_elevation_m=base)
        assert np.allclose(t.elevation_grid, base, atol=1e-10)


# ---------------------------------------------------------------------------
# slope_grid and aspect_grid
# ---------------------------------------------------------------------------

class TestSlopeAspect:
    def test_flat_terrain_zero_slope(self):
        t = Terrain.from_array(np.zeros((6, 6)))
        np.testing.assert_allclose(t.slope_grid, 0.0, atol=1e-10)

    def test_slope_grid_shape(self):
        t = Terrain.procedural(rows=8, cols=10, generator="undulating", seed=42)
        assert t.slope_grid.shape == (8, 10)
        assert t.aspect_grid.shape == (8, 10)

    def test_uniform_slope_correct_degrees(self):
        # grade_pct=100 → rise/run=1 → arctan(1)=45 degrees
        t = Terrain.procedural(
            rows=10, cols=10, generator="slope",
            grade_pct=100.0, direction_deg=0.0,
        )
        # Interior rows should have ~45° slope (edge cells use forward/backward diff, same here)
        # Central rows are accurate; test the mean of interior
        interior_slopes = t.slope_grid[1:-1, :]
        assert np.allclose(interior_slopes, 45.0, atol=0.5)

    def test_aspect_range(self):
        t = Terrain.procedural(rows=10, cols=10, generator="undulating", seed=99)
        assert t.aspect_grid.min() >= 0.0
        assert t.aspect_grid.max() < 360.0 + 1e-6


# ---------------------------------------------------------------------------
# Field.set_terrain() integration
# ---------------------------------------------------------------------------

class TestSetTerrain:
    def test_set_terrain_updates_elevation_grid(self):
        field = _flat_field(rows=5, cols=5)
        t = Terrain.procedural(rows=5, cols=5, generator="slope", grade_pct=5.0)
        field.set_terrain(t)
        np.testing.assert_array_equal(field.elevation_grid, t.elevation_grid)

    def test_field_state_has_terrain(self):
        field = _flat_field(rows=5, cols=5)
        t = Terrain.procedural(rows=5, cols=5, generator="bowl")
        field.set_terrain(t)
        state = field._init_field_state()
        assert state.terrain is t

    def test_set_terrain_wrong_type_raises(self):
        field = _flat_field()
        with pytest.raises(TypeError):
            field.set_terrain(np.zeros((4, 5)))

    def test_legacy_set_elevation_still_works(self):
        field = _flat_field(rows=3, cols=4)
        dem = np.ones((3, 4)) * 1.5
        field.set_elevation(dem)
        np.testing.assert_array_equal(field.elevation_grid, dem)
        # set_elevation now also builds a Terrain internally
        state = field._init_field_state()
        assert state.terrain is not None
        np.testing.assert_array_equal(state.terrain.elevation_grid, dem)
