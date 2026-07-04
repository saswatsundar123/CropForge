"""
tests/test_modifiers_land_prep.py
==================================
Tests for LandPrep modifiers (PRD v0.6.0 §6).

Covers:
  - RidgeFurrow: alternating geometry on flat grid (pure math)
  - ContourBund: bunds placed at correct elevation intervals
  - Terrace: staircase pattern, n platforms
  - ZeroTillage / ConventionalTill: no geometry change
  - Macro-topology preserved: base terrain shape survives modification
  - Custom LandPrep subclass: apply() is called and output used
  - Soil deltas written to SoilVoxelState (porosity_delta etc.)
  - No land prep → SoilVoxelState deltas remain 0.0 (backward compat)
  - Integration: field.set_land_prep() changes FieldState.elevation_grid
  - Bad input: set_land_prep(wrong_type) raises TypeError
"""

import numpy as np
import pytest

from cropforge import (
    Field, Terrain,
    LandPrep, RidgeFurrow, ContourBund, Terrace, ZeroTillage, ConventionalTill,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_field(rows=10, cols=10, resolution_m=1.0):
    f = Field(name="T", rows=rows, cols=cols, area_ha=0.1)
    terrain = Terrain.from_array(np.zeros((rows, cols)), resolution_m=resolution_m)
    f.set_terrain(terrain)
    return f


def _sloped_field(rows=10, cols=10, resolution_m=1.0):
    """10×10 field with a linear north-to-south slope (0 at top, 9 at bottom)."""
    elev = np.array([[float(r)] * cols for r in range(rows)])
    f = Field(name="T", rows=rows, cols=cols, area_ha=0.1)
    terrain = Terrain.from_array(elev, resolution_m=resolution_m)
    f.set_terrain(terrain)
    return f


# ---------------------------------------------------------------------------
# RidgeFurrow — pure math tests
# ---------------------------------------------------------------------------

class TestRidgeFurrow:
    def test_alternating_pattern_on_flat(self):
        """On a flat grid, RidgeFurrow must produce alternating high/low cols."""
        rows, cols = 5, 20
        spacing = 4.0   # 4 m per cycle, resolution=1 m → exactly 5 cycles in 20 cols
        height = 0.4    # 0.4 m ridge height → peaks at +0.2, troughs at -0.2

        mod = RidgeFurrow(ridge_spacing_m=spacing, ridge_height_m=height)
        base = np.zeros((rows, cols))
        modified, soil = mod.apply(base, resolution_m=1.0)

        # Modified grid must differ from base
        assert not np.array_equal(modified, base)

        # Pattern is identical for every row (wave only varies over columns)
        for r in range(1, rows):
            np.testing.assert_array_equal(modified[r], modified[0])

        # Peak columns (col=0, col=4, col=8 ... with freq=2π/4 → cos=1 at multiples of 4)
        # wave = midpoint + half_range * cos(2π/4 * col_pos)
        # midpoint = (peak - trough)/2 = (0.2 - 0.2)/2 = 0.0  (since furrow_depth = height/2 = 0.2)
        # half_range = (0.2 + 0.2)/2 = 0.2
        peak_val = modified[0, 0]   # col=0 → cos(0)=1 → wave = 0 + 0.2*1 = 0.2
        trough_val = modified[0, int(spacing / 2)]  # col=2 → cos(π)=-1 → wave = -0.2
        assert peak_val > 0.0, f"Peak should be above 0, got {peak_val}"
        assert trough_val < 0.0, f"Trough should be below 0, got {trough_val}"
        assert peak_val > trough_val

    def test_peak_amplitude_correct(self):
        """Ridge peak should equal ridge_height_m/2 above base (on flat grid)."""
        mod = RidgeFurrow(ridge_spacing_m=2.0, ridge_height_m=1.0)
        base = np.zeros((3, 20))
        modified, _ = mod.apply(base, resolution_m=1.0)
        # Col=0: cos(0)=1, peak = 0.5m (ridge_height/2)
        assert np.isclose(modified[0, 0], 0.5, atol=1e-6)

    def test_custom_furrow_depth(self):
        """Explicit furrow_depth_m is respected."""
        mod = RidgeFurrow(ridge_spacing_m=4.0, ridge_height_m=0.4, furrow_depth_m=0.0)
        base = np.zeros((4, 16))
        modified, _ = mod.apply(base, resolution_m=1.0)
        # With furrow_depth=0: trough = 0, peak = ridge_height/2 = 0.2
        # midpoint = (0.2 - 0)/2 = 0.1, half_range = (0.2 + 0)/2 = 0.1
        # At col=0: wave = 0.1 + 0.1*1 = 0.2
        # At col=2 (half cycle): wave = 0.1 + 0.1*(-1) = 0.0
        assert modified[0, 0] > 0.0       # ridge raises above base
        assert modified[0, 2] >= 0.0 - 1e-9  # trough at base level (not below)

    def test_invalid_spacing_raises(self):
        with pytest.raises(ValueError):
            RidgeFurrow(ridge_spacing_m=0.0)

    def test_soil_deltas_present(self):
        mod = RidgeFurrow()
        _, soil = mod.apply(np.zeros((3, 3)), 1.0)
        assert "porosity_delta" in soil
        assert "surface_roughness_index" in soil

    def test_base_not_modified_in_place(self):
        """apply() must not mutate the input array."""
        base = np.zeros((4, 8))
        original = base.copy()
        mod = RidgeFurrow()
        mod.apply(base, 1.0)
        np.testing.assert_array_equal(base, original)


# ---------------------------------------------------------------------------
# ContourBund
# ---------------------------------------------------------------------------

class TestContourBund:
    def test_bunds_raised_above_base(self):
        """On a sloped grid, at least one bund row should be higher than base."""
        rows, cols = 15, 5
        elev = np.array([[float(r)] * cols for r in range(rows)], dtype=float)
        mod = ContourBund(bund_height_m=0.5, interval_m=3.0)
        modified, _ = mod.apply(elev, resolution_m=1.0)
        # Some rows should be raised (modified > base)
        raised = (modified > elev).any(axis=1)
        assert raised.any(), "At least one bund row should be raised above base"

    def test_non_bund_rows_unchanged(self):
        """Rows between bunds must retain their base elevation."""
        rows, cols = 10, 4
        elev = np.array([[float(r) * 0.5] * cols for r in range(rows)], dtype=float)
        mod = ContourBund(bund_height_m=1.0, interval_m=5.0, bund_width_cells=1)
        modified, _ = mod.apply(elev, resolution_m=1.0)
        # Check that at least some rows are EQUAL to base (not all raised)
        unchanged = np.all(np.isclose(modified, elev), axis=1)
        assert unchanged.any(), "Some rows should remain unmodified between bunds"

    def test_flat_field_no_bunds(self):
        """Flat field has no elevation gain → no threshold crossed → no bunds."""
        base = np.zeros((8, 5))
        mod = ContourBund(bund_height_m=0.5, interval_m=1.0)
        modified, _ = mod.apply(base, resolution_m=1.0)
        # All row means are 0, threshold (0+1.0) never reached → no bunds
        np.testing.assert_array_equal(modified, base)

    def test_soil_deltas_present(self):
        _, soil = ContourBund().apply(np.zeros((5, 5)), 1.0)
        assert "porosity_delta" in soil


# ---------------------------------------------------------------------------
# Terrace
# ---------------------------------------------------------------------------

class TestTerrace:
    def test_n_platforms_created(self):
        """Terrace with n_terraces=3 should produce exactly 3 distinct row-mean levels."""
        rows, cols = 9, 5
        elev = np.array([[float(r) * 0.5] * cols for r in range(rows)], dtype=float)
        mod = Terrace(n_terraces=3)
        modified, _ = mod.apply(elev, resolution_m=1.0)
        row_means = np.round(modified.mean(axis=1), 8)
        unique_levels = np.unique(row_means)
        assert len(unique_levels) == 3, f"Expected 3 terrace levels, got {len(unique_levels)}"

    def test_staircase_monotonic(self):
        """Terrace platforms should be at strictly increasing elevation levels
        when the base terrain slopes upward from row 0."""
        rows, cols = 12, 5
        elev = np.array([[float(r)] * cols for r in range(rows)], dtype=float)
        mod = Terrace(n_terraces=4)
        modified, _ = mod.apply(elev, resolution_m=1.0)
        # Take one value per band (band is level, so mean == all values in band)
        bands = np.array_split(np.arange(rows), 4)
        platform_elevs = [modified[b[0], 0] for b in bands]
        assert all(
            platform_elevs[i] < platform_elevs[i + 1]
            for i in range(len(platform_elevs) - 1)
        ), f"Terrace levels not monotonically increasing: {platform_elevs}"

    def test_within_band_uniform_elevation(self):
        """Every cell within one terrace band should have identical elevation."""
        rows, cols = 9, 6
        elev = np.array([[float(r)] * cols for r in range(rows)], dtype=float)
        mod = Terrace(n_terraces=3)
        modified, _ = mod.apply(elev, resolution_m=1.0)
        bands = np.array_split(np.arange(rows), 3)
        for band in bands:
            band_vals = modified[band, :]
            assert np.allclose(band_vals, band_vals[0, 0]), "Band should be level"

    def test_soil_deltas_present(self):
        _, soil = Terrace().apply(np.zeros((6, 4)), 1.0)
        assert "porosity_delta" in soil
        assert "bulk_density_delta" in soil


# ---------------------------------------------------------------------------
# Geometry-free modifiers
# ---------------------------------------------------------------------------

class TestZeroTillageConventionalTill:
    def test_zerotillage_no_elevation_change(self):
        base = np.random.default_rng(1).random((5, 5))
        modified, soil = ZeroTillage().apply(base, 1.0)
        np.testing.assert_array_equal(modified, base)
        assert soil["porosity_delta"] == 0.0

    def test_conventional_till_no_elevation_change(self):
        base = np.random.default_rng(2).random((5, 5))
        modified, soil = ConventionalTill().apply(base, 1.0)
        np.testing.assert_array_equal(modified, base)
        assert soil["surface_roughness_index"] > 0.0


# ---------------------------------------------------------------------------
# Macro-topology preservation
# ---------------------------------------------------------------------------

class TestMacroTopologyPreservation:
    def test_ridge_furrow_preserves_slope(self):
        """On a sloped base, ridge-furrow should not erase the large-scale slope."""
        rows, cols = 5, 20
        slope_m_per_row = 1.0
        elev = np.array([[float(r) * slope_m_per_row] * cols for r in range(rows)], dtype=float)

        mod = RidgeFurrow(ridge_spacing_m=2.0, ridge_height_m=0.2)
        modified, _ = mod.apply(elev, resolution_m=1.0)

        # Row-mean of modified should still be monotonically increasing
        row_means = modified.mean(axis=1)
        assert all(
            row_means[i] < row_means[i + 1] for i in range(rows - 1)
        ), f"Slope macro-topology was erased. Row means: {row_means}"

    def test_terrace_total_elevation_range_bounded(self):
        """Terrace shouldn't create elevation values outside the base range."""
        rows, cols = 8, 5
        elev = np.array([[float(r) * 0.5] * cols for r in range(rows)], dtype=float)
        mod = Terrace(n_terraces=4)
        modified, _ = mod.apply(elev, resolution_m=1.0)
        assert modified.min() >= elev.min() - 1e-9
        assert modified.max() <= elev.max() + 1e-9


# ---------------------------------------------------------------------------
# Custom LandPrep subclass
# ---------------------------------------------------------------------------

class TestCustomLandPrepSubclass:
    def test_custom_subclass_apply_called(self):
        """A researcher-defined subclass must be accepted and its apply() used."""
        class FlattenEverything(LandPrep):
            def apply(self, elevation_grid, resolution_m):
                return np.zeros_like(elevation_grid), {"porosity_delta": 0.99}

        elev = np.ones((5, 5)) * 3.0
        field = Field(name="Custom", rows=5, cols=5, area_ha=0.1)
        terrain = Terrain.from_array(elev, resolution_m=1.0)
        field.set_terrain(terrain)
        field.set_land_prep(FlattenEverything())
        state = field._init_field_state()
        # elevation_grid should now be all zeros
        np.testing.assert_array_equal(state.elevation_grid, np.zeros((5, 5)))

    def test_custom_soil_delta_propagated(self):
        class HighPorosity(LandPrep):
            def apply(self, elevation_grid, resolution_m):
                return elevation_grid.copy(), {"porosity_delta": 0.42}

        field = Field(name="HP", rows=3, cols=3, area_ha=0.1)
        terrain = Terrain.from_array(np.zeros((3, 3)))
        field.set_terrain(terrain)
        field.set_land_prep(HighPorosity())
        state = field._init_field_state()
        # Every voxel should have porosity_delta = 0.42
        for row_soils in state.soil:
            for cell_soils in row_soils:
                for voxel in cell_soils:
                    assert np.isclose(voxel.porosity_delta, 0.42)


# ---------------------------------------------------------------------------
# Soil property delta stamping
# ---------------------------------------------------------------------------

class TestSoilDeltaStamping:
    def test_ridge_furrow_stamps_soil_deltas(self):
        field = _flat_field(rows=4, cols=8)
        field.set_land_prep(RidgeFurrow())
        state = field._init_field_state()
        for row_soils in state.soil:
            for cell_soils in row_soils:
                for voxel in cell_soils:
                    assert voxel.porosity_delta > 0.0
                    assert voxel.surface_roughness_index > 0.0

    def test_no_land_prep_deltas_zero(self):
        """Without land prep, soil deltas must remain 0.0 (backward compat)."""
        field = _flat_field(rows=3, cols=3)
        state = field._init_field_state()
        for row_soils in state.soil:
            for cell_soils in row_soils:
                for voxel in cell_soils:
                    assert voxel.porosity_delta == 0.0
                    assert voxel.bulk_density_delta == 0.0
                    assert voxel.surface_roughness_index == 0.0

    def test_terrace_stamps_correct_deltas(self):
        field = _sloped_field(rows=8, cols=5)
        field.set_land_prep(Terrace(n_terraces=4))
        state = field._init_field_state()
        voxel = state.soil[0][0][0]
        assert np.isclose(voxel.porosity_delta, 0.08)
        assert np.isclose(voxel.bulk_density_delta, -0.15)


# ---------------------------------------------------------------------------
# Integration: set_land_prep changes FieldState.elevation_grid
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_ridge_furrow_changes_elevation_grid(self):
        """Applying RidgeFurrow must produce a FieldState.elevation_grid
        that differs from the flat base terrain."""
        field = _flat_field(rows=5, cols=20)
        field.set_land_prep(RidgeFurrow(ridge_spacing_m=4.0, ridge_height_m=0.4))
        state = field._init_field_state()
        assert not np.array_equal(state.elevation_grid, np.zeros((5, 20)))

    def test_base_terrain_unchanged_after_land_prep(self):
        """Terrain.elevation_grid must be the unmodified base after land prep."""
        base_elev = np.zeros((5, 10))
        field = Field(name="T", rows=5, cols=10, area_ha=0.1)
        terrain = Terrain.from_array(base_elev.copy(), resolution_m=1.0)
        field.set_terrain(terrain)
        field.set_land_prep(RidgeFurrow(ridge_spacing_m=2.0, ridge_height_m=1.0))
        state = field._init_field_state()

        # FieldState.elevation_grid = modified (not flat)
        assert not np.allclose(state.elevation_grid, 0.0)
        # Terrain object still has original flat grid
        np.testing.assert_array_equal(state.terrain.elevation_grid, base_elev)

    def test_no_land_prep_elevation_unchanged(self):
        """Without land prep, FieldState.elevation_grid == terrain.elevation_grid."""
        base_elev = np.ones((4, 5)) * 2.5
        field = Field(name="T", rows=4, cols=5, area_ha=0.1)
        terrain = Terrain.from_array(base_elev.copy())
        field.set_terrain(terrain)
        state = field._init_field_state()
        np.testing.assert_array_equal(state.elevation_grid, base_elev)

    def test_set_land_prep_wrong_type_raises(self):
        field = _flat_field()
        with pytest.raises(TypeError):
            field.set_land_prep("not a land prep")

    def test_set_land_prep_wrong_instance_raises(self):
        field = _flat_field()
        with pytest.raises(TypeError):
            field.set_land_prep(42)

    def test_terrace_on_sloped_terrain_integration(self):
        """End-to-end: sloped terrain + terraces → stepped elevation in state."""
        field = _sloped_field(rows=9, cols=5)
        field.set_land_prep(Terrace(n_terraces=3))
        state = field._init_field_state()
        row_means = state.elevation_grid.mean(axis=1)
        unique_levels = np.unique(np.round(row_means, 6))
        assert len(unique_levels) == 3
