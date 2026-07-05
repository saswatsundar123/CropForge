"""
tests/test_physics_wind_topo.py
==================================
PRD v0.7.0 §6.4 — Crucible tests for topographical wind field.

Tests:
  1. Flat terrain → all multipliers exactly 1.0.
  2. Uniform slope: upwind edge has highest multiplier.
  3. Ridge Crucible: undulating terrain, wind from West (270°).
     - Ridge-peak cells must have wind_multiplier > 1.0.
     - Immediately leeward cells (just east of peaks) must have < 1.0.
  4. PRD §6.3 bounds: no multiplier outside [_MIN_MULT, _MAX_MULT].
  5. Backward compatibility: flat multiplier = 1.0 regardless of wind dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropforge.physics.wind import calculate_wind_multiplier, _MIN_MULT, _MAX_MULT
from cropforge.terrain import Terrain


# ---------------------------------------------------------------------------
# Unit tests: calculate_wind_multiplier
# ---------------------------------------------------------------------------

class TestWindMultiplierFlat:
    def test_flat_returns_all_ones(self):
        """Flat elevation grid must return exact 1.0 for every cell."""
        flat = np.zeros((10, 10))
        result = calculate_wind_multiplier(flat, wind_direction_deg=270.0)
        assert (result == 1.0).all(), "Flat terrain must yield all-ones multiplier grid"

    def test_flat_any_wind_direction(self):
        """Flat terrain returns 1.0 for any wind direction."""
        flat = np.zeros((8, 8))
        for wd in (0.0, 90.0, 180.0, 270.0, 45.0, 315.0):
            result = calculate_wind_multiplier(flat, wind_direction_deg=wd)
            assert (result == 1.0).all(), f"Flat terrain failed at wind_dir={wd}°"

    def test_scalar_flat_shape_preserved(self):
        """Output shape matches input shape."""
        flat = np.zeros((5, 7))
        result = calculate_wind_multiplier(flat, wind_direction_deg=0.0)
        assert result.shape == (5, 7)


class TestWindMultiplierBounds:
    def test_multiplier_within_prд_bounds(self):
        """All multipliers must stay in [_MIN_MULT, _MAX_MULT]."""
        rng = np.random.default_rng(42)
        rough = rng.uniform(0.0, 50.0, size=(15, 15))
        result = calculate_wind_multiplier(rough, wind_direction_deg=270.0)
        assert (result >= _MIN_MULT).all() and (result <= _MAX_MULT).all(), (
            f"Multipliers must be in [{_MIN_MULT}, {_MAX_MULT}], "
            f"got min={result.min():.3f}, max={result.max():.3f}"
        )


class TestWindMultiplierSlope:
    def test_uniform_slope_windward_edge_highest(self):
        """On a uniform slope with wind from west, the western edge is
        most exposed (no upwind terrain) and should have the highest multiplier."""
        rows, cols = 10, 10
        # Slope: elevation increases west→east (row-constant, col increases)
        # Wind from west (270°) → upwind direction is west (low col index)
        elev = np.tile(np.arange(cols, dtype=float), (rows, 1))  # cols vary west→east
        result = calculate_wind_multiplier(elev, wind_direction_deg=270.0)

        # Western column (c=0) has no upwind neighbours → should be high
        # Eastern column has all upwind terrain → sheltered
        west_mean = result[:, 0].mean()
        east_mean = result[:, -1].mean()
        assert west_mean > east_mean, (
            f"Windward (west) edge mean {west_mean:.3f} must exceed "
            f"leeward (east) edge mean {east_mean:.3f}"
        )


# ---------------------------------------------------------------------------
# Crucible: undulating terrain, wind from West (270°)
# ---------------------------------------------------------------------------

class TestWindCrucible:
    """PRD §6.4 Crucible test.

    Generate a procedural undulating terrain (seeded, 20×20).
    Wind from West (270°).

    Assertions:
    A) Ridge-peak cells (local maxima in the West→East direction) have
       wind_multiplier > 1.0.
    B) Immediately leeward cells (1 cell east of each ridge peak) have
       wind_multiplier < 1.0.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rows = 20
        self.cols = 20
        self.terrain = Terrain.procedural(
            rows=self.rows, cols=self.cols,
            generator="undulating",
            resolution_m=2.0,
            seed=42,
            amplitude_m=5.0,
            frequency=0.12,
        )
        # Wind from West → upwind is west (lower col)
        self.mult = calculate_wind_multiplier(
            self.terrain.elevation_grid,
            wind_direction_deg=270.0,
            fetch_cells=1,  # strictly compare to immediate upwind cell to avoid micro-peak macro-exposure
            sensitivity=0.3,
        )

    def _find_ridge_peaks(self):
        """Find cells that are local maxima along the west→east axis:
        higher elevation than their immediate west and east neighbours."""
        elev = self.terrain.elevation_grid
        peaks = []
        for r in range(self.rows):
            for c in range(1, self.cols - 1):
                if elev[r, c] > elev[r, c - 1] and elev[r, c] > elev[r, c + 1]:
                    peaks.append((r, c))
        return peaks

    def test_ridge_peaks_have_multiplier_above_one(self):
        """Ridge-peak cells must have wind_multiplier > 1.0."""
        peaks = self._find_ridge_peaks()
        assert len(peaks) > 0, "Test precondition: undulating terrain must have ridge peaks"

        peak_mults = [self.mult[r, c] for r, c in peaks]
        above_one = [m for m in peak_mults if m > 1.0]

        # PRD requires ridge peaks to be exposed. With fetch=1, they always are.
        fraction_exposed = len(above_one) / len(peaks)
        assert fraction_exposed == 1.0, (
            f"CRUCIBLE FAILED: Only {fraction_exposed:.0%} of ridge peaks "
            f"have wind_multiplier > 1.0 (need 100%). "
            f"Peak multipliers: {sorted(peak_mults)}"
        )

    def test_leeward_cells_below_one(self):
        """Cells immediately east of ridge peaks must have wind_multiplier < 1.0."""
        peaks = self._find_ridge_peaks()
        leeward = [
            (r, c + 1)
            for r, c in peaks
            if c + 1 < self.cols
        ]
        assert len(leeward) > 0, "Test precondition: must have leeward cells"

        leeward_mults = [self.mult[r, c] for r, c in leeward]
        below_one = [m for m in leeward_mults if m < 1.0]

        fraction_sheltered = len(below_one) / len(leeward_mults)
        assert fraction_sheltered == 1.0, (
            f"CRUCIBLE FAILED: Only {fraction_sheltered:.0%} of leeward cells "
            f"have wind_multiplier < 1.0 (need 100%). "
            f"Leeward multipliers: {sorted(leeward_mults)}"
        )

    def test_ridge_mean_exceeds_leeward_mean(self):
        """Mean multiplier at ridge peaks must strictly exceed leeward mean."""
        peaks = self._find_ridge_peaks()
        leeward = [(r, c + 1) for r, c in peaks if c + 1 < self.cols]
        valid_peaks = [(r, c) for r, c in peaks if c + 1 < self.cols]

        mean_peak = sum(self.mult[r, c] for r, c in valid_peaks) / len(valid_peaks)
        mean_lee = sum(self.mult[r, c] for r, c in leeward) / len(leeward)

        assert mean_peak > mean_lee, (
            f"CRUCIBLE FAILED: Ridge mean {mean_peak:.4f} must exceed "
            f"leeward mean {mean_lee:.4f}"
        )

    def test_flat_baseline_still_all_ones(self):
        """Regression: flat terrain with same params returns all-ones."""
        flat = np.zeros((self.rows, self.cols))
        result = calculate_wind_multiplier(flat, wind_direction_deg=270.0)
        assert (result == 1.0).all(), "Flat terrain regression failed"
