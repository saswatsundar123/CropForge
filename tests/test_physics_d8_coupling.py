"""
tests/test_physics_d8_coupling.py
====================================
PRD v0.6.0 Phase 6: D8 Terrain Coupling integration tests.

Validates that route_surface_water (D8) correctly respects
land-preparation geometry (RidgeFurrow) by routing water off
ridge peaks into adjacent furrow troughs.

Success criteria (PRD v0.6.0 §8.4):
  - Water applied to a ridge cell must accumulate in the furrow cell,
    not cross to the next ridge.
  - Inflow to furrow > inflow to ridge (flow along furrow axis).
  - Flat-grid fallback: no D8 routing when elevation_grid is uniform.

All tests operate on plain Python lists — no engine state imports.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cropforge.physics.hydrology import route_surface_water
from cropforge.land_prep import RidgeFurrow


# ---------------------------------------------------------------------------
# Helper: build a ridge-furrow elevation grid via the real LandPrep class
# ---------------------------------------------------------------------------

def _make_ridge_furrow_grid(
    rows: int = 4,
    cols: int = 6,
    ridge_height_m: float = 0.30,
    ridge_spacing_m: float = 1.0,
    resolution_m: float = 0.5,
) -> list[list[float]]:
    """Return a 2-D Python list with RidgeFurrow geometry applied."""
    base = np.zeros((rows, cols), dtype=float)
    rf = RidgeFurrow(
        ridge_height_m=ridge_height_m,
        ridge_spacing_m=ridge_spacing_m,
    )
    modified, _ = rf.apply(base, resolution_m)
    return modified.tolist()


# ---------------------------------------------------------------------------
# Test 1: Water flows OFF a ridge peak into the adjacent furrow trough
# ---------------------------------------------------------------------------

class TestRidgeToFurrowFlow:
    """PRD §8.4 — water placed on ridge must end up in furrow."""

    def test_ridge_runoff_reaches_furrow(self):
        """High runoff on the ridge column flows to the lowest adjacent column."""
        elev = _make_ridge_furrow_grid(rows=4, cols=6)

        # Identify which column is a ridge peak and which is a furrow trough
        # by looking at column 0 elevation pattern (same for all rows)
        col_elevs = [elev[0][c] for c in range(6)]
        ridge_col = int(col_elevs.index(max(col_elevs)))
        furrow_col = int(col_elevs.index(min(col_elevs)))

        # Place 20 mm of runoff on every ridge cell; zero elsewhere
        runoff = [[0.0] * 6 for _ in range(4)]
        for r in range(4):
            runoff[r][ridge_col] = 20.0

        inflow = route_surface_water(runoff, elev)

        # The furrow column must receive lateral inflow; ridge receives none
        total_furrow_inflow = sum(inflow[r][furrow_col] for r in range(4))
        total_ridge_inflow  = sum(inflow[r][ridge_col]  for r in range(4))

        assert total_furrow_inflow > 0.0, (
            "D8 routing must deliver runoff from ridge into furrow"
        )
        assert total_ridge_inflow == 0.0, (
            "Ridge column must not receive inflow when it is the highest cell"
        )

    def test_furrow_receives_more_than_ridge(self):
        """Furrow inflow > ridge inflow — water drains to valley, not uphill."""
        elev = _make_ridge_furrow_grid(rows=4, cols=6)
        col_elevs = [elev[0][c] for c in range(6)]
        ridge_col = int(col_elevs.index(max(col_elevs)))
        furrow_col = int(col_elevs.index(min(col_elevs)))

        runoff = [[5.0] * 6 for _ in range(4)]  # uniform runoff everywhere

        inflow = route_surface_water(runoff, elev)

        furrow_total = sum(inflow[r][furrow_col] for r in range(4))
        ridge_total  = sum(inflow[r][ridge_col]  for r in range(4))

        assert furrow_total > ridge_total, (
            "Furrow (trough) must accumulate more inflow than ridge (peak)"
        )

    def test_mass_conservation_ridge_furrow(self):
        """Total inflow <= total runoff (boundary cells can lose water)."""
        elev = _make_ridge_furrow_grid(rows=4, cols=6)

        runoff = [[10.0] * 6 for _ in range(4)]
        inflow = route_surface_water(runoff, elev)

        total_runoff = sum(runoff[r][c] for r in range(4) for c in range(6))
        total_inflow = sum(inflow[r][c]  for r in range(4) for c in range(6))

        assert total_inflow <= total_runoff + 1e-9, (
            "D8 must not create water; inflow <= runoff"
        )


# ---------------------------------------------------------------------------
# Test 2: Flat grid — D8 routes nothing (no downslope exists)
# ---------------------------------------------------------------------------

class TestFlatGridFallback:
    """PRD backward-compat: flat grid produces zero lateral inflow."""

    def test_flat_grid_zero_inflow(self):
        """On a perfectly flat field, D8 has no downslope and routes nothing."""
        rows, cols = 3, 3
        elev   = [[0.0] * cols for _ in range(rows)]
        runoff = [[5.0] * cols for _ in range(rows)]

        inflow = route_surface_water(runoff, elev)

        total_inflow = sum(inflow[r][c] for r in range(rows) for c in range(cols))
        assert total_inflow == 0.0, (
            "Flat grid: no lower neighbour exists, so no water is routed"
        )

    def test_empty_grid_returns_empty(self):
        assert route_surface_water([], []) == []


# ---------------------------------------------------------------------------
# Test 3: Simple 2-cell slope — deterministic routing
# ---------------------------------------------------------------------------

class TestSimpleSlope:
    """Minimal 1×2 case: uphill cell drains to downhill cell."""

    def test_two_cell_drains_downhill(self):
        elev   = [[1.0, 0.0]]   # left is higher
        runoff = [[8.0, 0.0]]

        inflow = route_surface_water(runoff, elev)

        assert inflow[0][1] == pytest.approx(8.0), (
            "All runoff from the high cell must flow to the low cell"
        )
        assert inflow[0][0] == pytest.approx(0.0), (
            "High cell receives no inflow"
        )

    def test_no_uphill_routing(self):
        """D8 must never route water uphill."""
        elev   = [[0.0, 1.0]]   # right is HIGHER — no valid downslope from left
        runoff = [[5.0, 0.0]]

        inflow = route_surface_water(runoff, elev)
        # Left cell (row=0, col=0) has no lower neighbour → water is lost at boundary
        assert inflow[0][1] == pytest.approx(0.0), (
            "Water must not route uphill to the higher cell"
        )


# ---------------------------------------------------------------------------
# Test 4: RidgeFurrow geometry sanity
# ---------------------------------------------------------------------------

class TestRidgeFurrowGeometry:
    """Confirm that RidgeFurrow.apply produces the expected ridge/furrow pattern."""

    def test_ridge_higher_than_furrow(self):
        elev = _make_ridge_furrow_grid()
        col_elevs = [elev[0][c] for c in range(6)]
        assert max(col_elevs) > min(col_elevs), (
            "Ridge-furrow must create elevation contrast between columns"
        )

    def test_ridge_height_respected(self):
        """Peak-to-trough amplitude matches ridge_height_m + furrow_depth_m."""
        ridge_h = 0.30
        elev = _make_ridge_furrow_grid(ridge_height_m=ridge_h)
        col_elevs = [elev[0][c] for c in range(6)]
        amplitude = max(col_elevs) - min(col_elevs)
        # RidgeFurrow default: furrow_depth = ridge_height/2, total = 1.5 × ridge_height/2
        # wave half-range = (peak + trough)/2 = (ridge_h/2 + furrow_depth/2)/2
        # Total peak-to-trough = peak + trough = ridge_h/2 + furrow_depth/2
        expected = ridge_h / 2.0 + ridge_h / 2.0 / 2.0   # = 0.225 m for ridge_h=0.3
        assert amplitude == pytest.approx(expected, abs=0.01), (
            f"Expected peak-to-trough {expected:.3f} m, got {amplitude:.3f} m"
        )
