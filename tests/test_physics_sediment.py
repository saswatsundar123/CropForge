"""
tests/test_physics_sediment.py
================================
Phase 2 crucible tests for PRD v0.8.0 §5.2 — Sediment Dynamics.

Crucible invariants:
  1. calculate_sediment_transport pure-math contracts.
  2. Flat cells always produce zero sediment (mass-conservation floor).
  3. Mass conservation: sum(eroded) == sum(deposited) + boundary_escaped.
  4. CRUCIBLE: On a slope-to-valley terrain, after 30 days of heavy rain:
       - Slope cells: effective_soil_depth DECREASES (erosion).
       - Valley cells: effective_soil_depth INCREASES (deposition).

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropforge.physics.soil import calculate_sediment_transport
from cropforge.physics.hydrology import route_surface_water



# ---------------------------------------------------------------------------
# Unit tests: calculate_sediment_transport
# ---------------------------------------------------------------------------

class TestCalculateSedimentTransport:
    def test_zero_erosion_index(self):
        assert calculate_sediment_transport(0.0, 20.0, 0.5) == (0.0, 0.0)

    def test_zero_runoff(self):
        assert calculate_sediment_transport(5.0, 0.0, 0.5) == (0.0, 0.0)

    def test_flat_cell_zero_output(self):
        """Slope=0 → transport_cap=0 → no sediment moved (mass-conservation floor)."""
        assert calculate_sediment_transport(5.0, 20.0, 0.0) == (0.0, 0.0)

    def test_normal_case(self):
        eroded, cap = calculate_sediment_transport(10.0, 20.0, 0.5)
        assert eroded == pytest.approx(0.05)   # min(10*0.005, 0.02*20*0.5)
        assert cap    == pytest.approx(0.2)

    def test_transport_limited(self):
        """High erosion potential but low runoff: capped by transport capacity."""
        eroded, cap = calculate_sediment_transport(1000.0, 1.0, 0.1)
        assert eroded == pytest.approx(cap)    # limited by transport, not supply
        assert eroded < 1000.0 * 0.005

    def test_supply_limited(self):
        """Very high runoff: eroded is capped by erodibility × erosion_index."""
        eroded, cap = calculate_sediment_transport(0.01, 1000.0, 0.9)
        potential = 0.01 * 0.005
        assert eroded == pytest.approx(potential)   # supply-limited

    def test_custom_k_values(self):
        eroded, cap = calculate_sediment_transport(
            5.0, 10.0, 0.5, k_erodibility=0.01, k_transport=0.05
        )
        assert eroded == pytest.approx(min(5.0 * 0.01, 0.05 * 10.0 * 0.5))


# ---------------------------------------------------------------------------
# Unit tests: mass conservation via route_surface_water reuse
# ---------------------------------------------------------------------------

class TestSedimentMassConservation:
    """Verify that the D8 routing step preserves mass.

    sediment routed into cells == sediment that left donor cells that had
    a valid downslope neighbour (boundary-escaped sediment is intentionally lost).
    """

    def _eroded_slope_grid(self):
        """4×4 grid: top two rows have sediment, bottom two are flat (no eroded soil)."""
        grid = [
            [0.02, 0.02, 0.02, 0.02],   # row 0  (steep)
            [0.01, 0.01, 0.01, 0.01],   # row 1  (moderate)
            [0.0,  0.0,  0.0,  0.0 ],   # row 2  (flat valley)
            [0.0,  0.0,  0.0,  0.0 ],   # row 3  (flat valley)
        ]
        return grid

    def _slope_elev(self):
        """Strictly descending rows so D8 routes row 0 → row 1 → row 2 → lost at row 3."""
        return [
            [3.0, 3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]

    def test_deposited_le_total_eroded(self):
        eroded = self._eroded_slope_grid()
        elev = self._slope_elev()
        deposit = route_surface_water(eroded, elev)

        total_eroded   = sum(eroded[r][c] for r in range(4) for c in range(4))
        total_deposited = sum(deposit[r][c] for r in range(4) for c in range(4))
        # deposited ≤ eroded (boundary escape only, no magic creation)
        assert total_deposited <= total_eroded + 1e-9

    def test_mass_conservation_with_interior_sink(self):
        """When all runoff routes to an interior flat cell, all deposited == all eroded."""
        # 2×1 grid: top cell has sediment, bottom cell is the sink (lower, no neighbour)
        eroded = [[0.05], [0.0]]
        elev   = [[1.0],  [0.0]]
        deposit = route_surface_water(eroded, elev)
        assert deposit[1][0] == pytest.approx(0.05)

    def test_flat_grid_no_transport(self):
        """Completely flat terrain: no lower neighbour → no deposition anywhere."""
        eroded = [[0.1] * 4 for _ in range(4)]
        elev   = [[0.0] * 4 for _ in range(4)]
        deposit = route_surface_water(eroded, elev)
        assert all(deposit[r][c] == 0.0 for r in range(4) for c in range(4))


# ---------------------------------------------------------------------------
# CRUCIBLE TEST: slope→valley terrain, 30 days heavy rain
# ---------------------------------------------------------------------------

class TestCrucible:
    """PRD v0.8.0 §5.2 Crucible.

    Terrain: 8×4 grid.
      Rows 0-5: steep slope  (elevation 5→0.5 m, step 0.9 m/row)
      Rows 6-7: flat valley  (elevation 0 m)

    Simulation: 30 days, 50 mm/day rainfall.
    Physics:    et0=True, water_balance=True, erosion=True, sediment_transport=True

    Expected outcome:
      - mean effective_soil_depth on slope cells < initial depth  (EROSION)
      - mean effective_soil_depth on valley cells > initial depth  (DEPOSITION)
    """

    def _build_farm(self):
        from cropforge import Farm, Field, Crop
        from cropforge.loaders import Weather

        rows, cols = 8, 4
        # Steep slope rows 0-5 (elevation 5.0→0.5), flat valley rows 6-7
        elev = np.zeros((rows, cols))
        for r in range(6):
            elev[r, :] = 5.0 - r * 0.9   # 5.0, 4.1, 3.2, 2.3, 1.4, 0.5
        elev[6:, :] = 0.0

        # Heavy rain every day — same column schema as test_physics_erosion.py
        weather_rows = [
            {
                "day": d, "doy": d,
                "temp_max_c": 32.0, "temp_min_c": 20.0, "temp_mean_c": 26.0,
                "radiation_mj_m2": 20.0, "rainfall_mm": 50.0,
                "et0_mm": 6.0, "wind_speed_ms": 2.0, "humidity_pct": 65.0,
                "co2_ppm": 415.0,
            }
            for d in range(1, 31)
        ]
        weather = Weather(pd.DataFrame(weather_rows).set_index("day"))

        field = Field(name="SlopeValley", rows=rows, cols=cols)
        field.set_elevation(elev)
        field.set_crop(Crop(species="Zea mays", variety="CrucibleCrop"))
        field.set_weather(weather)

        farm = Farm(name="CrucibleFarm", location=(20.0, 85.0))
        farm.add_field(field)
        farm.use_physics(erosion=True, sediment_transport=True)
        return farm, field

    def test_crucible_erosion_on_slope(self):
        """Slope cells MUST lose soil depth after 30 days of heavy rain."""
        farm, field = self._build_farm()
        farm.run(days=30)
        state = field._field_state
        depth_grid = state.custom.get("effective_soil_depth_cm_grid")
        assert depth_grid is not None, "effective_soil_depth_cm_grid not written by sediment hook"

        # Default soil layer is 20cm deep (farm.py line 532).
        # Slope rows 0-5 should have DECREASED depth after erosion.
        slope_depths = [depth_grid[r][c] for r in range(6) for c in range(4)]
        mean_slope_depth = sum(slope_depths) / len(slope_depths)
        assert mean_slope_depth < 20.0, (
            f"Slope cells should have eroded (depth < 20cm), got mean={mean_slope_depth:.4f}cm"
        )

    def test_crucible_deposition_in_valley(self):
        """Valley cells MUST gain soil depth after 30 days of heavy rain."""
        farm, field = self._build_farm()
        farm.run(days=30)
        state = field._field_state
        depth_grid = state.custom.get("effective_soil_depth_cm_grid")
        assert depth_grid is not None

        # Valley rows 6-7 should have INCREASED depth (> 20cm default)
        valley_depths = [depth_grid[r][c] for r in range(6, 8) for c in range(4)]
        mean_valley_depth = sum(valley_depths) / len(valley_depths)
        assert mean_valley_depth > 20.0, (
            f"Valley cells should have gained sediment (depth > 20cm), got mean={mean_valley_depth:.4f}cm"
        )

    def test_crucible_mass_conservation(self):
        """Total field soil must not increase (boundary escape is OK, creation is not)."""
        farm, field = self._build_farm()
        farm.run(days=30)
        state = field._field_state
        depth_grid = state.custom.get("effective_soil_depth_cm_grid")
        assert depth_grid is not None

        rows, cols = 8, 4
        # Read actual initial depth from the grid before any erosion
        # (grid is initialized at run() start; we verify conservation holds)
        initial_total = 20.0 * rows * cols   # 20cm default × 32 cells
        final_total = sum(depth_grid[r][c] for r in range(rows) for c in range(cols))
        # Conservation: final ≤ initial (boundary escape allowed, no creation)
        assert final_total <= initial_total + 1e-6, (
            f"Mass conservation violated: final={final_total:.4f} > initial={initial_total:.4f}"
        )

    def test_sediment_transport_requires_erosion(self):
        """use_physics(sediment_transport=True, erosion=False) must raise."""
        from cropforge import Farm
        from cropforge.runtime import CropForgeConfigError
        farm = Farm(name="X", location=(20.0, 85.0))
        with pytest.raises(CropForgeConfigError, match="requires erosion=True"):
            farm.use_physics(erosion=False, sediment_transport=True)

    def test_cumulative_grids_written(self):
        """Season-total grids must be present and have positive values on slope."""
        farm, field = self._build_farm()
        farm.run(days=30)
        state = field._field_state

        cum_eroded    = state.custom.get("cumulative_sediment_eroded_mm_grid")
        cum_deposited = state.custom.get("cumulative_sediment_deposited_mm_grid")
        assert cum_eroded    is not None, "cumulative_sediment_eroded_mm_grid missing"
        assert cum_deposited is not None, "cumulative_sediment_deposited_mm_grid missing"

        # At least some slope cell has eroded material
        slope_eroded = max(cum_eroded[r][c] for r in range(6) for c in range(4))
        assert slope_eroded > 0.0, "No erosion recorded on slope after 30 days"

        # At least some valley cell received deposits
        valley_deposited = max(cum_deposited[r][c] for r in range(6, 8) for c in range(4))
        assert valley_deposited > 0.0, "No deposition recorded in valley after 30 days"

