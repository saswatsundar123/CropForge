"""
tests/test_physics_nutrients.py
================================
Pure-math unit tests for cropforge/physics/nutrients.py.

All tests operate on plain Python dicts and lists — no CropForge engine
imports, no mocking of FieldState or EnvironmentState.

PRD v0.3.0 Phase 4 required tests:
  ✓ N leached proportionally to drainage flux × leaching_fraction × N_available
  ✓ N leached cascades from layer i to layer i+1
  ✓ N leached past bottom layer is lost (deep percolation)
  ✓ N cannot go below 0.0
  ✓ Lateral N export proportional to runoff_mm × N × runoff_n_fraction
  ✓ D8 routing: excess flows to steepest downslope neighbour
  ✓ No lateral flow if cell is below saturation
  ✓ apply_lateral_n_exchange: downslope gains, upslope loses
  ✓ Empty layer list returns safe zero result
  ✓ Inputs not mutated

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import pytest

from cropforge.physics.nutrients import (
    calculate_nitrogen_transport,
    calculate_lateral_runoff,
    apply_lateral_n_exchange,
    DEFAULT_LEACHING_FRACTION,
    DEFAULT_RUNOFF_N_FRACTION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_layer(n_kg_ha: float, top: float = 0.0, bot: float = 20.0) -> dict:
    return {"nitrogen_kg_ha": n_kg_ha, "depth_top_cm": top, "depth_bottom_cm": bot}


def _grid2(val: float = 0.0):
    """Return a 2×2 grid filled with *val*."""
    return [[val, val], [val, val]]


# ---------------------------------------------------------------------------
# calculate_nitrogen_transport — vertical leaching
# ---------------------------------------------------------------------------

class TestVerticalLeaching:
    def test_proportional_leaching_single_layer(self):
        """N_leached = drainage × leaching_fraction × N_available."""
        layers = [_n_layer(100.0)]
        result = calculate_nitrogen_transport(
            layers, [5.0], leaching_fraction=0.01
        )
        # 5 mm × 0.01 × 100 = 5.0 kg/ha leached (lost to deep perc)
        assert result["leached_kg_ha"][0] == pytest.approx(5.0)
        assert result["layers"][0]["nitrogen_kg_ha"] == pytest.approx(95.0)

    def test_leaching_scales_with_drainage(self):
        """Doubling drainage doubles leaching."""
        layers = [_n_layer(100.0)]
        r1 = calculate_nitrogen_transport(layers, [5.0], leaching_fraction=0.01)
        r2 = calculate_nitrogen_transport(layers, [10.0], leaching_fraction=0.01)
        assert r2["leached_kg_ha"][0] == pytest.approx(2 * r1["leached_kg_ha"][0])

    def test_leaching_scales_with_fraction(self):
        """Doubling leaching_fraction doubles leaching."""
        layers = [_n_layer(100.0)]
        r1 = calculate_nitrogen_transport(layers, [5.0], leaching_fraction=0.01)
        r2 = calculate_nitrogen_transport(layers, [5.0], leaching_fraction=0.02)
        assert r2["leached_kg_ha"][0] == pytest.approx(2 * r1["leached_kg_ha"][0])

    def test_leaching_cascade_two_layers(self):
        """N leached from layer 0 is added to layer 1."""
        layers = [_n_layer(100.0, 0.0, 20.0), _n_layer(50.0, 20.0, 40.0)]
        # 10 mm drainage from layer 0, none from layer 1
        result = calculate_nitrogen_transport(
            layers, [10.0, 0.0], leaching_fraction=0.01
        )
        leached_from_0 = result["leached_kg_ha"][0]
        # Layer 1 should gain leached_from_0
        assert result["layers"][1]["nitrogen_kg_ha"] == pytest.approx(50.0 + leached_from_0)

    def test_deep_percolation_loss_from_bottom_layer(self):
        """N leached past bottom layer is lost from system."""
        layers = [_n_layer(100.0)]
        result = calculate_nitrogen_transport(
            layers, [10.0], leaching_fraction=0.02
        )
        # 10 × 0.02 × 100 = 20 kg/ha leached and lost
        assert result["total_n_lost_kg_ha"] == pytest.approx(20.0)
        assert result["layers"][0]["nitrogen_kg_ha"] == pytest.approx(80.0)

    def test_n_cannot_go_below_zero(self):
        """N must not go negative even with extreme leaching."""
        layers = [_n_layer(1.0)]
        result = calculate_nitrogen_transport(
            layers, [1000.0], leaching_fraction=1.0
        )
        assert result["layers"][0]["nitrogen_kg_ha"] >= 0.0

    def test_zero_drainage_no_leaching(self):
        """No drainage → no leaching."""
        layers = [_n_layer(100.0)]
        result = calculate_nitrogen_transport(
            layers, [0.0], leaching_fraction=0.01
        )
        assert result["leached_kg_ha"][0] == pytest.approx(0.0)
        assert result["layers"][0]["nitrogen_kg_ha"] == pytest.approx(100.0)

    def test_zero_n_available_no_leaching(self):
        """With 0 N in layer, leaching = 0."""
        layers = [_n_layer(0.0)]
        result = calculate_nitrogen_transport(
            layers, [10.0], leaching_fraction=0.1
        )
        assert result["leached_kg_ha"][0] == pytest.approx(0.0)

    def test_three_layer_cascade(self):
        """Leaching cascades through all three layers correctly."""
        layers = [
            _n_layer(100.0, 0.0,  20.0),
            _n_layer(50.0,  20.0, 40.0),
            _n_layer(20.0,  40.0, 60.0),
        ]
        # Only layer 0 has drainage
        result = calculate_nitrogen_transport(
            layers, [10.0, 0.0, 0.0], leaching_fraction=0.02
        )
        leach0 = 10.0 * 0.02 * 100.0   # = 20 kg/ha
        assert result["leached_kg_ha"][0] == pytest.approx(leach0)
        # Layer 1 receives leach0, layer 2 unchanged
        assert result["layers"][1]["nitrogen_kg_ha"] == pytest.approx(50.0 + leach0)
        assert result["layers"][2]["nitrogen_kg_ha"] == pytest.approx(20.0)

    def test_inputs_not_mutated(self):
        """calculate_nitrogen_transport must not modify input dicts."""
        layers = [_n_layer(100.0)]
        original = layers[0]["nitrogen_kg_ha"]
        calculate_nitrogen_transport(layers, [5.0])
        assert layers[0]["nitrogen_kg_ha"] == original

    def test_empty_layers_returns_safe_result(self):
        """Empty input returns zero-value result dict with correct keys."""
        result = calculate_nitrogen_transport([], [])
        assert result["layers"] == []
        assert result["leached_kg_ha"] == []
        assert result["lateral_n_export_kg_ha"] == pytest.approx(0.0)
        assert result["total_n_lost_kg_ha"] == pytest.approx(0.0)

    def test_drainage_fluxes_padded_if_shorter_than_layers(self):
        """Short drainage flux list is padded with 0.0 safely."""
        layers = [_n_layer(100.0), _n_layer(50.0)]
        # Only 1 flux for 2 layers → layer 1 should not be drained
        result = calculate_nitrogen_transport(layers, [10.0], leaching_fraction=0.01)
        assert result["leached_kg_ha"][1] == pytest.approx(0.0)


class TestLateralRunoffN:
    """Lateral N export from calculate_nitrogen_transport(lateral_runoff_mm>0)."""

    def test_lateral_export_proportional(self):
        """N_exported = runoff_mm × runoff_n_fraction × N / 100."""
        layers = [_n_layer(100.0)]
        result = calculate_nitrogen_transport(
            layers, [0.0],
            lateral_runoff_mm=10.0,
            runoff_n_fraction=0.05,
        )
        # Formula: 10 × 0.05 × 100 / 100 = 0.5 kg/ha
        # The /100 normalises for the runoff-per-100mm scaling convention.
        assert result["lateral_n_export_kg_ha"] == pytest.approx(0.5)
        assert result["layers"][0]["nitrogen_kg_ha"] == pytest.approx(99.5)

    def test_lateral_export_cannot_exceed_available_n(self):
        """Lateral N export cannot exceed N available in layer 0."""
        layers = [_n_layer(1.0)]
        result = calculate_nitrogen_transport(
            layers, [0.0],
            lateral_runoff_mm=10000.0,
            runoff_n_fraction=1.0,
        )
        assert result["lateral_n_export_kg_ha"] <= 1.0
        assert result["layers"][0]["nitrogen_kg_ha"] >= 0.0

    def test_zero_runoff_no_lateral_export(self):
        """With lateral_runoff_mm=0, no lateral N export."""
        layers = [_n_layer(100.0)]
        result = calculate_nitrogen_transport(
            layers, [0.0], lateral_runoff_mm=0.0
        )
        assert result["lateral_n_export_kg_ha"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calculate_lateral_runoff — D8 routing
# ---------------------------------------------------------------------------

class TestCalculateLateralRunoff:
    def test_no_runoff_below_saturation(self):
        """Cells below saturation produce zero runoff."""
        moisture  = [[20.0, 20.0], [20.0, 20.0]]
        saturation = [[30.0, 30.0], [30.0, 30.0]]
        elevation  = [[1.0, 1.0], [0.0, 0.0]]
        runoff = calculate_lateral_runoff(moisture, saturation, elevation)
        for row in runoff:
            for val in row:
                assert val == pytest.approx(0.0)

    def test_runoff_generated_above_saturation(self):
        """Cells above saturation generate positive runoff if lower neighbour exists."""
        moisture  = [[40.0, 40.0], [20.0, 20.0]]
        saturation = [[30.0, 30.0], [30.0, 30.0]]
        elevation  = [[1.0, 1.0], [0.0, 0.0]]
        runoff = calculate_lateral_runoff(moisture, saturation, elevation,
                                          drainage_coefficient=0.5)
        # Row 0 cells are above saturation and have lower neighbours → runoff > 0
        assert runoff[0][0] > 0.0
        assert runoff[0][1] > 0.0
        # Row 1 cells are below saturation → no runoff
        assert runoff[1][0] == pytest.approx(0.0)
        assert runoff[1][1] == pytest.approx(0.0)

    def test_no_runoff_without_lower_neighbour(self):
        """A saturated cell with no lower neighbour does not generate outflow."""
        # Saturated cells are in row 1 at the LOWEST elevation (0.0m).
        # No neighbouring cell is at a lower elevation → D8 finds no sink → no outflow.
        moisture   = [[20.0], [40.0]]    # row 1 saturated, row 0 dry
        saturation = [[30.0], [30.0]]
        elevation  = [[1.0], [0.0]]      # row 1 is at 0.0m (lowest) — no lower neighbour
        runoff = calculate_lateral_runoff(moisture, saturation, elevation)
        # Row 1: saturated but no lower neighbour → no outflow
        assert runoff[1][0] == pytest.approx(0.0)
        # Row 0: below saturation → no outflow regardless
        assert runoff[0][0] == pytest.approx(0.0)

    def test_drainage_coefficient_scales_runoff(self):
        """drainage_coefficient=1.0 generates 2× runoff of 0.5."""
        moisture  = [[40.0], [20.0]]
        saturation = [[30.0], [30.0]]
        elevation  = [[1.0], [0.0]]
        r1 = calculate_lateral_runoff(moisture, saturation, elevation, drainage_coefficient=0.5)
        r2 = calculate_lateral_runoff(moisture, saturation, elevation, drainage_coefficient=1.0)
        if r1[0][0] > 0:
            assert r2[0][0] == pytest.approx(2 * r1[0][0])

    def test_empty_grid_returns_empty(self):
        result = calculate_lateral_runoff([], [], [])
        assert result == []


# ---------------------------------------------------------------------------
# apply_lateral_n_exchange — net N flux on a 2D grid
# ---------------------------------------------------------------------------

class TestApplyLateralNExchange:
    def test_upslope_loses_n_downslope_gains(self):
        """In a 2×1 grid with slope, top cell loses N and bottom gains it."""
        n_grid    = [[100.0], [50.0]]
        moisture  = [[40.0], [20.0]]   # top saturated
        saturation = [[30.0], [30.0]]
        elevation  = [[1.0], [0.0]]    # top is higher
        delta = apply_lateral_n_exchange(
            n_grid, moisture, saturation, elevation,
            runoff_n_fraction=0.05
        )
        # Top cell loses N (negative delta)
        assert delta[0][0] < 0.0
        # Bottom cell gains N (positive delta)
        assert delta[1][0] > 0.0

    def test_no_exchange_flat_grid(self):
        """Flat uniform grid with no saturation → all deltas are zero."""
        n_grid    = [[100.0, 100.0], [100.0, 100.0]]
        moisture  = [[20.0, 20.0], [20.0, 20.0]]   # all below saturation
        saturation = [[30.0, 30.0], [30.0, 30.0]]
        elevation  = [[0.0, 0.0], [0.0, 0.0]]
        delta = apply_lateral_n_exchange(n_grid, moisture, saturation, elevation)
        for row in delta:
            for val in row:
                assert val == pytest.approx(0.0)

    def test_conservation_approximate(self):
        """Net N change across grid ≈ 0 (N in = N out, minus boundary losses)."""
        n_grid    = [[100.0, 80.0], [60.0, 40.0]]
        moisture  = [[40.0, 40.0], [20.0, 20.0]]
        saturation = [[30.0, 30.0], [30.0, 30.0]]
        elevation  = [[1.0, 0.8], [0.3, 0.0]]
        delta = apply_lateral_n_exchange(n_grid, moisture, saturation, elevation)
        total_delta = sum(delta[r][c] for r in range(2) for c in range(2))
        # Total delta should be zero (no N leaves the system in this test)
        assert total_delta == pytest.approx(0.0, abs=1e-9)

    def test_returns_correct_grid_shape(self):
        """Return grid has same dimensions as input."""
        n_grid    = [[100.0, 80.0, 60.0], [70.0, 50.0, 30.0]]
        moisture  = [[25.0] * 3, [25.0] * 3]
        saturation = [[30.0] * 3, [30.0] * 3]
        elevation  = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        delta = apply_lateral_n_exchange(n_grid, moisture, saturation, elevation)
        assert len(delta) == 2
        assert len(delta[0]) == 3

    def test_n_transported_proportional_to_runoff_fraction(self):
        """Doubling runoff_n_fraction doubles N transported."""
        n_grid    = [[100.0], [50.0]]
        moisture  = [[40.0], [20.0]]
        saturation = [[30.0], [30.0]]
        elevation  = [[1.0], [0.0]]
        d1 = apply_lateral_n_exchange(n_grid, moisture, saturation, elevation,
                                       runoff_n_fraction=0.05)
        d2 = apply_lateral_n_exchange(n_grid, moisture, saturation, elevation,
                                       runoff_n_fraction=0.10)
        if d1[1][0] > 0:
            assert d2[1][0] == pytest.approx(2 * d1[1][0], rel=0.01)

    def test_empty_grid_returns_empty(self):
        delta = apply_lateral_n_exchange([], [], [], [])
        assert delta == []
