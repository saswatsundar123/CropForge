"""
tests/test_physics_soil.py
===========================
Tests for cropforge.physics.soil.calculate_root_impedance

Verification against PRD v0.2.0 Section 5.3 three-regime model:
  - MPa < 1.0  --> multiplier = 1.0 (unrestricted)
  - 1.0 <= MPa < 2.5  --> linear decline 1.0 -> 0.0
  - MPa >= 2.5  --> multiplier = 0.0 (hard pan)
"""

import pytest
from cropforge.physics.soil import calculate_root_impedance


# ---------------------------------------------------------------------------
# 1. Regime 1: unrestricted growth (MPa < 1.0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mpa", [0.0, 0.1, 0.5, 0.99])
def test_unrestricted_growth_below_1_mpa(mpa):
    """Any penetration resistance below 1.0 MPa must return multiplier 1.0."""
    assert calculate_root_impedance(mpa) == pytest.approx(1.0), (
        f"Expected 1.0 at {mpa} MPa, got {calculate_root_impedance(mpa)}"
    )


# ---------------------------------------------------------------------------
# 2. Regime 3: hard pan block (MPa >= 2.5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mpa", [2.5, 3.0, 5.0, 10.0, 100.0])
def test_hard_pan_block_at_or_above_2_5_mpa(mpa):
    """Any penetration resistance >= 2.5 MPa must return multiplier 0.0."""
    assert calculate_root_impedance(mpa) == pytest.approx(0.0), (
        f"Expected 0.0 at {mpa} MPa, got {calculate_root_impedance(mpa)}"
    )


# ---------------------------------------------------------------------------
# 3. Regime 2: linear decline (1.0 <= MPa < 2.5)
# ---------------------------------------------------------------------------

def test_linear_decline_at_1_mpa_boundary():
    """At the lower boundary (1.0 MPa) multiplier should be exactly 1.0."""
    m = calculate_root_impedance(1.0)
    assert m == pytest.approx(1.0), f"Expected 1.0 at 1.0 MPa, got {m}"


def test_linear_decline_at_midpoint():
    """At the midpoint (1.75 MPa) multiplier should be 0.5."""
    m = calculate_root_impedance(1.75)
    assert m == pytest.approx(0.5, abs=1e-9), f"Expected 0.5 at 1.75 MPa, got {m}"


def test_linear_decline_at_2_0_mpa():
    """At 2.0 MPa multiplier should be 1/3 (~0.333)."""
    # (2.5 - 2.0) / (2.5 - 1.0) = 0.5 / 1.5 = 1/3
    expected = (2.5 - 2.0) / (2.5 - 1.0)
    m = calculate_root_impedance(2.0)
    assert m == pytest.approx(expected, abs=1e-9), (
        f"Expected {expected:.4f} at 2.0 MPa, got {m}"
    )


def test_linear_decline_at_2_25_mpa():
    """At 2.25 MPa multiplier should be 1/6 (~0.1667)."""
    expected = (2.5 - 2.25) / (2.5 - 1.0)
    m = calculate_root_impedance(2.25)
    assert m == pytest.approx(expected, abs=1e-9), (
        f"Expected {expected:.4f} at 2.25 MPa, got {m}"
    )


def test_linear_decline_just_below_hard_pan():
    """Just below 2.5 MPa (e.g. 2.499) multiplier should be very small but > 0."""
    m = calculate_root_impedance(2.499)
    assert 0.0 < m < 0.01, (
        f"Expected tiny positive value just below hard-pan threshold, got {m}"
    )


# ---------------------------------------------------------------------------
# 4. Monotonic decrease in Regime 2
# ---------------------------------------------------------------------------

def test_monotonic_decrease_across_linear_regime():
    """Multiplier must strictly decrease as resistance increases in [1, 2.5)."""
    mpa_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.499]
    multipliers = [calculate_root_impedance(m) for m in mpa_values]
    for i in range(len(multipliers) - 1):
        assert multipliers[i] > multipliers[i + 1], (
            f"Multiplier not monotonically decreasing: "
            f"m({mpa_values[i]:.3f})={multipliers[i]:.4f} "
            f"<= m({mpa_values[i+1]:.3f})={multipliers[i+1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Output bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mpa", [0.0, 0.5, 1.0, 1.5, 2.0, 2.49, 2.5, 3.0])
def test_multiplier_always_in_0_to_1(mpa):
    """Root impedance multiplier must always be in [0.0, 1.0]."""
    m = calculate_root_impedance(mpa)
    assert 0.0 <= m <= 1.0, (
        f"Multiplier {m} at {mpa} MPa is outside [0, 1]"
    )


# ---------------------------------------------------------------------------
# 6. PRD specification spot checks (exact values stated in PRD v0.2.0 §5.3)
# ---------------------------------------------------------------------------

def test_prd_spec_0_5_mpa():
    """PRD v0.2.0 implicit spec: 0.5 MPa -> 1.0 (unrestricted)."""
    assert calculate_root_impedance(0.5) == pytest.approx(1.0)


def test_prd_spec_2_0_mpa():
    """PRD v0.2.0 Example: 2.0 MPa -> linearly declining (between 0 and 1)."""
    m = calculate_root_impedance(2.0)
    assert 0.0 < m < 1.0


def test_prd_spec_3_0_mpa():
    """PRD v0.2.0 spec: 3.0 MPa -> 0.0 (hard pan block)."""
    assert calculate_root_impedance(3.0) == pytest.approx(0.0)
