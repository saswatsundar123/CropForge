"""
cropforge/physics/soil.py
==========================
Soil-physics helper functions for CropForge v0.2.0.

All functions are pure mathematical transformations: no state objects are
read or written, making them trivially testable and reusable.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Root impedance thresholds (PRD v0.2.0 Section 5.3)
# ---------------------------------------------------------------------------

_IMPEDANCE_FREE_THRESHOLD_MPA = 1.0    # below this: unrestricted growth
_IMPEDANCE_HARD_PAN_MPA       = 2.5    # at or above this: growth = 0


def calculate_root_impedance(penetration_resistance_mpa: float) -> float:
    """Return a root growth multiplier (0.0 -- 1.0) for a given soil layer.

    Implements the three-regime model specified in PRD v0.2.0 Section 5.3:

    * **Regime 1 -- unrestricted** (MPa < 1.0):
        Returns 1.0. Roots extend at their maximum thermal-time rate.

    * **Regime 2 -- linear decline** (1.0 <= MPa < 2.5):
        Returns a multiplier that declines linearly from 1.0 at 1.0 MPa
        to 0.0 at 2.5 MPa.  Formula::

            multiplier = (2.5 - MPa) / (2.5 - 1.0)

    * **Regime 3 -- hard pan block** (MPa >= 2.5):
        Returns 0.0. Vertical root extension is completely blocked.

    Parameters
    ----------
    penetration_resistance_mpa:
        Mechanical resistance of the soil layer (MPa) from
        ``SoilVoxelState.penetration_resistance``.

    Returns
    -------
    float
        A multiplier in [0.0, 1.0] applied to the daily root extension rate.
        1.0 = unrestricted, 0.0 = completely blocked.

    Examples
    --------
    >>> calculate_root_impedance(0.5)
    1.0
    >>> calculate_root_impedance(1.75)   # midpoint of declining range
    0.5
    >>> calculate_root_impedance(3.0)
    0.0
    """
    mpa = penetration_resistance_mpa

    if mpa < _IMPEDANCE_FREE_THRESHOLD_MPA:
        return 1.0

    if mpa >= _IMPEDANCE_HARD_PAN_MPA:
        return 0.0

    # Linear decline from 1.0 (at free threshold) to 0.0 (at hard-pan threshold)
    span = _IMPEDANCE_HARD_PAN_MPA - _IMPEDANCE_FREE_THRESHOLD_MPA
    return (_IMPEDANCE_HARD_PAN_MPA - mpa) / span


# ---------------------------------------------------------------------------
# Clod dynamics (PRD v0.7.0 §7.4)
# ---------------------------------------------------------------------------

_ROUGHNESS_MIN: float = 0.1   # sealed surface floor; rain cannot break clods below this


def calculate_roughness_decay(
    roughness: float,
    rainfall_mm: float,
    decay_factor: float = 0.05,
    min_roughness: float = _ROUGHNESS_MIN,
) -> float:
    """Simulate clod breakdown: decay surface_roughness_index with rainfall.

    Each 10 mm of rain reduces (roughness - min_roughness) by decay_factor.
    Decay stops at min_roughness (sealed surface).

    Parameters
    ----------
    roughness:
        Current surface_roughness_index (0.0 -- 1.0).
    rainfall_mm:
        Today's rainfall (mm). No decay when 0.
    decay_factor:
        Fraction of the excess roughness lost per 10 mm rain. Default 0.05.
    min_roughness:
        Floor value; roughness cannot drop below this. Default 0.1.

    Returns
    -------
    float
        Updated surface_roughness_index >= min_roughness.

    Examples
    --------
    >>> round(calculate_roughness_decay(0.8, 20.0), 4)
    0.793
    >>> calculate_roughness_decay(0.1, 20.0)   # already at floor
    0.1
    >>> calculate_roughness_decay(0.8, 0.0)    # no rain
    0.8
    """
    if rainfall_mm <= 0.0 or roughness <= min_roughness:
        return roughness
    decay = decay_factor * (rainfall_mm / 10.0) * (roughness - min_roughness)
    return max(min_roughness, roughness - decay)


# ---------------------------------------------------------------------------
# Erosion index (PRD v0.7.0 §7.5 — simplified RUSLE)
# ---------------------------------------------------------------------------

def calculate_erosion_index(
    runoff_mm: float,
    slope_frac: float,
    surface_roughness: float = 0.0,
    vegetation_cover_frac: float = 0.0,
) -> float:
    """Daily soil erosion index for one cell (dimensionless, simplified RUSLE).

    Erosion is the product of runoff intensity and slope transport potential,
    damped independently by surface roughness (clods / tillage barriers) and
    vegetation cover (roots / canopy). Both dampers are multiplicative.

    Returns 0.0 when either *runoff_mm* or *slope_frac* is zero or negative,
    guaranteeing **exactly zero erosion on flat fields** (PRD §7.5 criterion).

    Parameters
    ----------
    runoff_mm:
        Surface runoff today (mm). Use ``SoilVoxelState.custom['surface_runoff_mm_today']``
        when water_balance is enabled; otherwise pass ``env.rainfall_mm`` as proxy.
    slope_frac:
        Normalised local slope (0.0 -- 1.0) from ``_compute_slope_normalized``.
    surface_roughness:
        Current ``surface_roughness_index`` (0.0 -- 1.0). Higher = more clod
        barriers = less erosion. Default 0.0 (bare tilled surface).
    vegetation_cover_frac:
        Fraction of ground covered by vegetation (0.0 -- 1.0).
        Derived from LAI: ``min(1, LAI / 3)``.  Default 0.0 (bare ground).

    Returns
    -------
    float
        Dimensionless erosion index >= 0.0 for accumulation over the season.

    Examples
    --------
    >>> calculate_erosion_index(0.0, 0.5)   # no runoff
    0.0
    >>> calculate_erosion_index(10.0, 0.0)  # flat field
    0.0
    >>> round(calculate_erosion_index(10.0, 0.5, surface_roughness=0.3), 4)
    3.5
    """
    if runoff_mm <= 0.0 or slope_frac <= 0.0:
        return 0.0
    roughness_damper = max(0.0, 1.0 - surface_roughness)
    veg_damper = max(0.0, 1.0 - vegetation_cover_frac)
    return runoff_mm * slope_frac * roughness_damper * veg_damper
