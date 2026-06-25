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
