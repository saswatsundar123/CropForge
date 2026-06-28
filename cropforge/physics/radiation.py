"""
cropforge/physics/radiation.py
================================
Radiation Interception Engine for CropForge v0.5.0.

Implements the Beer-Lambert law for canopy light interception:

    PAR_intercepted = solar_rad × 0.5 × (1 − e^(−k × LAI))

where:
    solar_rad  = incoming total solar radiation (MJ/m²/day)
    0.5        = PAR fraction of total solar radiation (standard assumption)
    k          = extinction coefficient (crop-specific; typically 0.4–0.6)
    LAI        = Leaf Area Index (m²/m²)

This module is a pure math module. It carries no simulation state. All
simulation-state interactions are handled by the hook factory
``make_radiation_hook()`` in builtin_hooks.py.

PRD Reference: §7.2, §7.3 (v0.5.0)

Scientific References:
    Monteith, J.L. (1977). Climate and the efficiency of crop production
        in Britain. Phil. Trans. R. Soc. Lond. B, 281, 277–294.
    FAO-56 (Allen et al. 1998). Chapter 10, Radiation interception.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# PAR fraction of total solar radiation (standard agronomy constant)
# ---------------------------------------------------------------------------

_PAR_FRACTION = 0.5   # 50 % of total solar radiation is PAR


def calculate_intercepted_par(
    solar_rad_mj: float,
    lai: float,
    k_extinction: float = 0.45,
) -> float:
    """Beer-Lambert canopy light interception.

    Returns the photosynthetically active radiation (PAR) intercepted by
    the crop canopy on a given day, in MJ m⁻² day⁻¹.

    Formula (PRD §7.2)::

        PAR_int = solar_rad × 0.5 × (1 − exp(−k × LAI))

    Parameters
    ----------
    solar_rad_mj:
        Incoming total solar radiation (MJ m⁻² day⁻¹). Maps to
        ``EnvironmentState.radiation_mj_m2``.
    lai:
        Leaf Area Index of the plant (m² leaf / m² ground). Must be ≥ 0.
        Negative values are clamped to zero.
    k_extinction:
        Beer-Lambert extinction coefficient (dimensionless).
        Typical range: 0.4–0.6. Default 0.45 (wheat / C3 crops).
        Use ~0.50 for C4 crops (maize).

    Returns
    -------
    float
        Intercepted PAR (MJ m⁻² day⁻¹). Always in [0, solar_rad × 0.5].

    Raises
    ------
    ValueError
        If ``solar_rad_mj < 0`` or ``k_extinction <= 0``.

    Examples
    --------
    >>> calculate_intercepted_par(solar_rad_mj=15.0, lai=0.0, k_extinction=0.45)
    0.0
    >>> # Crucible test (PRD §7.4): LAI=3.0, k=0.45, rad=15 MJ
    >>> round(calculate_intercepted_par(15.0, 3.0, 0.45), 6)
    6.256938
    >>> # High LAI approaches PAR_FRACTION × solar_rad
    >>> calculate_intercepted_par(15.0, 20.0, 0.45) < 15.0 * 0.5
    True
    """
    if solar_rad_mj < 0:
        raise ValueError(
            f"solar_rad_mj must be ≥ 0, got {solar_rad_mj!r}."
        )
    if k_extinction <= 0:
        raise ValueError(
            f"k_extinction must be > 0, got {k_extinction!r}."
        )

    lai_clamped = max(0.0, lai)

    intercepted = (
        solar_rad_mj
        * _PAR_FRACTION
        * (1.0 - math.exp(-k_extinction * lai_clamped))
    )
    return intercepted


def par_at_lais(
    solar_rad_mj: float,
    lai_values: list[float],
    k_extinction: float = 0.45,
) -> list[float]:
    """Vectorised convenience: compute intercepted PAR for a list of LAI values.

    Useful for batch processing an entire plant grid without calling
    calculate_intercepted_par() in a Python loop.

    Parameters
    ----------
    solar_rad_mj:
        Incoming solar radiation (MJ m⁻² day⁻¹).
    lai_values:
        Sequence of LAI values (one per plant).
    k_extinction:
        Extinction coefficient. Applied uniformly.

    Returns
    -------
    list[float]
        Intercepted PAR for each input LAI value.

    Examples
    --------
    >>> par_at_lais(15.0, [0.0, 1.0, 3.0], k_extinction=0.45)
    [0.0, ...]
    """
    return [
        calculate_intercepted_par(solar_rad_mj, lai, k_extinction)
        for lai in lai_values
    ]
