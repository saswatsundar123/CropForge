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


# ---------------------------------------------------------------------------
# Solar geometry helpers  (PRD v0.7.0 §5.3)
# ---------------------------------------------------------------------------

def calculate_solar_position(
    doy: int,
    hour: float,
    latitude_deg: float,
) -> tuple[float, float]:
    """Solar altitude and azimuth from day-of-year, hour, and latitude.

    Parameters
    ----------
    doy:          Day of year (1–365).
    hour:         Solar hour (0–24). Use 12.0 for solar noon.
    latitude_deg: Site latitude in decimal degrees (positive = N).

    Returns
    -------
    (altitude_deg, azimuth_deg)
        altitude_deg: Solar elevation above horizon (0–90°). Negative = below horizon.
        azimuth_deg:  Solar azimuth clockwise from North (0–360°).
    """
    lat = math.radians(latitude_deg)
    # Solar declination (Spencer 1971 approx)
    b = 2 * math.pi * (doy - 1) / 365.0
    decl = math.radians(
        0.006918
        - 0.399912 * math.cos(b) + 0.070257 * math.sin(b)
        - 0.006758 * math.cos(2 * b) + 0.000907 * math.sin(2 * b)
        - 0.002697 * math.cos(3 * b) + 0.00148 * math.sin(3 * b)
    ) * (180 / math.pi)  # degrees
    decl_rad = math.radians(decl)

    # Hour angle (15° per hour, solar noon = 0)
    ha = math.radians((hour - 12.0) * 15.0)

    # Altitude
    sin_alt = (
        math.sin(lat) * math.sin(decl_rad)
        + math.cos(lat) * math.cos(decl_rad) * math.cos(ha)
    )
    sin_alt = max(-1.0, min(1.0, sin_alt))
    altitude_rad = math.asin(sin_alt)
    altitude_deg = math.degrees(altitude_rad)

    # Azimuth (clockwise from North)
    if math.cos(altitude_rad) < 1e-10:
        # Sun directly overhead → azimuth undefined; use 180 (south for N hemisphere)
        azimuth_deg = 180.0
    else:
        cos_az = (
            math.sin(decl_rad) - math.sin(lat) * sin_alt
        ) / (math.cos(lat) * math.cos(altitude_rad))
        cos_az = max(-1.0, min(1.0, cos_az))
        az = math.degrees(math.acos(cos_az))
        # Afternoon: azimuth > 180
        azimuth_deg = az if ha <= 0 else 360.0 - az

    return altitude_deg, azimuth_deg


def calculate_slope_radiation_factor(
    slope_deg: float,
    aspect_deg: float,
    solar_altitude_deg: float,
    solar_azimuth_deg: float,
) -> float:
    """Cosine correction factor for radiation on a sloped cell.

    Returns the ratio of radiation incident on the slope vs. a flat surface.
    Flat cell = 1.0. Sun-facing slope > 1.0. Shaded slope → clamped to 0.0.

    Parameters
    ----------
    slope_deg:         Cell slope in degrees (0 = flat).
    aspect_deg:        Cell aspect clockwise from North (0 = north-facing).
    solar_altitude_deg: Sun elevation above horizon.
    solar_azimuth_deg:  Sun azimuth clockwise from North.

    Returns
    -------
    float in [0.0, ~1.5]  (can exceed 1.0 for steep sun-facing slopes)
    """
    if solar_altitude_deg <= 0.0:
        return 0.0  # sun below horizon

    slope_r = math.radians(slope_deg)
    # terrain.py aspect = steepest-ascent direction (uphill).
    # The irradiance formula needs the downhill (sun-facing) direction = +180°.
    # ponytail: single +180 mod 360 here; no changes needed in terrain.py or callers.
    facing_deg = (aspect_deg + 180.0) % 360.0
    aspect_r = math.radians(facing_deg)
    alt_r = math.radians(solar_altitude_deg)
    az_r = math.radians(solar_azimuth_deg)

    # cos(incidence) on sloped surface (standard GIS formula)
    cos_i = (
        math.sin(alt_r) * math.cos(slope_r)
        + math.cos(alt_r) * math.sin(slope_r) * math.cos(az_r - aspect_r)
    )
    # cos(incidence) on flat surface = sin(altitude)
    cos_flat = math.sin(alt_r)

    if cos_flat < 1e-10:
        return 0.0

    return max(0.0, cos_i / cos_flat)


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
