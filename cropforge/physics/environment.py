"""
cropforge/physics/environment.py
=================================
Penman-Monteith reference evapotranspiration (FAO-56 Equation 6).

All equations and constants follow:
    Allen R.G., Pereira L.S., Raes D., Smith M. (1998).
    FAO Irrigation and Drainage Paper No. 56 -- Crop Evapotranspiration.
    Food and Agriculture Organisation of the United Nations, Rome.

This module contains only pure mathematical functions.  It does not read
from or write to any CropForge state objects, making it trivially testable
and usable outside the CropForge engine loop.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# FAO-56 constants
# ---------------------------------------------------------------------------

_ALBEDO_REFERENCE = 0.23        # Eq. 38  -- reference grass surface
_STEFAN_BOLTZMANN = 4.903e-9    # MJ K-4 m-2 day-1  (Eq. 39)
_SOIL_HEAT_FLUX_G = 0.0         # Daily timestep approximation (FAO-56 p.54)


def _sat_vapour_pressure(temp_c: float) -> float:
    """Saturation vapour pressure at *temp_c* (kPa) -- FAO-56 Eq. 11."""
    return 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))


def _slope_svp(temp_mean_c: float) -> float:
    """Slope of SVP curve Delta (kPa/degC) at *temp_mean_c* -- FAO-56 Eq. 13."""
    es = _sat_vapour_pressure(temp_mean_c)
    return 4098.0 * es / (temp_mean_c + 237.3) ** 2


def _psychrometric_constant(elevation_m: float) -> float:
    """Psychrometric constant gamma (kPa/degC) -- FAO-56 Eq. 8.

    Atmospheric pressure P = 101.3 * ((293 - 0.0065*z) / 293)^5.26 (Eq. 7).
    gamma = 0.000665 * P  (Eq. 8).
    """
    p_kpa = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    return 0.000665 * p_kpa


def _actual_vapour_pressure(
    temp_max_c: float, temp_min_c: float, humidity_pct: float
) -> float:
    """Actual vapour pressure ea (kPa) -- FAO-56 Eq. 17.

    ea = (es_Tmin * RHmax + es_Tmax * RHmin) / 2

    For daily data where only mean RH is available, FAO-56 Eq. 19 is used:
        ea = (RHmean / 100) * (es_Tmax + es_Tmin) / 2
    """
    es_tmax = _sat_vapour_pressure(temp_max_c)
    es_tmin = _sat_vapour_pressure(temp_min_c)
    es_mean = (es_tmax + es_tmin) / 2.0
    return (humidity_pct / 100.0) * es_mean


def _wind_speed_at_2m(wind_speed_ms: float, measurement_height_m: float = 2.0) -> float:
    """Convert wind speed from measurement height to 2 m using log profile.

    FAO-56 Eq. 47: u2 = uz * 4.87 / ln(67.8*z - 5.42)
    Valid for z > 0.1 m.
    """
    if abs(measurement_height_m - 2.0) < 1e-6:
        return wind_speed_ms
    if measurement_height_m <= 0.1:
        raise ValueError(
            f"measurement_height_m must be > 0.1 m, got {measurement_height_m}"
        )
    return wind_speed_ms * 4.87 / math.log(67.8 * measurement_height_m - 5.42)


def _net_radiation(
    radiation_mj_m2: float,
    temp_max_c: float,
    temp_min_c: float,
    vp_kpa: float,
    doy: int,
    latitude_deg: float,
    elevation_m: float,
) -> float:
    """Net radiation Rn (MJ m-2 day-1) at the reference crop surface.

    Rns = (1 - alpha) * Rs                              (Eq. 38)
    Rnl = sigma * (T4_max + T4_min)/2 * (0.34 - 0.14*sqrt(ea)) * (1.35*Rs/Rso - 0.35)
                                                         (Eq. 39)
    Rn  = Rns - Rnl                                      (Eq. 40)

    Clear-sky radiation Rso:
        Rso = (0.75 + 2e-5 * elevation_m) * Ra           (Eq. 37)
    where Ra = extraterrestrial radiation (Eq. 21).
    """
    # --- Net short-wave radiation (Rns) ---
    rns = (1.0 - _ALBEDO_REFERENCE) * radiation_mj_m2

    # --- Extraterrestrial radiation (Ra) -- FAO-56 Eq. 21 ---
    lat_rad = math.radians(latitude_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)          # Eq. 23
    delta_sol = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)  # Eq. 24
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta_sol))            # Eq. 25
    ra = (
        24.0 * 60.0 / math.pi    # 24 h × 60 min/h — Gsc is in MJ m-2 min-1 (FAO-56 Eq. 21)
        * 0.0820                  # Gsc = 0.0820 MJ m-2 min-1
        * dr
        * (
            ws * math.sin(lat_rad) * math.sin(delta_sol)
            + math.cos(lat_rad) * math.cos(delta_sol) * math.sin(ws)
        )
    )  # Ra: extraterrestrial radiation, MJ m-2 day-1


    # --- Clear-sky solar radiation (Rso) -- Eq. 37 ---
    rso = (0.75 + 2e-5 * elevation_m) * ra
    rso = max(rso, 1e-6)   # avoid division by zero on polar winter days

    # --- Net long-wave radiation (Rnl) -- Eq. 39 ---
    t_max_k4 = (temp_max_c + 273.16) ** 4
    t_min_k4 = (temp_min_c + 273.16) ** 4
    humidity_term = 0.34 - 0.14 * math.sqrt(max(vp_kpa, 0.0))
    cloudiness_term = 1.35 * (radiation_mj_m2 / rso) - 0.35
    cloudiness_term = max(0.05, min(cloudiness_term, 1.0))  # clamp to physical range
    rnl = _STEFAN_BOLTZMANN * (t_max_k4 + t_min_k4) / 2.0 * humidity_term * cloudiness_term

    return rns - rnl


def calculate_fao56_et0(
    *,
    temp_max_c: float,
    temp_min_c: float,
    humidity_pct: float,
    wind_speed_ms: float,
    radiation_mj_m2: float,
    elevation_m: float,
    latitude_deg: float,
    doy: int,
    anemometer_height_m: float = 2.0,
) -> dict:
    """Compute FAO-56 Penman-Monteith reference ET0 and all intermediates.

    Parameters
    ----------
    temp_max_c:
        Daily maximum air temperature (degC).
    temp_min_c:
        Daily minimum air temperature (degC).
    humidity_pct:
        Mean daily relative humidity (%).
    wind_speed_ms:
        Mean daily wind speed at *anemometer_height_m* (m/s).
    radiation_mj_m2:
        Incoming solar short-wave radiation (MJ m-2 day-1).
    elevation_m:
        Site elevation above mean sea level (m). Used for psychrometric
        constant and clear-sky radiation.
    latitude_deg:
        Site latitude in decimal degrees (positive = N, negative = S).
        Used for extraterrestrial radiation calculation.
    doy:
        Day-of-year (1-366). Used for solar geometry.
    anemometer_height_m:
        Height of wind measurement above ground surface (m). Defaults to
        2.0 m (standard). Supply actual height if weather station is at
        a different elevation -- the logarithmic correction (FAO-56 Eq. 47)
        is applied automatically.

    Returns
    -------
    dict with keys:
        et0_mm            -- Reference ET0 (mm/day)  [primary result]
        vp_kpa            -- Actual vapour pressure (kPa)
        psychrometric_kpa -- Psychrometric constant gamma (kPa/degC)
        slope_svp         -- Slope of SVP curve Delta (kPa/degC)
        net_radiation_mj  -- Net radiation Rn (MJ m-2 day-1)
        es_kpa            -- Saturation vapour pressure (kPa)
        vpd_kpa           -- Vapour pressure deficit (kPa)
        u2_ms             -- Wind speed corrected to 2 m height (m/s)

    Notes
    -----
    FAO-56 Equation 6::

        ET0 = (0.408 * Delta * (Rn - G) + gamma * (900/(T+273)) * u2 * VPD)
              / (Delta + gamma * (1 + 0.34 * u2))

    where G = 0 for daily time step.

    Raises
    ------
    ValueError
        If ``radiation_mj_m2 < 0`` or ``humidity_pct`` is outside [0, 100].
    """
    if radiation_mj_m2 < 0:
        raise ValueError(
            f"radiation_mj_m2 must be >= 0, got {radiation_mj_m2}"
        )
    if not (0.0 <= humidity_pct <= 100.0):
        raise ValueError(
            f"humidity_pct must be in [0, 100], got {humidity_pct}"
        )

    temp_mean_c = (temp_max_c + temp_min_c) / 2.0

    # Intermediates
    u2 = _wind_speed_at_2m(wind_speed_ms, anemometer_height_m)
    delta = _slope_svp(temp_mean_c)
    gamma = _psychrometric_constant(elevation_m)
    ea = _actual_vapour_pressure(temp_max_c, temp_min_c, humidity_pct)
    es_tmax = _sat_vapour_pressure(temp_max_c)
    es_tmin = _sat_vapour_pressure(temp_min_c)
    es = (es_tmax + es_tmin) / 2.0
    vpd = max(es - ea, 0.0)
    rn = _net_radiation(
        radiation_mj_m2, temp_max_c, temp_min_c, ea, doy, latitude_deg, elevation_m
    )
    g = _SOIL_HEAT_FLUX_G

    # FAO-56 Eq. 6
    numerator = (
        0.408 * delta * (rn - g)
        + gamma * (900.0 / (temp_mean_c + 273.0)) * u2 * vpd
    )
    denominator = delta + gamma * (1.0 + 0.34 * u2)

    et0_mm = max(numerator / denominator, 0.0)   # ET0 cannot be negative

    return {
        "et0_mm": et0_mm,
        "vp_kpa": ea,
        "psychrometric_kpa": gamma,
        "slope_svp": delta,
        "net_radiation_mj": rn,
        "es_kpa": es,
        "vpd_kpa": vpd,
        "u2_ms": u2,
    }
