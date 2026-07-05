"""
tests/test_physics_radiation_topo.py
======================================
PRD v0.7.0 §5.5 — Crucible tests for slope-aspect radiation correction.

Tests:
  1. Flat terrain → multiplier = 1.0, PAR matches v0.6.0 scalar baseline.
  2. South-facing slope at solar noon (lat=45°N, day=180) → factor > 1.0.
  3. North-facing slope → factor < 1.0.
  4. Sun below horizon → factor = 0.0.
  5. Integration Crucible: South-facing plants get strictly more intercepted_par_mj
     than North-facing plants on the same field, same day, same LAI.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropforge.physics.radiation import (
    calculate_intercepted_par,
    calculate_slope_radiation_factor,
    calculate_solar_position,
)


# ---------------------------------------------------------------------------
# Unit tests: solar geometry
# ---------------------------------------------------------------------------

class TestSolarPosition:
    def test_noon_altitude_positive_lat(self):
        """At solar noon on summer solstice at 45°N, sun should be high."""
        alt, az = calculate_solar_position(doy=172, hour=12.0, latitude_deg=45.0)
        assert alt > 45.0, f"Expected altitude > 45°, got {alt:.2f}°"

    def test_winter_noon_lower_than_summer(self):
        """Winter sun lower than summer sun (same latitude)."""
        alt_summer, _ = calculate_solar_position(doy=172, hour=12.0, latitude_deg=45.0)
        alt_winter, _ = calculate_solar_position(doy=355, hour=12.0, latitude_deg=45.0)
        assert alt_summer > alt_winter

    def test_sun_below_horizon_at_midnight(self):
        """Sun should be below horizon at midnight (hour=0 → hour angle = -180°)."""
        alt, _ = calculate_solar_position(doy=180, hour=0.0, latitude_deg=45.0)
        assert alt < 0.0, f"Expected negative altitude at midnight, got {alt:.2f}°"

    def test_azimuth_near_south_at_noon_northern_hemisphere(self):
        """At solar noon in northern hemisphere, sun is roughly due south."""
        _, az = calculate_solar_position(doy=180, hour=12.0, latitude_deg=45.0)
        # Should be close to 180° (south), allow ±20° for declination effects
        assert 160.0 <= az <= 200.0, f"Expected azimuth ~180°, got {az:.2f}°"


# ---------------------------------------------------------------------------
# Unit tests: slope radiation factor
# ---------------------------------------------------------------------------

class TestSlopeRadiationFactor:
    def test_flat_surface_returns_1(self):
        """slope_deg=0 must return exactly 1.0 regardless of aspect."""
        alt, az = calculate_solar_position(doy=180, hour=12.0, latitude_deg=45.0)
        factor = calculate_slope_radiation_factor(
            slope_deg=0.0, aspect_deg=180.0,
            solar_altitude_deg=alt, solar_azimuth_deg=az,
        )
        assert abs(factor - 1.0) < 1e-9, f"Flat surface factor should be 1.0, got {factor}"

    def test_south_facing_slope_greater_than_one_at_noon_day180(self):
        """South-facing 20° slope at 45°N, day 180, solar noon → factor > 1.0.
        terrain.py convention: aspect = uphill direction.
        South-facing = HIGH north, LOW south = uphill toward north = aspect=0°.
        """
        alt, az = calculate_solar_position(doy=180, hour=12.0, latitude_deg=45.0)
        factor = calculate_slope_radiation_factor(
            slope_deg=20.0, aspect_deg=0.0,   # 0 = north = uphill = south-facing slope
            solar_altitude_deg=alt, solar_azimuth_deg=az,
        )
        assert factor > 1.0, (
            f"South-facing slope should receive > flat, got factor={factor:.4f} "
            f"(solar alt={alt:.1f}°, az={az:.1f}°)"
        )

    def test_north_facing_slope_less_than_one(self):
        """North-facing 20° slope at 45°N, day 180 → factor < 1.0.
        terrain.py convention: north-facing = uphill toward south = aspect=180°.
        """
        alt, az = calculate_solar_position(doy=180, hour=12.0, latitude_deg=45.0)
        factor = calculate_slope_radiation_factor(
            slope_deg=20.0, aspect_deg=180.0,  # 180 = south = uphill = north-facing slope
            solar_altitude_deg=alt, solar_azimuth_deg=az,
        )
        assert factor < 1.0, (
            f"North-facing slope should receive < flat, got factor={factor:.4f}"
        )

    def test_sun_below_horizon_returns_zero(self):
        """Negative solar altitude → factor = 0.0."""
        factor = calculate_slope_radiation_factor(
            slope_deg=20.0, aspect_deg=180.0,
            solar_altitude_deg=-5.0, solar_azimuth_deg=180.0,
        )
        assert factor == 0.0

    def test_south_vs_north_ordering(self):
        """South-facing slope must receive strictly more radiation than north-facing.
        terrain.py: south-facing=aspect 0° (uphill north), north-facing=aspect 180° (uphill south).
        """
        alt, az = calculate_solar_position(doy=180, hour=12.0, latitude_deg=45.0)
        f_south = calculate_slope_radiation_factor(20.0,   0.0, alt, az)  # south-facing
        f_north = calculate_slope_radiation_factor(20.0, 180.0, alt, az)  # north-facing
        assert f_south > f_north, (
            f"South factor {f_south:.4f} must exceed north factor {f_north:.4f}"
        )

    def test_hand_calculation_reference(self):
        """Verify against manual trig for known inputs.

        At 45°N, day 180, solar noon:
          altitude ≈ 68°, azimuth ≈ 180° (south)
          South-facing 10° slope: terrain aspect=0° (uphill=north), facing=180° (downhill=south)
            cos_i = sin(68°)cos(10°) + cos(68°)sin(10°)cos(180°-180°)
                  = sin(68°)cos(10°) + cos(68°)sin(10°)
                  = sin(68°+10°) = sin(78°)
            factor = sin(78°)/sin(68°) ≈ 1.053
        """
        alt, az = calculate_solar_position(doy=180, hour=12.0, latitude_deg=45.0)
        # aspect=0° = south-facing (uphill=north in terrain.py convention)
        factor = calculate_slope_radiation_factor(10.0, 0.0, alt, az)
        # Factor should be > 1 and within reasonable range of manual estimate
        assert 1.0 < factor < 1.20, (
            f"Expected factor in (1.0, 1.20) for 10° south slope, got {factor:.4f}"
        )


# ---------------------------------------------------------------------------
# Integration Crucible: full field-level PAR comparison
# ---------------------------------------------------------------------------

class TestCrucibleSouthVsNorthPAR:
    """PRD §5.5 Crucible: south-facing plants get more intercepted_par_mj than
    north-facing plants at 45°N, day 180, same LAI."""

    def _make_field_with_ew_ridge(self):
        """10×10 field with an East-West ridge: left half slopes South, right half North."""
        import numpy as np
        from cropforge.terrain import Terrain

        rows, cols = 10, 10
        elev = np.zeros((rows, cols))
        # Left 5 columns: south-facing (high in north, low in south)
        for r in range(rows):
            for c in range(5):
                elev[r, c] = (rows - 1 - r) * 0.5  # decreases going south
        # Right 5 columns: north-facing (high in south, low in north)
        for r in range(rows):
            for c in range(5, cols):
                elev[r, c] = r * 0.5  # decreases going north
        return Terrain.from_array(elev, resolution_m=1.0)

    def test_south_slope_more_par_than_north_slope(self):
        """Crucible: south-facing plants must receive strictly more PAR."""
        terrain = self._make_field_with_ew_ridge()

        alt_deg, az_deg = calculate_solar_position(doy=180, hour=12.0, latitude_deg=45.0)
        fixed_lai = 2.0
        fixed_k = 0.45
        solar_rad = 20.0  # MJ/m²/day

        # Collect mean PAR for south-facing and north-facing cells
        south_pars = []
        north_pars = []
        rows, cols = terrain.slope_grid.shape

        for r in range(rows):
            for c in range(cols):
                factor = calculate_slope_radiation_factor(
                    slope_deg=float(terrain.slope_grid[r, c]),
                    aspect_deg=float(terrain.aspect_grid[r, c]),
                    solar_altitude_deg=alt_deg,
                    solar_azimuth_deg=az_deg,
                )
                corrected_rad = solar_rad * factor
                par = calculate_intercepted_par(corrected_rad, fixed_lai, fixed_k)
                if c < 5:
                    south_pars.append(par)
                else:
                    north_pars.append(par)

        mean_south = sum(south_pars) / len(south_pars)
        mean_north = sum(north_pars) / len(north_pars)

        assert mean_south > mean_north, (
            f"CRUCIBLE FAILED: South-facing mean PAR {mean_south:.4f} MJ/m² "
            f"must exceed north-facing mean PAR {mean_north:.4f} MJ/m²"
        )

    def test_flat_baseline_matches_v060_exact(self):
        """Flat terrain (slope=0) must return same intercepted_par as v0.6.0 scalar path."""
        solar_rad = 15.0
        lai = 3.0
        k = 0.45

        # v0.6.0 path: direct scalar
        expected = calculate_intercepted_par(solar_rad, lai, k)

        # v0.7.0 path: factor=1.0 for flat slope
        factor = calculate_slope_radiation_factor(0.0, 0.0, 60.0, 180.0)
        assert abs(factor - 1.0) < 1e-9
        result = calculate_intercepted_par(solar_rad * factor, lai, k)

        assert abs(result - expected) < 1e-10, (
            f"Flat terrain must produce identical PAR: expected={expected}, got={result}"
        )
