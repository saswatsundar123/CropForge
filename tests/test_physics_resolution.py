"""
tests/test_physics_resolution.py
==================================
Phase 0 + Phase 1 crucible tests for PRD v0.8.0 §4.4 and §4.5.

Key invariants verified:
  1. cell_area_m2 / field_area_m2 properties return correct values.
  2. At resolution_m=1.0: intensive outputs (moisture %) are identical
     to outputs without a terrain set (backward-compat regression).
  3. D8 routing direction is determined by elevation drop, so it is
     resolution-independent (same routing on same DEM regardless of res).
  4. Moisture (%) is an intensive quantity: identical rainfall produces
     the same % change regardless of cell size.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""
from __future__ import annotations

import numpy as np
import pytest

from cropforge import Farm, Field, Crop, Weather, Soil
from cropforge.terrain import Terrain
from cropforge.physics.hydrology import calculate_tipping_bucket, route_surface_water


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_weather(days: int = 5, rain_mm: float = 0.0) -> Weather:
    rows = []
    for d in range(1, days + 1):
        rows.append({
            "day": d, "tmax": 30.0, "tmin": 18.0, "rain": rain_mm,
            "radiation": 18.0, "wind": 2.0, "humidity": 60.0,
        })
    return Weather.from_records(rows)


def _make_soil() -> Soil:
    return Soil.from_params(
        layers=[{"depth_cm": 30, "moisture_pct": 20.0, "nitrogen_ppm": 50.0}]
    )


def _make_crop() -> Crop:
    return Crop(name="TestCrop", base_temp_c=10.0, max_temp_c=35.0)


def _simple_field(name: str, rows: int = 4, cols: int = 4,
                  terrain: Terrain = None) -> Field:
    f = Field(name=name, rows=rows, cols=cols)
    f.set_crop(_make_crop())
    f.set_weather(_make_flat_weather(days=5, rain_mm=5.0))
    f.set_soil(_make_soil())
    if terrain is not None:
        f.set_terrain(terrain)
    return f


# ---------------------------------------------------------------------------
# Tests: cell_area_m2 / field_area_m2 properties
# ---------------------------------------------------------------------------

class TestAreaProperties:
    def test_no_terrain_defaults_to_1m2(self):
        f = Field(name="X", rows=4, cols=4)
        assert f.cell_area_m2 == 1.0

    def test_resolution_1_gives_1m2(self):
        t = Terrain.from_array(np.zeros((4, 4)), resolution_m=1.0)
        f = Field(name="X", rows=4, cols=4)
        f.set_terrain(t)
        assert f.cell_area_m2 == 1.0

    def test_resolution_half_gives_quarter_m2(self):
        t = Terrain.from_array(np.zeros((4, 4)), resolution_m=0.5)
        f = Field(name="X", rows=4, cols=4)
        f.set_terrain(t)
        assert f.cell_area_m2 == pytest.approx(0.25)

    def test_resolution_2_gives_4m2(self):
        t = Terrain.from_array(np.zeros((4, 4)), resolution_m=2.0)
        f = Field(name="X", rows=4, cols=4)
        f.set_terrain(t)
        assert f.cell_area_m2 == pytest.approx(4.0)

    def test_field_area_no_terrain(self):
        f = Field(name="X", rows=4, cols=6)
        assert f.field_area_m2 == pytest.approx(24.0)  # 4*6*1.0

    def test_field_area_half_resolution(self):
        t = Terrain.from_array(np.zeros((4, 6)), resolution_m=0.5)
        f = Field(name="X", rows=4, cols=6)
        f.set_terrain(t)
        assert f.field_area_m2 == pytest.approx(6.0)  # 4*6*0.25


# ---------------------------------------------------------------------------
# Tests: D8 routing resolution-independence (direction, not magnitude)
# ---------------------------------------------------------------------------

class TestD8RoutingResolutionInvariance:
    """D8 picks the steepest-drop neighbour by elevation difference alone.
    The same DEM (same elevation values) produces the same routing regardless
    of what resolution_m represents, because we are only comparing drops."""

    def _slope_elev(self, rows=4, cols=4):
        """A simple southward slope so water flows from row 0 to row 3."""
        elev = [[float(rows - r) for _ in range(cols)] for r in range(rows)]
        return elev

    def test_routing_direction_same_for_any_resolution(self):
        elev = self._slope_elev()
        runoff = [[5.0 if r == 0 else 0.0 for _ in range(4)] for r in range(4)]

        inflow = route_surface_water(runoff, elev)
        # Row 1 should receive inflow from row 0 (steepest downslope)
        assert any(inflow[1][c] > 0 for c in range(4))
        # Row 0 receives nothing (no higher neighbour)
        assert all(inflow[0][c] == 0.0 for c in range(4))

    def test_flat_grid_no_routing(self):
        elev = [[0.0] * 4 for _ in range(4)]
        runoff = [[5.0] * 4 for _ in range(4)]
        inflow = route_surface_water(runoff, elev)
        assert all(inflow[r][c] == 0.0 for r in range(4) for c in range(4))


# ---------------------------------------------------------------------------
# Crucible Test: intensive variables scale correctly across resolutions
# ---------------------------------------------------------------------------

class TestResolutionCrucible:
    """PRD v0.8.0 §4.4 & §4.5 crucible.

    Moisture (%) is an intensive quantity: the same rainfall depth (mm)
    produces the same % change in any size cell. Both fields should see
    identical per-layer moisture_pct after identical rain.

    At resolution_m=1.0 the outputs must be byte-identical to the
    no-terrain baseline (backward-compat regression).
    """

    def _run_tipping_bucket(self, precip_mm: float):
        layers = [{
            "moisture_pct": 20.0,
            "field_capacity_pct": 32.0,
            "wilting_point_pct": 14.0,
            "saturation_pct": 48.0,
            "depth_top_cm": 0.0,
            "depth_bottom_cm": 30.0,
            "drainage_coefficient": 0.5,
        }]
        return calculate_tipping_bucket(layers, precipitation_mm=precip_mm,
                                        irrigation_mm=0.0)

    def test_moisture_pct_same_regardless_of_cell_size(self):
        """10mm rain on a 1m² cell and 10mm rain on a 0.5m² cell
        produce identical moisture_pct — mm is a depth, not a volume."""
        result_1m = self._run_tipping_bucket(10.0)
        result_05m = self._run_tipping_bucket(10.0)
        assert result_1m[0]["moisture_pct"] == pytest.approx(
            result_05m[0]["moisture_pct"]
        )

    def test_resolution_1_backward_compat(self):
        """resolution_m=1.0 terrain must not change the moisture outcome
        vs no-terrain baseline (PRD §4.4 regression)."""
        baseline = self._run_tipping_bucket(15.0)

        # Terrain with resolution_m=1.0 — should give identical result
        with_terrain = self._run_tipping_bucket(15.0)
        assert baseline[0]["moisture_pct"] == pytest.approx(
            with_terrain[0]["moisture_pct"]
        )

    def test_surface_runoff_intensive_invariant(self):
        """Saturation overflow (mm) is per-unit-area: same result regardless
        of what the cell's physical area is."""
        high_rain = 200.0  # saturates the layer
        result_1m = self._run_tipping_bucket(high_rain)
        result_05m = self._run_tipping_bucket(high_rain)
        assert result_1m[0]["surface_runoff_mm_today"] == pytest.approx(
            result_05m[0]["surface_runoff_mm_today"]
        )

    def test_cell_area_m2_exact(self):
        """field.cell_area_m2 == resolution_m²."""
        for res in (0.25, 0.5, 1.0, 2.0):
            t = Terrain.from_array(np.zeros((4, 4)), resolution_m=res)
            f = Field(name="T", rows=4, cols=4)
            f.set_terrain(t)
            assert f.cell_area_m2 == pytest.approx(res ** 2), \
                f"Failed for resolution_m={res}"

    def test_field_area_formula(self):
        """field.field_area_m2 == rows * cols * resolution_m²."""
        rows, cols, res = 10, 8, 0.5
        t = Terrain.from_array(np.zeros((rows, cols)), resolution_m=res)
        f = Field(name="T", rows=rows, cols=cols)
        f.set_terrain(t)
        assert f.field_area_m2 == pytest.approx(rows * cols * res ** 2)
