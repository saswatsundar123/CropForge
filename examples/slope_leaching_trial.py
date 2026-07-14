"""
examples/slope_leaching_trial.py
==================================
PRD v0.3.0 -- Slope N Leaching Trial (The Crucible)

Scientific Scenario
-------------------
A 3x3 grid field with a distinct North-to-South elevation gradient
(upper cells at 5 m, lower cells at 0 m). A single heavy fertilizer
event (120 kg N/ha) is applied uniformly on day 1. A high-rainfall
event (80 mm on day 3) drives saturated surface runoff, causing
nitrogen to move laterally downslope via the D8 routing algorithm.

Expected spatial outcome:
  - Top row cells (high elevation): nitrogen DEPLETED vs initial
  - Bottom row cells (low elevation): nitrogen ACCUMULATED vs initial

The test suite (test_integration_spatial.py) uses the same setup to
assert this spatial differentiation programmatically.

Usage
-----
  python examples/slope_leaching_trial.py

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropforge import Farm, Field, Crop, Event
from cropforge.state import EnvironmentState, SoilVoxelState


# ---------------------------------------------------------------------------
# Weather source: heavy rain on day 3, dry otherwise
# ---------------------------------------------------------------------------

class SlopeWeather:
    """Dry weather with one heavy rainfall event on day 3."""

    def get_day(self, day: int) -> EnvironmentState:
        return EnvironmentState(
            day=day, doy=day,
            temp_max_c=30.0, temp_min_c=18.0, temp_mean_c=24.0,
            radiation_mj_m2=18.0,
            rainfall_mm=80.0 if day == 3 else 0.0,  # heavy rain drives runoff
            et0_mm=5.0,
            wind_speed_ms=2.0,
            humidity_pct=60.0,
        )


# ---------------------------------------------------------------------------
# Soil: uniform N = 120 kg/ha, high moisture (near FC)
# ---------------------------------------------------------------------------

INITIAL_N_KG_HA = 120.0
INITIAL_MOISTURE = 38.0   # above FC=30% -> ensures saturation triggers runoff after rain

class LoamSoil:
    """Single-layer loam soil with uniform N across all cells."""

    def build_grid(self, rows: int, cols: int):
        return [
            [
                [SoilVoxelState(
                    row=r, col=c, layer=0,
                    depth_top_cm=0.0, depth_bottom_cm=20.0,
                    moisture_pct=INITIAL_MOISTURE,
                    nitrogen_kg_ha=INITIAL_N_KG_HA,
                    bulk_density=1.30,
                    penetration_resistance=0.5,
                )]
                for c in range(cols)
            ]
            for r in range(rows)
        ]


# ---------------------------------------------------------------------------
# Main trial
# ---------------------------------------------------------------------------

def run_slope_leaching_trial() -> dict:
    """Run the slope leaching trial and return final N grid."""
    print("=" * 65)
    print("  CropForge v0.3.0 -- Slope N Leaching Trial")
    print("  3x3 grid | North-to-South elevation gradient")
    print("=" * 65)

    farm = Farm(name="SlopeTrial", location=(23.5, 77.0))

    field = Field(name="SlopeField", rows=3, cols=3, area_ha=0.5)
    field.set_crop(Crop(species="Zea mays", variety="K1"))
    field.set_weather(SlopeWeather())
    field.set_soil(LoamSoil())

    # North-to-South slope: row 0 is highest (5 m), row 2 is lowest (0 m)
    # Columns are uniform in elevation within each row
    dem = np.array([
        [5.0, 5.0, 5.0],   # row 0 -- upslope
        [2.5, 2.5, 2.5],   # row 1 -- mid-slope
        [0.0, 0.0, 0.0],   # row 2 -- downslope / receiving
    ], dtype=float)
    field.set_elevation(dem)

    field.set_water_params(
        field_capacity_pct=30.0,
        wilting_point_pct=12.0,
        saturation_pct=44.0,
        drainage_coefficient=0.8,   # fast-draining to trigger runoff quickly
        crop_coefficient=1.0,
        stress_increment_per_day=0.05,
    )
    field.set_nitrogen_params(
        leaching_fraction=0.02,     # moderate leaching
        runoff_n_fraction=0.08,     # high runoff N fraction for visible lateral signal
    )
    farm.add_field(field)

    farm.use_physics(
        et0=True,
        water_balance=True,
        nutrients=True,
        lateral_flow=True,
    )

    # Uniform heavy fertilizer on day 1 (all cells get 120 kg N/ha)
    farm.add_event(Event.fertiliser(
        field="SlopeField", day=1, amount_kg_ha=120, layer=0
    ))

    @farm.step(phase=0)
    def grow(state, env):
        for plant in state.plants:
            plant.root_depth_cm = 20.0
        return state

    farm.run(days=10)

    # Extract final N grid
    final_n = field._field_state.soil
    n_grid = [[final_n[r][c][0].nitrogen_kg_ha for c in range(3)] for r in range(3)]

    print("\nInitial N (all cells):", INITIAL_N_KG_HA, "kg/ha")
    print("\nFinal N grid (kg/ha) after 10 days:")
    print("  (row 0 = upslope, row 2 = downslope)")
    print()
    for r in range(3):
        row_str = "  ".join(f"{n_grid[r][c]:7.2f}" for c in range(3))
        label = ["^ upslope  ", "  mid-slope", "v downslope"][r]
        print(f"  Row {r} [{label}]:  {row_str}  kg/ha")

    # Summary stats
    top_n    = sum(n_grid[0]) / 3
    bottom_n = sum(n_grid[2]) / 3

    print(f"\n  Mean N row 0 (upslope):   {top_n:.2f} kg/ha")
    print(f"  Mean N row 2 (downslope): {bottom_n:.2f} kg/ha")
    print(f"  Accumulation at downslope: {bottom_n - INITIAL_N_KG_HA:+.2f} kg/ha")
    print(f"  Depletion at upslope:     {top_n - INITIAL_N_KG_HA:+.2f} kg/ha")

    # PRD success criterion
    spatial_differentiation = bottom_n > top_n
    print("\n" + "=" * 65)
    print("  Spatial N differentiation (downslope > upslope):")
    print(f"  {'CONFIRMED OK' if spatial_differentiation else 'NOT OBSERVED FAIL'}")
    print("=" * 65)

    return {
        "n_grid": n_grid,
        "top_mean_n": top_n,
        "bottom_mean_n": bottom_n,
        "spatial_differentiation": spatial_differentiation,
    }


if __name__ == "__main__":
    result = run_slope_leaching_trial()
    sys.exit(0 if result["spatial_differentiation"] else 1)
