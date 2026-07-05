"""
docs/tutorials/geomorphology.py
================================
CropForge v0.8.0 — Geomorphology Tutorial

Demonstrates the full erosion-to-deposition feedback loop:
  1. Build a sloped, undulating terrain with sub-metre resolution.
  2. Apply TiedRidges for conservation effect comparison.
  3. Enable full sediment physics (erosion + transport + depth feedback).
  4. Run 30 days of heavy monsoon rainfall.
  5. Print a spatial report: zones of net erosion vs. net deposition.

This tutorial showcases three v0.8.0 subsystems working together:
  - resolution_m propagation (all spatial fluxes in physical units)
  - Sediment transport + D8 routing (mass-conserved)
  - Dynamic elevation feedback (geomorphological loop)

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

import numpy as np
import pandas as pd
from cropforge import Farm, Field, Terrain, TiedRidges, ZeroTillage
from cropforge.loaders import Weather
from cropforge.plugins import StandardMaize

# ---------------------------------------------------------------------------
# 1. Terrain — 50×50 field at 0.5 m resolution (25m × 25m)
# ---------------------------------------------------------------------------
ROWS, COLS = 50, 50
RES = 0.5   # metres per cell

np.random.seed(42)
xs = np.linspace(0, 2 * np.pi, COLS)
zs = np.linspace(0, 2 * np.pi, ROWS)
xg, zg = np.meshgrid(xs, zs)
# Sloped terrain (north-high, south-low) with gentle undulation
elev = 5.0 - zg * (5.0 / (2 * np.pi)) + 0.4 * np.sin(xg * 2) * np.cos(zg)
elev = elev.astype(np.float32)

terrain = Terrain.from_array(elev, resolution_m=RES)

print(f"Terrain: {ROWS}×{COLS} cells at {RES} m resolution "
      f"({ROWS * RES:.0f}m × {COLS * RES:.0f}m)")
print(f"  Elevation range: {elev.min():.2f} – {elev.max():.2f} m")
print(f"  Mean slope: {terrain.slope_grid.mean():.1f}°")

# ---------------------------------------------------------------------------
# 2. Weather — 30 days of monsoon (heavy rainfall every 3 days)
# ---------------------------------------------------------------------------

DAYS = 30
weather_rows = []
for d in range(1, DAYS + 1):
    rain = 45.0 if d % 3 == 0 else 4.0   # heavy event every 3 days
    weather_rows.append({
        "day": d, "doy": d,
        "temp_max_c": 34.0, "temp_min_c": 22.0, "temp_mean_c": 28.0,
        "radiation_mj_m2": 20.0, "rainfall_mm": rain,
        "et0_mm": 6.5, "wind_speed_ms": 3.5, "humidity_pct": 75.0,
        "co2_ppm": 415.0,
    })

weather_df = pd.DataFrame(weather_rows).set_index("day")

# ---------------------------------------------------------------------------
# 3. Fields — TiedRidges (conservation) vs. ZeroTillage (control)
# ---------------------------------------------------------------------------

def make_field(name, land_prep):
    f = Field(name=name, rows=ROWS, cols=COLS)
    f.set_terrain(Terrain.from_array(elev.copy(), resolution_m=RES))
    f.set_land_prep(land_prep)
    f.set_weather(Weather(weather_df))
    f.use_plugin(StandardMaize)
    return f

field_tr  = make_field("TiedRidges_Field",  TiedRidges(
    ridge_height_m=0.20, ridge_spacing_m=2.0,
    tie_spacing_m=4.0,   tie_height_m=0.10,
))
field_ctl = make_field("Control_Field", ZeroTillage())

# ---------------------------------------------------------------------------
# 4. Farm — run with full v0.8.0 physics
# ---------------------------------------------------------------------------

farm = Farm(name="Geomorphology_Tutorial", location=(23.0, 82.0))
farm.add_field(field_tr)
farm.add_field(field_ctl)
farm.use_physics(
    et0=True, soil_water_balance=True,
    clod_dynamics=True, erosion=True, sediment_transport=True,
)

print(f"\nRunning {DAYS}-day monsoon simulation...")
farm.run(days=DAYS)
print("Done.")

# ---------------------------------------------------------------------------
# 5. Post-analysis — spatial sediment budget report
# ---------------------------------------------------------------------------

def sediment_summary(field_state, label):
    rows_n, cols_n = len(field_state.soil), len(field_state.soil[0])
    total_loss    = 0.0
    total_deposit = 0.0
    net_erosion_cells    = 0
    net_deposition_cells = 0

    for r in range(rows_n):
        for c in range(cols_n):
            loss  = field_state.soil[r][c][0].cumulative_sediment_loss_kg_m2
            depos = field_state.soil[r][c][0].cumulative_deposition_kg_m2
            net   = loss - depos
            total_loss    += loss
            total_deposit += depos
            if net > 0.001:
                net_erosion_cells += 1
            elif net < -0.001:
                net_deposition_cells += 1

    total_cells = rows_n * cols_n
    cell_area   = RES ** 2
    print(f"\n{label}")
    print(f"  Cells with net erosion    : {net_erosion_cells} / {total_cells}")
    print(f"  Cells with net deposition : {net_deposition_cells} / {total_cells}")
    print(f"  Total sediment loss       : {total_loss * cell_area:.4f} kg")
    print(f"  Total deposition          : {total_deposit * cell_area:.4f} kg")
    print(f"  Mass balance check        : loss={total_loss:.4f}, deposit={total_deposit:.4f}")
    return total_loss * cell_area

print("\n" + "=" * 60)
print("SEDIMENT BUDGET REPORT")
print("=" * 60)
loss_tr  = sediment_summary(farm._fields[0]._field_state, "TiedRidges (treatment)")
loss_ctl = sediment_summary(farm._fields[1]._field_state, "ZeroTillage (control)")

if loss_ctl > 0:
    reduction_pct = (loss_ctl - loss_tr) / loss_ctl * 100
    print(f"\n  TiedRidges reduced total sediment export by {reduction_pct:.1f}%")

print("=" * 60)
print("\nGeomorphology tutorial complete.")
print("Run `farm.visualize()` to inspect the dynamic elevation changes in 3D.")
