"""
examples/maize_dual_plot_v2.py
================================
CropForge v0.5.0 -- Maize Dual-Plot Crucible (Plugin Edition)

Same dual-plot hard-pan scenario as maize_dual_plot.py, but the root-
impedance clamping and water-stress logic are now encapsulated inside
StandardMaize instead of manual @farm.step functions.

Demonstrates:
    field.use_plugin(StandardMaize())

Expected outcomes are identical to maize_dual_plot.py:
    Plot A max root depth > 19 cm  (no hard pan)
    Plot B max root depth <= 19 cm  (hard pan at 19 cm blocks growth)

Usage::
    python examples/maize_dual_plot_v2.py

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from cropforge.farm import Farm, Field
from cropforge.crop import Crop
from cropforge.loaders import Weather, Soil
from cropforge.plugins import StandardMaize

# ---------------------------------------------------------------------------
# Constants (kept identical to maize_dual_plot.py for parity)
# ---------------------------------------------------------------------------
DAYS              = 90
GRID_ROWS         = 10
GRID_COLS         = 10
AREA_HA           = 0.5
HARD_PAN_DEPTH_CM = 19.0
LATITUDE_DEG      = 23.3
ELEVATION_M       = 300.0

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR   = SCRIPT_DIR / "data"

# ---------------------------------------------------------------------------
# Farm + fields (identical setup to maize_dual_plot.py)
# ---------------------------------------------------------------------------
farm = Farm(name="Maize_DualPlot_v2_2026", location=(LATITUDE_DEG, 85.3))

field_a = Field(
    name="Plot_A_Slope",
    rows=GRID_ROWS, cols=GRID_COLS, area_ha=AREA_HA,
    elevation_profile="slope_2pct_N",
)
field_a.set_crop(Crop(species="Zea mays", variety="DH_Maize"))
field_a.set_weather(Weather.from_csv(str(DATA_DIR / "maize_weather_90d.csv")))
field_a.set_soil(Soil.from_csv(str(DATA_DIR / "maize_soil_plotA_slope.csv"), apply="uniform"))
farm.add_field(field_a)

field_b = Field(
    name="Plot_B_Hardpan",
    rows=GRID_ROWS, cols=GRID_COLS, area_ha=AREA_HA,
    elevation_profile=None,
)
field_b.set_crop(Crop(species="Zea mays", variety="DH_Maize"))
field_b.set_weather(Weather.from_csv(str(DATA_DIR / "maize_weather_90d.csv")))
field_b.set_soil(Soil.from_csv(str(DATA_DIR / "maize_soil_plotB_hardpan.csv"), apply="uniform"))
farm.add_field(field_b)

# ---------------------------------------------------------------------------
# Physics engines (same as original)
# ---------------------------------------------------------------------------
farm.use_physics(
    et0=True,
    root_impedance=True,
    elevation_m=ELEVATION_M,
    anemometer_height_m=2.0,
)

# ---------------------------------------------------------------------------
# Attach StandardMaize plugin to BOTH fields
# Each field gets its own plugin instance (isolated state)
# ---------------------------------------------------------------------------
field_a.use_plugin(StandardMaize)
field_b.use_plugin(StandardMaize)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("=" * 60)
print("CropForge v0.5.0 -- Maize Dual-Plot Crucible (Plugin Edition)")
print("=" * 60)
print(f"  Plot A: {field_a.name}  ({GRID_ROWS}x{GRID_COLS} grid, slope 2%)")
print(f"  Plot B: {field_b.name}  ({GRID_ROWS}x{GRID_COLS} grid, hard pan at 19cm)")
print(f"  Days  : {DAYS}")
print(f"  Plugin: StandardMaize  (root impedance + water stress)")
print()

farm.run(days=DAYS)

print(f"\nSimulation complete. Log -> {farm._last_log_path}\n")

# ---------------------------------------------------------------------------
# Summary (same structure as maize_dual_plot.py for diff comparison)
# ---------------------------------------------------------------------------
for fname, field in [("Plot_A_Slope", field_a), ("Plot_B_Hardpan", field_b)]:
    plants = field._field_state.plants
    alive_count = sum(1 for p in plants if p.alive)
    max_root    = max((p.root_depth_cm for p in plants), default=0.0)
    mean_root   = sum(p.root_depth_cm for p in plants) / len(plants) if plants else 0.0
    print(f"  {fname}:")
    print(f"    Alive plants     : {alive_count} / {len(plants)}")
    print(f"    Max root depth   : {max_root:.2f} cm")
    print(f"    Mean root depth  : {mean_root:.2f} cm")

print()
print(f"Expected: Plot A max root > {HARD_PAN_DEPTH_CM} cm")
print(f"Expected: Plot B max root <= {HARD_PAN_DEPTH_CM} cm")
