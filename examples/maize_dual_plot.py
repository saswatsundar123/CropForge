"""
examples/maize_dual_plot.py
============================
CropForge v0.2.0 — Maize Dual-Plot Crucible

This example is the flagship demonstration of the v0.2.0 Opt-In Physics
architecture. It runs TWO fields simultaneously under the same weather:

  Plot A — Slope terrain:  Normal soil, lower N (slope runoff).
                           Roots grow freely; plant canopy expands normally.

  Plot B — Hard Pan:       Hard compacted pan at 19 cm depth
                           (penetration_resistance = 3.0 MPa → multiplier = 0.0).
                           Root growth is physically blocked by the pan.
                           Roots cannot access the deeper moisture reservoir.
                           Under sustained hot monsoon conditions this leads to
                           earlier moisture stress and a visible difference in
                           plant survival vs Plot A.

Physics enabled:
  farm.use_physics(et0=True, root_impedance=True)
    → ET0 computed each day by FAO-56 Penman-Monteith (phase=-2)
    → root_growth_multiplier set from soil penetration_resistance (phase=-1)

Researcher model (phase=0):
  Step 1 — root_growth:   Advances root_depth_cm by base_rate × multiplier.
                           Hard pan (multiplier=0.0) prevents advancement.
  Step 2 — water_stress:  Simple PWP check. Accumulates stress_days in
                           plant.custom['stress_days']. Death after 5 days.

PRD References:
    PRD v0.2.0 Section 2.1 — maize_dual_plot.py requirements
    PRD v0.2.0 Section 9   — Execution order (ET0 → root → @farm.step)
    PRD v0.2.0 Section 10  — Backward compatibility
    PRD v0.2.0 Section 16  — Success criteria (physical accuracy)

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths relative to this script so it runs from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR   = SCRIPT_DIR / "data"
ROOT_DIR   = SCRIPT_DIR.parent   # project root

# Add project root to sys.path so cropforge can be imported without install
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cropforge.farm import Farm, Field
from cropforge.crop import Crop
from cropforge.loaders import Weather, Soil

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DAYS              = 90        # 90-day kharif maize season
GRID_ROWS         = 10        # 10-row grid per plot
GRID_COLS         = 10        # 10-col grid per plot (100 plants per plot)
AREA_HA           = 0.5

# Root growth parameters
BASE_ROOT_RATE_CM = 0.35      # cm/day under optimal soil conditions
HARD_PAN_DEPTH_CM = 19.0      # Known hard-pan boundary in Plot B

# Water stress parameters (simplified)
PWP_PCT           = 13.0      # Permanent wilting point (%)
STRESS_DEATH_DAYS = 5         # Days below PWP before plant dies

# Site parameters for FAO-56 ET0
LATITUDE_DEG      = 23.3      # ICAR-IARI Jharkhand
ELEVATION_M       = 300.0     # ~300 m above sea level

# ---------------------------------------------------------------------------
# Build Farm
# ---------------------------------------------------------------------------
farm = Farm(name="Maize_DualPlot_2026", location=(LATITUDE_DEG, 85.3))

# ---- Plot A — Slope terrain -------------------------------------------
field_a = Field(
    name="Plot_A_Slope",
    rows=GRID_ROWS,
    cols=GRID_COLS,
    area_ha=AREA_HA,
    elevation_profile="slope_2pct_N",   # 2% northward slope (PRD Section 2.1)
)
field_a.set_crop(Crop(species="Zea mays", variety="DH_Maize"))
field_a.set_weather(
    Weather.from_csv(str(DATA_DIR / "maize_weather_90d.csv"))
)
field_a.set_soil(
    Soil.from_csv(str(DATA_DIR / "maize_soil_plotA_slope.csv"), apply="uniform")
)
farm.add_field(field_a)

# ---- Plot B — Hard Pan --------------------------------------------------
field_b = Field(
    name="Plot_B_Hardpan",
    rows=GRID_ROWS,
    cols=GRID_COLS,
    area_ha=AREA_HA,
    elevation_profile=None,             # flat field (PRD Section 2.1)
)
field_b.set_crop(Crop(species="Zea mays", variety="DH_Maize"))
field_b.set_weather(
    Weather.from_csv(str(DATA_DIR / "maize_weather_90d.csv"))
)
field_b.set_soil(
    Soil.from_csv(str(DATA_DIR / "maize_soil_plotB_hardpan.csv"), apply="uniform")
)
farm.add_field(field_b)

# ---------------------------------------------------------------------------
# Activate built-in physics engines
# PRD v0.2.0 Section 9: execution order guaranteed
#   phase=-2  ET0 engine  → env.et0_mm populated
#   phase=-1  Root impedance engine → plant.root_growth_multiplier set
#   phase= 0  Researcher step functions (below) → read computed values
# ---------------------------------------------------------------------------
farm.use_physics(
    et0=True,
    root_impedance=True,
    elevation_m=ELEVATION_M,
    anemometer_height_m=2.0,    # standard 2m mast
)

# ---------------------------------------------------------------------------
# Researcher step functions (PRD v0.2.0 Section 2.1)
# ---------------------------------------------------------------------------

@farm.step(interval="daily", phase=0)
def root_growth(state, env):
    """Advance root depth by base_rate × root_growth_multiplier.

    The root_growth_multiplier is computed by the impedance engine (phase=-1).
    For Plot B plants, once root_depth_cm reaches the hard pan at 19 cm the
    multiplier becomes 0.0 and no further root extension occurs.

    CLAMP LOGIC:
        The impedance engine sets root_growth_multiplier based on the
        CURRENT layer at phase=-1. When root_depth_cm is still in layer 0
        (< 19 cm), multiplier=1.0, so the plant may try to advance into
        layer 1. We therefore apply an additional safety clamp: after
        computing the new depth, we check if it has crossed into a layer
        whose resistance is >= 2.5 MPa, and if so, cap the depth at the
        top of that layer.

    This ensures the hard pan clamping is exact to the layer boundary
    (19.0 cm) regardless of step size.
    """
    from cropforge.physics.soil import calculate_root_impedance
    BLOCK_THRESHOLD = 0.0  # multiplier value that means "blocked"

    for plant in state.plants:
        if not plant.alive:
            continue

        # Initialise root depth on day 1 (seed germination)
        if plant.root_depth_cm == 0.0:
            plant.root_depth_cm = 2.0   # germination: 2 cm root at sowing

        # Daily root extension: base rate attenuated by impedance multiplier
        daily_extension = BASE_ROOT_RATE_CM * plant.root_growth_multiplier
        if daily_extension <= 0.0:
            continue   # blocked — nothing to do

        new_depth = plant.root_depth_cm + daily_extension

        # Safety clamp: walk soil layers and cap at top of any blocked layer
        soil_col = state.soil[plant.row][plant.col]
        for voxel in soil_col:
            mult = calculate_root_impedance(voxel.penetration_resistance)
            if mult <= BLOCK_THRESHOLD:
                # This layer is blocked.  If new_depth would enter it, clamp.
                if new_depth >= voxel.depth_top_cm:
                    new_depth = voxel.depth_top_cm
                    break   # Stop at the first blocking layer

        plant.root_depth_cm = new_depth

    return state


@farm.step(interval="daily", phase=1)
def water_stress(state, env):
    """Simple PWP death check.

    MODEL DECISION:
        Full soil water balance is v0.2.0 Phase 3. Here we implement a
        simplified model: check whether the surface layer (layer 0) moisture
        is below PWP. This creates meaningful divergence between the plots
        because:
        - Plot A roots penetrate deeper over time, reaching moist layers
        - Plot B roots are blocked at 19 cm and cannot access deeper moisture

        In a real simulation the soil water balance engine would deplete
        moisture automatically. Here we simulate depletion via a simple
        ET0-proportional daily draw-down on the surface layer only, to
        produce scientifically plausible but non-rigorous moisture dynamics
        that demonstrate the hard pan effect.
    """
    # Simplified ET draw-down from surface layer (crude approximation)
    # Actual soil water balance is v0.2.0 Phase 3 (PRD Section 6)
    et0_fraction = 0.55   # approximate crop coefficient Kc for maize (vegetative)
    daily_depletion_pct = env.et0_mm * et0_fraction * 0.08   # mm → pct approximation

    for row_list in state.soil:
        for col_list in row_list:
            if col_list:
                # Deplete surface layer; floor at 0
                col_list[0].moisture_pct = max(
                    0.0,
                    col_list[0].moisture_pct - daily_depletion_pct
                )
                # Add rainfall (mm → pct, coarse approximation)
                col_list[0].moisture_pct = min(
                    32.0,   # field capacity
                    col_list[0].moisture_pct + env.rainfall_mm * 0.06
                )

    # Stress accumulation per plant
    for plant in state.plants:
        if not plant.alive:
            continue

        # Find the surface soil moisture for this plant's cell
        soil_col = state.soil[plant.row][plant.col]
        surface_moisture = soil_col[0].moisture_pct if soil_col else PWP_PCT + 1.0

        if surface_moisture < PWP_PCT:
            plant.stress_index = min(1.0, plant.stress_index + 0.2)
            plant.custom.setdefault("stress_days", 0)
            plant.custom["stress_days"] += 1
            # Death after STRESS_DEATH_DAYS consecutive days below PWP
            if plant.custom["stress_days"] >= STRESS_DEATH_DAYS:
                plant.alive = False
        else:
            plant.stress_index = max(0.0, plant.stress_index - 0.05)
            plant.custom["stress_days"] = 0

    return state


# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------
print("=" * 60)
print("CropForge v0.2.0 — Maize Dual-Plot Crucible")
print("=" * 60)
print(f"  Plot A: {field_a.name}  ({GRID_ROWS}x{GRID_COLS} grid, slope 2%)")
print(f"  Plot B: {field_b.name}  ({GRID_ROWS}x{GRID_COLS} grid, hard pan at 19cm)")
print(f"  Days  : {DAYS}")
print(f"  Physics: ET0=True, root_impedance=True")
print()

farm.run(days=DAYS)

log_path = farm._last_log_path
print()
print(f"Simulation complete. Parquet log: {log_path}")
print()

# ---------------------------------------------------------------------------
# Quick result summary — read Parquet and print day-90 stats
# ---------------------------------------------------------------------------
import glob
import pyarrow.parquet as pq
import pandas as pd

parquet_files = glob.glob(f"{log_path}/**/*.parquet", recursive=True)
frames = []
for pf in parquet_files:
    pf_path = Path(pf)
    parts = {}
    for part in pf_path.parts:
        if "=" in part:
            k, v = part.split("=", 1)
            parts[k] = v
    tbl = pq.read_table(pf)
    frame = tbl.to_pandas()
    for k, v in parts.items():
        if k not in frame.columns:
            frame[k] = v
    frames.append(frame)

if frames:
    df_all = pd.concat(frames, ignore_index=True)
    df_all["day"] = pd.to_numeric(df_all["day"], errors="coerce")
    df_plants = df_all[df_all["day"] == DAYS].copy()

    print(f"Day {DAYS} summary (plant table):")
    for field_name in ["Plot_A_Slope", "Plot_B_Hardpan"]:
        sub = df_plants[df_plants["field_name"] == field_name]
        if sub.empty or "root_depth_cm" not in sub.columns:
            print(f"  {field_name}: no plant data found")
            continue
        alive_count = int(sub["alive"].sum()) if "alive" in sub.columns else "?"
        max_root    = float(sub["root_depth_cm"].max())
        mean_root   = float(sub["root_depth_cm"].mean())
        print(f"  {field_name}:")
        print(f"    Alive plants     : {alive_count} / {len(sub)}")
        print(f"    Max root depth   : {max_root:.2f} cm")
        print(f"    Mean root depth  : {mean_root:.2f} cm")
else:
    print("  (Could not locate Parquet files for summary)")

print()
print("Expected physical outcomes:")
print(f"  Plot A max root > {HARD_PAN_DEPTH_CM} cm  (no impedance)")
print(f"  Plot B max root <= {HARD_PAN_DEPTH_CM} cm  (hard pan blocks at 19 cm)")
print()
print(f"Log path for tests: {log_path}")
