"""
examples/disease_outbreak_trial.py
=====================================
CropForge v0.5.0 -- Spatial Disease Outbreak Trial

Demonstrates the Wind-driven Anisotropic Disease Spread engine working
alongside the StandardWheat plugin and the Beer-Lambert Radiation engine.

Scenario
--------
* A 30x30 wheat field (900 plants), 90-day season.
* StandardWheat plugin drives phenology and biomass accumulation.
* Beer-Lambert radiation engine (k=0.45) writes intercepted_par_mj
  to every plant daily.
* On Day 40, a localised blight outbreak is seeded at the field center
  (row 15, col 15) via the farm.use_physics(disease=True, disease_foci=...)
  parameter -- exactly as the PRD sec.8 scenario specifies.
* Wind blows steadily EAST at 270 deg (from the West), anisotropy=0.80,
  amplifying eastward spread.
* The simulation prints a spatial disease map every 30 days showing
  which plants are Susceptible (.), Infected (I), or Resistant (R).
* Outputs a full Parquet log for dashboard playback.

Usage::
    python examples/disease_outbreak_trial.py

Expected output::
    Day 40: one center plant infected (seeded).
    Day 60: infection cluster spread eastward.
    Day 90: eastern half significantly more infected than western half.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from cropforge import Crop, Farm, Field, Soil, Weather
from cropforge.plugins import StandardWheat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Field geometry
# ---------------------------------------------------------------------------

ROWS, COLS = 30, 30         # 900 plants
CENTER_ROW  = ROWS // 2     # 15
CENTER_COL  = COLS // 2     # 15

# Disease parameters (sec.8.2)
OUTBREAK_DAY           = 40
WIND_FROM_DEGREES      = 270.0   # West wind -> blows East
SPREAD_RATE            = 0.20    # 20% daily transmission per neighbour
LATENCY_DAYS           = 3       # 3-day latent period before contagious
STRESS_INCREMENT       = 0.04    # +0.04 stress_index per infected day
ANISOTROPY             = 0.80    # Strong directional bias

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  CropForge v0.5.0 -- Disease Outbreak Trial")
print(f"  Field: {ROWS}x{COLS} ({ROWS*COLS} plants)  |  Duration: 90 days")
print(f"  Disease: seeded at ({CENTER_ROW},{CENTER_COL}) on Day {OUTBREAK_DAY}")
print(f"  Wind: {WIND_FROM_DEGREES} deg (from West -> blows East)")
print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Farm and field setup
# ---------------------------------------------------------------------------

farm  = Farm(name="DiseaseOutbreakFarm", location=(28.6, 77.2))
field = Field(
    name="WheatPlot_Blight",
    rows=ROWS,
    cols=COLS,
    area_ha=4.0,
    elevation_profile="flat",
)

_data_dir = Path(__file__).parent / "data"

field.set_crop(Crop(species="wheat", variety="HD-2967", sowing_doy=60))
field.set_weather(
    Weather.from_csv(
        str(_data_dir / "wheat_synthetic_weather_90d.csv"),
        date_col="date",
        tmax_col="tmax_c",
        tmin_col="tmin_c",
        radiation_col="radiation_mj",
        rainfall_col="rainfall_mm",
        humidity_col="humidity_pct",
        wind_col="wind_ms",
        wind_unit="m/s",
    )
)
field.set_soil(Soil.from_csv(str(_data_dir / "wheat_uniform_soil_3layer.csv"), apply="uniform"))
farm.add_field(field)

# ---------------------------------------------------------------------------
# Attach StandardWheat plugin
# ---------------------------------------------------------------------------
field.use_plugin(StandardWheat)

# ---------------------------------------------------------------------------
# Enable Advanced Physics:
#   - Radiation interception (Beer-Lambert, k=0.45) at phase=-2
#   - Spatial disease spread (SIR, wind=270 deg) at phase=-1
#
# NOTE: disease_foci seeds the center plant on simulation Day 1 of the hook,
#       but the infection only becomes VISIBLE after OUTBREAK_DAY.
#       We use farm.add_event to seed the outbreak on the correct day instead.
# ---------------------------------------------------------------------------

farm.use_physics(
    radiation=True,
    k_extinction=0.45,
    # Disease engine configured but foci seeded via Event below
    disease=True,
    disease_foci=None,          # No automatic day-1 seeding
    disease_spread_rate=SPREAD_RATE,
    disease_latency_days=LATENCY_DAYS,
    disease_stress_increment=STRESS_INCREMENT,
    disease_wind_direction_deg=WIND_FROM_DEGREES,
    disease_anisotropy=ANISOTROPY,
    disease_seed=42,            # Reproducible spread
)

# ---------------------------------------------------------------------------
# Day 40 Event: Introduce blight at field center
# ---------------------------------------------------------------------------

from cropforge.events import Event

@farm.add_event(Event.custom(field="WheatPlot_Blight", day=OUTBREAK_DAY))
def introduce_blight(field_state, env_state):
    """Seed the center plant with blight on Day 40."""
    center = next(
        p for p in field_state.plants
        if p.row == CENTER_ROW and p.col == CENTER_COL
    )
    center.custom["disease_state"] = "I"
    center.custom["days_infected"]  = 0
    center.custom["disease_stress"] = 0.0
    field_state.events_fired.append(
        f"Day {OUTBREAK_DAY}: Blight seeded at ({CENTER_ROW},{CENTER_COL})"
    )
    print(f"\n  [!] Day {OUTBREAK_DAY}: Blight outbreak seeded at center ({CENTER_ROW},{CENTER_COL})!\n")
    return field_state

# ---------------------------------------------------------------------------
# Diagnostic step: print spatial map every 30 days
# ---------------------------------------------------------------------------

def _disease_map(field_state, cols: int = COLS) -> str:
    """Render a compact grid showing S (.), I, R disease states."""
    rows = len(field_state.plants) // cols
    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            p = field_state.plants[r * cols + c]
            state = p.custom.get("disease_state", "S")
            row_chars.append("I" if state == "I" else ("R" if state == "R" else "."))
        lines.append("  " + "".join(row_chars))
    return "\n".join(lines)


@farm.step(interval="daily", phase=10)
def print_disease_snapshot(state, env):
    """Print spatial disease map at days 40, 60, 90."""
    if state.day in (40, 60, 90):
        infected = sum(1 for p in state.plants if p.custom.get("disease_state") == "I")
        resistant = sum(1 for p in state.plants if p.custom.get("disease_state") == "R")
        mid_col = COLS // 2
        east_i = sum(1 for p in state.plants if p.col >= mid_col and p.custom.get("disease_state") == "I")
        west_i = sum(1 for p in state.plants if p.col < mid_col and p.custom.get("disease_state") == "I")
        print(f"\n  -- Day {state.day} Disease Snapshot --")
        print(f"     Infected:  {infected:4d}  |  Resistant: {resistant}")
        print(f"     East half: {east_i:4d} I  |  West half:  {west_i:4d} I")
        print(f"     Wind-bias ratio (E/W+1): {east_i / (west_i + 1):.2f}x")
        print(_disease_map(state))
    return state

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("Starting 90-day simulation ...\n")
farm.run(days=90)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

final_state = farm._fields[0]._field_state
plants = final_state.plants
alive  = sum(1 for p in plants if p.alive)
mid_col = COLS // 2

infected  = sum(1 for p in plants if p.custom.get("disease_state") == "I")
resistant = sum(1 for p in plants if p.custom.get("disease_state") == "R")
east_i    = sum(1 for p in plants if p.col >= mid_col and p.custom.get("disease_state") == "I")
west_i    = sum(1 for p in plants if p.col < mid_col  and p.custom.get("disease_state") == "I")

avg_biomass = sum(p.biomass_g for p in plants) / len(plants) if plants else 0.0
avg_grain   = sum(p.custom.get("grain_biomass_g", 0.0) for p in plants) / len(plants) if plants else 0.0
avg_par     = sum(p.custom.get("intercepted_par_mj", 0.0) for p in plants) / len(plants) if plants else 0.0
stage_sample = plants[0].custom.get("phenological_stage", "?") if plants else "?"

print(f"\n{'=' * 60}")
print(f"  Final Summary -- Day 90")
print(f"{'=' * 60}")
print(f"  Plants alive:             {alive} / {len(plants)}")
print(f"  Mean biomass:             {avg_biomass:.2f} g/plant")
print(f"  Mean grain biomass:       {avg_grain:.2f} g/plant")
print(f"  Mean intercepted PAR:     {avg_par:.3f} MJ/m^2/day")
print(f"  Final phenol. stage:      {stage_sample}")
print(f"")
print(f"  Disease outcome (Day 90):")
print(f"    Total infected:         {infected}")
print(f"    Total resistant:        {resistant}")
print(f"    East half infected:     {east_i}")
print(f"    West half infected:     {west_i}")
ratio = east_i / (west_i + 1)
dominance = "[OK] East dominates (wind anisotropy confirmed)" if east_i > west_i else "-> Spread roughly isotropic"
print(f"    E/W ratio:              {ratio:.2f}x  {dominance}")
print(f"")
print(f"  Parquet log:  {farm._last_log_path or 'logging disabled'}")
print(f"{'=' * 60}\n")

farm.visualize(quality="enhanced")
