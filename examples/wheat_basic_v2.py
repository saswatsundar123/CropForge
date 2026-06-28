"""
examples/wheat_basic_v2.py
===========================
CropForge v0.5.0 — Wheat Basic (Plugin Edition)

Identical scenario to wheat_basic.py but model logic is now provided by
the StandardWheat plugin instead of a manual @farm.step function.

Demonstrates the single-line plugin invocation pattern:
    field.use_plugin(StandardWheat())

The soil/weather management events (irrigation, fertiliser) are kept as
explicit @farm.step so the simulation remains comparable to v1.

Usage::
    python examples/wheat_basic_v2.py

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
# Constants
# ---------------------------------------------------------------------------
FC  = 35.0
PWP = 12.0
STRESS_INCREMENT = 0.04
IRRIGATION_DAY   = 30
IRRIGATION_MM    = 30.0
FERTILISER_DAY   = 15
FERTILISER_KG_HA = 40.0

# ---------------------------------------------------------------------------
# Farm setup (identical to wheat_basic.py)
# ---------------------------------------------------------------------------
print("\n" + "=" * 56)
print("  CropForge v0.5.0 -- wheat_basic_v2.py (Plugin Edition)")
print("  Field: 20x30 (600 plants)  |  Duration: 90 days")
print("=" * 56 + "\n")

farm = Farm(name="WheatBasic_v2", location=(28.6, 77.2))

field_a = Field(name="Plot A", rows=20, cols=30, area_ha=2.4, elevation_profile="flat")

_data_dir = Path(__file__).parent / "data"
field_a.set_crop(Crop(species="wheat", variety="HD-2967", sowing_doy=60))
field_a.set_weather(
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
field_a.set_soil(Soil.from_csv(str(_data_dir / "wheat_uniform_soil_3layer.csv"), apply="uniform"))
farm.add_field(field_a)

# ---------------------------------------------------------------------------
# Attach the StandardWheat plugin — replaces the manual @farm.step growth model
# ---------------------------------------------------------------------------
field_a.use_plugin(StandardWheat)

# ---------------------------------------------------------------------------
# Soil management events (irrigation + fertiliser) remain as researcher steps
# ---------------------------------------------------------------------------

@farm.step(interval="daily", phase=1)
def soil_and_management(state, env):
    """Soil water balance and management events."""
    for row_cells in state.soil:
        for col_voxels in row_cells:
            topsoil = col_voxels[0]
            topsoil.moisture_pct = min(FC, topsoil.moisture_pct + env.rainfall_mm * 0.1)
            evap_loss = env.et0_mm * 0.4
            topsoil.moisture_pct = max(0.0, topsoil.moisture_pct - evap_loss)

    if state.day == IRRIGATION_DAY:
        for row_cells in state.soil:
            for col_voxels in row_cells:
                topsoil = col_voxels[0]
                topsoil.moisture_pct = min(FC, topsoil.moisture_pct + IRRIGATION_MM * 0.1)
        state.events_fired.append(f"Irrigation +{IRRIGATION_MM}mm on day {state.day}")

    if state.day == FERTILISER_DAY:
        for row_cells in state.soil:
            for col_voxels in row_cells:
                topsoil = col_voxels[0]
                topsoil.nitrogen_kg_ha += FERTILISER_KG_HA
        state.events_fired.append(f"Fertiliser +{FERTILISER_KG_HA}kg N/ha on day {state.day}")

    return state

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("Starting 90-day simulation ...\n")
farm.run(days=90)

final_state = farm._fields[0]._field_state
alive  = sum(1 for p in final_state.plants if p.alive)
plants = final_state.plants
avg_biomass = sum(p.biomass_g for p in plants) / len(plants) if plants else 0
avg_grain   = sum(p.extra.get("grain_biomass_g", 0.0) for p in plants) / len(plants) if plants else 0
stage_sample = plants[0].extra.get("phenological_stage", "?") if plants else "?"

print(f"\n  Simulation complete. Log -> {farm._last_log_path}")
print(f"  Field: Plot A  ({len(plants)} plants)")
print(f"    Alive:                {alive}")
print(f"    Mean biomass:         {avg_biomass:.2f} g/plant")
print(f"    Mean grain biomass:   {avg_grain:.2f} g/plant")
print(f"    Final stage (p[0]):   {stage_sample}")
print("-" * 56 + "\n")
