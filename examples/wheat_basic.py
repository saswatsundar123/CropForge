"""
examples/wheat_basic.py
=======================
CropForge Phase 1 Deliverable — Reference Wheat Simulation

This script is the primary onboarding artifact for CropForge (PRD Section 10).
It is a complete, self-contained simulation that demonstrates the full Phase 1
API:

  - Farm / Field / Crop construction
  - Weather.from_csv and Soil.from_csv loaders
  - The @farm.step decorator with phase ordering
  - Parquet state logging

Scenario (PRD Section 10.1):
  - Crop       : Wheat (generic variety)
  - Field size : 20 rows × 30 cols (600 plant positions)
  - Duration   : 90 days (tillering → grain fill)
  - Soil       : 3 layers (0–20 cm, 20–40 cm, 40–60 cm), uniform profile
  - Weather    : 90-day synthetic CSV (SYNTHETIC DATA — NOT FROM A WEATHER STATION)
  - Management : Irrigation at day 30 (30 mm), Fertiliser at day 15 (40 kg N/ha)

Model equations (PRD Section 10.2):
  Biomass:    delta_biomass = RUE × APAR × (1 − water_stress_factor)
              APAR = radiation × (1 − exp(−k × lai))
              RUE = 1.2 g/MJ,  k = 0.5
  LAI:        delta_lai = max(0, temp_mean_c − T_BASE) × LAI_SLOPE
              T_BASE = 0 °C, LAI_SLOPE = 0.001 m²/m²/°C·day
  Water stress: stress = max(0, min(1, (moisture − PWP) / (FC − PWP)))
  Height:     height_cm = K_H × sqrt(biomass_g)   (K_H = 0.8)
  Death:      if stress_index >= 1.0 → plant.alive = False

Usage::
    python examples/wheat_basic.py

Expected output (PRD Section 10.4):
    cropforge_output/WheatBasic_<timestamp>/
        plants/    — partitioned Parquet files
        soil/      — partitioned Parquet files
        environment/ — partitioned Parquet files

Success criterion: completes in < 10 seconds on a 4-core / 8 GB laptop.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

import logging
import math
import os
import sys
from pathlib import Path

# ---- Ensure the package is importable when run directly from examples/ ----
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from cropforge import Crop, Farm, Field, Soil, Weather

# Configure logging so the researcher sees engine messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Soil / weather constants (embedded in this script for reproducibility)
# ---------------------------------------------------------------------------

# Soil thresholds (match the header comments in wheat_uniform_soil_3layer.csv)
FC  = 35.0   # Field capacity (% volumetric water content)
PWP = 12.0   # Permanent wilting point (% VWC)

# RUE model constants (PRD Section 10.2)
RUE = 1.2    # Radiation Use Efficiency (g/MJ)
K   = 0.5    # Light extinction coefficient
K_H = 0.8    # Height allometry coefficient

# LAI thermal accumulation
T_BASE    = 0.0     # Base temperature (°C)
LAI_SLOPE = 0.001   # LAI increment per °C·day above T_BASE

# Stress accumulation per timestep under wilting conditions
STRESS_INCREMENT = 0.04   # Chosen so ~25 consecutive wilting days kill a plant

# ---------------------------------------------------------------------------
# 1. Define the farm
# ---------------------------------------------------------------------------

print("\n" + "="*56)
print("  CropForge -- wheat_basic.py")
print("  Field: 20x30 (600 plants)  |  Duration: 90 days")
print("="*56 + "\n")

farm = Farm(name="WheatBasic", location=(28.6, 77.2))

field_a = Field(
    name="Plot A",
    rows=20,
    cols=30,
    area_ha=2.4,
    elevation_profile="flat",
)

# ---- Locate data files relative to this script -------------------------
_data_dir = Path(__file__).parent / "data"
_weather_csv = _data_dir / "wheat_synthetic_weather_90d.csv"
_soil_csv    = _data_dir / "wheat_uniform_soil_3layer.csv"

# ---- Attach inputs (PRD Section 6.1) -----------------------------------
field_a.set_crop(Crop(species="wheat", variety="generic", sowing_doy=60))
field_a.set_weather(
    Weather.from_csv(
        str(_weather_csv),
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
field_a.set_soil(Soil.from_csv(str(_soil_csv), apply="uniform"))

farm.add_field(field_a)

# ---------------------------------------------------------------------------
# 2. Register management events (PRD Section 6.3)
# ---------------------------------------------------------------------------
# Events are fired inside the step functions below for v0.1 (the full Event
# class is implemented in Phase 2). We handle them via explicit day checks
# so this script is fully self-contained.

IRRIGATION_DAY    = 30    # Day to add 30 mm to topsoil moisture
IRRIGATION_MM     = 30.0
FERTILISER_DAY    = 15    # Day to add 40 kg N/ha to topsoil
FERTILISER_KG_HA  = 40.0

# ---------------------------------------------------------------------------
# 3. Step functions (PRD Section 6.2)
# ---------------------------------------------------------------------------
# Phase 1 — Soil processes (evaporation + irrigation + fertilisation)
# Phase 2 — Crop growth (LAI, biomass, height, stress, death)

@farm.step(interval="daily", phase=1)
def soil_and_management(state, env):
    """Soil water balance and management event application.

    Runs BEFORE the plant growth model (phase=1 < phase=2) so that updated
    soil moisture is available to the transpiration / stress calculation.
    """
    for row_cells in state.soil:
        for col_voxels in row_cells:
            topsoil = col_voxels[0]

            # ---- Simple drainage: rainfall refills topsoil (capped at FC) ----
            topsoil.moisture_pct = min(FC, topsoil.moisture_pct + env.rainfall_mm * 0.1)

            # ---- Daily bare-soil evaporation (simplified Penman proxy) -------
            evap_loss = env.et0_mm * 0.4   # 40 % of ET₀ from bare soil
            topsoil.moisture_pct = max(0.0, topsoil.moisture_pct - evap_loss)

    # ---- Irrigation event (day 30) ----------------------------------------
    if state.day == IRRIGATION_DAY:
        for row_cells in state.soil:
            for col_voxels in row_cells:
                topsoil = col_voxels[0]
                topsoil.moisture_pct = min(FC, topsoil.moisture_pct + IRRIGATION_MM * 0.1)
        state.events_fired.append(f"Irrigation +{IRRIGATION_MM}mm on day {state.day}")

    # ---- Fertiliser event (day 15) -----------------------------------------
    if state.day == FERTILISER_DAY:
        for row_cells in state.soil:
            for col_voxels in row_cells:
                topsoil = col_voxels[0]
                topsoil.nitrogen_kg_ha += FERTILISER_KG_HA
        state.events_fired.append(f"Fertiliser +{FERTILISER_KG_HA}kg N/ha on day {state.day}")

    return state


@farm.step(interval="daily", phase=2)
def wheat_growth_model(state, env):
    """Wheat growth model — RUE, LAI, water stress, height, plant death.

    Model equations (PRD Section 10.2):
        delta_biomass = RUE × APAR × (1 − water_stress_factor)
        APAR          = radiation × (1 − exp(−k × lai))
        delta_lai     = max(0, temp_mean − T_BASE) × LAI_SLOPE
        height_cm     = K_H × sqrt(biomass_g)
        stress_factor = (moisture − PWP) / (FC − PWP)    clamped [0, 1]
    """
    for plant in state.plants:
        if not plant.alive:
            continue

        # ---- Topsoil moisture for this plant's position -------------------
        topsoil = state.soil[plant.row][plant.col][0]
        moisture = topsoil.moisture_pct

        # ---- Water stress factor (PRD Section 10.2) ----------------------
        if FC > PWP:
            stress_factor = max(0.0, min(1.0, (moisture - PWP) / (FC - PWP)))
        else:
            stress_factor = 0.0

        # ---- LAI increment: thermal time above base temperature ----------
        delta_lai = max(0.0, env.temp_mean_c - T_BASE) * LAI_SLOPE
        plant.lai = max(0.0, plant.lai + delta_lai)

        # ---- Absorbed PAR (Beer–Lambert law) ------------------------------
        apar = env.radiation_mj_m2 * (1.0 - math.exp(-K * plant.lai))

        # ---- Biomass accumulation (PRD Section 10.2) ----------------------
        delta_biomass = RUE * apar * stress_factor
        plant.biomass_g += delta_biomass

        # ---- Height (allometric proxy) ------------------------------------
        plant.height_cm = K_H * math.sqrt(max(0.0, plant.biomass_g))

        # ---- Phenological stage (simplified thermal-time milestones) ------
        if plant.age_days < 20:
            plant.phenological_stage = "germination"
        elif plant.age_days < 40:
            plant.phenological_stage = "tillering"
        elif plant.age_days < 65:
            plant.phenological_stage = "stem_elongation"
        elif plant.age_days < 80:
            plant.phenological_stage = "heading"
        else:
            plant.phenological_stage = "grain_fill"

        # ---- Water stress accumulation ------------------------------------
        if moisture < PWP:
            plant.stress_index = min(1.0, plant.stress_index + STRESS_INCREMENT)
        else:
            # Gradual recovery from mild stress
            plant.stress_index = max(0.0, plant.stress_index - 0.01)

        # ---- Plant death (PRD Section 10.2) --------------------------------
        if plant.stress_index >= 1.0:
            plant.alive = False

    return state


# ---------------------------------------------------------------------------
# 4. Run the simulation
# ---------------------------------------------------------------------------

print("Starting 90-day simulation ...\n")
farm.run(days=90)

# ---------------------------------------------------------------------------
# 5. Summary report
# ---------------------------------------------------------------------------

print(f"\n" + "-"*56)
print("  Simulation complete.")
print(f"  Parquet log -> {farm._last_log_path}")

final_state = farm._fields[0]._field_state
alive  = sum(1 for p in final_state.plants if p.alive)
dead   = sum(1 for p in final_state.plants if not p.alive)
plants = final_state.plants

if plants:
    avg_biomass = sum(p.biomass_g for p in plants) / len(plants)
    avg_height  = sum(p.height_cm for p in plants) / len(plants)
    avg_lai     = sum(p.lai        for p in plants) / len(plants)
    print(f"\n  Field: Plot A  ({len(plants)} plants)")
    print(f"    Alive:          {alive}")
    print(f"    Dead:           {dead}")
    print(f"    Mean biomass:   {avg_biomass:.2f} g/plant")
    print(f"    Mean height:    {avg_height:.2f} cm")
    print(f"    Mean LAI:       {avg_lai:.4f} m2/m2")
print("-"*56 + "\n")

# ---------------------------------------------------------------------------
# 6. Launch the dashboard
# ---------------------------------------------------------------------------

print("Launching CropForge dashboard ...")
print("  Press Ctrl-C in this terminal to stop the server.\n")
farm.visualize()
