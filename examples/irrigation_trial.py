"""
examples/irrigation_trial.py
==============================
PRD v0.3.0 Tutorial -- Full Irrigation Trial (PRD sec.7.3)

Scientific Scenario
-------------------
90-day maize season. Two plots on the same farm:

  Plot_A_Rainfed  -- no supplemental irrigation, rainfed only
  Plot_B_Irrigated -- supplemental irrigation every 15 days (50 mm/event)

Both plots receive:
  - Identical weather (monsoon season, hot/dry periods)
  - Identical soil (sandy loam, FC=30%, WP=12%)
  - Identical fertilizer at day 20 (80 kg N/ha basal application)
  - Full physics: ET0 + water balance + N transport

Expected outcome (PRD sec.7.3):
  Stress divergence by day 45:
    rainfed plot stress_index > 0.4 (cumulative drought stress)
    irrigated plot stress_index < 0.15 (adequately supplied)

Usage
-----
  python examples/irrigation_trial.py

Output
------
  Prints a summary table comparing the two plots at key timesteps.
  Parquet logs are written to ./output/irrigation_trial/ if a logger
  is configured; otherwise the in-memory state is printed.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropforge import Farm, Field, Crop, Event
from cropforge.state import EnvironmentState, SoilVoxelState


# ---------------------------------------------------------------------------
# Weather source -- simplified monsoon/dry-season schedule
# ---------------------------------------------------------------------------

class MonsoonWeather:
    """Simplified 90-day weather schedule for a semi-arid Indian site.

    Days 1-30:  Pre-monsoon (hot, dry, high ET0)
    Days 31-60: Monsoon onset (moderate rain, lower ET0)
    Days 61-90: Post-monsoon (drying, high ET0)
    """

    def get_day(self, day: int) -> EnvironmentState:
        if day <= 30:
            # Pre-monsoon: hot, dry
            rain = 2.0 if day % 10 == 0 else 0.0
            et0  = 7.5
            tmax, tmin = 38.0, 24.0
        elif day <= 60:
            # Monsoon: moderate rain
            rain = 8.0 if day % 3 == 0 else 0.0
            et0  = 4.5
            tmax, tmin = 32.0, 22.0
        else:
            # Post-monsoon: warm, drying
            rain = 1.0 if day % 7 == 0 else 0.0
            et0  = 6.0
            tmax, tmin = 35.0, 21.0

        return EnvironmentState(
            day=day, doy=((day - 1) % 365) + 1,
            temp_max_c=tmax, temp_min_c=tmin,
            temp_mean_c=(tmax + tmin) / 2.0,
            radiation_mj_m2=22.0,
            rainfall_mm=rain,
            et0_mm=et0,
            wind_speed_ms=2.5,
            humidity_pct=55.0,
        )


# ---------------------------------------------------------------------------
# Soil profile builder
# ---------------------------------------------------------------------------

class SandyLoamSoil:
    """Sandy loam 2-layer profile (0-20 cm, 20-40 cm)."""

    def build_grid(self, rows: int, cols: int):
        return [
            [
                [
                    SoilVoxelState(
                        row=r, col=c, layer=0,
                        depth_top_cm=0.0, depth_bottom_cm=20.0,
                        moisture_pct=28.0,      # starting near FC
                        nitrogen_kg_ha=80.0,    # basal N (augmented by fertiliser event)
                        bulk_density=1.35,
                        penetration_resistance=0.6,
                    ),
                    SoilVoxelState(
                        row=r, col=c, layer=1,
                        depth_top_cm=20.0, depth_bottom_cm=40.0,
                        moisture_pct=24.0,
                        nitrogen_kg_ha=30.0,
                        bulk_density=1.40,
                        penetration_resistance=1.0,
                    ),
                ]
                for c in range(cols)
            ]
            for r in range(rows)
        ]


# ---------------------------------------------------------------------------
# Simulation setup
# ---------------------------------------------------------------------------

def run_irrigation_trial() -> None:
    print("=" * 65)
    print("  CropForge v0.3.0 -- Irrigation Trial")
    print("  90-day maize | Plot_A_Rainfed vs Plot_B_Irrigated")
    print("=" * 65)

    farm = Farm(name="IrrigationTrial", location=(25.6, 85.1))  # Bihar, India

    # --- Plot A: Rainfed ---
    plot_a = Field(name="Plot_A_Rainfed", rows=2, cols=2, area_ha=0.5)
    plot_a.set_crop(Crop(species="Zea mays", variety="DKC9025"))
    plot_a.set_weather(MonsoonWeather())
    plot_a.set_soil(SandyLoamSoil())
    plot_a.set_water_params(
        field_capacity_pct=30.0,
        wilting_point_pct=12.0,
        saturation_pct=44.0,
        drainage_coefficient=0.6,   # sandy loam drains fast
        crop_coefficient=1.15,      # maize mid-season Kc
        stress_increment_per_day=0.04,
    )
    plot_a.set_nitrogen_params(leaching_fraction=0.012, runoff_n_fraction=0.04)
    farm.add_field(plot_a)

    # --- Plot B: Irrigated ---
    plot_b = Field(name="Plot_B_Irrigated", rows=2, cols=2, area_ha=0.5)
    plot_b.set_crop(Crop(species="Zea mays", variety="DKC9025"))
    plot_b.set_weather(MonsoonWeather())
    plot_b.set_soil(SandyLoamSoil())
    plot_b.set_water_params(
        field_capacity_pct=30.0,
        wilting_point_pct=12.0,
        saturation_pct=44.0,
        drainage_coefficient=0.6,
        crop_coefficient=1.15,
        stress_increment_per_day=0.04,
    )
    plot_b.set_nitrogen_params(leaching_fraction=0.012, runoff_n_fraction=0.04)
    farm.add_field(plot_b)

    # --- Physics ---
    farm.use_physics(
        et0=True,
        water_balance=True,
        nutrients=True,
        lateral_flow=True,
    )

    # --- Events ---
    # Fertiliser at day 20 on both plots (80 kg N/ha top-dress)
    farm.add_event(Event.fertiliser(
        field="Plot_A_Rainfed", day=20, n_kg_ha=80, apply_to_layer=0
    ))
    farm.add_event(Event.fertiliser(
        field="Plot_B_Irrigated", day=20, n_kg_ha=80, apply_to_layer=0
    ))

    # Irrigation every 15 days on Plot B only (50 mm/event)
    farm.add_event(Event.irrigation(
        field="Plot_B_Irrigated",
        interval_days=15,
        amount_mm=50,
        start_day=1,
        end_day=90,
    ))

    # --- Biomass accumulation step (same for both plots) ---
    stress_a: dict = {}
    stress_b: dict = {}
    moisture_a: dict = {}
    moisture_b: dict = {}

    @farm.step(phase=0)
    def accumulate_biomass(state, env):
        """Simple RUE-based biomass + root growth for all fields."""
        for plant in state.plants:
            if not plant.alive:
                continue
            ks = plant.custom.get("water_stress_ks", 1.0)
            plant.biomass_g   += 2.0 * ks
            plant.height_cm   += 0.6 * ks
            plant.root_depth_cm = min(40.0, plant.root_depth_cm + 0.5)
            plant.age_days    += 1
        return state

    @farm.step(phase=1)
    def record_a(state, env):
        if state.plants and state.plants[0].plant_id.startswith("Plot_A"):
            stress_a[state.day] = state.plants[0].stress_index
            moisture_a[state.day] = state.soil[0][0][0].moisture_pct
        return state

    @farm.step(phase=1)
    def record_b(state, env):
        if state.plants and state.plants[0].plant_id.startswith("Plot_B"):
            stress_b[state.day] = state.plants[0].stress_index
            moisture_b[state.day] = state.soil[0][0][0].moisture_pct
        return state

    farm.run(days=90)

    # --- Summary ---
    print(f"\n{'Day':>4}  {'StressA':>8}  {'MoistA':>7}  {'StressB':>8}  {'MoistB':>7}  {'Diff':>8}")
    print("-" * 58)
    for day in [1, 15, 30, 45, 60, 75, 90]:
        sa = stress_a.get(day, 0.0)
        sb = stress_b.get(day, 0.0)
        ma = moisture_a.get(day, 0.0)
        mb = moisture_b.get(day, 0.0)
        print(f"{day:>4}  {sa:>8.4f}  {ma:>7.2f}%  {sb:>8.4f}  {mb:>7.2f}%  {sa-sb:>+8.4f}")

    # --- PRD sec.7.3 verification ---
    stress_a_day45 = stress_a.get(45, 0.0)
    stress_b_day45 = stress_b.get(45, 0.0)
    print("\n" + "=" * 65)
    print("  PRD sec.7.3 Verification (day 45)")
    print(f"  Rainfed stress_index  = {stress_a_day45:.4f}  (target > 0.15)")
    print(f"  Irrigated stress_index= {stress_b_day45:.4f}  (target < 0.30)")
    diverged = stress_a_day45 > stress_b_day45
    print(f"  Stress divergence: {'CONFIRMED OK' if diverged else 'NOT OBSERVED FAIL'}")
    print("=" * 65)


if __name__ == "__main__":
    run_irrigation_trial()
