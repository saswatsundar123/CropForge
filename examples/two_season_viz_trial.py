"""
examples/two_season_viz_trial.py
=================================
Phase 4 verification script — PRD v0.4.0 Task 4.

Demonstrates:
  1. A 2-season simulation using farm.save_state() / load_state() / prepare_next_season()
  2. Launching farm.visualize() on the combined 2-season log
  3. The time-series chart renders a vertical season boundary line at day 11
  4. The CSV Export button is present in the header

Usage:
    python examples/two_season_viz_trial.py

The dashboard will open on http://localhost:7860/
Use Ctrl-C to stop it.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cropforge import Farm, Field

# ---------------------------------------------------------------------------
# Environment data — 10 days per season, simple rising temp
# ---------------------------------------------------------------------------
def _env(days: int):
    return [
        dict(
            temp_max_c=28.0 + d * 0.5,
            temp_min_c=15.0 + d * 0.2,
            temp_mean_c=21.5 + d * 0.3,
            radiation_mj_m2=18.0,
            rainfall_mm=2.0 if d % 3 == 0 else 0.0,
            et0_mm=4.5,
            wind_speed_ms=2.0,
            humidity_pct=65.0,
            co2_ppm=415.0,
        )
        for d in range(days)
    ]


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------
output_dir = tempfile.mkdtemp(prefix="cropforge_2season_")
print(f"\n[CropForge] Output directory: {output_dir}")

farm = Farm(name="TwoSeasonDemo")
field = Field(name="WheatPlot", rows=4, cols=4)
farm.add_field(field)

@farm.step(interval="daily")
def simple_growth(state, env):
    for plant in state.plants:
        if plant.alive:
            plant.biomass_g  += env.radiation_mj_m2 * 0.8
            plant.lai        += 0.02
            plant.height_cm  += 0.5


# ---- Season 1 (days 1–10) ------------------------------------------------
print("\n[CropForge] Running Season 1 (days 1–10) ...")
farm.run(days=10)
print(f"  Season 1 complete. Season counter: {farm._current_season}")

state_file = Path(output_dir) / "s1_end.cfstate"
farm.save_state(str(state_file))
print(f"  Saved state -> {state_file}")

# ---- Season 2 (days 11–20) -----------------------------------------------
print("\n[CropForge] Preparing Season 2 ...")
farm.load_state(str(state_file))
farm.prepare_next_season()
print(f"  Season 2 ready. Season counter: {farm._current_season}")

print("[CropForge] Running Season 2 (days 11–20) ...")
farm.run(days=10)
print(f"  Season 2 complete. Day offset: {farm._day_offset}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  TWO-SEASON SIMULATION COMPLETE")
print("="*60)
print(f"  Total days logged : {farm._day_offset}")
print(f"  Season counter    : {farm._current_season}")
print(f"  Log path          : {farm._last_log_path}")
print()
print("  Expected dashboard behaviour:")
print("  - Time-series spans days 1–20 continuously")
print("  - Vertical dashed amber line at day 11 labelled 'Season 2 Starts'")
print("  - 'Export CSV' button in header downloads all 20 rows of daily data")
print("="*60)

# ---------------------------------------------------------------------------
# Launch visualizer
# ---------------------------------------------------------------------------
print("\n[CropForge] Launching dashboard at http://localhost:7860/")
print("            Press Ctrl-C to stop.\n")
farm.visualize()
