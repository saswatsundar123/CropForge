"""
examples/terraced_slope_trial.py
==================================
PRD v0.6.0 — Terraced Slope Trial

Scientific Scenario
-------------------
A 30×30 grid field with a procedural sloped terrain.
A ContourBund (or Terrace) land preparation modifier is applied to
interrupt the downslope water flow. 
A heavy rain event on day 10 drives lateral flow via the D8 engine.
The simulation is run for 30 days to test D8 terrain coupling.

Usage
-----
  python examples/terraced_slope_trial.py

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropforge import Farm, Field, Terrain, ContourBund
from cropforge.plugins import StandardWheat
from cropforge.state import EnvironmentState


class HeavyRainWeather:
    """Dry weather with one heavy rainfall event on day 10."""
    def get_day(self, day: int) -> EnvironmentState:
        return EnvironmentState(
            day=day, doy=day,
            temp_max_c=25.0, temp_min_c=10.0, temp_mean_c=17.5,
            radiation_mj_m2=15.0,
            rainfall_mm=100.0 if day == 10 else 0.0,  # heavy rain on day 10
            et0_mm=4.0,
            wind_speed_ms=1.5,
            humidity_pct=50.0,
        )


def main():
    print("Setting up Terraced Slope Trial...")
    farm = Farm(name="Terraced Slope Trial")
    
    # 1. Create a field
    field = Field(name="TerracedField", rows=30, cols=30, area_ha=1.0)
    
    # 2. Generate procedural slope topography
    terrain = Terrain.procedural(rows=30, cols=30, generator="slope", drop_m=5.0)
    field.set_terrain(terrain)
    
    # 3. Apply ContourBund land prep modifier
    #    bund every 1m drop
    field.set_land_prep(ContourBund(bund_height_m=0.3, interval_m=1.0))
    
    # 4. Use StandardWheat plugin
    field.use_plugin(StandardWheat)

    # 5. Set Weather
    field.set_weather(HeavyRainWeather())
    
    farm.add_field(field)
    
    # 6. Enable Physics (hydrology and lateral flow)
    farm.use_physics(
        et0=True,
        water_balance=True,
        nutrients=False,
        lateral_flow=True,  # Test D8 coupling
    )
    
    # 7. Run for 30 days
    print("Running simulation for 30 days...")
    farm.run(days=30)
    
    # 8. Save state (for researchers to inspect the modified elevation grid and hydrology)
    farm.save_state()
    print("Simulation complete. State saved.")


if __name__ == "__main__":
    main()
