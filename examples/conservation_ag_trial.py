"""
conservation_ag_trial.py
========================
CropForge v0.7.0 Capstone Showcase Script

This script demonstrates the v0.7.0 Topographical Physics engine,
specifically comparing Conventional Tillage with Conservation Agriculture
(Contour Bunding) on a sloped terrain.

It generates an undulating terrain and simulates 60 days of maize growth
under heavy rainfall. By comparing the cumulative erosion between the
two fields, researchers can quantitatively observe the soil conservation
benefits of contour bunds.

Run this script to generate the logs:
    python examples/conservation_ag_trial.py

Then start the visualiser to explore the results in 3D:
    cropforge-dash
"""
import sys
from pathlib import Path

# Ensure we can import cropforge if running directly from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from cropforge.farm import Farm, Field, Crop
from cropforge.loaders import Weather
from cropforge.terrain import Terrain
from cropforge.land_prep import ConventionalTill, ContourBund


def main():
    print("Initialize CropForge v0.7.0 Conservation Ag Trial...")
    
    # Generate procedural undulating terrain
    # The terrain has a built-in slope/undulation to drive runoff and erosion
    terrain = Terrain.procedural(generator="undulating", rows=30, cols=30, resolution_m=1.0)
    
    # Create the farm
    farm = Farm(name="Conservation Agriculture Trial", location=(28.6, 77.2))
    
    # We will simulate 60 days of heavy rain.
    print("Generating weather with frequent heavy rainfall...")
    weather_df = pd.DataFrame({
        "day": range(1, 61),
        "doy": range(150, 210),
        "temp_max_c": 32.0,
        "temp_min_c": 22.0,
        "temp_mean_c": 27.0,
        "radiation_mj_m2": 20.0,
        # Heavy rain every 5 days
        "rainfall_mm": [30.0 if i % 5 == 0 else 2.0 for i in range(1, 61)],
        "et0_mm": 5.0,
        "wind_speed_ms": 3.5, # moderate wind
        "humidity_pct": 75.0,
        "co2_ppm": 415.0
    })
    weather = Weather(weather_df.set_index("day"))
    
    # Standard maize crop for both fields
    maize = Crop(species="Zea mays", variety="StandardHybrid")
    
    # -----------------------------------------------------------------------
    # Field A: Conventional Tillage (High roughness initially, no bunds)
    # -----------------------------------------------------------------------
    field_a = Field(name="Conventional Till", rows=30, cols=30)
    field_a.set_terrain(terrain)
    field_a.set_weather(weather)
    field_a.set_crop(maize)
    field_a.set_land_prep(ConventionalTill())
    farm.add_field(field_a)
    
    # -----------------------------------------------------------------------
    # Field B: Conservation Agriculture (Contour Bunds)
    # -----------------------------------------------------------------------
    field_b = Field(name="Contour Bunds", rows=30, cols=30)
    field_b.set_terrain(terrain)
    field_b.set_weather(weather)
    field_b.set_crop(maize)
    # Bunds trap water, reducing runoff velocity and erosion
    field_b.set_land_prep(ContourBund(bund_height_m=0.3, interval_m=1.0))
    farm.add_field(field_b)
    
    # -----------------------------------------------------------------------
    # Enable Physics Engines (v0.7.0 Features)
    # -----------------------------------------------------------------------
    print("Enabling topographical physics engines...")
    farm.use_physics(
        et0=True,
        water_balance=True,              # Required for runoff
        lateral_flow=True,               # D8 runoff routing
        clod_dynamics=True,              # Rain melts the clods over time
        erosion=True,                    # RUSLE-based erosion index
        slope_radiation_correction=True, # Solar incidence adjusted by slope/aspect
        terrain_wind=True,               # Topographical wind field
        root_clamping=True,              # Topographical root constraints
        wind_direction_deg=270.0         # Wind coming from the West
    )
    
    # Run simulation
    print("\nRunning simulation for 60 days...")
    farm.run(days=60)
    
    print("\nSimulation Complete!")
    print(f"Results saved to: {farm._last_log_path}")
    print("\nTo view the results:")
    print("1. Run: cropforge-dash")
    print("2. Open http://localhost:8050")
    print("3. Click 'Open 3D Terrain View' on day 60")
    print("4. Compare 'Cumulative Erosion Index' across both fields.")

if __name__ == "__main__":
    main()
