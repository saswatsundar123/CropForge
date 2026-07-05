# Modeling Conservation Agriculture

CropForge v0.7.0 introduces Topographical Physics, enabling the simulation of land preparation effects on water dynamics, soil erosion, and plant growth in a true 3D spatial context. 

This tutorial walks through the `conservation_ag_trial.py` example script, demonstrating how to set up a comparison between Conventional Tillage and Conservation Agriculture (Contour Bunding) on a sloped terrain.

## The Goal

We want to quantitatively demonstrate how contour bunds (a conservation agriculture technique) reduce soil erosion compared to conventional tillage on an undulating, sloped field under heavy rainfall.

## Step 1: Terrain Generation

First, we generate a procedural undulating terrain. This provides the slope necessary to drive runoff and erosion.

```python
from cropforge.terrain import Terrain

# The terrain has a built-in slope/undulation to drive runoff and erosion
terrain = Terrain.procedural(generator="undulating", rows=30, cols=30, resolution_m=1.0)
```

## Step 2: Weather Setup

We simulate 60 days with frequent heavy rainfall events to stress test the erosion mechanics.

```python
import pandas as pd
from cropforge.loaders import Weather

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
    "wind_speed_ms": 3.5,
    "humidity_pct": 75.0,
    "co2_ppm": 415.0
})
weather = Weather(weather_df.set_index("day"))
```

## Step 3: Setting Up the Fields

We create two fields side-by-side using the identical terrain and weather, but differing `LandPrep` configurations.

### Field A: Conventional Tillage
Conventional tillage starts with a high surface roughness, but it decays quickly under heavy rain. It offers no physical barriers to lateral flow.

```python
from cropforge.farm import Field
from cropforge.land_prep import ConventionalTill

field_a = Field(name="Conventional Till", rows=30, cols=30)
field_a.set_terrain(terrain)
field_a.set_weather(weather)
field_a.set_crop(maize)
field_a.set_land_prep(ConventionalTill())
farm.add_field(field_a)
```

### Field B: Contour Bunds
Contour bunds are physical barriers placed along the contour lines of the slope. They trap water, reducing runoff velocity and minimizing erosion.

```python
from cropforge.land_prep import ContourBund

field_b = Field(name="Contour Bunds", rows=30, cols=30)
field_b.set_terrain(terrain)
field_b.set_weather(weather)
field_b.set_crop(maize)
# Bunds trap water, reducing runoff velocity and erosion
field_b.set_land_prep(ContourBund(bund_height_m=0.3, interval_m=1.0))
farm.add_field(field_b)
```

## Step 4: Enabling Topographical Physics

To activate the 3D physics engines, we must explicitly enable them before running the simulation.

```python
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
```

## Step 5: Run and Visualize

Run the simulation:
```bash
python examples/conservation_ag_trial.py
```

Then, launch the CropForge Dashboard:
```bash
cropforge-dash
```

Open your browser to `http://localhost:8050`. 
1. Select "Field: Conventional Till".
2. Scrub to Day 60.
3. Click "Open 3D Terrain View".
4. In the Surface Overlay Variable dropdown, select **Cumulative Erosion Index**.
5. Observe the high erosion channels forming on the steep sections of the terrain.
6. Switch to "Field: Contour Bunds" and observe how the bunds physically block the flow paths, significantly lowering the overall erosion index.
