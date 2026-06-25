# Quickstart

This guide reproduces the `examples/wheat_basic.py` example step by step.

## 1. Prepare Input Data

CropForge expects two CSV files: a weather file and a soil profile.
See [Section 8 of the PRD](../prd.md) for the full column specification.

## 2. Define the Simulation

```python
from cropforge import Farm, Field, Crop
from cropforge.loaders import Weather, Soil
from cropforge.runtime import step

# ---- Crop ----------------------------------------------------------------
wheat = Crop(species="wheat", variety="HD-2967", sowing_doy=300)

# ---- Field ---------------------------------------------------------------
weather = Weather.from_csv(
    "examples/data/weather_sample.csv",
    date_col="date", tmax_col="tmax_c", tmin_col="tmin_c",
    radiation_col="radiation_mj", rainfall_col="rainfall_mm",
)
soil = Soil.from_csv("examples/data/soil_sample.csv", apply="uniform")

field = Field(
    "Plot A",
    rows=20, cols=30,
    crop=wheat,
    spacing_m=0.2,
    weather=weather,
    soil=soil,
)

# ---- Custom model step ---------------------------------------------------
@step(phase="plant", order=10)
def grow(plant, env, soil_voxel, dt):
    """Simple biomass accumulation driven by radiation."""
    rad = env.get("radiation_mj", 10.0)
    plant.biomass_g  += rad * 2.5 * dt
    plant.height_cm  += rad * 0.08 * dt
    plant.lai        += rad * 0.003 * dt
    plant.lai         = min(plant.lai, 5.0)

# ---- Farm ----------------------------------------------------------------
farm = Farm("WheatBasic")
farm.add_field(field)
farm.register_step(grow)

# ---- Run -----------------------------------------------------------------
farm.run(days=90)
```

## 3. Launch the Dashboard

```python
farm.visualize()
# Opens http://localhost:7860 in your default browser.
# Press Ctrl-C to stop.
```

## 4. Interact

- **Play/Pause** the timeline at 1×, 2×, or 5× speed.
- **Drag** the scrubber to any day.
- **Click** any plant cylinder to open the Farm Inspector (Panel 4).
- **Change** the colour variable (Biomass / LAI / Height / Stress) from the dropdown.
