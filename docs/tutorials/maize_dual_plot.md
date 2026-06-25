# Maize Dual-Plot Tutorial

This tutorial demonstrates the power of CropForge's **Multi-Field** capabilities and **Opt-In Physics** to simulate a Genotype x Environment (GxE) scenario. We will simulate two adjacent maize fields with identical weather and genetics, but differing soil profiles.

You can find the full runnable script in `examples/maize_dual_plot.py`.

## The GxE Scenario

We have two fields:
1. **Plot A (Slope)**: Well-drained soil on a slope. It has no physical restrictions, allowing roots to grow freely.
2. **Plot B (Hardpan)**: A flat plot with a dense, compacted layer of soil (a "hard pan") starting at 19cm depth.

We want to observe how this physical difference in the soil affects the root architecture of the maize plants over a 90-day season.

## Setting up the Fields

First, we define the crop and the two fields. Notice how we load different soil CSV files for each field.

```python
from cropforge import Farm, Field, Crop
from cropforge.loaders import Weather, Soil

# 1. Define the Crop (Identical genetics)
maize = Crop(species="maize", variety="Kharif-1", sowing_doy=150)

# 2. Load identical weather
weather = Weather.from_csv("data/maize_weather_90d.csv")

# 3. Create Plot A (Slope - No constraints)
field_a = Field(
    name="Plot A (Slope)",
    rows=10, cols=10, crop=maize, spacing_m=0.3,
    weather=weather,
    soil=Soil.from_csv("data/maize_soil_plotA_slope.csv", apply="uniform")
)

# 4. Create Plot B (Hardpan - 19cm restriction)
field_b = Field(
    name="Plot B (Hardpan)",
    rows=10, cols=10, crop=maize, spacing_m=0.3,
    weather=weather,
    soil=Soil.from_csv("data/maize_soil_plotB_hardpan.csv", apply="uniform")
)
```

In `maize_soil_plotB_hardpan.csv`, the penetration resistance spikes to $3.0$ MPa at the 19cm layer.

## Enabling Opt-In Physics

We create the farm and add both fields. To ensure the soil's penetration resistance actually affects the plants, we enable the built-in Root Impedance physics solver.

```python
farm = Farm("Maize_DualPlot")
farm.add_field(field_a)
farm.add_field(field_b)

@farm.use_physics(et0=True, root_impedance=True)
def apply_physics():
    pass
```

Because `root_impedance=True`, the engine will automatically calculate a `root_growth_multiplier` for every plant, every day, based on the soil resistance at the plant's current root depth.

## Defining the Plant Logic

Now we write our custom plant step. We use the `plant.root_growth_multiplier` calculated by the physics engine (which ran at `phase=-1`) to throttle the plant's potential root growth.

```python
@farm.step(phase=0)
def grow_maize(state):
    env = state.env
    
    # Simple temperature-driven growth
    gdd = max(0, env.tmean_c - 10.0)
    
    for plant in state.plants:
        if not plant.alive:
            continue
            
        # Biomass driven by GDD
        plant.biomass_g += gdd * 0.5
        
        # Root growth driven by GDD, restricted by physics multiplier
        potential_root_growth = gdd * 0.2
        actual_root_growth = potential_root_growth * plant.root_growth_multiplier
        plant.root_depth_cm += actual_root_growth
```

## Execution and Analysis

We run the simulation and launch the visualization dashboard.

```python
farm.run(days=90)
farm.visualize()
```

### Using the Multi-Field Dashboard

When the dashboard opens, you will notice the new **Field Selector Dropdown** in the upper right.

1. **Compare using the Time-Series Chart**:
   - The time-series chart automatically displays lines for *all* fields simultaneously.
   - Select `mean_root_depth_cm`. You will immediately see the lines diverge. Plot A's roots continue growing downward (reaching ~33.5cm), while Plot B's roots hit a hard ceiling exactly at 19cm.
   
2. **Visualise the difference in 3D**:
   - The 3D viewport and the spatial Heatmap display data for the *active* field.
   - Change the Field Selector from "Plot A" to "Plot B". The 3D scene will instantly tear down and rebuild, displaying the stunted crops of Plot B.
   
3. **Inspect the Soil Profile**:
   - Use the `Click to Inspect` tool in the 3D viewport on a plant in Plot B.
   - In the Farm Inspector panel, the Soil Depth Cross-Section will clearly visualize the high-resistance hard pan layer starting at 19cm.
