# Tutorial: Terrain and Land Preparation

In this tutorial, we will learn how to apply physical terrain transformations to a field and visualize their impact using CropForge's 3D Plotly UI.

## Why Land Preparation?

Real agricultural fields are rarely flat. Topography drives water movement, causing runoff, leaching, and spatial variability in crop stress. Farmers counteract this through **Land Preparation**—physically reshaping the soil to retain water or prevent erosion.

CropForge v0.6.0 introduces the `LandPrep` modifier class to dynamically adjust the field's `elevation_grid` and soil physics properties prior to planting.

## Step 1: Create a Field with Topography

First, we need a field that isn't perfectly flat. We'll generate a procedural slope.

```python
from cropforge import Farm, Field, Terrain

farm = Farm(name="Hillside Farm")
field = Field(name="Terraced Plot", rows=50, cols=50)

# Create a slope that drops 5 metres from North to South
terrain = Terrain.procedural(rows=50, cols=50, generator="slope", drop_m=5.0)
field.set_terrain(terrain)
```

## Step 2: Apply a Land Prep Modifier

If we rain on this slope, water will rapidly run off the bottom edge. Let's add contour bunds to intercept the flow.

```python
from cropforge import ContourBund

# Place a 30cm high earthen bund every 1 metre of elevation drop
bunds = ContourBund(bund_height_m=0.3, interval_m=1.0)
field.set_land_prep(bunds)

farm.add_field(field)
```

Other available modifiers include:
- `RidgeFurrow`: For row crops, creates alternating peaks and troughs.
- `Terrace`: Cuts step-like flat terraces into steep hillsides.
- `DeepTillage`: Alters soil bulk density and porosity (no elevation change).

## Step 3: Enable D8 Physics

To ensure water actually obeys the new terrain geometry, we must enable lateral flow in the physics engine.

```python
farm.use_physics(
    water_balance=True,
    lateral_flow=True  # Enables D8 terrain routing!
)
```

## Step 4: Visualise in 3D

After running your simulation, launch the dashboard:

```python
farm.run(days=30)
farm.visualize()
```

### Using the 3D Plotly Modal

1. Open the CropForge dashboard in your browser.
2. In the top-left UI panel, toggle **Terrain View**.
3. The 2D spatial heatmap will seamlessly transition into a 3D procedural WebGL viewport.
4. **Surface Overlay Variable**: Use this dropdown to colorise the 3D plants based on physical variables (e.g., `nitrogen_kg_ha`, `moisture_mm`, `stress_index`). 
5. Zoom and pan to observe how water pooled behind your contour bunds has resulted in taller, lower-stress plants compared to the dry slopes between the bunds!
