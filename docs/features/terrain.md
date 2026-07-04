# Terrain Engine

CropForge v0.6.0 introduces full 3D spatial topography mapping and D8-coupled lateral water flow. The physical elevation of your field is no longer assumed to be perfectly flat. This fundamentally alters how hydrology behaves, moving water across the surface grid.

## Generating Terrain

The `Terrain` class allows you to ingest real-world elevation models or generate synthetic topographies.

### 1. Procedural Generation
Useful for synthetic trials or exploring idealised slopes.

```python
from cropforge import Terrain

# Generate a uniform slope with a 5m drop
terrain = Terrain.procedural(rows=30, cols=30, generator="slope", drop_m=5.0)

# Generate rolling hills
terrain = Terrain.procedural(rows=30, cols=30, generator="undulating", amplitude_m=2.0)
```

### 2. CSV Import
Load elevation grids from surveys or other modelling tools.

```python
terrain = Terrain.from_csv("survey_data.csv", resolution_m=1.0)
```

### 3. GeoTIFF Ingestion
Directly ingest high-resolution drone or satellite Digital Elevation Models (DEMs).

```python
terrain = Terrain.from_geotiff("field_dem_30cm.tif", resolution_m=1.0)
```

## Physics Integration

When `farm.use_physics(lateral_flow=True)` is enabled, the hydrology engine uses the **D8 Steepest-Descent Routing Algorithm**.
Surface runoff (from rainfall that exceeds the soil infiltration rate) is routed downslope from higher cells to their steepest lower neighbour. Water pools in local depressions or exits the field boundary.

## Land Preparation

CropForge supports modifying the raw terrain with agronomic land preparation techniques. See the [Terrain & Land Prep Tutorial](../tutorials/terrain_and_land_prep.md) for more details.
