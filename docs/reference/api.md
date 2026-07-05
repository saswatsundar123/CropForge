# CropForge v0.8.0 — API Reference

Auto-extracted from source docstrings. For conceptual guides see the [tutorials](../tutorials/).

---

## Core Classes

### `Farm`

```python
class Farm(name: str, location: tuple[float, float])
```

The top-level simulation container. Holds one or more `Field` objects, a step registry, and the physics configuration.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable farm identifier. |
| `location` | `tuple[float, float]` | `(latitude_deg, longitude_deg)` used by solar engines. |

**Key Methods**

| Method | Signature | Description |
|---|---|---|
| `add_field` | `(field: Field) → None` | Register a field for simulation. |
| `use_physics` | `(**kwargs) → None` | Enable opt-in physics engines. See [Physics Reference](#physics-use_physics). |
| `add_event` | `(event: Event) → None` | Schedule an agronomic event. |
| `run` | `(days: int, season: int = 1) → None` | Execute the daily simulation loop. |
| `visualize` | `() → None` | Launch the Dash+FastAPI dashboard. |
| `save_state` | `(path: str) → None` | Persist soil and plant state to `.cfstate`. |
| `load_state` | `(path: str) → None` | Restore a previous `.cfstate` checkpoint. |
| `prepare_next_season` | `() → None` | Reset plants, retain soil, advance season counter. |

**`use_physics` flags** {#physics-use_physics}

```python
farm.use_physics(
    et0=False,                     # FAO-56 Penman-Monteith ET0
    root_impedance=False,          # Root growth limited by penetration resistance
    soil_water_balance=False,      # Tipping-bucket soil water model  (requires et0=True)
    nutrients=False,               # Nitrogen mineralisation + uptake
    lateral_flow=False,            # D8 lateral N leaching
    radiation=False,               # Beer-Lambert PAR interception
    biotic_stress=False,           # SIR wind-driven disease model
    clod_dynamics=False,           # Rainfall-driven roughness decay
    erosion=False,                 # RUSLE-based erosion index
    sediment_transport=False,      # D8 sediment routing (v0.8.0; requires erosion=True)
    slope_radiation_correction=False,  # Slope-aspect solar incidence
    terrain_wind=False,            # Terrain-modulated wind field
    # Additional kwargs passed through to individual engines
)
```

---

### `Field`

```python
class Field(
    name: str,
    rows: int,
    cols: int,
    area_ha: float = None,
    crop: Crop = None,
    weather: Weather = None,
    soil: Soil = None,
)
```

A spatial grid of `rows × cols` plant/soil cells. Each cell maps to one plant and one soil column.

**Key Methods**

| Method | Description |
|---|---|
| `set_terrain(terrain: Terrain)` | Attach a topographic grid. |
| `set_land_prep(lp: LandPrep)` | Apply pre-simulation land modification. |
| `set_crop(crop: Crop)` | Set the crop species and parameters. |
| `set_weather(weather: Weather)` | Attach daily weather data. |
| `set_soil(soil: Soil)` | Attach layered soil profile. |
| `set_water_params(**kwargs)` | Override soil hydraulic parameters. |
| `set_nitrogen_params(**kwargs)` | Override nitrogen model parameters. |
| `set_clod_params(**kwargs)` | Set clod dynamics decay constants. |
| `set_erosion_params(**kwargs)` | Set soil erodibility K-factor. |
| `use_plugin(plugin: type)` | Attach a `CropPlugin` to this field. |

---

### `Terrain`

```python
class Terrain
```

Stores a digital elevation model (DEM) and derived grids (slope, aspect, D8 flow, flow accumulation).

**Construction**

```python
# Procedural generation
Terrain.procedural(
    rows: int, cols: int,
    generator: str = "undulating",   # "flat" | "slope" | "undulating" | "ridge"
    amplitude_m: float = 2.0,
    frequency: float = 0.08,
    seed: int = 42,
    resolution_m: float = 1.0,       # Physical cell size (v0.8.0)
) → Terrain

# From NumPy array (v0.8.0)
Terrain.from_array(
    arr: np.ndarray,                 # shape (rows, cols), dtype float32
    resolution_m: float = 1.0,
) → Terrain

# From CSV
Terrain.from_csv(path: str, resolution_m: float = 1.0) → Terrain

# From GeoTIFF (requires rasterio)
Terrain.from_geotiff(path: str) → Terrain   # resolution_m read from CRS
```

**Key Attributes**

| Attribute | Type | Description |
|---|---|---|
| `resolution_m` | `float` | Physical cell edge length in metres. |
| `elevation_grid` | `np.ndarray` | Base DEM, shape `(rows, cols)`. |
| `slope_grid` | `np.ndarray` | Slope in degrees per cell. |
| `aspect_grid` | `np.ndarray` | Aspect in degrees from North. |
| `d8_flow_grid` | `np.ndarray` | D8 flow direction index per cell. |
| `flow_accumulation_grid` | `np.ndarray` | Upslope contributing area in cells. |

---

### Land Preparation

All `LandPrep` subclasses implement:
```python
def apply(rows, cols, soil_template, terrain) → (elevation_grid, soil_mods)
# or (v0.8.0 extension):
def apply(...) → (elevation_grid, soil_mods, per_cell_mods)
```

| Class | Constructor | Description |
|---|---|---|
| `RidgeFurrow` | `(ridge_height_m, ridge_spacing_m, furrow_depth_m)` | Raised beds in alternating rows. |
| `ContourBund` | `(vertical_interval_m, bund_height_cm)` | Elevation-following embankments. |
| `Terrace` | `(vertical_interval_m, terrace_width_m)` | Horizontal cut-and-fill terraces. |
| `ZeroTillage` | `()` | High-residue, no soil disturbance. |
| `ConventionalTill` | `(roughness_index)` | Standard tillage, variable roughness. |
| `TiedRidges` | `(ridge_height_m, ridge_spacing_m, tie_spacing_m, tie_height_m)` | Ridge-furrow + periodic tie-dams. Micro-catchments block D8 runoff. ✅ v0.8.0 |
| `VegetativeFilterStrip` | `(strip_width_m, n_strip_rows, position)` | Dense grass strip at field edge. Reduces detachment by ~95%. ✅ v0.8.0 |

---

### `Event`

```python
Event.irrigation(field: str, interval_days: int, amount_mm: float) → Event
Event.fertiliser(field: str, interval_days: int, amount_n_ppm: float) → Event
Event.custom(field: str, interval_days: int, fn: Callable) → Event
```

---

### `Crop` / `Weather` / `Soil`

```python
Crop(species: str, **params)

Weather(df: pd.DataFrame)           # index = day int
Weather.from_csv(path, ...)

Soil(layers: list[dict])
Soil.from_csv(path, apply: str)     # apply = "uniform" | "layered"
```

---

## Physics Engines

### Hydrology (`cropforge/physics/hydrology.py`)

**Inputs:** daily `precip_mm`, `et0_mm`, `slope_grid`, `roughness_index`  
**Outputs (per cell, layer 0):** `surface_runoff_mm_today`, `moisture_pct`, `infiltration_rate_mm_hr`, `clod_dynamics` roughness update

Surface runoff uses a modified Green-Ampt approach with slope-dependent velocity. `resolution_m` scales lateral flux between cells.

---

### Erosion & Sediment Transport (`cropforge/physics/soil.py`)

```
erosion_index = runoff_mm × slope_frac × (1 − roughness) × (1 − veg_cover)

eroded_depth_mm = erodibility_k × erosion_index × cell_area_m2

transport_capacity = runoff_mm × slope_frac × flow_velocity_factor

net = min(eroded_depth_mm, transport_capacity)
  → excess deposited locally
  → carried sediment routed D8 downslope
```

**Mass conservation invariant:** `Σ(erosion) = Σ(field_export) + Σ(deposition)` to floating-point precision.

**v0.8.0 SoilState fields added:**

| Field | Unit | Description |
|---|---|---|
| `sediment_flux_kg_m2` | kg/m² | Sediment passing through cell today |
| `sediment_deposited_kg_m2` | kg/m² | Deposited sediment today |
| `cumulative_sediment_loss_kg_m2` | kg/m² | Total loss since start |
| `cumulative_deposition_kg_m2` | kg/m² | Total deposition since start |

---

### Geomorphological Feedback (`cropforge/physics/builtin_hooks.py`)

After sediment transport, `elevation_grid_m` is updated daily:
```
Δz = (deposited_mm − eroded_mm) / 1000   # m
elevation[r, c] += Δz
```
Slope and aspect grids recompute automatically for the next timestep. Layer 0 depth expands/contracts with a 1 mm bedrock floor.

---

## Exceptions

| Exception | Base | Raised when |
|---|---|---|
| `CropForgeConfigError` | `ValueError` | Physics flags conflict; `prepare_next_season()` called during run |
| `CropForgeStepError` | `RuntimeError` | Researcher's `@farm.step` crashes with a Python exception |
| `CropForgeVisualizeError` | `RuntimeError` | `farm.visualize()` preflight fails (missing Parquet, wrong version) |
| `CropForgeStateError` | `ValueError` | `.cfstate` file incompatible, or `save_state()` called before `run()` |
| `CropForgeEventError` | `ValueError` | Event misconfigured (e.g. `interval_days=0`) |
| `CropForgePluginError` | `ValueError` | Plugin registration conflict or unregistered plugin |

All exceptions include an actionable message with the corrective action.

---

## 3D Visualisation

**LOD Terrain Renderer (v0.8.0)**

The Three.js viewport chunks terrain into 64×64-cell tiles. Each chunk holds two geometries:
- **Hi-res** (step=1): 1:1 cell-to-vertex mapping — used when camera is within 55 world units.
- **Lo-res** (step=4): 16× fewer vertices — used when camera is beyond 80 world units.

Hysteresis (55/80 unit thresholds) prevents geometry flickering at the LOD boundary. Frustum culling drops off-screen chunks from GPU draw calls.

**Raycasting** uses pure grid math (`instanceId = row × cols + col`) — unaffected by the LOD geometry level.

---

*Generated from source: `cropforge/` v0.8.0 — 2026-07-05*
