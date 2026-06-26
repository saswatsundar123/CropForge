# Farm & Field

`cropforge.Farm` and `cropforge.Field` are the top-level orchestration classes.

---

## Farm

```python
from cropforge import Farm

farm = Farm(name="Trial 2026-A", location=(23.4, 85.3))
```

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Human-readable trial identifier. Used as the output directory prefix. |
| `location` | `Tuple[float, float]` | `(0.0, 0.0)` | `(latitude_deg, longitude_deg)`. Latitude is used by the FAO-56 ET0 engine. |

### Methods

#### `farm.add_field(field)`
Attach a configured `Field` to the farm. Field names must be unique.

#### `farm.add_event(event)`
Register a management event (`Event.irrigation()`, `Event.fertiliser()`, or `Event.custom()`). See [Events](events.md).

#### `@farm.step(phase=0)`
Decorator to register a model step function. All step functions receive `(field_state, env_state)` and may return a modified `field_state`.

```python
@farm.step(phase=0)
def grow(state, env):
    for plant in state.plants:
        plant.biomass_g += max(0, env.temp_mean_c - 10) * 0.5
    return state
```

**Phase rules:**
- `phase` must be a non-negative integer (0, 1, 2, …).
- Built-in physics hooks occupy negative phases (`-2`, `-1`). They always run before user steps.
- Multiple steps with the same phase execute in non-deterministic order. A warning is emitted.

#### `farm.use_physics(et0=False, root_impedance=False, elevation_m=0.0, anemometer_height_m=2.0)`
Enable built-in physics engines. See [Opt-In Physics](../features/physics.md).

#### `farm.run(days=N)`
Execute the simulation for `N` days. Writes a partitioned Parquet log to `cropforge_output/<session>/`.

#### `farm.visualize()`
Launch the FastAPI + Dash dashboard at `http://localhost:7860`. Reads the most recent Parquet log.

---

## Field

```python
from cropforge import Field, Crop
from cropforge.loaders import Weather, Soil

field = Field(name="Plot A", rows=10, cols=10, area_ha=1.0)
field.set_crop(Crop(species="maize", variety="Kharif-1", sowing_doy=150))
field.set_weather(Weather.from_csv("data/weather.csv"))
field.set_soil(Soil.from_csv("data/soil.csv", apply="uniform"))
```

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Unique field identifier within the farm. |
| `rows` | `int` | — | Number of rows in the plant grid (N–S axis). |
| `cols` | `int` | — | Number of columns in the plant grid (E–W axis). |
| `area_ha` | `float` | `1.0` | Physical area (ha). Used for per-hectare quantity calculations. |
| `elevation_profile` | `str \| np.ndarray \| None` | `None` | Terrain model. `None` / `"flat"` = flat. `"slope_1pct_N"` / `"slope_2pct_N"` = built-in slopes. Pass a NumPy array for a custom DEM. |

### Methods

| Method | Description |
|---|---|
| `field.set_crop(crop)` | Attach a `Crop` instance. |
| `field.set_weather(weather)` | Attach a `Weather` data source. |
| `field.set_soil(soil)` | Attach a `Soil` profile. |
| `field.set_elevation(dem)` | Replace the elevation grid with a NumPy array of shape `(rows, cols)`. |
