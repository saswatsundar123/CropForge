# Logger

The Parquet state logger (`cropforge/logger.py`) serialises the full simulation state at every timestep.

---

## Output Location

Parquet data is written to:
```
cropforge_output/<session_name>/
```

Where `session_name` is `<farm_name>_<UTC_timestamp>`, e.g. `WheatBasic_20260626T183000`.

---

## Table Schemas

The logger writes three partitioned Parquet tables:

### `plants/`
One row per plant per day per field.

| Column | Type | Description |
|---|---|---|
| `day` | int32 | Simulation day. |
| `field_name` | string | Field identifier. |
| `plant_id` | string | e.g. `"r00c03"`. |
| `row`, `col` | int16 | Grid position. |
| `age_days` | int16 | Days since sowing. |
| `lai` | float32 | Leaf Area Index. |
| `biomass_g` | float32 | Dry biomass (g). |
| `height_cm` | float32 | Canopy height (cm). |
| `root_depth_cm` | float32 | Root depth (cm). |
| `stress_index` | float32 | Cumulative stress (0–1). |
| `alive` | bool | Plant viability. |
| `phenological_stage` | string | Growth stage label. |
| `custom_json` | string | JSON-serialised `PlantState.custom` dict. |

### `soil/`
One row per soil voxel (row × col × layer) per day per field.

| Column | Type | Description |
|---|---|---|
| `day` | int32 | Simulation day. |
| `field_name` | string | Field identifier. |
| `row`, `col` | int16 | Grid position. |
| `layer` | int8 | Layer index. |
| `depth_top_cm`, `depth_bottom_cm` | float32 | Layer boundaries (cm). |
| `moisture_pct` | float32 | Volumetric water content (%). |
| `nitrogen_kg_ha` | float32 | Available N (kg ha⁻¹). |
| `bulk_density` | float32 | Bulk density (g cm⁻³). |
| `penetration_resistance` | float32 | Soil strength (MPa). |
| `custom_json` | string | JSON-serialised custom dict. |

### `environment/`
One row per field per day.

| Column | Type | Description |
|---|---|---|
| `day` | int32 | Simulation day. |
| `field_name` | string | Field identifier. |
| `doy` | int16 | Day of year. |
| `temp_max_c`, `temp_min_c`, `temp_mean_c` | float32 | Temperature (°C). |
| `radiation_mj_m2` | float32 | Solar radiation (MJ m⁻²). |
| `rainfall_mm` | float32 | Rainfall (mm). |
| `et0_mm` | float32 | Reference ET (mm). |
| `wind_speed_ms` | float32 | Wind speed (m s⁻¹). |
| `humidity_pct` | float32 | Relative humidity (%). |
| `co2_ppm` | float32 | CO₂ concentration (ppm). |
| `events_fired` | string | JSON array of event names that fired this day. |
| `custom_json` | string | JSON-serialised custom dict. |

---

## Reading Logs Externally

```python
import pandas as pd

plants = pd.read_parquet("cropforge_output/MyFarm_20260626T120000/plants/")
soil   = pd.read_parquet("cropforge_output/MyFarm_20260626T120000/soil/")
env    = pd.read_parquet("cropforge_output/MyFarm_20260626T120000/environment/")

# Filter to one field and one day
day45_plotA = plants[
    (plants["field_name"] == "Plot_A") & (plants["day"] == 45)
]
```

Partitioning by `field_name` and `day` means queries on a single day/field are very fast — PyArrow only reads the relevant partition files.
