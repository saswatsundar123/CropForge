# Loaders

`cropforge.loaders` provides CSV-based data sources for weather and soil profiles.

---

## Weather

Load daily meteorological data from a CSV file.

```python
from cropforge.loaders import Weather

weather = Weather.from_csv("data/weather.csv")
```

### `Weather.from_csv(path)`

**Required columns:**

| Column | Unit | Description |
|---|---|---|
| `day` | integer | Simulation day (1-indexed). |
| `tmax_c` | °C | Maximum air temperature. |
| `tmin_c` | °C | Minimum air temperature. |
| `tmean_c` | °C | Mean air temperature. |
| `solar_rad_mj` | MJ m⁻² | Incoming solar radiation. |
| `precip_mm` | mm | Daily precipitation. |
| `wind_speed_ms` | m s⁻¹ | Wind speed at 2 m height. |
| `humidity_pct` | % | Relative humidity. |

**Additional columns required for `use_physics(et0=True)`:**

| Column | Unit | Description |
|---|---|---|
| `latitude_deg` | decimal degrees | Site latitude (for solar angle calculation). |
| `elevation_m` | m | Site elevation above sea level (for psychrometric constant). |

---

## Soil

Load a soil profile from a CSV file and apply it spatially to a field grid.

```python
from cropforge.loaders import Soil

soil = Soil.from_csv("data/soil.csv", apply="uniform")
```

### `Soil.from_csv(path, apply="uniform")`

**Required columns:**

| Column | Unit | Description |
|---|---|---|
| `layer` | integer | Layer index (0 = topsoil). |
| `depth_top_cm` | cm | Top boundary of layer. |
| `depth_bottom_cm` | cm | Bottom boundary of layer. |
| `moisture_pct` | % | Initial volumetric water content. |
| `nitrogen_kg_ha` | kg ha⁻¹ | Initial available nitrogen. |
| `bulk_density` | g cm⁻³ | Soil bulk density. |
| `penetration_resistance_mpa` | MPa | Mechanical resistance. Values ≥ 2.5 block root growth. |

**`apply` strategies:**

| Value | Behaviour |
|---|---|
| `"uniform"` | Same soil profile applied identically to every grid cell. |
| `"row"` | Profile varies by row (requires one CSV block per row). |
| `"col"` | Profile varies by column (requires one CSV block per col). |
