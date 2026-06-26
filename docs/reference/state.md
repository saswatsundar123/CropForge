# State Schema

CropForge's simulation state is defined as a set of Python dataclasses in `cropforge/state.py`. Every step function, physics hook, and event handler reads from and writes to these objects.

---

## PlantState

Represents the physiological state of a single plant at one timestep.

| Field | Type | Default | Description |
|---|---|---|---|
| `plant_id` | `str` | — | Unique identifier (e.g. `"r00c03"`). |
| `row` | `int` | — | Grid row index. |
| `col` | `int` | — | Grid column index. |
| `age_days` | `int` | `0` | Days since sowing. |
| `lai` | `float` | `0.0` | Leaf Area Index (m² m⁻²). |
| `biomass_g` | `float` | `0.0` | Total above-ground dry biomass (g). |
| `height_cm` | `float` | `0.0` | Canopy height (cm). |
| `root_depth_cm` | `float` | `0.0` | Rooting depth (cm). |
| `stress_index` | `float` | `0.0` | Cumulative stress (0 = no stress, 1 = dead). |
| `alive` | `bool` | `True` | Plant viability flag. |
| `phenological_stage` | `str` | `"germination"` | Current growth stage label. |
| `root_growth_multiplier` | `float` | `1.0` | **(v0.2.0)** Engine-set impedance multiplier. `1.0` = free growth, `0.0` = hard-pan block. Set by `use_physics(root_impedance=True)`. |
| `custom` | `dict` | `{}` | Arbitrary researcher fields. Serialised to Parquet as JSON. |

---

## SoilVoxelState

Represents one soil layer at one grid position (row, col, layer).

| Field | Type | Description |
|---|---|---|
| `row`, `col`, `layer` | `int` | Grid position and layer index. |
| `depth_top_cm` | `float` | Top of this layer (cm). |
| `depth_bottom_cm` | `float` | Bottom of this layer (cm). |
| `moisture_pct` | `float` | Volumetric water content (%). |
| `nitrogen_kg_ha` | `float` | Available nitrogen (kg ha⁻¹). |
| `bulk_density` | `float` | Bulk density (g cm⁻³). |
| `penetration_resistance` | `float` | Soil strength (MPa). Values ≥ 2.5 trigger `root_growth_multiplier = 0.0`. |
| `custom` | `dict` | Arbitrary researcher fields. |

---

## EnvironmentState

Meteorological conditions for one field on one day.

| Field | Type | Description |
|---|---|---|
| `day` | `int` | Simulation day (1-indexed). |
| `doy` | `int` | Day of year (1–365). |
| `temp_max_c` | `float` | Maximum air temperature (°C). |
| `temp_min_c` | `float` | Minimum air temperature (°C). |
| `temp_mean_c` | `float` | Mean air temperature (°C). |
| `radiation_mj_m2` | `float` | Incoming solar radiation (MJ m⁻² day⁻¹). |
| `rainfall_mm` | `float` | Daily rainfall (mm). |
| `et0_mm` | `float` | Reference evapotranspiration (mm). Computed by `use_physics(et0=True)` or provided by the Weather CSV. |
| `wind_speed_ms` | `float` | Wind speed at 2 m height (m s⁻¹). |
| `humidity_pct` | `float` | Relative humidity (%). |
| `co2_ppm` | `float` | Atmospheric CO₂ (ppm). Default `415.0`. |
| `vp_kpa` | `float` | **(v0.2.0)** Actual vapour pressure (kPa). FAO-56 intermediate. |
| `psychrometric_kpa` | `float` | **(v0.2.0)** Psychrometric constant γ (kPa °C⁻¹). FAO-56 intermediate. |
| `slope_svp` | `float` | **(v0.2.0)** Slope of SVP curve Δ (kPa °C⁻¹). FAO-56 intermediate. |
| `net_radiation_mj` | `float` | **(v0.2.0)** Net radiation Rn (MJ m⁻² day⁻¹). FAO-56 intermediate. |
| `custom` | `dict` | Arbitrary researcher fields. |

---

## FieldState

Complete state of one field on one day.

| Field | Type | Description |
|---|---|---|
| `day` | `int` | Current simulation day. |
| `plants` | `List[PlantState]` | Flat list of all plants (row-major). |
| `soil` | `List[List[List[SoilVoxelState]]]` | Soil grid indexed as `[row][col][layer]`. |
| `elevation_grid` | `np.ndarray` | Shape `(rows, cols)` elevation surface (m). |
| `events_fired` | `List[str]` | Names of events that fired on this day. |
| `custom` | `dict` | Arbitrary researcher fields. |
