# Opt-In Physics

CropForge v0.2.0 introduced **Opt-In Physics** — pre-built, mathematically verified solvers that plug into the simulation's negative phase execution slots. 
v0.5.0 extended this with **Beer-Lambert Radiation Interception** and **Wind-driven Spatial Disease Spread**.
v0.7.0 introduces a massive update with **Topographical Physics**, natively coupling 3D terrain geometry with environment and soil mechanics (Solar Incidence, Wind Shadow, Clod Dynamics, and RUSLE-based Erosion).

## The Philosophy

CropForge was built to let researchers write their *own* mathematical models in Python. However, many models require standard environmental, soil, or epidemiological calculations. Writing these from scratch for every experiment is tedious and error-prone.

Opt-In Physics provides standard, robust implementations. Crucially, they are **gated** — if you never call `farm.use_physics()`, CropForge behaves identically to v0.1.0: a blank canvas for your own math.

---

## Enabling Physics

All engines are enabled through a single call before `farm.run()`:

```python
farm.use_physics(
    et0=True,               # FAO-56 Penman-Monteith ET0
    root_impedance=True,    # Soil penetration resistance
    water_balance=True,     # Soil water balance (requires et0=True)
    lateral_flow=True,      # D8 runoff routing
    radiation=True,         # Beer-Lambert light interception (v0.5.0)
    disease=True,           # Wind-anisotropic SIR disease spread (v0.5.0)
    # --- disease configuration ---
    disease_foci=[(15, 15)],
    disease_wind_direction_deg=270.0,
    disease_spread_rate=0.20,
    
    # --- Topographical Physics (v0.7.0) ---
    slope_radiation_correction=True, # Solar incidence adjusted by slope/aspect
    terrain_wind=True,               # Topographical wind field
    root_clamping=True,              # Topographical root constraints
    clod_dynamics=True,              # Rain melts the clods over time
    erosion=True,                    # RUSLE-based erosion index
)
```

No decorator syntax is needed. Any combination of engines can be enabled independently.

---

## Execution Order

Physics engines run at **negative phases**, guaranteed before any researcher `@farm.step` (which default to `phase=0` or higher):

| Phase | Engine |
|-------|--------|
| `-4`  | Lateral flow + nitrogen transport |
| `-3`  | Soil water balance (FAO-56 hydrology) |
| `-2`  | ET0 Penman-Monteith + Radiation Interception |
| `-1`  | Root impedance + Spatial disease spread |
| `0+`  | Researcher `@farm.step` functions |

---

## FAO-56 Penman-Monteith ET0 (`et0=True`)

Reads daily weather from `EnvironmentState` and computes reference evapotranspiration. Writes:

- `env.et0_mm` — reference ET (mm/day)
- `env.vp_kpa`, `env.psychrometric_kpa`, `env.slope_svp`, `env.net_radiation_mj` — FAO-56 intermediates

---

## Root Impedance (`root_impedance=True`)

Maps each plant's `root_depth_cm` to the soil grid's `penetration_resistance`. Writes `plant.root_growth_multiplier`:

- `1.0` — unrestricted growth
- `→ 0.0` — hard-pan block when resistance ≥ 2.5 MPa

---

## Soil Water Balance (`water_balance=True`)

Closes the ET₀ → soil moisture → plant stress loop automatically. Requires `et0=True`. Computes daily drainage, runoff, and writes `Ks` (water stress coefficient) to soil voxels.

---

## Beer-Lambert Radiation Interception (`radiation=True`) — *v0.5.0*

Implements the standard canopy light interception equation for every living plant:

$$\text{PAR}_{\text{int}} = \text{solar\_rad} \times 0.5 \times \left(1 - e^{-k \times \text{LAI}}\right)$$

where:

- `solar_rad` = `env.radiation_mj_m2` (MJ m⁻² day⁻¹)
- `0.5` = PAR fraction of total solar radiation
- `k` = extinction coefficient (default `0.45` for C3 crops; use `0.50` for C4/maize)
- `LAI` = `plant.lai`

**Output:** `plant.custom['intercepted_par_mj']` — readable by any plugin or `@farm.step`.

### Enabling

```python
farm.use_physics(radiation=True, k_extinction=0.45)
```

### Reading the result in a plugin or step

```python
@farm.step(interval="daily")
def use_par(state, env):
    for plant in state.plants:
        par = plant.custom.get("intercepted_par_mj", 0.0)
        # Use intercepted PAR for RUE-based biomass accumulation
        delta_biomass = par * 1.5  # example: RUE = 1.5 g/MJ
        plant.biomass_g += delta_biomass
    return state
```

### Backward compatibility

When `radiation=False` (the default), `plant.custom` will never have an `intercepted_par_mj` key. Always use `.get("intercepted_par_mj", 0.0)` in any code that may run with or without this engine.

---

## Wind-driven Anisotropic Disease Spread (`disease=True`) — *v0.5.0*

A spatially explicit **SIR (Susceptible–Infected–Resistant) grid model** that simulates disease or pest pressure propagating across the plant grid.

### The model

Each plant has one of three states stored in `plant.custom['disease_state']`:

| State | Meaning |
|-------|---------|
| `'S'` | Susceptible (default; healthy) |
| `'I'` | Infected (spreading; accumulating stress) |
| `'R'` | Resistant / removed |

Each day, every infected plant attempts to infect its 4-connected neighbours. The **probability is weighted by wind direction**: downwind neighbours receive a much higher infection probability than upwind neighbours.

### Wind direction convention

`disease_wind_direction_deg` follows the **meteorological bearing**: the direction *from* which the wind blows.

| Value | Wind from | Spreads toward |
|-------|-----------|----------------|
| `0°`  | North     | South |
| `90°` | East      | West |
| `180°`| South     | North |
| `270°`| West      | **East** ← typical trial scenario |

### Enabling

```python
farm.use_physics(
    disease=True,
    disease_foci=[(15, 15)],          # (row, col) infected on Day 1
    disease_spread_rate=0.15,          # base daily infection probability
    disease_latency_days=5,            # days before plant becomes contagious
    disease_stress_increment=0.04,     # daily stress_index increase per infected plant
    disease_wind_direction_deg=270.0,  # wind from West → spreads East
    disease_anisotropy=0.80,           # 0=isotropic, 1=fully directional
    disease_seed=42,                   # optional reproducibility seed
)
```

Or seed the outbreak on a specific day using the Event system:

```python
from cropforge.events import Event

@farm.add_event(Event.custom(field="MyField", day=40))
def introduce_blight(field_state, env_state):
    center = next(p for p in field_state.plants if p.row == 15 and p.col == 15)
    center.custom["disease_state"] = "I"
    center.custom["days_infected"]  = 0
    center.custom["disease_stress"] = 0.0
    return field_state
```

### Schema keys written (`plant.custom`)

| Key | Type | Description |
|-----|------|-------------|
| `disease_state` | `str` | `'S'`, `'I'`, or `'R'` |
| `disease_stress` | `float` | Cumulative disease stress (0–1) |
| `days_infected` | `int` | Days since first infection |

`disease_stress` integrates into `plant.stress_index` automatically (at 50% weight) to couple disease pressure with the growth model.

### Backward compatibility

When `disease=False` (the default), no `disease_state` key is ever written. Always use `.get("disease_state", "S")` in portable code.

---

## Combining Engines

All engines are fully composable. A complete v0.5.0 research setup:

```python
farm.use_physics(
    et0=True,
    water_balance=True,
    root_impedance=True,
    radiation=True,
    k_extinction=0.45,
    disease=True,
    disease_foci=[(10, 10)],
    disease_wind_direction_deg=270.0,
    disease_spread_rate=0.20,
)
```

See [`examples/disease_outbreak_trial.py`](https://github.com/saswatsundar123/cropforge/blob/main/examples/disease_outbreak_trial.py) for a complete working script and the [Disease Modeling Tutorial](../tutorials/disease_modeling.md) for a step-by-step walkthrough.

---

## Topographical Physics (`v0.7.0`)

With v0.7.0, CropForge bridges the gap between flat-field simulations and true 3D spatial modelling. If a field has a `Terrain` object and land preparation applied, the engine uses the resulting elevation grid to modify physics behaviour.

### Solar Incidence (`slope_radiation_correction=True`)
Adjusts `env.radiation_mj_m2` at a per-cell level based on the slope, aspect, and the sun's declination for that day of the year. South-facing slopes in the Northern Hemisphere receive a radiation multiplier > 1.0, while North-facing slopes are shadowed.

### Topographical Wind Shadow (`terrain_wind=True`)
Adjusts `env.wind_speed_ms` spatially based on `wind_direction_deg`. Windward slopes receive a multiplier > 1.0, ridge tops experience the highest multiplier, and leeward slopes are sheltered. This heavily influences spatial disease spread and ET0.

### Clod Dynamics & Infiltration (`clod_dynamics=True`)
Simulates the physical "melting" of soil clods. `LandPrep` (like `ConventionalTill`) sets an initial `surface_roughness_index`. Heavy rain exponentially decays this roughness over time. High roughness traps water, increasing infiltration and reducing runoff velocity.

### RUSLE-based Erosion (`erosion=True`)
Computes a daily erosion index based on a simplified Revised Universal Soil Loss Equation (RUSLE). Erosion scales multiplicatively with `surface_runoff_mm_today` and slope fraction, but is heavily dampened by `surface_roughness` and vegetation cover. The cumulative erosion index is logged and can be viewed dynamically in the 3D Terrain Modal.

### Effective Soil Depth (`root_clamping=True`)
When LandPrep (like Furrows or Bunds) cuts into or piles up soil, it modifies the `effective_soil_depth_m` of that cell. This engine strictly limits root depth to the new effective depth, simulating the physical boundaries of carved terrain.
