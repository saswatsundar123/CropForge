# Disease Modeling Tutorial

> **Prerequisites:** CropForge v0.5.0, StandardWheat plugin installed, `examples/data/` directory present.

This tutorial walks through [`examples/disease_outbreak_trial.py`](https://github.com/saswatsundar123/cropforge/blob/main/examples/disease_outbreak_trial.py) — a complete simulation of a spatially spreading crop disease driven by wind direction.

---

## What you will build

A 30×30 wheat field (900 plants) where:

1. **StandardWheat** drives phenology and biomass accumulation.
2. **Beer-Lambert radiation** writes `intercepted_par_mj` to every plant.
3. On **Day 40**, a blight outbreak is seeded at the field center.
4. **Wind blowing East** (270° meteorological bearing) preferentially spreads the disease eastward over the following 50 days.
5. At Day 90, the eastern half of the field has **~2.5× more infections** than the western half.

---

## Step 1: Set up the farm and field

```python
from cropforge import Crop, Farm, Field, Soil, Weather
from cropforge.plugins import StandardWheat

farm  = Farm(name="DiseaseOutbreakFarm", location=(28.6, 77.2))
field = Field(name="WheatPlot_Blight", rows=30, cols=30, area_ha=4.0)

field.set_crop(Crop(species="wheat", variety="HD-2967", sowing_doy=60))
field.set_weather(Weather.from_csv("examples/data/wheat_synthetic_weather_90d.csv", ...))
field.set_soil(Soil.from_csv("examples/data/wheat_uniform_soil_3layer.csv", apply="uniform"))
farm.add_field(field)
```

The field is a flat 30×30 grid. The `location` tuple `(lat, lon)` is used internally by the ET0 engine when enabled.

---

## Step 2: Attach the StandardWheat plugin

```python
field.use_plugin(StandardWheat)
```

`StandardWheat` provides the 6-stage phenology model (thermal-time driven), RUE-based biomass accumulation, and grain-fill partitioning. It runs at `phase=0` — after all built-in physics hooks.

---

## Step 3: Enable the physics engines

```python
farm.use_physics(
    radiation=True,
    k_extinction=0.45,        # C3 wheat extinction coefficient
    disease=True,
    disease_foci=None,        # We'll seed via Event for day-accurate control
    disease_spread_rate=0.20,
    disease_latency_days=3,
    disease_stress_increment=0.04,
    disease_wind_direction_deg=270.0,   # From West -> blows East
    disease_anisotropy=0.80,
    disease_seed=42,
)
```

**Why `disease_foci=None`?** The `disease_foci` parameter seeds plants on the *very first simulation day*. In this scenario, we want the outbreak to begin on **Day 40**, which requires the Event system.

### Understanding wind direction

`disease_wind_direction_deg=270.0` means the wind blows *from* the West. Disease spores travel *with* the wind — eastward. The model gives the eastern neighbour of each infected plant an infection probability ~3–4× higher than the western neighbour.

---

## Step 4: Seed the Day-40 outbreak via an Event

```python
from cropforge.events import Event

@farm.add_event(Event.custom(field="WheatPlot_Blight", day=40))
def introduce_blight(field_state, env_state):
    center = next(
        p for p in field_state.plants
        if p.row == 15 and p.col == 15
    )
    center.custom["disease_state"] = "I"
    center.custom["days_infected"]  = 0
    center.custom["disease_stress"] = 0.0
    return field_state
```

Events fire at the **end of the specified day**, after all `@farm.step` functions. This ensures the disease engine picks up the newly infected center plant on **Day 41**.

!!! tip "Seeding multiple foci"
    To model a realistic multi-point outbreak (e.g. from field edges or infected seed lots), simply loop over multiple plants in the event handler and set their `disease_state` to `'I'`.

---

## Step 5: Add a diagnostic snapshot

```python
@farm.step(interval="daily", phase=10)
def print_disease_snapshot(state, env):
    if state.day in (40, 60, 90):
        infected = sum(1 for p in state.plants if p.custom.get("disease_state") == "I")
        mid_col  = 15
        east_i   = sum(1 for p in state.plants if p.col >= mid_col and p.custom.get("disease_state") == "I")
        west_i   = sum(1 for p in state.plants if p.col < mid_col  and p.custom.get("disease_state") == "I")
        print(f"Day {state.day}: {infected} infected | East={east_i} West={west_i}")
    return state
```

`phase=10` ensures this runs *after* the disease engine (`phase=-1`) and the StandardWheat plugin (`phase=0`).

---

## Step 6: Run and interpret

```python
farm.run(days=90)
```

### Expected output (excerpt)

```
  -- Day 60 Disease Snapshot --
     Infected:    26  |  Resistant: 0
     East half:   19 I  |  West half:     7 I
     Wind-bias ratio (E/W+1): 2.38x

  -- Day 90 Disease Snapshot --
     Infected:   132  |  Resistant: 0
     East half:   95 I  |  West half:    37 I
     Wind-bias ratio (E/W+1): 2.50x

  E/W ratio: 2.50x  [OK] East dominates (wind anisotropy confirmed)
```

The spatial map printed at each snapshot shows the infection front advancing eastward — exactly the PRD §8 Crucible criterion.

---

## Step 7: Read the Parquet log

The simulation writes a full Parquet log under `cropforge_output/`. Every plant's `disease_state`, `disease_stress`, `intercepted_par_mj`, and all other `plant.custom` fields are serialised as JSON on every day:

```python
import pandas as pd
plants = pd.read_parquet("cropforge_output/DiseaseOutbreakFarm_.../plants.parquet")
# Filter for Day 90 infected plants
infected_day90 = plants[(plants["day"] == 90) & (plants["disease_state"] == "I")]
```

---

## Key schema fields written by the disease engine

| `plant.custom` key | Type | Description |
|--------------------|------|-------------|
| `disease_state` | `str` | `'S'` (susceptible), `'I'` (infected), `'R'` (resistant) |
| `disease_stress` | `float` | Cumulative stress 0–1; integrates into `stress_index` |
| `days_infected` | `int` | Days since first infection |
| `intercepted_par_mj` | `float` | Daily intercepted PAR (from radiation engine) |

---

## Extending the model

### Changing wind direction mid-season

The disease engine uses the `wind_direction_deg` closure value captured at `farm.use_physics()` time. To model a shifting wind, you can instead call `calculate_disease_spread()` directly in a custom `@farm.step`, passing a time-varying direction from your weather data.

### Adding a recovery (R) state

The current engine transitions plants `S → I` but not `I → R`. To add recovery, add a custom step:

```python
@farm.step(interval="daily", phase=5)
def disease_recovery(state, env):
    for plant in state.plants:
        if plant.custom.get("disease_state") == "I":
            if plant.custom.get("days_infected", 0) >= 21:  # 21-day infection period
                plant.custom["disease_state"] = "R"
    return state
```

### Visualising disease state in the dashboard

When `disease=True`, you can colour the 3D viewport by `disease_state` using the variable selector. The standard colour mapping is: `S` = health gradient, `I` = orange (`#FF6F00`), `R` = grey (`#9E9E9E`).

---

## Full script

See [`examples/disease_outbreak_trial.py`](https://github.com/saswatsundar123/cropforge/blob/main/examples/disease_outbreak_trial.py) for the complete, runnable version.
