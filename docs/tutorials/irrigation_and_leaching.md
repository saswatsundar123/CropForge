# Tutorial 3: Irrigation Trial & Slope Leaching

**CropForge v0.3.0** — *The full-stack experiment*

This tutorial demonstrates the two landmark experiments introduced in v0.3.0.
Together, they exercise every physics engine in CropForge — ET₀, soil water balance,
root impedance, nitrogen transport, and spatial lateral flow — in a single researcher
workflow.

---

## What You Will Learn

| Experiment | Engines Used | Key Output |
|---|---|---|
| **Irrigation Trial** | ET₀ · Water Balance · Events | Stress divergence between rainfed and irrigated maize |
| **Slope Leaching Trial** | Water Balance · Nutrients · Lateral Flow | Spatial N redistribution on a sloped field |

---

## Part 1 — Irrigation Trial

### Scientific Scenario

A 90-day maize season on a semi-arid site (Bihar, India). Two plots:

- **Plot_A_Rainfed** — relies entirely on monsoon rainfall
- **Plot_B_Irrigated** — receives 50 mm supplemental irrigation every 15 days

Both plots start with identical soil and fertilization. The only difference is water
supply. By day 45 you will see a clear divergence in plant stress.

### Code

The full script is at [`examples/irrigation_trial.py`](../../examples/irrigation_trial.py).
Here is the critical setup:

```python
from cropforge import Farm, Field, Crop, Event

farm = Farm(name="IrrigationTrial", location=(25.6, 85.1))

# Plot A — Rainfed
plot_a = Field(name="Plot_A_Rainfed", rows=2, cols=2, area_ha=0.5)
plot_a.set_crop(Crop(species="Zea mays", variety="DKC9025"))
plot_a.set_weather(MonsoonWeather())
plot_a.set_water_params(
    field_capacity_pct=30.0,
    wilting_point_pct=12.0,
    saturation_pct=44.0,
    drainage_coefficient=0.6,
    crop_coefficient=1.15,
    stress_increment_per_day=0.04,
)
farm.add_field(plot_a)

# Plot B — Irrigated (identical setup + an irrigation event)
plot_b = Field(name="Plot_B_Irrigated", rows=2, cols=2, area_ha=0.5)
# ... same soil/weather config ...
farm.add_field(plot_b)

# Enable physics stack
farm.use_physics(
    et0=True,          # Penman-Monteith ET₀
    water_balance=True, # FAO-56 tipping-bucket drainage
    nutrients=True,    # Nitrogen leaching
    lateral_flow=True, # Surface runoff N transport
)

# Events
farm.add_event(Event.fertiliser(field="Plot_A_Rainfed",   day=20, amount_kg_ha=80, layer=0))
farm.add_event(Event.fertiliser(field="Plot_B_Irrigated", day=20, amount_kg_ha=80, layer=0))
farm.add_event(Event.irrigation(
    field="Plot_B_Irrigated",
    interval_days=15,
    amount_mm=50,
    start_day=1,
    end_day=90,
))

# Researcher model — biomass accumulation gated by water stress
@farm.step(phase=0)
def accumulate_biomass(state, env):
    for plant in state.plants:
        if not plant.alive:
            continue
        ks = plant.custom.get("water_stress_ks", 1.0)  # written by water balance engine
        plant.biomass_g += 2.0 * ks
        plant.height_cm += 0.6 * ks
        plant.root_depth_cm = min(40.0, plant.root_depth_cm + 0.5)
        plant.age_days += 1
    return state

farm.run(days=90)
```

### Running It

```bash
python examples/irrigation_trial.py
```

Expected terminal output (Day 45):

```
 Day  StressA    MoistA  StressB    MoistB      Diff
----  --------  -------  --------  -------  --------
   1    0.0000   28.00%    0.0000   28.00%    0.0000
  15    0.0800   18.40%    0.0000   30.00%   +0.0800
  30    0.2000   14.20%    0.0000   28.50%   +0.2000
  45    0.4200   12.10%    0.0000   27.80%   +0.4200
  ...
```

!!! note "PRD §7.3 Criterion"
    By day 45, rainfed `stress_index > 0.15` and irrigated `stress_index < 0.10`.
    The divergence confirms the soil water balance and event system are correctly
    coupled.

### What the Physics Engines Did

```
Every day (each field independently):
  phase=-4  Nitrogen leached downward proportional to yesterday's drainage
  phase=-3  Rainfall added → ET₀ extracted from root zone → drainage cascades
            → Ks computed → stress_index updated
  phase=-2  Penman-Monteith ET₀ calculated from weather
  phase=-1  Root impedance multiplier applied (if enabled)
  phase= 0  Your @farm.step runs (biomass accumulation)
  end-of-day  Event.irrigation fires if today matches interval
```

---

## Part 2 — Slope Leaching Trial

### Scientific Scenario

A 3×3 grid field with a pronounced **North-to-South elevation gradient** (5 m → 0 m).
All cells receive identical fertilisation (120 kg N/ha on day 1). A single heavy rainfall
event (80 mm on day 3) exceeds soil saturation, generating surface overland flow.
The D8 routing algorithm moves water — and dissolved nitrogen — downslope.

**Expected outcome:** Top-row cells (upslope) **deplete** nitrogen. Bottom-row cells
(downslope) **accumulate** nitrogen above neighbouring values.

### Code

The full script is at [`examples/slope_leaching_trial.py`](../../examples/slope_leaching_trial.py).
The critical spatial setup:

```python
import numpy as np
from cropforge import Farm, Field, Crop

farm = Farm(name="SlopeTrial", location=(23.5, 77.0))
field = Field(name="SlopeField", rows=3, cols=3, area_ha=0.5)

# North-to-South DEM: row 0 = 5 m (upslope), row 2 = 0 m (downslope)
dem = np.array([
    [5.0, 5.0, 5.0],   # row 0 — high elevation
    [2.5, 2.5, 2.5],   # row 1 — mid-slope
    [0.0, 0.0, 0.0],   # row 2 — low elevation, receives runoff
], dtype=float)
field.set_elevation(dem)

field.set_water_params(
    field_capacity_pct=30.0,
    wilting_point_pct=12.0,
    saturation_pct=44.0,
    drainage_coefficient=0.8,
)
field.set_nitrogen_params(
    leaching_fraction=0.02,    # 2% of available N leached per mm drainage
    runoff_n_fraction=0.08,    # 8% of layer-0 N exported per 100 mm runoff
)
farm.add_field(field)

farm.use_physics(
    et0=True,
    water_balance=True,
    nutrients=True,
    lateral_flow=True,
)

# Uniform fertilisation on day 1
farm.add_event(Event.fertiliser(field="SlopeField", day=1, amount_kg_ha=120, layer=0))
```

### Running It

```bash
python examples/slope_leaching_trial.py
```

Expected terminal output (after 10 days):

```
Initial N (all cells): 120.0 kg/ha

Final N grid (kg/ha) after 10 days:
  (row 0 = upslope, row 2 = downslope)

  Row 0 [↑ upslope  ]:   71.24   71.24   71.24  kg/ha
  Row 1 [  mid-slope]:   75.88   75.88   75.88  kg/ha
  Row 2 [↓ downslope]:   79.41   79.41   79.41  kg/ha

  Mean N row 0 (upslope):   71.24 kg/ha
  Mean N row 2 (downslope): 79.41 kg/ha
  Accumulation at downslope: -40.59 kg/ha  ← net after leaching
  Depletion at upslope:     -48.76 kg/ha

Spatial N differentiation (downslope > upslope): CONFIRMED ✓
```

### How the Lateral Flow Works

The heavy rain (80 mm on day 3) exceeds saturation (44%) in all cells. The excess water
above saturation is tracked as `surface_runoff_mm_today`. The next day (day 4), the
nutrients hook (phase=-4) reads this runoff volume and applies D8 routing:

```
For each cell with surface_runoff_mm > 0:
  1. Find the steepest downslope neighbour (D8 algorithm)
  2. N_transported = runoff_mm × runoff_n_fraction × N_available / 100
  3. Subtract N from source cell, add to sink cell
```

The cascade effect — row 0 → row 1 → row 2 — creates a measurable nitrogen
gradient that would be detected in a real soil survey.

---

## Combining Both Experiments

For a complete study, you can run both on the same farm:

```python
# Farm with both fields
farm = Farm(name="FullStudy", location=(25.6, 85.1))
farm.add_field(irrigated_plot)
farm.add_field(slope_field)

# One use_physics() call enables all engines for all fields
farm.use_physics(
    et0=True,
    water_balance=True,
    nutrients=True,
    lateral_flow=True,
)

farm.run(days=90)
farm.visualize()   # opens the multi-field dashboard
```

---

## Key API Reference

| Method | Description |
|---|---|
| `Field.set_water_params(field_capacity_pct, wilting_point_pct, saturation_pct, drainage_coefficient, crop_coefficient)` | Configure soil hydraulic properties |
| `Field.set_nitrogen_params(leaching_fraction, runoff_n_fraction)` | Configure N transport parameters |
| `Field.set_elevation(dem)` | Set a NumPy DEM for D8 lateral routing |
| `farm.use_physics(et0, water_balance, nutrients, lateral_flow)` | Enable opt-in physics engines |
| `Event.irrigation(field, interval_days, amount_mm, start_day, end_day)` | Recurring irrigation event |
| `Event.fertiliser(field, day, amount_kg_ha, layer)` | One-off or multi-day fertilisation |

---

## Physics Dependencies

```
lateral_flow=True  ─── requires ──→  water_balance=True
nutrients=True     ─── requires ──→  water_balance=True
water_balance=True ─── requires ──→  et0=True
```

Violating any dependency raises `CropForgeConfigError` at `farm.run()` with an
explicit message showing exactly what to enable.

---

*Next: [API Reference — Farm](../reference/farm.md)*
