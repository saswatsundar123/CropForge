# Events

The `Event` class (v0.3.0) allows researchers to schedule discrete management actions — irrigation, fertiliser application, and custom interventions — that fire on specific days or at repeating intervals.

---

## Execution Contract

Events fire **at the end of each day**, after all physics hooks and `@farm.step` functions have run. State modifications made by an event are visible to model logic from **day+1** onwards.

```
Day N execution order:
  1. Physics hooks  (phase=-2, -1)
  2. @farm.step     (phase=0, 1, 2, …)
  3. Events fire    ← here
  4. Logger records the completed day
```

---

## API

```python
from cropforge import Farm, Event

farm = Farm(name="Irrigation Trial")

# Interval-based irrigation (fires every 15 days)
farm.add_event(Event.irrigation(
    field="Plot_A",
    interval_days=15,
    amount_mm=30,
    start_day=1,
    end_day=90,
))

# One-time fertiliser
farm.add_event(Event.fertiliser(
    field="Plot_A",
    day=20,
    n_kg_ha=40.0,
    apply_to_layer=0,
))

# Fertiliser on multiple days (split application)
farm.add_event(Event.fertiliser(
    field="Plot_B",
    days=[20, 45],
    n_kg_ha=25.0,
    apply_to_layer=0,
))

# Custom event — arbitrary function called on a specific day
@farm.add_event(Event.custom(field="Plot_A", day=50))
def stress_test(field_state, env_state):
    for plant in field_state.plants:
        plant.custom["drought_stressed"] = True

farm.run(days=90)
```

---

## `Event.irrigation()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `field` | `str` | — | Target field name. |
| `interval_days` | `int` | — | Days between irrigations. Must be ≥ 1. |
| `amount_mm` | `float` | — | Water added to layer 0 moisture (mm). |
| `start_day` | `int` | `1` | First day on which the event fires. |
| `end_day` | `int \| None` | `None` | Last day (inclusive). `None` = end of simulation. |

Moisture is capped at the soil's saturation percentage. Irrigation does not affect other fields.

---

## `Event.fertiliser()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `field` | `str` | — | Target field name. |
| `day` | `int` | — | Single day on which to fire. Mutually exclusive with `days`. |
| `days` | `List[int]` | — | List of days on which to fire. Mutually exclusive with `day`. |
| `n_kg_ha` | `float` | — | Nitrogen added per hectare. |
| `apply_to_layer` | `int` | `0` | Soil layer index to receive nitrogen. |

---

## `Event.custom()`

| Parameter | Type | Description |
|---|---|---|
| `field` | `str` | Target field name. |
| `day` | `int` | Day on which to fire. |

The decorated function receives `(field_state: FieldState, env_state: EnvironmentState)`. It may return `None` or a modified `field_state`.

---

## Event Log

Every fired event writes one entry to the Event Log panel in the dashboard and to `FieldState.events_fired`. Format:

```
Day 15 | Plot_A | irrigation | +30mm (layer 0)
Day 20 | Plot_A | fertiliser | +40 kg/ha N (layer 0)
Day 50 | Plot_A | custom | stress_test()
```

---

## Error Handling

| Error | When raised |
|---|---|
| `CropForgeEventError` | `interval_days=0` or other invalid configuration. Raised at `farm.run()` time. |
| Custom event exception | Caught, logged to Event Log as an error entry. Simulation **continues**. |
