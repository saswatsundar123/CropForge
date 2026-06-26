# Runtime

The CropForge runtime engine (`cropforge/runtime.py`) drives the time-stepping loop and defines the error-handling contract.

---

## Execution Order (per day)

For each simulation day, for each field, the engine executes in this exact order:

| Order | Phase | What runs |
|---|---|---|
| 1 | `phase=-2` | ET0 physics hook (if `use_physics(et0=True)`) |
| 2 | `phase=-1` | Root impedance hook (if `use_physics(root_impedance=True)`) |
| 3 | `phase=0` | Researcher `@farm.step` functions (sorted by phase) |
| 4 | End of day | Registered `Event` objects that fire on this day |
| 5 | — | Parquet logger records the completed timestep |

> **Key principle**: Events fire *after* all step functions. State modifications made by an event are visible to model logic from day+1 onwards. This mirrors real farming: the crop responds to today's environment, then the farmer acts.

---

## Error Handling

When a `@farm.step` function raises an unhandled exception:

1. The simulation **halts immediately**.
2. A crash log is written to `cropforge_crash.log` in the working directory.
3. All completed timesteps are **flushed to the partial Parquet log** so data is not lost.
4. A `CropForgeStepError` is raised.

### `CropForgeStepError`

```python
from cropforge.runtime import CropForgeStepError
```

| Attribute | Type | Description |
|---|---|---|
| `day` | `int` | Day on which the failure occurred. |
| `step_name` | `str` | Name of the step function that raised. |
| `crash_log_path` | `str` | Absolute path to the crash log file. |
| `original_exception` | `BaseException` | The original exception. |

### `CropForgeVisualizeError`

Raised by `farm.visualize()` if the pre-flight check fails (e.g. no Parquet log found, port already in use).

### `CropForgeEventError`

Raised when an `Event` is configured with invalid parameters (e.g. `interval_days=0`). Raised at `farm.run()` time, not at event registration time.
