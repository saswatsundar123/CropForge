# Opt-In Physics

CropForge v0.2.0 introduces **Opt-In Physics**, providing built-in, mathematically verified biological solvers that can be enabled with a single decorator.

## The Philosophy

CropForge was originally built to let researchers write their *own* mathematical models in Python. However, many models require standard environmental and soil calculations (like ET0 or root impedance). Writing these from scratch for every experiment is tedious.

Opt-In Physics provides standard, robust implementations of these solvers. Importantly, they are *opt-in*. If you don't enable them, CropForge behaves exactly as it did in v0.1.0—as a blank canvas for your custom math.

## The `@farm.use_physics()` Decorator

You can enable built-in physics solvers by decorating your farm instance:

```python
from cropforge import Farm

farm = Farm("MyFarm")

@farm.use_physics(et0=True, root_impedance=True)
def init_physics():
    pass
```

### FAO-56 Penman-Monteith ET0 (`et0=True`)
When `et0=True` is provided, CropForge automatically registers a built-in step function that runs at `phase=-2`. This function reads the daily weather data and calculates the reference evapotranspiration (ET0) using the FAO-56 Penman-Monteith equation. The result is written to `EnvironmentState.et0_mm` (along with intermediate variables like `vp_kpa`, `psychrometric_kpa`, `slope_svp`, and `net_radiation_mj`).

### Root Impedance (`root_impedance=True`)
When `root_impedance=True` is provided, CropForge registers a built-in step function at `phase=-1`. This step simulates physical soil constraints on root growth. It maps the current root depth to the soil grid's penetration resistance and calculates a `root_growth_multiplier`. If the penetration resistance is $\ge 2.5$ MPa, root growth is severely restricted (multiplier $\to 0.0$).

## Backward Compatibility via Negative Phases

The core innovation of the Opt-In Physics architecture is the use of negative phases.

User-defined `@farm.step` functions default to `phase=0`. By scheduling the built-in physics solvers at `phase=-2` (Environment) and `phase=-1` (Soil), we guarantee that they execute *before* any of the user's custom plant logic. 

This means a researcher can enable Opt-In Physics and immediately use `state.env.et0_mm` or `plant.root_growth_multiplier` in their `phase=0` custom steps, without altering the execution order or breaking legacy scripts.

## ⚠️ Water Stress Coupling — Manual Wiring Required (v0.2.0)

> **Important for v0.2.0 users.** ET₀ computation and plant stress coupling are **separate concerns**.

In v0.2.0, enabling `use_physics(et0=True)` correctly computes `EnvironmentState.et0_mm` every day using the FAO-56 Penman-Monteith equation. However, **this value does not automatically affect `PlantState.stress_index`**. The engine calculates evapotranspiration demand but does not yet implement a soil water balance to determine how much of that demand is met.

If you need stress coupling in v0.2.0, you must wire it manually in your `@farm.step` function:

```python
@farm.step(phase=1)
def compute_stress(state, env):
    # Manual ET0 → stress wiring (v0.2.0 workaround)
    # This is a simplified example; use your own empirical coefficients.
    for plant in state.plants:
        if env.et0_mm > 0:
            # Fraction of ET demand unmet (no soil water available yet)
            unmet_fraction = min(1.0, env.et0_mm / 10.0)
            plant.stress_index = min(1.0, plant.stress_index + unmet_fraction * 0.05)
    return state
```

**This limitation is resolved in v0.3.0** by the Soil Water Balance subsystem, which will automatically close the ET₀ → soil moisture → plant stress loop. When `use_physics(soil_water_balance=True)` is enabled in v0.3.0, `plant.stress_index` will be updated automatically — no manual wiring required.
