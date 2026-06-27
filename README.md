# CropForge

> Open-source, code-first virtual farm runtime for agricultural researchers.

CropForge lets you define a crop simulation entirely in Python. You write the model equations; CropForge handles time-stepping, spatial state management, logging, and visual playback.

## What's New in v0.4.0
* **Plugin Ecosystem**: Extensible architecture allowing third-party researchers to publish and install custom crop models (e.g., `cropforge-wheat`) via PyPI.
* **Multi-Season Rotations**: Preserve soil state, moisture, and nitrogen between consecutive simulation runs on the same field to study long-term soil health and crop rotations.
* **Compare Dashboard**: Visually compare multiple farm configurations on the same time-series chart and export session data directly to CSV.
* **Spatial Hydrology**: D8 lateral surface water routing across gridded fields based on elevation.

## What's New in v0.2.0
* **Opt-In Physics**: Built-in, mathematically verified FAO-56 Penman-Monteith ET0 resolution and Root Impedance models. Enabled via the `@farm.use_physics(et0=True, root_impedance=True)` decorator.
* **Multi-Field Dashboard**: Compare divergent physical environments and scenarios (GxE) directly in the UI with a new Field Selector Dropdown, filtering Heatmaps, 3D viewport, and Inspector panels per-field, while retaining a unified Time-Series comparative view.

```bash
pip install cropforge
```

See `examples/wheat_basic.py` for a minimal working simulation.

## Licence
MIT — Saswat Sundar Rath, ICAR-IARI Jharkhand, 2026
