# CropForge

> Open-source, code-first virtual farm runtime for agricultural researchers.

[![PyPI version](https://badge.fury.io/py/cropforge.svg)](https://badge.fury.io/py/cropforge)
[![Tests](https://github.com/saswatsundar123/cropforge/actions/workflows/ci.yml/badge.svg)](https://github.com/saswatsundar123/cropforge/actions)
[![Docs](https://readthedocs.org/projects/cropforge/badge/?version=latest)](https://cropforge.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CropForge lets you define a crop simulation entirely in Python. You write the model equations; CropForge handles time-stepping, spatial state management, logging, and visual playback.

```bash
pip install cropforge
```

---

## What's New in v0.5.0

### White Minimal Scientific Dashboard
A complete rework of the visualisation layer built on Material Design:
- Sleek, clean white minimal theme with refined borders and ample whitespace.
- Persistent Farm Inspector (always visible, no click-to-reveal).
- Left sidebar with all controls; right sidebar with live charts.
- Animated preloader with per-field progress tracking.
- All panels renamed to researcher-facing terminology.

### Official Crop Plugins — StandardWheat & StandardMaize
First-party, PRD-verified crop plugins distributed as `cropforge.plugins`:

```python
from cropforge.plugins import StandardWheat, StandardMaize

field.use_plugin(StandardWheat)    # CERES-Wheat phenology + RUE biomass + grain fill
field.use_plugin(StandardMaize)    # C4 parameters + root impedance + water stress
```

- **StandardWheat**: 6-stage thermal-time phenology, RUE biomass, grain-fill partitioning into `plant.custom['grain_biomass_g']`.
- **StandardMaize**: C4 photosynthesis parameters, hard-pan root clamping, water-stress mortality.
- Both plugins are fully isolated per field — no cross-talk when running dual-plot comparisons.

### Advanced Physics Engines
Two new opt-in physics engines, mathematically verified against PRD Crucible criteria:

#### Beer-Lambert Radiation Interception
```python
farm.use_physics(radiation=True, k_extinction=0.45)
# Writes plant.custom['intercepted_par_mj'] for every plant every day
```
Implements: `PAR_int = solar_rad × 0.5 × (1 − e^(−k × LAI))`
Crucible verified: LAI=3.0, k=0.45, rad=15 MJ → 5.5557 MJ ±0.001.

#### Wind-driven Anisotropic Disease Spread
```python
farm.use_physics(
    disease=True,
    disease_foci=[(15, 15)],          # Initial outbreak coordinates
    disease_wind_direction_deg=270.0,  # From West → spreads East
    disease_spread_rate=0.20,
    disease_anisotropy=0.80,
)
```
A spatially explicit SIR grid model where infection probability is heavily weighted by wind direction. Crucible verified: eastern half has **2.5× more infections** than western half when wind blows East.

---

## Quick Start

```python
from cropforge import Farm, Field, Crop, Soil, Weather
from cropforge.plugins import StandardWheat

farm  = Farm(name="MyFarm", location=(28.6, 77.2))
field = Field(name="Plot A", rows=20, cols=30, area_ha=2.4)
field.set_crop(Crop(species="wheat"))
field.set_weather(Weather.from_csv("data/weather.csv", ...))
field.set_soil(Soil.from_csv("data/soil.csv", apply="uniform"))
farm.add_field(field)

field.use_plugin(StandardWheat)
farm.use_physics(radiation=True, disease=True, disease_wind_direction_deg=270.0)

farm.run(days=90)
```

See `examples/wheat_basic_v2.py`, `examples/maize_dual_plot_v2.py`, and `examples/disease_outbreak_trial.py` for complete working scripts.

---

## What's New in v0.4.0
- **Plugin Ecosystem**: Extensible architecture for third-party crop models via PyPI.
- **Multi-Season Rotations**: Preserve soil state between consecutive runs.
- **Compare Dashboard**: Compare multiple farm configs and export to CSV.
- **Spatial Hydrology**: D8 lateral surface water routing across gridded fields.

## What's New in v0.2.0
- **Opt-In Physics**: FAO-56 Penman-Monteith ET0 and Root Impedance models.
- **Multi-Field Dashboard**: Field Selector, comparative time-series, GxE analysis.

---

## Documentation

Full documentation at [cropforge.readthedocs.io](https://cropforge.readthedocs.io).

## Licence

MIT — Saswat Sundar Rath, ICAR-IARI Jharkhand, 2026
