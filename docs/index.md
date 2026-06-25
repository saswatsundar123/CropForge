# CropForge

**Open-source, code-first virtual farm runtime for agricultural researchers.**

CropForge is a Python 3.12+ simulation engine that lets crop science researchers
define, run, and visualise virtual farm experiments as pure Python code.

## Key Features

- **Code-first**: Define crops, fields, and management events in Python — no
  configuration files, no GUIs required.
- **Opt-In Physics** (v0.2.0): Built-in, mathematically verified FAO-56 ET0 and root impedance physics that seamlessly integrate with custom scripts.
- **Multi-Field Dashboard** (v0.2.0): Compare divergent physical environments and GxE scenarios directly in the UI with the new Field Selector.
- **Decoupled architecture**: The time-stepping engine runs headless to
  completion; the visual dashboard reads the Parquet log afterward.
- **Binary streaming**: Three.js 3D viewport receives Float32Array buffers
  directly — no JSON overhead for 3D data.
- **Phase-ordered steps**: Attach model steps to any phase (environment,
  plant, soil, output) using the `@step` decorator; the runtime enforces
  execution order.
- **Interactive dashboard**: FastAPI + Plotly Dash dashboard with 3D instanced
  plant renderer, time-series metrics, event log, and per-plant inspector.

## Quick Start

```bash
pip install cropforge
```

```python
from cropforge import Farm, Field, Crop
from cropforge.loaders import Weather, Soil

wheat = Crop(species="wheat", variety="HD-2967", sowing_doy=300)
field = Field("Plot A", rows=20, cols=30, crop=wheat,
              spacing_m=0.2, weather=Weather.from_csv("weather.csv"),
              soil=Soil.from_csv("soil.csv", apply="uniform"))

farm = Farm("WheatBasic")
farm.add_field(field)
farm.run(days=90)
farm.visualize()      # opens http://localhost:7860
```

## Documentation

- [Installation](getting_started/installation.md)
- [Quickstart](getting_started/quickstart.md)
- [API Reference](reference/crop.md)

## Licence

MIT © Saswat Sundar Rath, ICAR-IARI Jharkhand
