# Changelog

All notable changes to CropForge are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning: [Semantic Versioning](https://semver.org/)

---

## [0.3.0] — In Development

### Added
- **Event System** (`cropforge.Event`): `Event.irrigation()`, `Event.fertiliser()`, `Event.custom()` — schedule management actions on specific days or at repeating intervals.
- **API Reference Docs**: Seven complete reference pages (`crop`, `farm`, `state`, `loaders`, `runtime`, `logger`, `viz`, `events`).
- `CropForgeEventError` for invalid event configuration.

### Changed
- Event execution order: events now fire **after** all `@farm.step` functions (end-of-day), not before. This correctly mirrors the PRD Section 4.3 contract.

---

## [0.2.0] — 2026-06-25

### Added
- **Opt-In Physics**: FAO-56 Penman-Monteith ET0 engine (`phase=-2`) and Root Impedance engine (`phase=-1`).
- `@farm.use_physics(et0=True, root_impedance=True)` decorator.
- `EnvironmentState` extended with four FAO-56 intermediate fields.
- **Multi-Field Dashboard**: Field Selector dropdown, comparative time-series, `FieldBufferRegistry`.
- `cf_set_field` postMessage support in Three.js viewport (full scene teardown and re-bootstrap).
- Dual-plot GxE example (`examples/maize_dual_plot.py`).
- Documentation: `docs/features/physics.md`, `docs/tutorials/maize_dual_plot.md`.
- Published to PyPI: `pip install cropforge`.

### Fixed
- Replaced broken `pytest-dash` with `dash[testing]` in dev dependencies.
- Moved PRD files to `internal_dev/` (gitignored, not public).

---

## [0.1.0] — 2026-06-24

### Added
- **Core Engine**: `Farm`, `Field`, `Crop`, `Weather`, `Soil` public API.
- `@farm.step(phase=N)` decorator and phase-ordered step dispatcher.
- `EnvironmentState`, `PlantState`, `SoilVoxelState`, `FieldState` dataclasses.
- `Weather.from_csv()` and `Soil.from_csv()` with `uniform`, `row`, `col` apply strategies.
- `StateLogger` — partitioned Parquet output with Snappy compression.
- **Three.js 3D Viewport**: `InstancedMesh` binary streaming, raycaster click-to-inspect, `OrbitControls`.
- **Plotly Dash Dashboard**: 4-panel layout (3D viewport, time-series, heatmap, farm inspector).
- FastAPI binary buffer server with `Float32Array` frame API.
- postMessage bridge: `cf_set_day`, `cf_set_variable`, `cf_deselect`, `PLANT_CLICKED`.
- GitHub Actions CI: test matrix (Python 3.12/3.13) → build → OIDC publish to PyPI on `v*` tags.
- 304 passing tests.
