# Product Requirements — CropForge

This page provides a high-level overview of CropForge's design goals and requirements across versions. Full PRD documents are maintained in the project's `internal_dev/` directory.

---

## Vision

CropForge is an **open-source, code-first virtual farm runtime** for agricultural researchers. Researchers write the model; CropForge faithfully executes it and visualises the result.

**Design principles:**

1. **Code-first**: No config files, no GUI to configure. Simulation logic lives in Python.
2. **Compute-then-visualise**: The engine runs fully headless to completion, emitting a structured Parquet log. The visual dashboard reads that log separately.
3. **Opt-in complexity**: Physics solvers, water balance, nitrogen balance are all opt-in. A researcher who doesn't need them is not burdened by them.
4. **Backward compatibility**: Every new version must run all existing user scripts unchanged.

---

## v0.1.0 — Foundation

Core simulation engine (`Farm`, `Field`, `Crop`, `Weather`, `Soil`), phase-ordered step dispatcher, Parquet logger, Three.js 3D viewport, and Plotly Dash dashboard.

## v0.2.0 — Opt-In Physics

FAO-56 ET0 (Penman-Monteith) and Root Impedance engines using negative-phase hooks. Multi-field dashboard with `FieldBufferRegistry`. Published to PyPI.

## v0.3.0 — Events & Water Balance *(In Development)*

Farm Event System (`Event.irrigation()`, `Event.fertiliser()`, `Event.custom()`), Soil Water Balance (closes ET0→stress loop), Soil Nitrogen Balance, complete API reference documentation, and ReadTheDocs deployment.

## v0.4.0 — Deferred

Lateral water flow (D8 algorithm), Plugin API, R bindings, multi-year carry-over state, live weather API integration (ERA5, IMD), full N-P-K model, WebGL leaf geometry deformation.
