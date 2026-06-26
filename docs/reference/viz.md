# Visualization

CropForge's visualization layer is launched via `farm.visualize()` after a simulation run.

---

## Launching the Dashboard

```python
farm.run(days=90)
farm.visualize()  # Opens http://localhost:7860
```

`farm.visualize()` boots a FastAPI binary buffer server and a Plotly Dash dashboard in the same process, then opens the browser.

---

## Dashboard Panels

| Panel | Description |
|---|---|
| **Panel 1: 3D Viewport** | Three.js `InstancedMesh` renderer. Displays all plants colour-coded by a selected variable. Supports click-to-inspect. Day scrubber and playback controls. |
| **Panel 2: Time-Series** | Plotly line chart of any plant metric over the simulation period. For multi-field sessions, all fields are plotted simultaneously on the same chart for divergence comparison. |
| **Panel 3: Heatmap** | Spatial plant grid coloured by a selected variable at the current day. |
| **Panel 4: Farm Inspector** | Per-plant detail panel activated by clicking a plant in the 3D viewport. Shows all metrics, 90-day sparkline history, and soil depth cross-section. |

---

## Multi-Field Support (v0.2.0)

When a session contains more than one field, a **Field Selector** dropdown appears above Panel 2. Selecting a field:

1. Updates the Heatmap and Farm Inspector to show only that field's data.
2. Sends a `cf_set_field` postMessage to the 3D viewport iframe, which tears down the current Three.js scene and re-bootstraps it with the new field's binary buffers.

The Time-Series chart always shows **all fields simultaneously** to enable visual divergence comparison.

---

## Binary Buffer API

The FastAPI server exposes these endpoints (used internally by the Three.js viewport):

| Endpoint | Description |
|---|---|
| `GET /api/fields` | Returns `{fields: [...], default_field: "..."}`. |
| `GET /api/buffer/meta?field=<name>` | JSON metadata: `n_plants`, `n_days`, `days`, `field_name`. |
| `GET /api/buffer?day=<N>&field=<name>` | Raw binary frame: `Float32Array` with 9 floats per plant. |
| `GET /api/buffer/rebuild?variable=<v>&field=<name>` | Triggers hot-reload of colour data for the viewport. |

---

## postMessage Bridge

The Dash parent and the Three.js iframe communicate via `window.postMessage`:

| Message type | Direction | Payload | Effect |
|---|---|---|---|
| `cf_set_day` | Dash → iframe | `{day: N}` | Jump to day N in 3D viewport. |
| `cf_set_field` | Dash → iframe | `{field: "name"}` | Tear down and reload 3D scene for new field. |
| `cf_set_variable` | Dash → iframe | `{variable: "biomass_g"}` | Switch colour mapping in 3D viewport. |
| `cf_deselect` | Dash → iframe | `{}` | Clear plant selection. |
| `PLANT_CLICKED` | iframe → Dash | `{plant_id: "r02c04"}` | Open Farm Inspector for the clicked plant. |

---

## Memory Ceiling

Binary buffer memory usage:

```
bytes = n_plants × 9 floats × 4 bytes/float × n_days
```

For a 100-plant, 90-day run: `100 × 9 × 4 × 90 ≈ 324 KB`. For a 10,000-plant, 365-day run: `≈ 1.3 GB`. Plan field grid sizes accordingly.
