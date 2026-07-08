# Visual Engine - CropForge v0.9.5

CropForge v0.9.5 completes the visual architecture arc. The simulation engine
still runs first and writes Parquet logs; the dashboard then reconstructs the
field from logged plant, terrain, weather, and machinery data.

---

## Quality Modes

The dashboard accepts a `quality` parameter through `farm.visualize()`:

| Mode | Description |
|---|---|
| `"standard"` (default) | Compatibility path for broad hardware support. Rendering avoids expensive post-processing and weather effects. |
| `"enhanced"` | PBR materials, shadows, SSAO, bloom, ACES tone mapping, disease shader coloring, machinery shadows, and bounded weather particles. |

```python
farm.run(days=90)
farm.visualize()                       # standard
farm.visualize(quality="enhanced")     # PBR + visual effects
```

`quality="standard"` remains the compatibility baseline. Expensive effects are
strictly gated behind `quality="enhanced"`.

---

## First-Party Asset Bundles

The built-in `StandardWheat` and `StandardMaize` plugins auto-register bundled
GLTF stage assets when imported. Researchers can still register custom models
through `cropforge.models.ModelRegistry`.

```python
from cropforge.plugins import StandardWheat

field.use_plugin(StandardWheat)
# GLTF assets are registered automatically.
```

Cylinder fallback is always active. If a stage model is absent, a GLTF fails to
load, or a custom registry entry is invalid, the viewport silently renders the
safe cylinder representation instead of breaking the session.

---

## Binary Viewport Buffer

The plant viewport buffer is a `Float32Array` with 14 floats per plant:

| Slot | Field |
|---:|---|
| 0 | x |
| 1 | y |
| 2 | z |
| 3 | scale_y |
| 4 | radius |
| 5-7 | RGB color |
| 8 | alive |
| 9 | model_index |
| 10 | stage_progress |
| 11 | morph_weight |
| 12 | stress_ks |
| 13 | disease_severity |

Morph, wilt, and disease data travel with each plant instance. Third-party
binary consumers must track this stride.

---

## Morph Targets And Stress Visuals

When a GLTF contains morph targets, the Three.js shader interpolates between the
registered start and end geometry using `morph_weight`. Drought stress drives a
wilt deformation from `stress_ks`.

Disease severity is logged from the disease engine and packed into the viewport
buffer. In enhanced mode, `material.onBeforeCompile` injects fragment shader
logic that blends the natural plant color with a necrotic brown-yellow tone
based on `disease_severity`.

Guard clauses keep models without morph targets renderable. Missing morph data
skips the morph calculation rather than black-screening WebGL.

---

## Machinery Animation Framework

Tillage, spray, and harvest events log their movement paths to a separate
`machinery` Parquet dataset. The Python runtime records event day, machine type,
and waypoints; it does not perform real-time rendering.

The `/api/buffer/day/{day}` endpoint adds active machinery metadata to the daily
JSON payload:

```json
{
  "day": 5,
  "field_name": "Plot_A",
  "machinery": [
    {
      "event_name": "tillage",
      "machine_type": "tractor",
      "path": [[0, 0], [10, 0], [10, 1]]
    }
  ]
}
```

The frontend loads a lightweight proxy mesh and animates it along the logged path
with `requestAnimationFrame`. Enhanced mode enables shadow casting for the
machine; standard mode keeps the proxy lightweight.

---

## Weather Particle Effects

Daily precipitation is read from the environment Parquet dataset and exposed as
`precipitation_mm` in `/api/buffer/day/{day}`. The frontend uses that value to
control rain visibility and intensity.

Rain particles are created only in enhanced mode:

- A bounded `THREE.Points` system is allocated once.
- Particle count is capped at 6000.
- `geometry.setDrawRange()` scales visible droplets by rainfall intensity.
- Rain hides when `precipitation_mm < 2.0`.
- Droplets fall in the animation loop and reset to the top of the field bounds.

This keeps weather visuals expressive without allowing unbounded WebGL buffers.

---

## Plotly Dash Terrain Panel

The Plotly terrain chart applies 4x upsampling before display. The physics grid
is never modified.

| Parameter | Rendering behavior |
|---|---|
| Elevation | Bicubic upsampling for smooth terrain |
| Color overlay | Linear upsampling to keep physics values honest |
| Lighting | Matte soil surface with low specular response |
| Axes | Hidden grid lines, tick labels, and background planes |

`z` and `surfacecolor` are always upsampled from the same source dimensions so
Plotly receives shape-compatible arrays.

---

## GLTF Export

Use `farm.export_scene()` for a terrain-aware GLB snapshot:

```python
farm.run(days=90)
farm.export_scene(day=45, filepath="mid_season_export.glb")
```

The dashboard also exposes a browser-side scene export button for the live
Three.js view.
