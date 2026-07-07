# Visual Engine — CropForge v0.9.0

CropForge v0.9.0 introduced a fully overhauled visual engine across both the WebGL Three.js viewport and the Plotly Dash terrain panel.

---

## Three.js WebGL Viewport

### Quality Modes

The dashboard accepts a `quality` parameter via `farm.visualize(quality=...)`:

| Mode | Description |
|---|---|
| `"standard"` (default) | Identical to v0.8.0. `MeshBasicMaterial` — no lighting computation. Maximum compatibility. |
| `"enhanced"` | PBR materials (`MeshStandardMaterial`), shadow casting, sun-angle lighting. |

```python
farm.run(days=90)
farm.visualize()                       # standard — identical to v0.8.0
farm.visualize(quality="enhanced")    # PBR — photorealistic
```

### PBR Material Parameters (`quality="enhanced"`)

**Terrain mesh:**
```js
roughness: 0.9,   // matte soil, not specular clay
metalness: 0.0,
color: 0x8B6914   // ochre base; tinted by overlay variable at runtime
```

**Plant instances (cylinder fallback):**
```js
roughness: 0.6,
metalness: 0.0,
color: 0x228B22   // forest green; overridden by stage-progress tint
```

Shadow caster: `DirectionalLight` positioned at the current solar azimuth/altitude (computed from simulation day and field latitude). Shadow map resolution: 2048×2048.

### Per-Stage InstancedMesh Routing

In v0.8.0, all plants shared one `InstancedMesh`. In v0.9.0, the renderer maintains a **dictionary of InstancedMeshes keyed by `model_index`**. Each day:

1. Binary buffer is parsed (`11 floats/plant` — x, y, z, height, r, g, b, alive, instance_id, `stage_index`, `lai_scale`)
2. Plant's `model_index` is read from the `/api/buffer/meta` endpoint
3. The plant matrix is applied to the correct `InstancedMesh` for that model

This allows different plant geometries per phenological stage with zero overhead for stages that share the same mesh.

### Model Registry Integration

When `quality="enhanced"` and a GLTF model is registered for a stage, the Three.js loader fetches it from `/api/model/<model_index>` and substitutes it for the cylinder. **Cylinder fallback is always active** — if no model is registered or the GLTF fails to load, cylinders render silently.

---

## Plotly Dash Terrain Panel (4× Visual Upsampling)

The `go.Surface` terrain chart now applies bicubic upsampling before display. The physics simulation grid is **never modified** — this is a pure UI-layer enhancement.

### What changes

| Parameter | Before v0.9.0 | v0.9.0 |
|---|---|---|
| Grid passed to `go.Surface` | Raw simulation grid (e.g. 20×30) | 4× upsampled (e.g. 80×120) |
| Upsampling method (elevation) | None | `scipy.ndimage.zoom(order=3)` — bicubic |
| Upsampling method (colour overlay) | None | `scipy.ndimage.zoom(order=1)` — linear |
| Lighting | Default Plotly plastic | `roughness=0.9, specular=0.1, ambient=0.7` |
| Axis clutter | Grid lines, tick labels, background | All hidden (`showbackground`, `showgrid`, `zeroline`, `showticklabels` = False) |

### Why two different zoom orders?

- **Elevation** (`order=3`, bicubic): produces smoothly curving terrain that looks like real topography even on coarse grids.
- **Colour overlay** (`order=1`, linear): prevents physics data (erosion index, moisture %) from being artificially smoothed — values stay honest while still aligning to the upsampled shape.

> [!NOTE]
> Bicubic interpolation on random data can produce minor overshoot at sharp boundaries. On physically meaningful, smooth terrain grids this is imperceptible. The physics data is never modified.

### Shape-match guarantee

`z` (elevation) and `surfacecolor` (overlay) **always have identical shape** — both are zoomed from the same source grid dimensions. This prevents the Plotly shape-mismatch error that would otherwise occur when passing a raw elevation grid with a separately-computed overlay.

---

## GLTF Export

See `docs/tutorials/3d_assets_and_export.md` for the full export workflow.

```python
# Quick reference
glb = farm.export_scene(day=15, filepath="output/day15.glb")
# Open in Blender → File → Import → GLTF 2.0
```
