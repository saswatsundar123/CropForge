# 3D Assets & GLTF Export — CropForge v0.9.0

CropForge v0.9.0 adds two complementary features for 3D content pipelines:

1. **Asset Registry** — register GLTF plant models per species and phenological stage; the WebGL viewport substitutes them for the default cylinder geometry.
2. **Scene Export** — export any simulation day as a binary GLB file, ready to open in Blender or Unreal Engine 5.

---

## Part 1: The Asset Registry

### How it works

The `ModelRegistry` maps `(species, stage)` pairs to GLTF file paths. At dashboard startup, the FastAPI server reads the registry and serves registered models at `/api/model/<index>`. The Three.js viewport fetches and caches them, then routes each plant to the correct `InstancedMesh` based on its current phenological stage.

**Cylinder fallback is always active.** Plants with no registered model — or whose GLTF fails to load — silently render as cylinders. No errors, no crashes.

### Quick start

```python
from cropforge.models import ModelRegistry

# Register a GLTF for wheat at the 'emergence' stage
ModelRegistry.register(
    species="Triticum aestivum",   # must match Field crop species
    stage="emergence",             # stage name from plugin._STAGE_ORDER
    gltf_path="models/wheat_emergence.gltf",
)

# Register multiple stages
for stage in ["tillering", "jointing", "heading"]:
    ModelRegistry.register(
        species="Triticum aestivum",
        stage=stage,
        gltf_path=f"models/wheat_{stage}.glb",
    )

# Inspect what's registered
print(ModelRegistry.list_registered())
# → {'Triticum aestivum': ['emergence', 'tillering', 'jointing', 'heading']}
```

### Stage names for built-in plugins

**StandardWheat** (`Triticum aestivum`):

| Index | Stage name |
|---|---|
| 0 | `germination` |
| 1 | `emergence` |
| 2 | `tillering` |
| 3 | `jointing` |
| 4 | `heading` |
| 5 | `grain_fill` |
| 6 | `maturity` |

**StandardMaize** (`Zea mays`) — same index order, stage names differ. Check `cropforge/plugins.py`.

### Using a model package (Option C)

First-party GLTF models are distributed as separate pip packages to keep the core `cropforge` package lightweight:

```bash
# First-party assets
pip install cropforge-models-wheat
```

```python
import cropforge_models_wheat   # auto-registers all 7 wheat stages on import
from cropforge.plugins import StandardWheat
```

### Bundling models with a custom plugin (Option B)

```python
from cropforge import CropPlugin, register_crop
from cropforge.models import ModelRegistry
import pathlib

@register_crop("MyBarley")
class MyBarleyPlugin(CropPlugin):
    _GLTF_DIR = pathlib.Path(__file__).parent / "models"
    _STAGE_ORDER = ["germination", "emergence", "tillering", "heading", "maturity"]
    species = "Hordeum vulgare"

    @classmethod
    def _auto_register(cls):
        for stage in cls._STAGE_ORDER:
            p = cls._GLTF_DIR / f"barley_{stage}.glb"
            if p.exists():
                ModelRegistry.register(cls.species, stage, str(p))

MyBarleyPlugin._auto_register()
```

---

## Part 2: GLB Scene Export

### Python API

```python
farm.run(days=90)

# Export the scene at day 45
glb_path = farm.export_scene(
    day=45,
    filepath="output/field_day45.glb",
    field="Plot_A",     # optional — defaults to first field
)
print(f"Exported: {glb_path}")
```

### What the GLB contains

| Geometry | Description |
|---|---|
| Terrain mesh | Triangulated grid from the simulation elevation data. Each quad is two triangles. |
| Plant boxes | Alive plants at the given day as coloured rectangular boxes (height = `height_cm / 100`). Dead plants are excluded. |

> [!NOTE]
> First-party StandardWheat and StandardMaize GLTF assets are auto-registered.
> If a crop or stage has no registered asset, `export_scene` keeps the cylinder
> fallback so exports remain valid.

### Install the export dependency

```bash
pip install cropforge[export]
# or directly:
pip install pygltflib>=1.16
```

### Opening in Blender

1. `File → Import → glTF 2.0 (.glb/.gltf)`
2. Select your exported `.glb`
3. Press Numpad 0 → camera view
4. Set up lighting and render

### Opening in Unreal Engine 5

1. In the Content Browser: `Import → Select .glb`
2. Accept the import settings
3. Drag into the scene — UE5 applies its own Nanite / Lumen pipeline automatically

### Exporting multiple days (animation prep)

```python
import os
os.makedirs("glb_frames", exist_ok=True)

for day in range(1, 91):
    farm.export_scene(day=day, filepath=f"glb_frames/day_{day:03d}.glb")

# Then in Blender: File → Import → script to batch-import and keyframe
```

---

## Frontend Export (Browser)

The dashboard also provides a one-click GLB export directly from the browser:

1. Run the dashboard: `farm.visualize()`
2. In the top bar, click **⬇ Export 3D Scene (.glb)**
3. The browser downloads `cropforge_scene_day_N.glb` using Three.js `GLTFExporter`

This captures the current Three.js scene exactly as rendered — including terrain geometry, plant instances, and any loaded GLTF models.
