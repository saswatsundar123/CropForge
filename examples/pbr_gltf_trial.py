"""
examples/pbr_gltf_trial.py
==========================
Phase 3 trial: PBR rendering + GLTF model registration for StandardWheat.

Demonstrates:
  - Registering a GLTF model at the 'emergence' stage
  - Launching the dashboard in 'enhanced' quality (PBR shadows on)
  - Confirming model_index_map is populated in buffer metadata

Run:
    python examples/pbr_gltf_trial.py
"""
from cropforge import Farm, Field
from cropforge.plugins import StandardWheat
from cropforge.models import ModelRegistry
from cropforge.state import EnvironmentState


# --- Register dummy GLTF for 'emergence' stage (index 1 = emergence in _STAGE_ORDER)
# In production: use a real .gltf path or pip install cropforge-models-wheat
DUMMY_GLTF = "tests/assets/dummy_plant.gltf"  # relative to project root

ModelRegistry.register(
    species="Triticum aestivum",
    stage="emergence",       # matches StandardWheat._STAGE_ORDER[1]
    gltf_path=DUMMY_GLTF,
)

print("Registered models:", ModelRegistry.list_registered())


# --- Build a small farm ---
class _Warm:
    def get_day(self, day):
        return EnvironmentState(
            day=day, doy=((day - 1) % 365) + 1,
            temp_max_c=28.0, temp_min_c=16.0, temp_mean_c=22.0,
            radiation_mj_m2=20.0, rainfall_mm=0.0,
            et0_mm=4.0, wind_speed_ms=2.0, humidity_pct=55.0,
        )


farm = Farm("PBR_Trial", location=(28.6, 77.2))
field = Field("F1", rows=10, cols=10)
field.set_weather(_Warm())
field.use_plugin(StandardWheat)
farm.add_field(field)
farm.run(days=30)

# Inspect buffer meta — should show model_index_map populated
from cropforge.viz.buffers import FIELD_REGISTRY
from cropforge.viz.app import _DATA, _load_parquet

_load_parquet(farm._last_log_path)
from cropforge.viz.app import create_dash_app
dash_app = create_dash_app(log_path=farm._last_log_path)
from cropforge.viz.server import create_fastapi_app
_DATA_PLANTS = _DATA.get("plants")
api = create_fastapi_app(
    dash_app=dash_app.server,
    log_path=farm._last_log_path,
    cropforge_version="0.9.0",
    plants_df=_DATA_PLANTS,
    quality="enhanced",
)
store = FIELD_REGISTRY.get(None)
if store and store.is_ready:
    print("quality_mode in meta:", store.meta.get("quality_mode"))
    print("model_index_map:", store.meta.get("model_index_map"))
    print("floats_per_plant:", store.meta.get("floats_per_plant"))

print("\nTo launch with enhanced PBR rendering:")
print("    farm.visualize(quality='enhanced')")
print("\nThis will open http://localhost:7860 with:")
print("  - MeshStandardMaterial on terrain (roughness=0.9)")
print("  - MeshStandardMaterial on plants (roughness=0.6)")
print("  - Shadow casting on sun, terrain, and plant meshes")
print("  - GLTF mesh loaded for 'emergence' stage plants (model_index=1)")
print("  - Cylinder fallback for all other stages (model_index=0)")
