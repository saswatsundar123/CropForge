"""
examples/photorealistic_twin_trial.py
======================================
v0.9.0 Capstone Deliverable — Photorealistic Digital Twin Trial.

Demonstrates the complete v0.9.0 feature stack working together:
  1. Procedural undulating terrain
  2. RidgeFurrow land preparation
  3. StandardWheat with full physics
  4. AssetRegistry — dummy GLTF model at 'emergence' stage
  5. GLB scene export via farm.export_scene()
  6. Photorealistic PBR dashboard via farm.visualize(quality="enhanced")

Run:
    python examples/photorealistic_twin_trial.py

Output:
    wheat_trial_day15.glb  — open in Blender / Unreal Engine 5
    Browser opens http://localhost:7860 with PBR lighting active.
"""
from cropforge import Farm, Field, Terrain, RidgeFurrow
from cropforge.plugins import StandardWheat
from cropforge.models import ModelRegistry
from cropforge.state import EnvironmentState

# ---------------------------------------------------------------------------
# 1. Register GLTF asset for 'emergence' stage
#    In production: pip install cropforge-models-wheat
# ---------------------------------------------------------------------------
DUMMY_GLTF = "tests/assets/dummy_plant.gltf"

ModelRegistry.register(
    species="Triticum aestivum",
    stage="emergence",          # stage index 1 in StandardWheat._STAGE_ORDER
    gltf_path=DUMMY_GLTF,
)
print("Model registry:", ModelRegistry.list_registered())


# ---------------------------------------------------------------------------
# 2. Weather stub — 15 warm spring days
# ---------------------------------------------------------------------------
class _SpringWeather:
    def get_day(self, day):
        return EnvironmentState(
            day=day,
            doy=((day - 1) % 365) + 1,
            temp_max_c=26.0 + (day % 5),
            temp_min_c=14.0 + (day % 3),
            temp_mean_c=20.0,
            radiation_mj_m2=19.5,
            rainfall_mm=3.0 if day % 5 == 0 else 0.0,
            et0_mm=3.8,
            wind_speed_ms=1.8,
            humidity_pct=58.0,
        )


# ---------------------------------------------------------------------------
# 3. Build farm: undulating terrain + ridge furrow + wheat
# ---------------------------------------------------------------------------
farm = Farm("WheatTrial_v090", location=(28.6, 77.2))

terrain = Terrain.procedural(
    rows=20, cols=30,
    generator="undulating",
    amplitude_m=1.5,
    resolution_m=1.0,
    seed=42,
)
land_prep = RidgeFurrow(
    ridge_height_m=0.15,
    furrow_width_m=0.6,
    row_spacing_m=0.75,
)

field = Field("Plot_A", rows=20, cols=30)
field.set_weather(_SpringWeather())
field.set_terrain(terrain)
field.set_land_prep(land_prep)
field.use_plugin(StandardWheat)

farm.add_field(field)

# Enable physics (soil_water_balance is enough to make the terrain meaningful)
@farm.use_physics(
    et0=True,
    soil_water_balance=True,
    erosion=True,
)
def _(f): pass

# ---------------------------------------------------------------------------
# 4. Run 15 days
# ---------------------------------------------------------------------------
print("Running 15-day wheat trial with PBR assets...")
farm.run(days=15)
print(f"Simulation complete. Log: {farm._last_log_path}")

# ---------------------------------------------------------------------------
# 5. Export GLB scene
# ---------------------------------------------------------------------------
glb_path = farm.export_scene(day=15, filepath="wheat_trial_day15.glb")
print(f"GLB exported → {glb_path}")
print("Open wheat_trial_day15.glb in Blender / Unreal Engine 5 for offline render.")

# ---------------------------------------------------------------------------
# 6. Launch PBR dashboard
# ---------------------------------------------------------------------------
print("\nLaunching photorealistic dashboard at http://localhost:7860 ...")
print("Press Ctrl+C to stop.\n")
farm.visualize(quality="enhanced")
