"""
tests/test_viz_registry.py
===========================
Crucible Test for PRD v0.9.0 Phase 1: AssetRegistry + model_id state pipeline.

Verifies:
1. AssetRegistry.register / get_model_path / list_registered
2. Crop A (registered) → model_id updates as plant advances through stages
3. Crop B (not registered) → model_id stays "" throughout (cylinder fallback)
4. Parquet schema contains model_id column without breaking legacy reads
5. Mulching, BroadBedFurrow, terrain_feedback flag, 3-tuple compat shim
"""

import math
import pytest
import numpy as np

from cropforge.viz.registry import AssetRegistry
from cropforge.land_prep import Mulching, BroadBedFurrow, RidgeFurrow, LandPrep
from cropforge.state import PlantState


# ---------------------------------------------------------------------------
# Fixture: clean registry between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    AssetRegistry.clear()
    yield
    AssetRegistry.clear()


# ---------------------------------------------------------------------------
# 1. AssetRegistry unit tests
# ---------------------------------------------------------------------------

def test_register_and_get():
    AssetRegistry.register("CropA", stage=4, uri="assets/crop_a_anthesis.gltf")
    assert AssetRegistry.get_model_path("CropA", 4) == "assets/crop_a_anthesis.gltf"


def test_register_documented_species_gltf_path_alias():
    AssetRegistry.register(
        species="Triticum aestivum",
        stage=4,
        gltf_path="assets/wheat_anthesis.gltf",
    )
    assert (
        AssetRegistry.get_model_path(species="Triticum aestivum", stage=4)
        == "assets/wheat_anthesis.gltf"
    )


def test_get_unregistered_returns_none():
    assert AssetRegistry.get_model_path("CropB", 4) is None


def test_list_registered():
    AssetRegistry.register("Wheat", stage=0, uri="w0.gltf")
    AssetRegistry.register("Wheat", stage=4, uri="w4.gltf")
    AssetRegistry.register("Maize", stage=2, uri="m2.gltf")
    result = AssetRegistry.list_registered()
    assert result["Wheat"] == [0, 4]
    assert result["Maize"] == [2]


def test_clear():
    AssetRegistry.register("X", stage=1, uri="x.gltf")
    AssetRegistry.clear()
    assert AssetRegistry.get_model_path("X", 1) is None


# ---------------------------------------------------------------------------
# 2. PlantState.model_id default
# ---------------------------------------------------------------------------

def test_plant_state_model_id_default():
    p = PlantState(plant_id="p1", row=0, col=0)
    assert p.model_id == ""


# ---------------------------------------------------------------------------
# 3. Crucible Test: two-crop simulation
#    Crop A registered → model_id updates with stage
#    Crop B not registered → model_id stays ""
# ---------------------------------------------------------------------------

def test_crucible_registry_two_crops(tmp_path):
    """
    Crucible Test (PRD v0.9.0 §Task 5):
    - CropA: register stage 0 and stage 2 URIs
    - CropB: no registration
    - Run 5 days; researcher bumps CropA to 'vegetative' on day 3
    - Assert CropA.model_id transitions; CropB.model_id stays ""
    """
    from cropforge.farm import Farm, Field

    AssetRegistry.register("CropA", stage=0, uri="mock://crop_a_germination.gltf")
    AssetRegistry.register("CropA", stage=2, uri="mock://crop_a_vegetative.gltf")
    # CropB: intentionally not registered

    farm = Farm("TestFarm", location=(20.0, 85.0))
    field = Field("TestField", rows=4, cols=4)
    farm.add_field(field)

    # Plant both crops; store plant refs via custom dict key
    @farm.step(phase=0)
    def setup(state, env):
        if env.day == 1:
            for plant in state.plants:
                r, c = plant.row, plant.col
                if r < 2:
                    plant.custom["crop_name"] = "CropA"
                    plant.phenological_stage = "germination"
                else:
                    plant.custom["crop_name"] = "CropB"
                    plant.phenological_stage = "germination"

        # On day 3: advance CropA plants to vegetative
        if env.day == 3:
            for plant in state.plants:
                if plant.custom.get("crop_name") == "CropA":
                    plant.phenological_stage = "vegetative"

    farm.run(days=5)

    final_state = field._field_state

    crop_a_plants = [p for p in final_state.plants if p.custom.get("crop_name") == "CropA"]
    crop_b_plants = [p for p in final_state.plants if p.custom.get("crop_name") == "CropB"]

    # CropA: stage=vegetative (index 2) → should have vegetative URI
    for plant in crop_a_plants:
        assert plant.phenological_stage == "vegetative"
        assert plant.model_id == "mock://crop_a_vegetative.gltf", (
            f"CropA plant {plant.plant_id} expected vegetative URI, got {plant.model_id!r}"
        )

    # CropB: no registration → must always be "" (cylinder fallback)
    for plant in crop_b_plants:
        assert plant.model_id == "", (
            f"CropB plant {plant.plant_id} expected empty model_id, got {plant.model_id!r}"
        )


def test_crucible_model_id_at_germination():
    """CropA registered at stage=0 → model_id set even on day 1."""
    from cropforge.farm import Farm, Field

    AssetRegistry.register("Wheat", stage=0, uri="assets/wheat_germ.gltf")

    farm = Farm("WF", location=(20.0, 85.0))
    field = Field("F1", rows=2, cols=2)
    farm.add_field(field)

    @farm.step(phase=0)
    def tag(state, env):
        for plant in state.plants:
            plant.custom["crop_name"] = "Wheat"
            plant.phenological_stage = "germination"

    farm.run(days=1)
    for plant in field._field_state.plants:
        assert plant.model_id == "assets/wheat_germ.gltf"


# ---------------------------------------------------------------------------
# 4. Parquet schema backward-compat: model_id column readable
# ---------------------------------------------------------------------------

def test_parquet_schema_contains_model_id(tmp_path):
    """Parquet log written by a v0.9.0 run contains model_id column."""
    import pyarrow.parquet as pq
    from cropforge.farm import Farm, Field

    AssetRegistry.register("Wheat", stage=0, uri="assets/wheat.gltf")

    farm = Farm("ParquetFarm", location=(20.0, 85.0))
    field = Field("PF", rows=2, cols=2)
    farm.add_field(field)

    @farm.step(phase=0)
    def tag(state, env):
        for p in state.plants:
            p.custom["crop_name"] = "Wheat"

    farm.run(days=2)

    log_path = farm._last_log_path
    assert log_path is not None
    # Logger writes partitioned datasets: session_dir/plants/, session_dir/soil/, etc.
    table = pq.read_table(log_path + "/plants")
    assert "model_id" in table.schema.names, "model_id column missing from Parquet log"


# ---------------------------------------------------------------------------
# 5. Phase 0 carry-over tests
# ---------------------------------------------------------------------------

def test_mulching_no_elevation_change():
    m = Mulching(cover_fraction=0.7, mulch_type="straw", thickness_cm=5.0)
    elev = np.zeros((10, 10))
    modified, soil_mods = m.apply(elev, resolution_m=1.0)
    np.testing.assert_array_equal(modified, elev)
    assert "mulch_cover_fraction" in soil_mods
    assert soil_mods["mulch_cover_fraction"] == pytest.approx(0.7)


def test_mulching_invalid_cover_fraction():
    with pytest.raises(ValueError):
        Mulching(cover_fraction=1.5)


def test_mulching_invalid_type():
    with pytest.raises(ValueError):
        Mulching(mulch_type="hay")


def test_bbf_bed_raises_terrain():
    bbf = BroadBedFurrow(bed_width_m=1.5, bed_height_cm=15.0,
                         furrow_width_m=0.5, furrow_depth_cm=20.0)
    elev = np.zeros((10, 20))
    modified, soil_mods = bbf.apply(elev, resolution_m=0.5)
    # Some cells should be raised (beds) and some depressed (furrows)
    assert modified.max() > 0.0, "Expected raised bed cells"
    assert modified.min() < 0.0, "Expected depressed furrow cells"
    assert soil_mods.get("p_factor") == pytest.approx(0.45)


def test_bbf_invalid_params():
    with pytest.raises(ValueError):
        BroadBedFurrow(bed_width_m=0.0)
    with pytest.raises(ValueError):
        BroadBedFurrow(furrow_width_m=-1.0)


def test_three_tuple_shim_on_old_two_tuple_subclass():
    """A subclass returning a 2-tuple still works via _apply_compat (PRD v0.9.0 §4.4)."""
    class OldStylePrep(LandPrep):
        def apply(self, elevation_grid, resolution_m):
            return elevation_grid.copy(), {"porosity_delta": 0.01}

    prep = OldStylePrep()
    elev = np.zeros((5, 5))
    result = prep._apply_compat(elev, resolution_m=1.0)
    assert len(result) == 3
    assert result[2] == {}   # per_cell_mods defaults to empty dict


def test_three_tuple_shim_preserves_native_three_tuple():
    """A subclass already returning 3-tuple passes through unchanged."""
    class NewStylePrep(LandPrep):
        def apply(self, elevation_grid, resolution_m):
            return elevation_grid.copy(), {}, {(0, 0): {"surface_roughness_index": 0.9}}

    prep = NewStylePrep()
    elev = np.zeros((5, 5))
    result = prep._apply_compat(elev, resolution_m=1.0)
    assert len(result) == 3
    assert (0, 0) in result[2]


def test_terrain_feedback_flag_accepted():
    """terrain_feedback=False should be accepted by use_physics without error."""
    from cropforge.farm import Farm, Field

    farm = Farm("TF_Farm", location=(20.0, 85.0))
    farm.use_physics(
        et0=False,
        erosion=True,
        sediment_transport=True,
        terrain_feedback=False,  # freeze terrain
    )
    # No exception raised = pass
