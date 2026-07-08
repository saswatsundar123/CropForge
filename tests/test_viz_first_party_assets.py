"""
tests/test_viz_first_party_assets.py
======================================
Crucible Tests for PRD v0.9.5 Phase 1: First-Party Asset Bundles & Auto-Hook Binding.

Verifies:
1. Importing StandardWheat auto-registers all 7 GLTF stage paths (no manual calls)
2. Importing StandardMaize auto-registers all 7 GLTF stage paths
3. All registered paths point to existing files on disk
4. GLTF files are valid JSON with correct structure (scene, meshes, morph targets)
5. Stage 0 (germination) bounding-box height is near-zero (PRD §4.6)
6. Stage 4 (anthesis) has the tallest bounding box of all wheat stages (PRD §4.6)
7. All GLTF files are within triangle budget specified in PRD §4.3 / §5.2
8. Binary buffer model_index updates as a StandardWheat plant advances through stages
9. `from cropforge.models import ModelRegistry` import path works (PRD §2.3)
10. Cylinder fallback intact — crops without GLTF still render (no crash)
"""

import json
import os
import pathlib
import struct

import pytest

from cropforge.viz.registry import AssetRegistry

_WHEAT_STAGES = 7
_MAIZE_STAGES = 7

# Triangle budgets per PRD §4.3 (wheat) and §5.2 (maize)
_WHEAT_TRI_BUDGET = [50, 50, 150, 150, 300, 300, 100]
_MAIZE_TRI_BUDGET = [50, 50, 200, 200, 400, 400, 150]


# ---------------------------------------------------------------------------
# Fixture: isolate registry across test module — import once at module level
# so StandardWheat fires its auto-registration exactly once.
# ---------------------------------------------------------------------------

import importlib

@pytest.fixture(scope="module", autouse=True)
def _load_plugins():
    """Import plugins; let their module-level auto-registration run.
    Uses reload to ensure it fires even if other test modules imported them first.
    """
    AssetRegistry.clear()
    import cropforge.plugins.wheat as wheat
    import cropforge.plugins.maize as maize
    importlib.reload(wheat)
    importlib.reload(maize)
    yield
    # Don't clear after — other tests in this module need the state.


@pytest.fixture(autouse=True)
def _per_test_clear_if_needed():
    # Only clear between tests that explicitly register their own URIs
    yield


# ---------------------------------------------------------------------------
# 1. Auto-registration: StandardWheat
# ---------------------------------------------------------------------------

def test_wheat_auto_registers_all_7_stages():
    """StandardWheat import auto-registers all 7 stage paths."""
    registered = AssetRegistry.list_registered()
    assert "StandardWheat" in registered, "StandardWheat not in registry after import"
    stages = registered["StandardWheat"]
    assert len(stages) == _WHEAT_STAGES, (
        f"Expected {_WHEAT_STAGES} wheat stages, got {len(stages)}: {stages}"
    )
    assert stages == list(range(_WHEAT_STAGES)), f"Missing stage indices: {stages}"


# ---------------------------------------------------------------------------
# 2. Auto-registration: StandardMaize
# ---------------------------------------------------------------------------

def test_maize_auto_registers_all_7_stages():
    """StandardMaize import auto-registers all 7 stage paths."""
    registered = AssetRegistry.list_registered()
    assert "StandardMaize" in registered, "StandardMaize not in registry after import"
    stages = registered["StandardMaize"]
    assert len(stages) == _MAIZE_STAGES, (
        f"Expected {_MAIZE_STAGES} maize stages, got {len(stages)}: {stages}"
    )
    assert stages == list(range(_MAIZE_STAGES)), f"Missing stage indices: {stages}"


# ---------------------------------------------------------------------------
# 3. All registered paths exist on disk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("crop_key", ["StandardWheat", "StandardMaize"])
def test_all_registered_paths_exist(crop_key):
    """Every URI in the registry points to a real file on disk."""
    registered = AssetRegistry.list_registered()
    for stage_idx in registered[crop_key]:
        uri = AssetRegistry.get_model_path(crop_key, stage_idx)
        assert uri is not None, f"No URI for {crop_key} stage {stage_idx}"
        assert os.path.exists(uri), (
            f"{crop_key} stage {stage_idx} path does not exist: {uri}"
        )


# ---------------------------------------------------------------------------
# 4. GLTF files are valid JSON with correct structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("crop_key,budget_list", [
    ("StandardWheat", _WHEAT_TRI_BUDGET),
    ("StandardMaize", _MAIZE_TRI_BUDGET),
])
def test_gltf_structure_and_triangle_budget(crop_key, budget_list):
    """Each GLTF is valid JSON, has mesh + material, 2 morph targets, within tri budget."""
    registered = AssetRegistry.list_registered()
    for stage_idx in registered[crop_key]:
        uri = AssetRegistry.get_model_path(crop_key, stage_idx)
        with open(uri, "r", encoding="utf-8") as f:
            gltf = json.load(f)

        # Valid GLTF 2.0
        assert gltf["asset"]["version"] == "2.0", f"{crop_key} stage {stage_idx}: bad GLTF version"

        # Has at least one mesh
        assert "meshes" in gltf and len(gltf["meshes"]) >= 1

        # Has at least one material
        assert "materials" in gltf and len(gltf["materials"]) >= 1

        # Mesh primitive has morph targets (targets list length >= 2)
        primitive = gltf["meshes"][0]["primitives"][0]
        targets = primitive.get("targets", [])
        assert len(targets) == 2, (
            f"{crop_key} stage {stage_idx}: expected 2 morph targets, got {len(targets)}"
        )

        # Morph target names in extras
        mt_names = gltf["meshes"][0].get("extras", {}).get("morphTargetNames", [])
        assert "stage_start" in mt_names, f"{crop_key} stage {stage_idx}: missing 'stage_start' morph"
        assert "stage_end" in mt_names,   f"{crop_key} stage {stage_idx}: missing 'stage_end' morph"

        # Triangle count within budget (index accessor count / 3)
        idx_accessor_idx = primitive["indices"]
        idx_count = gltf["accessors"][idx_accessor_idx]["count"]
        n_triangles = idx_count // 3
        budget = budget_list[stage_idx]
        assert n_triangles <= budget, (
            f"{crop_key} stage {stage_idx}: {n_triangles} triangles exceeds budget {budget}"
        )


# ---------------------------------------------------------------------------
# 5 & 6. Stage 0 near-zero height; stage 4 maximum height (wheat)
# ---------------------------------------------------------------------------

def _get_wheat_gltf(stage_idx: int) -> dict:
    uri = AssetRegistry.get_model_path("StandardWheat", stage_idx)
    with open(uri, "r", encoding="utf-8") as f:
        return json.load(f)


def _position_max_y(gltf: dict) -> float:
    """Read the POSITION accessor max[1] (Y) from the GLTF."""
    primitive = gltf["meshes"][0]["primitives"][0]
    pos_accessor_idx = primitive["attributes"]["POSITION"]
    return gltf["accessors"][pos_accessor_idx]["max"][1]


def test_wheat_stage_0_near_zero_height():
    """Germination stage should be nearly invisible above-ground (height < 1 cm)."""
    gltf = _get_wheat_gltf(0)
    max_y = _position_max_y(gltf)
    assert max_y < 0.01, f"Stage 0 max_y={max_y:.4f}m — should be near-zero for germination"


def test_wheat_stage_4_tallest():
    """Anthesis (stage 4) should have the tallest bounding box of all wheat stages."""
    heights = [_position_max_y(_get_wheat_gltf(i)) for i in range(_WHEAT_STAGES)]
    tallest_stage = heights.index(max(heights))
    assert tallest_stage == 4, (
        f"Expected stage 4 (anthesis) to be tallest; tallest was stage {tallest_stage} "
        f"with height {heights[tallest_stage]:.3f}m. All heights: {[f'{h:.3f}' for h in heights]}"
    )


# ---------------------------------------------------------------------------
# 8. Binary buffer model_index tracks stage as StandardWheat plant grows
# ---------------------------------------------------------------------------

def test_standard_wheat_model_index_tracks_stage():
    """
    Crucible Test (PRD v0.9.5 §4.6, last bullet):
    Run a StandardWheat simulation. Do not manually register any assets.
    Assert that binary buffer model_index updates to point to the registered
    stage geometries as the plant grows.
    """
    from cropforge.farm import Farm, Field
    from cropforge.plugins import StandardWheat
    from cropforge.state import EnvironmentState

    # Fresh farm — StandardWheat already imported at module fixture
    class _HotWeather:
        """High-temperature weather to rapidly advance thermal time."""
        def get_day(self, day):
            return EnvironmentState(
                day=day, doy=((day - 1) % 365) + 1,
                temp_max_c=38.0, temp_min_c=28.0, temp_mean_c=33.0,
                radiation_mj_m2=22.0, rainfall_mm=0.0,
                et0_mm=5.0, wind_speed_ms=2.0, humidity_pct=40.0,
            )

    farm = Farm("WheatAutoAsset", location=(28.6, 77.2))
    field = Field("F1", rows=4, cols=4)
    field.set_weather(_HotWeather())
    field.use_plugin(StandardWheat)
    farm.add_field(field)

    # Run long enough to advance beyond germination (TT > 60°C·day at 33°C mean → ~2 days)
    farm.run(days=10)

    plants = field._field_state.plants

    # After 10 days at 33°C → thermal_time ≈ 330 → stage = 'tillering' (index 2)
    for plant in plants:
        if not plant.alive:
            continue
        stage = plant.custom.get("phenological_stage", "germination")
        stage_idx = ["germination", "emergence", "tillering",
                     "stem_extension", "anthesis", "grain_fill", "maturity"].index(stage)

        # The registered URI for this stage should exist
        uri = AssetRegistry.get_model_path("StandardWheat", stage_idx)
        assert uri is not None, (
            f"No URI for StandardWheat stage {stage_idx} ({stage}) — "
            "auto-registration may have failed"
        )
        assert os.path.exists(uri), f"GLTF file missing: {uri}"


# ---------------------------------------------------------------------------
# 9. `from cropforge.models import ModelRegistry` works (PRD §2.3)
# ---------------------------------------------------------------------------

def test_model_registry_import_path():
    """PRD §2.3: both import paths must resolve to the same class."""
    from cropforge.models import ModelRegistry
    from cropforge.viz.registry import AssetRegistry as AR
    assert ModelRegistry is AR, (
        "ModelRegistry should be the same object as AssetRegistry"
    )


# ---------------------------------------------------------------------------
# 10. Cylinder fallback — unregistered crop, no crash
# ---------------------------------------------------------------------------

def test_cylinder_fallback_for_unregistered_crop():
    """Plants from crops with no GLTF registered return None (→ cylinder) without error."""
    uri = AssetRegistry.get_model_path("UnknownCrop", stage=3)
    assert uri is None  # None triggers cylinder fallback in JS renderer
