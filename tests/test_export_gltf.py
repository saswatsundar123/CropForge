"""
tests/test_export_gltf.py
=========================
Crucible tests for Phase 4: GLTF Scene Export.

Verifies:
  - export_scene() writes a .glb to disk without crashing
  - written file is > 0 bytes
  - farm.export_scene() API delegates correctly
  - export raises ValueError for missing day
  - pygltflib is importable (dependency check)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Warm:
    """Minimal weather stub — warm enough for wheat germination."""
    def get_day(self, day):
        from cropforge.state import EnvironmentState
        return EnvironmentState(
            day=day, doy=((day - 1) % 365) + 1,
            temp_max_c=28.0, temp_min_c=16.0, temp_mean_c=22.0,
            radiation_mj_m2=20.0, rainfall_mm=0.0,
            et0_mm=4.0, wind_speed_ms=2.0, humidity_pct=55.0,
        )


def _small_wheat_farm(tmpdir):
    """Run StandardWheat on a 5×5 sloped terrain for 10 days."""
    from cropforge import Farm, Field, Terrain
    from cropforge.plugins import StandardWheat

    farm = Farm("GLB_Test", location=(28.6, 77.2))
    field = Field("F_export", rows=5, cols=5)
    field.set_weather(_Warm())
    field.set_terrain(
        Terrain.procedural(rows=5, cols=5, generator="undulating",
                           amplitude_m=0.5, resolution_m=1.0, seed=99)
    )
    field.use_plugin(StandardWheat)
    farm.add_field(field)
    farm.run(days=10)
    return farm


# ---------------------------------------------------------------------------
# 1. pygltflib dependency check
# ---------------------------------------------------------------------------

def test_pygltflib_importable():
    import pygltflib
    assert hasattr(pygltflib, "GLTF2")


# ---------------------------------------------------------------------------
# 2. Crucible: export_scene writes a valid .glb
# ---------------------------------------------------------------------------

def test_export_scene_creates_file(tmp_path):
    """Main crucible — run 10-day wheat sim, export day=10, assert .glb exists."""
    farm = _small_wheat_farm(tmp_path)
    out = tmp_path / "test_export.glb"
    result = farm.export_scene(day=10, filepath=str(out))
    assert result.exists(), f".glb not found at {result}"
    assert result.stat().st_size > 0, ".glb is empty"


def test_export_scene_glb_header(tmp_path):
    """GLB files must start with magic bytes 0x46546C67 ('glTF')."""
    farm = _small_wheat_farm(tmp_path)
    out = tmp_path / "header_check.glb"
    farm.export_scene(day=5, filepath=str(out))
    with open(out, "rb") as f:
        magic = f.read(4)
    assert magic == b"glTF", f"Bad GLB magic: {magic!r}"


def test_export_scene_midday(tmp_path):
    """Export a mid-run day (day=5) — should not crash."""
    farm = _small_wheat_farm(tmp_path)
    out = tmp_path / "day5.glb"
    result = farm.export_scene(day=5, filepath=str(out))
    assert result.stat().st_size > 0


def test_export_scene_uses_logged_terrain_elevation(tmp_path):
    """Exported terrain vertices must preserve the non-flat logged terrain."""
    import numpy as np
    import pygltflib

    farm = _small_wheat_farm(tmp_path)
    out = tmp_path / "terrain.glb"
    farm.export_scene(day=5, filepath=str(out))

    gltf = pygltflib.GLTF2().load_binary(str(out))
    blob = gltf.binary_blob()
    pos_accessor = gltf.accessors[0]
    pos_view = gltf.bufferViews[pos_accessor.bufferView]
    offset = (pos_view.byteOffset or 0) + (pos_accessor.byteOffset or 0)
    terrain_positions = np.frombuffer(
        blob,
        dtype=np.float32,
        count=25 * 3,
        offset=offset,
    ).reshape(25, 3)

    assert not np.allclose(terrain_positions[:, 1], 0.0), (
        "GLB terrain vertices are flat; expected logged terrain elevation"
    )


def test_export_animation_creates_glb(tmp_path):
    """farm.export_animation() should exist and write a valid .glb."""
    farm = _small_wheat_farm(tmp_path)
    out = tmp_path / "season.glb"
    result = farm.export_animation(days=range(1, 4), filepath=str(out), fps=4)
    assert result.exists()
    with open(out, "rb") as f:
        assert f.read(4) == b"glTF"


# ---------------------------------------------------------------------------
# 3. Error paths
# ---------------------------------------------------------------------------

def test_export_scene_bad_day(tmp_path):
    """Requesting a day not in the log raises ValueError."""
    farm = _small_wheat_farm(tmp_path)
    out = tmp_path / "bad.glb"
    with pytest.raises(ValueError, match="not in log"):
        farm.export_scene(day=999, filepath=str(out))


def test_export_scene_no_run():
    """export_scene without a prior run raises CropForgeVisualizeError."""
    from cropforge import Farm, Field
    from cropforge.runtime import CropForgeVisualizeError
    farm = Farm("NoRun")
    with pytest.raises(CropForgeVisualizeError):
        farm.export_scene(day=1)


# ---------------------------------------------------------------------------
# 4. Low-level exporter
# ---------------------------------------------------------------------------

def test_export_gltf_direct(tmp_path):
    """Call cropforge.export_gltf.export_scene directly — same result."""
    from cropforge.export_gltf import export_scene as raw_export
    farm = _small_wheat_farm(tmp_path)
    out = tmp_path / "raw.glb"
    result = raw_export(log_path=farm._last_log_path, day=10, filepath=str(out))
    assert result.exists()
    assert result.stat().st_size > 0
