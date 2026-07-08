"""
tests/test_viz_buffer.py
========================
Crucible tests for Phase 2: Binary Buffer Expansion & Stage Animation.

Verifies:
  - stage_progress on PlantState defaults to 0.0
  - wheat/maize _get_stage_progress maths
  - BufferStore now packs 14 floats per plant
  - model_index_map present in BufferStore.meta
  - stage_progress values > 0.0 after thermal accumulation
  - buffer length == n_plants * 14 * 4 bytes
"""
from __future__ import annotations

import struct
import math

import pytest


# ---------------------------------------------------------------------------
# 1. PlantState defaults
# ---------------------------------------------------------------------------

def test_plant_state_stage_progress_default():
    from cropforge.state import PlantState
    p = PlantState(plant_id="p0", row=0, col=0)
    assert p.stage_progress == 0.0


# ---------------------------------------------------------------------------
# 2. _get_stage_progress math (wheat)
# ---------------------------------------------------------------------------

def test_wheat_stage_progress_germination_start():
    from cropforge.plugins.wheat import _get_stage_progress
    # At TT=0 we're at the very start of germination (0/60 = 0.0)
    assert _get_stage_progress(0.0, "germination") == pytest.approx(0.0, abs=1e-6)


def test_wheat_stage_progress_germination_half():
    from cropforge.plugins.wheat import _get_stage_progress
    # germination span: 0→60, halfway = 30 TT → 0.5
    assert _get_stage_progress(30.0, "germination") == pytest.approx(0.5, abs=1e-6)


def test_wheat_stage_progress_clamped_at_one():
    from cropforge.plugins.wheat import _get_stage_progress
    # Maturity is the final stage; any TT beyond start should clamp ≤ 1.0
    val = _get_stage_progress(9999.0, "maturity")
    assert val == pytest.approx(1.0)


def test_maize_stage_progress_vegetative():
    from cropforge.plugins.maize import _get_stage_progress
    # vegetative span: 250→600, at TT=425 → 0.5
    assert _get_stage_progress(425.0, "vegetative") == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. BufferStore: 14 floats per plant
# ---------------------------------------------------------------------------

def test_buffer_floats_per_plant_constant():
    from cropforge.viz.buffers import FLOATS_PER_PLANT, BYTES_PER_PLANT
    assert FLOATS_PER_PLANT == 14
    assert BYTES_PER_PLANT == 56


def test_buffer_frame_length_is_14_floats():
    """BufferStore.build() -> n_plants * 14 * 4 bytes per frame."""
    import pandas as pd
    from cropforge.viz.buffers import BufferStore

    rows, cols = 3, 3
    n_plants = rows * cols
    records = [
        {
            "day": 1, "field_name": "F", "plant_id": f"p{r}{c}",
            "row": r, "col": c,
            "height_cm": 10.0 + r, "biomass_g": 5.0 + c,
            "lai": 0.5, "alive": True,
            "model_id": "", "stage_progress": 0.3,
        }
        for r in range(rows) for c in range(cols)
    ]
    df = pd.DataFrame(records)

    store = BufferStore(field_name="F")
    store.build(df, variable="biomass_g")

    frame = store.get_frame(1)
    assert frame is not None
    assert len(frame) == n_plants * 14 * 4


# ---------------------------------------------------------------------------
# 4. model_index_map in meta
# ---------------------------------------------------------------------------

def test_buffer_meta_has_model_index_map_empty():
    """No model_id column → empty map in meta."""
    import pandas as pd
    from cropforge.viz.buffers import BufferStore

    records = [
        {"day": 1, "field_name": "F", "plant_id": "p00", "row": 0, "col": 0,
         "height_cm": 5.0, "biomass_g": 2.0, "lai": 0.3, "alive": True,
         "model_id": "", "stage_progress": 0.0}
    ]
    df = pd.DataFrame(records)
    store = BufferStore("F")
    store.build(df)
    assert "model_index_map" in store.meta
    assert store.meta["model_index_map"] == {}
    assert store.meta["buffer_fields"][11] == "morph_weight"
    assert store.meta["buffer_fields"][12] == "stress_ks"
    assert store.meta["buffer_fields"][13] == "disease_severity"


def test_buffer_meta_model_index_map_populated():
    """model_id strings get unique ascending ints (1-based); empty → 0."""
    import pandas as pd
    from cropforge.viz.buffers import BufferStore

    records = [
        {"day": 1, "field_name": "F", "plant_id": f"p{i}", "row": 0, "col": i,
         "height_cm": 5.0, "biomass_g": 1.0, "lai": 0.2, "alive": True,
         "model_id": uri, "stage_progress": 0.1}
        for i, uri in enumerate(["assets/wheat_a.gltf", "", "assets/wheat_b.gltf"])
    ]
    df = pd.DataFrame(records)
    store = BufferStore("F")
    store.build(df)

    m = store.meta["model_index_map"]
    # Two non-empty URIs → indices 1 and 2 (sorted order)
    assert set(m.values()) == {1, 2}
    assert "assets/wheat_a.gltf" in m
    assert "assets/wheat_b.gltf" in m
    assert "" not in m  # empty string is NOT in the map


def test_buffer_model_index_zero_for_cylinder_fallback():
    """Plants with model_id='' must pack 0.0 at offset [9]."""
    import pandas as pd
    from cropforge.viz.buffers import BufferStore

    records = [
        {"day": 1, "field_name": "F", "plant_id": "p00", "row": 0, "col": 0,
         "height_cm": 5.0, "biomass_g": 1.0, "lai": 0.2, "alive": True,
         "model_id": "", "stage_progress": 0.25}
    ]
    df = pd.DataFrame(records)
    store = BufferStore("F")
    store.build(df)

    frame = store.get_frame(1)
    floats = struct.unpack(f"{14}f", frame)
    assert floats[9] == pytest.approx(0.0)   # model_index = 0 (cylinder)
    assert floats[10] == pytest.approx(0.25)  # stage_progress packed
    assert floats[11] == pytest.approx(0.25)  # morph_weight defaults to stage_progress
    assert floats[12] == pytest.approx(1.0)   # no stress logged => unstressed
    assert floats[13] == pytest.approx(0.0)   # no disease logged => healthy


# ---------------------------------------------------------------------------
# 5. Crucible: 30-day StandardWheat run → stage_progress in [0, 1]
# ---------------------------------------------------------------------------

def test_crucible_wheat_30day_stage_progress_in_buffer():
    """
    Run StandardWheat for 30 days, build a BufferStore from the parquet log,
    and verify:
      - model_index_map key exists in meta
      - buffer length = n_plants × 11 × 4 for day 15
      - at least some plants have stage_progress > 0.0 and < 1.0
    """
    import pyarrow.parquet as pq
    import pandas as pd
    from cropforge.farm import Farm, Field
    from cropforge.plugins import StandardWheat
    from cropforge.viz.buffers import BufferStore

    farm = Farm("BufferCrucible", location=(28.6, 77.2))
    field = Field("BF", rows=4, cols=4)
    farm.add_field(field)

    from cropforge.state import EnvironmentState

    class _Warm:
        def get_day(self, day):
            return EnvironmentState(
                day=day, doy=((day - 1) % 365) + 1,
                temp_max_c=28.0, temp_min_c=16.0, temp_mean_c=22.0,
                radiation_mj_m2=20.0, rainfall_mm=0.0,
                et0_mm=4.0, wind_speed_ms=2.0, humidity_pct=55.0,
            )

    field.set_weather(_Warm())
    field.use_plugin(StandardWheat)
    farm.run(days=30)

    # Read the Parquet log
    log_dir = farm._last_log_path
    assert log_dir is not None
    df = pq.read_table(log_dir + "/plants").to_pandas()

    assert "stage_progress" in df.columns, "stage_progress column missing from Parquet"

    # Build a BufferStore from the run data
    store = BufferStore("BF")
    store.build(df[df["field_name"] == "BF"].copy())

    # Day 15 frame
    frame = store.get_frame(15)
    assert frame is not None, "Day 15 frame missing"

    n_plants = store.n_plants
    assert len(frame) == n_plants * 14 * 4, (
        f"Expected {n_plants * 14 * 4} bytes, got {len(frame)}"
    )

    # model_index_map in meta
    assert "model_index_map" in store.meta

    # Unpack all stage_progress values for day 15
    floats_per = 14
    all_floats = struct.unpack(f"{n_plants * floats_per}f", frame)
    stage_progresses = [all_floats[i * floats_per + 10] for i in range(n_plants)]

    # After 15 days × 22°C/day = 330 TT — plants should be in tillering stage
    # (tillering starts at TT=200, ends at TT=500) → stage_progress ≈ 0.43
    # Any value strictly in (0.0, 1.0) is a valid pass
    assert any(0.0 < sp < 1.0 for sp in stage_progresses), (
        f"Expected some stage_progress in (0,1), got: {stage_progresses}"
    )
