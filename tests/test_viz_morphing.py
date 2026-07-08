"""
tests/test_viz_morphing.py
==========================
Crucible tests for v0.9.5 Phase 2 morph-target interpolation.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pandas as pd
import pytest


def test_wheat_mid_vegetative_morph_weight_transmitted():
    """A wheat plant halfway through tillering packs morph_weight=0.5."""
    from cropforge.plugins.wheat import _get_stage_progress
    from cropforge.viz.buffers import BYTES_PER_PLANT, FLOATS_PER_PLANT, BufferStore

    thermal_time = 350.0  # wheat tillering span is 200 -> 500 degree-days
    stage = "tillering"
    stage_progress = _get_stage_progress(thermal_time, stage)
    assert stage_progress == pytest.approx(0.5)

    df = pd.DataFrame([
        {
            "day": 1,
            "field_name": "MorphField",
            "plant_id": "wheat_50pct",
            "row": 0,
            "col": 0,
            "height_cm": 22.0,
            "biomass_g": 9.0,
            "lai": 1.4,
            "alive": True,
            "phenological_stage": stage,
            "model_id": "standard_wheat/stage_2_tillering.gltf",
            "stage_progress": stage_progress,
            "custom_json": json.dumps({
                "thermal_time": thermal_time,
                "phenological_stage": stage,
                "water_stress_ks": 0.8,
            }),
        }
    ])

    store = BufferStore("MorphField")
    store.build(df)

    frame = store.get_frame(1)
    assert frame is not None
    assert len(frame) == BYTES_PER_PLANT

    floats = struct.unpack(f"{FLOATS_PER_PLANT}f", frame)
    assert FLOATS_PER_PLANT == 14
    assert BYTES_PER_PLANT == 56
    assert store.meta["buffer_fields"][11] == "morph_weight"
    assert store.meta["buffer_fields"][12] == "stress_ks"
    assert store.meta["buffer_fields"][13] == "disease_severity"
    assert floats[10] == pytest.approx(0.5)
    assert floats[11] == pytest.approx(0.5)
    assert floats[12] == pytest.approx(0.8)
    assert floats[13] == pytest.approx(0.0)


def test_viewport_reads_morph_fields_and_guards_missing_targets():
    """The browser contract includes morph fields and a no-target shader guard."""
    js = Path("cropforge/viz/static/main.js").read_text(encoding="utf-8")

    assert "const FLOATS = 14" in js
    assert "frame[base + 11]" in js
    assert "frame[base + 12]" in js
    assert "cfMorphWeight" in js
    assert "cfWiltWeight" in js
    assert "const hasMorphTargets = morphPositions.length >= 2" in js
    assert "!hasMorphTargets && !useDiseaseShader" in js
    assert "cfMorphTargetsEnabled = false" in js
    assert "cfWiltUsesTarget" in js
    assert "smoothstep(0.05, 1.0, position.y)" in js
