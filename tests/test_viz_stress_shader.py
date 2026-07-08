"""
tests/test_viz_stress_shader.py
================================
Crucible tests for v0.9.5 Phase 3 disease stress visual expression.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pyarrow.parquet as pq
import pytest


def test_disease_engine_day20_severity_reaches_buffer_slot():
    """Disease severity logged from the disease engine is packed at float slot 13."""
    from cropforge.farm import Farm, Field
    from cropforge.viz.buffers import BYTES_PER_PLANT, FLOATS_PER_PLANT, BufferStore

    farm = Farm("DiseaseShaderCrucible")
    field = Field("DiseasePlot", rows=3, cols=3)
    farm.add_field(field)
    farm.use_physics(
        disease=True,
        disease_foci=[(0, 0)],
        disease_spread_rate=0.0,
        disease_latency_days=0,
        disease_stress_increment=0.04,
        disease_seed=42,
    )

    farm.run(days=20)
    assert farm._last_log_path is not None

    df = pq.read_table(farm._last_log_path + "/plants").to_pandas()
    assert "disease_severity" in df.columns

    infected = df[
        (df["field_name"] == "DiseasePlot")
        & (df["day"] == 20)
        & (df["row"] == 0)
        & (df["col"] == 0)
    ].iloc[0]
    assert infected["disease_severity"] > 0.5

    store = BufferStore("DiseasePlot")
    store.build(df[df["field_name"] == "DiseasePlot"].copy())

    frame = store.get_frame(20)
    assert frame is not None
    assert FLOATS_PER_PLANT == 14
    assert BYTES_PER_PLANT == 56
    assert store.meta["buffer_fields"][13] == "disease_severity"
    assert len(frame) == store.n_plants * BYTES_PER_PLANT

    floats = struct.unpack(f"{store.n_plants * FLOATS_PER_PLANT}f", frame)
    plant_index = 0 * store.cols + 0
    disease_offset = plant_index * FLOATS_PER_PLANT + 13
    assert floats[disease_offset] == pytest.approx(infected["disease_severity"])
    assert floats[disease_offset] > 0.5


def test_enhanced_fragment_shader_injects_necrosis_blend():
    """The enhanced shader path carries per-instance disease severity to fragment mix()."""
    js = Path("cropforge/viz/static/main.js").read_text(encoding="utf-8")

    assert "const FLOATS = 14" in js
    assert "frame[base + 13]" in js
    assert "cfDiseaseSeverity" in js
    assert "vCfDiseaseSeverity" in js
    assert "CF_USE_DISEASE_SHADER" in js
    assert "qualityEnhanced" in js
    assert "vec3(0.55, 0.35, 0.10)" in js
    assert "mix(diffuseColor.rgb, cfNecroticColor" in js
