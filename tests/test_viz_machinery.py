"""
tests/test_viz_machinery.py
===========================
Crucible tests for v0.9.5 Phase 4 machinery path logging and viewport payloads.
"""
from __future__ import annotations

import json

import pyarrow.parquet as pq
from fastapi.testclient import TestClient


def _dummy_wsgi_app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"CropForge test app"]


def test_tillage_event_logs_machinery_path_and_api_payload():
    from cropforge.events import Event
    from cropforge.farm import Farm, Field
    from cropforge.viz.server import create_fastapi_app

    rows, cols = 4, 5
    farm = Farm("MachineryCrucible")
    field = Field("MachinePlot", rows=rows, cols=cols)
    farm.add_field(field)
    farm.add_event(Event.tillage(field="MachinePlot", day=5))
    farm.run(days=5)

    assert farm._last_log_path is not None
    machinery = pq.read_table(farm._last_log_path + "/machinery").to_pandas()
    rec = machinery[
        (machinery["field_name"] == "MachinePlot")
        & (machinery["day"].astype(int) == 5)
    ].iloc[0]
    assert rec["machine_type"] == "tractor"

    path = json.loads(rec["path_json"])
    assert len(path) >= rows * 2
    assert all(0.0 <= point[0] <= cols - 1 for point in path)
    assert all(0.0 <= point[1] <= rows - 1 for point in path)

    plants = pq.read_table(farm._last_log_path + "/plants").to_pandas()
    app = create_fastapi_app(
        dash_app=_dummy_wsgi_app,
        log_path=farm._last_log_path,
        cropforge_version="test",
        plants_df=plants,
    )
    client = TestClient(app)

    response = client.get("/api/buffer/day/5?field=MachinePlot")
    assert response.status_code == 200
    payload = response.json()
    assert payload["day"] == 5
    assert payload["field_name"] == "MachinePlot"
    assert payload["machinery"]
    assert payload["machinery"][0]["machine_type"] == "tractor"
    assert payload["machinery"][0]["path"] == path


def test_frontend_contains_machinery_animation_layer():
    js = open("cropforge/viz/static/main.js", encoding="utf-8").read()

    assert "dayMeta" in js
    assert "/api/buffer/day/${day}" in js
    assert "_animateMachineryForDay(day)" in js
    assert "new THREE.BoxGeometry(0.8, 0.35, 0.45)" in js
    assert "requestAnimationFrame(tick)" in js
    assert "machine.castShadow = qualityEnhanced" in js
