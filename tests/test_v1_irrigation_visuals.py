from __future__ import annotations

import json

import pyarrow.parquet as pq
from fastapi.testclient import TestClient


def _dummy_wsgi_app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"CropForge test app"]


def test_irrigation_event_logs_sprinkler_path_and_api_payload():
    from cropforge.events import Event
    from cropforge.farm import Farm, Field
    from cropforge.viz.server import create_fastapi_app

    rows, cols = 5, 6
    farm = Farm("IrrigationVisualCrucible")
    field = Field("IrrigatedPlot", rows=rows, cols=cols)
    farm.add_field(field)
    farm.add_event(Event.irrigation(
        field="IrrigatedPlot",
        interval_days=999,
        amount_mm=15.0,
        start_day=10,
        end_day=10,
    ))
    farm.run(days=10)

    assert farm._last_log_path is not None
    machinery = pq.read_table(farm._last_log_path + "/machinery").to_pandas()
    rec = machinery[
        (machinery["field_name"] == "IrrigatedPlot")
        & (machinery["day"].astype(int) == 10)
    ].iloc[0]
    assert rec["event_name"] == "irrigation"
    assert rec["machine_type"] == "sprinkler"

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

    response = client.get("/api/buffer/day/10?field=IrrigatedPlot")
    assert response.status_code == 200
    payload = response.json()
    assert payload["day"] == 10
    assert payload["field_name"] == "IrrigatedPlot"
    assert payload["machinery"]
    assert payload["machinery"][0]["machine_type"] == "sprinkler"
    assert payload["machinery"][0]["path"] == path


def test_frontend_contains_enhanced_sprinkler_particles():
    js = open("cropforge/viz/static/main.js", encoding="utf-8").read()

    assert "_ensureSprinklerParticles" in js
    assert "_emitSprinklerBurst" in js
    assert "machineType === 'sprinkler'" in js
    assert "if (!qualityEnhanced)" in js
    assert "new THREE.PointsMaterial" in js


def test_yield_dashboard_panel_is_present():
    app_py = open("cropforge/viz/app.py", encoding="utf-8").read()

    assert "Yield Metrics" in app_py
    assert "calculate-yield-btn" in app_py
    assert "yield-summary-panel" in app_py
    assert "build_yield_summary" in app_py
