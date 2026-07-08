"""
tests/test_viz_weather.py
=========================
Crucible tests for v0.9.5 Phase 5 daily weather metadata and rain particles.
"""
from __future__ import annotations

import pyarrow.parquet as pq
from fastapi.testclient import TestClient


def _dummy_wsgi_app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"CropForge test app"]


class _RainWeather:
    def get_day(self, day: int):
        from cropforge.state import EnvironmentState

        return EnvironmentState(
            day=day,
            doy=day,
            temp_max_c=28.0,
            temp_min_c=18.0,
            temp_mean_c=23.0,
            radiation_mj_m2=20.0,
            rainfall_mm=20.0 if day == 3 else 0.0,
            et0_mm=4.0,
            wind_speed_ms=1.5,
            humidity_pct=75.0,
        )


def test_daily_precipitation_metadata_reaches_buffer_day_payload():
    from cropforge.farm import Farm, Field
    from cropforge.viz.server import create_fastapi_app

    farm = Farm("WeatherCrucible")
    field = Field("RainPlot", rows=3, cols=3)
    field.set_weather(_RainWeather())
    farm.add_field(field)
    farm.run(days=4)

    assert farm._last_log_path is not None
    environment = pq.read_table(farm._last_log_path + "/environment").to_pandas()
    day3 = environment[
        (environment["field_name"] == "RainPlot")
        & (environment["day"].astype(int) == 3)
    ].iloc[0]
    day4 = environment[
        (environment["field_name"] == "RainPlot")
        & (environment["day"].astype(int) == 4)
    ].iloc[0]
    assert float(day3["rainfall_mm"]) == 20.0
    assert float(day4["rainfall_mm"]) == 0.0

    plants = pq.read_table(farm._last_log_path + "/plants").to_pandas()
    app = create_fastapi_app(
        dash_app=_dummy_wsgi_app,
        log_path=farm._last_log_path,
        cropforge_version="test",
        plants_df=plants,
    )
    client = TestClient(app)

    response_day3 = client.get("/api/buffer/day/3?field=RainPlot")
    response_day4 = client.get("/api/buffer/day/4?field=RainPlot")
    assert response_day3.status_code == 200
    assert response_day4.status_code == 200

    assert response_day3.json()["precipitation_mm"] == 20.0
    assert response_day4.json()["precipitation_mm"] == 0.0


def test_frontend_contains_enhanced_only_rain_particle_system():
    js = open("cropforge/viz/static/main.js", encoding="utf-8").read()

    assert "const _RAIN_PARTICLE_COUNT = 6000" in js
    assert "const _RAIN_HIDE_THRESHOLD_MM = 2.0" in js
    assert "if (qualityEnhanced) initRainSystem(fieldW, fieldD)" in js
    assert "new THREE.PointsMaterial" in js
    assert "new THREE.Points(geo, mat)" in js
    assert "dayMeta[day]?.precipitation_mm" in js
    assert "rainSystem.geometry.setDrawRange(0, rainActiveCount)" in js
    assert "_animateRain()" in js
    assert "rainSystem.visible = false" in js
