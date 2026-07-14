"""
cropforge/viz/server.py
=======================
FastAPI server bootstrap for the CropForge dashboard.

PRD References:
    Section 7.1  — Launch behaviour: local FastAPI server, port 7860,
                   opens default browser, shuts down on terminal interrupt.
    Section 7.3  — Binary Float32Array data serving: Parquet read-once at
                   startup, all days packed into FIELD_REGISTRY; per-day
                   slices served as raw binary (no JSON).
    PRD v0.2.0   — Multi-field: /api/buffer/meta?field=<n> and
                   /api/buffer?day=<d>&field=<n> return per-field data.
                   Default field = first field (alphabetically by name).

Endpoints:
    GET /api/health                         → {status, version}
    GET /api/log_path                       → {log_path}
    GET /api/fields                         → {fields: [...], default_field: ...}
    GET /api/buffer/meta[?field=<name>]     → JSON metadata (n_plants, n_days …)
    GET /api/buffer?day=<d>[&field=<name>]  → raw binary bytes (Float32Array)
    GET /api/buffer/rebuild?variable=<v>[&field=<name>]
                                            → rebuild colour mapping
    (all other routes)                      → Dash WSGI app

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

HOST     = "127.0.0.1"
PORT     = 7860
BASE_URL = f"http://localhost:{PORT}"

# Path to the static/ directory (index.html + main.js live here)
_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_fastapi_app(
    dash_app,
    log_path: str,
    cropforge_version: str,
    plants_df=None,
    quality: str = "standard",
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    dash_app:
        The Plotly Dash ``app.server`` (WSGI callable).
    log_path:
        Absolute path to the Parquet session directory.
    cropforge_version:
        CropForge version string.
    plants_df:
        Full plants DataFrame — used to build FIELD_REGISTRY.
    quality:
        ``"standard"`` or ``"enhanced"`` — injected into every BufferStore.meta
        so the JS frontend can read ``meta.quality_mode`` from /api/buffer/meta.
    """
    from cropforge.viz.buffers import FIELD_REGISTRY

    # ---- Pre-build per-field binary frames at startup ------------------
    if plants_df is not None and not plants_df.empty:
        FIELD_REGISTRY.build_all(plants_df, variable="biomass_g")
        # Inject quality_mode into every store's meta so JS can read it
        for store in FIELD_REGISTRY._stores.values():
            store._meta["quality_mode"] = quality
    else:
        logger.warning("No plants data available — FIELD_REGISTRY will be empty.")

    # ---- Create the FastAPI app ----------------------------------------
    api = FastAPI(
        title="CropForge Dashboard API",
        description="Local-only API serving the CropForge simulation dashboard.",
        version=cropforge_version,
        docs_url="/api/docs",
        redoc_url=None,
    )

    # ---- CORS headers for the iframe postMessage bridge ----------------
    from fastapi.middleware.cors import CORSMiddleware
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # API endpoints (must be registered BEFORE mounting WSGI)
    # ------------------------------------------------------------------

    @api.get("/api/health")
    async def health():
        return {"status": "ok", "version": cropforge_version}

    @api.get("/api/log_path")
    async def get_log_path():
        return {"log_path": log_path}

    @api.get("/api/fields")
    async def list_fields():
        """Return available field names and the default field.

        The Three.js client and Dash frontend use this to populate
        the Field Selector dropdown and request the correct buffer.
        """
        field_names = FIELD_REGISTRY.field_names
        default = field_names[0] if field_names else ""
        return {"fields": field_names, "default_field": default}

    @api.get("/api/buffer/meta")
    async def buffer_meta(
        field: Optional[str] = Query(None, description="Field name (default: first field)"),
    ):
        """Return JSON metadata describing the binary buffer layout for a field.

        The Three.js client fetches this first to know n_plants, grid size,
        and which days are available before issuing per-day requests.

        Query Parameters
        ----------------
        field : str, optional
            Field name as stored in the Parquet ``field_name`` column.
            Defaults to the first field (alphabetical) if omitted.
        """
        store = FIELD_REGISTRY.get(field)
        if store is None or not store.is_ready:
            available = FIELD_REGISTRY.field_names
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Buffer store not ready for field '{field}'. "
                    f"Available fields: {available}"
                ),
            )
        return JSONResponse(content=store.meta)

    @api.get("/api/buffer")
    async def buffer_day(
        day:   int           = Query(..., description="Simulation day (1-indexed)"),
        field: Optional[str] = Query(None, description="Field name (default: first field)"),
    ):
        """Return the binary Float32Array frame for a single simulation day.

        PRD Section 7.3:
            Per-timestep field state is served as flat binary Float32Array
            buffers … packed in the order that Three.js InstancedMesh
            attribute arrays expect.

        PRD v0.2.0 Multi-field:
            Pass ?field=<name> to get data for a specific field.
            Default is the first available field.

        Response Content-Type: application/octet-stream
        Body: n_plants x 56 bytes (14 float32 per plant)
              Layout is described by /api/buffer/meta["buffer_fields"].
        """
        store = FIELD_REGISTRY.get(field)
        if store is None or not store.is_ready:
            available = FIELD_REGISTRY.field_names
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Buffer store not ready for field '{field}'. "
                    f"Available fields: {available}"
                ),
            )
        frame = store.get_frame(day)
        if frame is None:
            raise HTTPException(
                status_code=404,
                detail=f"Day {day} not in buffer for field '{store.field_name}'.",
            )
        return Response(
            content=frame,
            media_type="application/octet-stream",
            headers={
                "Content-Length":   str(len(frame)),
                "X-CropForge-Day":  str(day),
                "X-N-Plants":       str(store.n_plants),
                "X-Field-Name":     store.field_name,
                "Cache-Control":    "no-cache",
            },
        )

    @api.get("/api/buffer/day/{day}")
    async def buffer_day_payload(
        day: int,
        field: Optional[str] = Query(None, description="Field name (default: first field)"),
    ):
        """Return JSON metadata for a day, including any machinery path."""
        import json as _json
        import pyarrow.parquet as _pq

        store = FIELD_REGISTRY.get(field)
        if store is None or not store.is_ready:
            available = FIELD_REGISTRY.field_names
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Buffer store not ready for field '{field}'. "
                    f"Available fields: {available}"
                ),
            )

        precipitation_mm = 0.0
        environment_dir = Path(log_path) / "environment"
        if environment_dir.exists():
            try:
                env_df = _pq.read_table(str(environment_dir)).to_pandas()
                if not env_df.empty:
                    day_series = env_df["day"].astype(int)
                    mask = (day_series == int(day)) & (env_df["field_name"] == store.field_name)
                    if mask.any():
                        rec = env_df[mask].iloc[0]
                        precipitation_mm = float(rec.get("rainfall_mm", 0.0) or 0.0)
            except Exception:
                logger.exception("Failed to read weather metadata from %s", environment_dir)

        machinery = []
        machinery_dir = Path(log_path) / "machinery"
        if machinery_dir.exists():
            try:
                df = _pq.read_table(str(machinery_dir)).to_pandas()
                if not df.empty:
                    day_series = df["day"].astype(int)
                    mask = (day_series == int(day)) & (df["field_name"] == store.field_name)
                    for rec in df[mask].to_dict("records"):
                        try:
                            path = _json.loads(rec.get("path_json", "[]"))
                        except (TypeError, ValueError):
                            path = []
                        machinery.append({
                            "event_name": rec.get("event_name", "machinery"),
                            "machine_type": rec.get("machine_type", "machine"),
                            "path": path,
                        })
            except Exception:
                logger.exception("Failed to read machinery metadata from %s", machinery_dir)

        weeds = []
        weed_dir = Path(log_path) / "weed_states"
        if weed_dir.exists():
            try:
                weed_df = _pq.read_table(str(weed_dir)).to_pandas()
                if not weed_df.empty:
                    day_series = weed_df["day"].astype(int)
                    mask = (day_series == int(day)) & (weed_df["field_name"] == store.field_name)
                    for rec in weed_df[mask].to_dict("records"):
                        weeds.append({
                            "row": int(rec.get("row", 0)),
                            "col": int(rec.get("col", 0)),
                            "alive": bool(rec.get("alive", True)),
                            "lai": float(rec.get("lai", 0.0) or 0.0),
                            "biomass_g": float(rec.get("biomass_g", 0.0) or 0.0),
                            "species": rec.get("species", "generic_grass"),
                        })
            except Exception:
                logger.exception("Failed to read weed metadata from %s", weed_dir)

        return JSONResponse(content={
            "day": day,
            "field_name": store.field_name,
            "precipitation_mm": precipitation_mm,
            "machinery": machinery,
            "weeds": weeds,
        })

    @api.get("/api/buffer/rebuild")
    async def buffer_rebuild(
        variable: str           = Query("biomass_g", description="Colour-map variable"),
        field:    Optional[str] = Query(None,        description="Field to rebuild (default: all)"),
    ):
        """Rebuild binary frames with a new colour variable.

        Pass ?field=<name> to rebuild only one field's store.
        Omit ?field to rebuild all fields simultaneously.
        """
        if plants_df is None or plants_df.empty:
            raise HTTPException(status_code=503, detail="No plant data loaded.")

        if field:
            store = FIELD_REGISTRY.rebuild_field(field, plants_df, variable)
            if store is None:
                raise HTTPException(status_code=404, detail=f"Unknown field '{field}'.")
            return {
                "status": "rebuilt",
                "variable": variable,
                "field": field,
                "n_days": store.n_days,
            }
        else:
            FIELD_REGISTRY.rebuild_all(plants_df, variable)
            return {
                "status": "rebuilt",
                "variable": variable,
                "fields": FIELD_REGISTRY.field_names,
                "n_days_per_field": {
                    fn: FIELD_REGISTRY.get(fn).n_days
                    for fn in FIELD_REGISTRY.field_names
                },
            }

    @api.get("/api/buffer/terrain")
    async def buffer_terrain(
        field: Optional[str] = Query(None, description="Field name (default: first field)"),
    ):
        """Return the (modified) elevation grid for a field as a flat JSON array.

        PRD v0.6.0 §7 — terrain binary stream.
        The returned grid is the effective elevation after any LandPrep modifier
        has been applied (not the raw DEM), so the Three.js ground plane
        displays furrows/terraces/bunds exactly as the D8 engine sees them.

        Response JSON:
            {
              "rows": int,
              "cols": int,
              "resolution_m": float,
              "elevation_flat": [float, ...]   // rows*cols values, row-major
            }
        Returns a flat zero grid if no terrain data was written (flat field).
        """
        from cropforge.viz.app import _DATA

        terrain_all = _DATA.get("terrain")

        # Resolve field name (default = first available)
        field_names = FIELD_REGISTRY.field_names
        resolved = field or (field_names[0] if field_names else None)

        if terrain_all and resolved and resolved in terrain_all:
            return JSONResponse(content=terrain_all[resolved])

        # Fallback: serve a flat zero grid using buffer meta dimensions
        store = FIELD_REGISTRY.get(field)
        if store is not None and store.is_ready:
            n = store.rows * store.cols
            return JSONResponse(content={
                "rows": store.rows,
                "cols": store.cols,
                "resolution_m": 1.0,
                "elevation_flat": [0.0] * n,
            })

        raise HTTPException(
            status_code=503,
            detail="No terrain data available and no field store ready.",
        )

    # ---- Serve Three.js static files at /viewport/ --------------------
    if _STATIC_DIR.exists():
        api.mount("/viewport", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    else:
        logger.warning("Static directory not found: %s", _STATIC_DIR)

    # ---- Mount Dash WSGI at root (must be LAST) -----------------------
    api.mount("/", WSGIMiddleware(dash_app))

    return api


# ---------------------------------------------------------------------------
# Boot function (called by farm.visualize())
# ---------------------------------------------------------------------------

def boot(log_path: str, cropforge_version: str = "0.1.0", quality: str = "standard") -> None:
    """Start the dashboard server and open the default browser.

    Blocks until the user presses Ctrl-C (PRD Section 7.1).
    """
    from cropforge.viz.app import _DATA, _load_parquet, create_dash_app

    # Load Parquet once — create_dash_app re-uses _DATA if already populated
    _load_parquet(log_path)

    dash_app = create_dash_app(log_path=log_path)

    # Build FastAPI with binary buffer (plants_df already in _DATA)
    fastapi_app = create_fastapi_app(
        dash_app=dash_app.server,
        log_path=log_path,
        cropforge_version=cropforge_version,
        plants_df=_DATA.get("plants"),
        quality=quality,
    )

    # Configure uvicorn
    config = uvicorn.Config(
        app=fastapi_app,
        host=HOST,
        port=PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    # Open browser after a short delay
    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(BASE_URL)

    threading.Thread(target=_open_browser, daemon=True).start()

    logger.info(
        "CropForge dashboard running at %s  (press Ctrl-C to stop)", BASE_URL
    )
    print(f"\n[CropForge] Dashboard -> {BASE_URL}")
    print("[CropForge] Press Ctrl-C to stop the server.\n")

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[CropForge] Server stopped.")
        logger.info("CropForge dashboard server stopped.")
