"""
cropforge/viz/app.py
====================
Plotly Dash application â€” CropForge Phase 2+3+4 Dashboard Frontend.

PRD References:
    Section 7.1 â€” Served by FastAPI on port 7860
    Section 7.2 â€” Four-panel layout
    Section 7.3 â€” Panel 1: Three.js iframe at /viewport/;
                  raycasting postMessage PLANT_CLICKED â†’ Panel 4
    Section 7.2 â€” Panel 4: Farm Inspector sidebar (collapsed by default)
    Section 16  â€” Parquet schema driving the data layer

Layout (PRD Section 7.2):
    +----------------------------------+--------------------+
    |  Panel 1: 3D Viewport            | Panel 2: Metrics   |
    |  (placeholder - Phase 3)  60%    | Dashboard     40%  |
    |                                  |--------------------|
    |                                  | Panel 3: Event Log |
    +----------------------------------+--------------------+
    Panel 4: Farm Inspector (right sidebar, collapsed by default)

Data flow:
    1. ``create_dash_app(log_path)`` reads all three Parquet tables once at
       startup into pandas DataFrames held in module-level ``_DATA``.
    2. Dash callbacks update Panel 2 charts in response to dropdown /
       slider interactions.  No Parquet reads happen during interaction.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level data cache (loaded once at startup)
# ---------------------------------------------------------------------------
_DATA: Dict[str, Optional[pd.DataFrame]] = {
    "plants":   None,
    "soil":     None,
    "env":      None,
    "log_path": None,
    "terrain":  None,   # v0.6.0 â€” {field_name: {rows, cols, resolution_m, elevation_flat}}
}


# ---------------------------------------------------------------------------
# Parquet reader
# ---------------------------------------------------------------------------

def _load_parquet(log_path: str) -> None:
    """Read all three Parquet tables into ``_DATA`` at startup.

    PRD Section 7.3 (Phase 3 note):
        The Parquet-to-binary conversion is performed once per session by
        the FastAPI server at startup and held in memory.

    For Phase 2 we load pandas DataFrames; Phase 3 will add the
    Float32Array binary conversion layer alongside.

    The files are written with partition_cols=[field_name, day], which
    encodes those values in the directory path (Hive partitioning format):
        plants/field_name=Plot%20A/day=1/part-0.parquet

    We use pyarrow's HivePartitioning to decode the path segments back
    into DataFrame columns.

    v0.4.0 â€” Season column:
        The environment table now carries a ``season`` (int32) column.
        Legacy single-season logs do not have this column; we default it
        to 1 so all downstream code can assume it always exists.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    p = Path(log_path)
    _DATA["log_path"] = log_path

    # Partition schema: both tables use the same two partition columns
    _PART_SCHEMA = pa.schema([
        pa.field("field_name", pa.string()),
        pa.field("day",        pa.int32()),
    ])
    partitioning = ds.partitioning(_PART_SCHEMA, flavor="hive")

    for table_name in ("plants", "soil", "environment"):
        subdir = p / table_name
        if subdir.exists():
            try:
                dataset = ds.dataset(
                    str(subdir),
                    format="parquet",
                    partitioning=partitioning,
                )
                df = dataset.to_table().to_pandas()

                # Normalise types after decode
                if "day" in df.columns:
                    df["day"] = df["day"].astype(int)
                if "field_name" in df.columns:
                    df["field_name"] = df["field_name"].astype(str)

                # v0.4.0: ensure 'season' column exists in env table.
                # Legacy logs don't have it; default to 1 so all callbacks work
                # without an isinstance guard.
                if table_name == "environment":
                    if "season" not in df.columns:
                        df["season"] = 1
                    else:
                        df["season"] = df["season"].astype(int)

                key = "env" if table_name == "environment" else table_name
                _DATA[key] = df
                logger.info(
                    "Loaded %s table: %d rows, %d cols",
                    table_name, len(df), len(df.columns)
                )
            except Exception:
                logger.exception("Failed to load %s Parquet from %s", table_name, subdir)
        else:
            logger.warning("Parquet subdirectory not found: %s", subdir)

    # v0.6.0 â€” Load terrain grids written by runtime.py at simulation end.
    import json as _json
    _terrain_file = p / "terrain.json"
    if _terrain_file.exists():
        try:
            _DATA["terrain"] = _json.loads(_terrain_file.read_text(encoding="utf-8"))
            logger.info("Loaded terrain.json from %s (%d fields)", _terrain_file, len(_DATA["terrain"]))
        except Exception:
            logger.exception("Failed to load terrain.json from %s", _terrain_file)
    else:
        logger.info("No terrain.json found at %s â€” flat terrain assumed.", p)



# ---------------------------------------------------------------------------
# Aggregated metrics builder (for Panel 2 time-series)
# ---------------------------------------------------------------------------

def _build_daily_metrics(plants_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate plant-level records to per-day field means.

    Returns a DataFrame with columns:
        day, field_name, mean_biomass_g, mean_lai, mean_height_cm,
        mean_root_depth_cm, mean_stress_index, alive_count, dead_count
    """
    agg = (
        plants_df
        .groupby(["day", "field_name"], observed=True)
        .agg(
            mean_biomass_g=("biomass_g",    "mean"),
            mean_lai=       ("lai",          "mean"),
            mean_height_cm= ("height_cm",    "mean"),
            mean_root_depth_cm=("root_depth_cm", "mean"),
            mean_stress_index= ("stress_index",  "mean"),
            alive_count=    ("alive",        "sum"),
            total_plants=   ("alive",        "count"),
        )
        .reset_index()
    )
    agg["dead_count"] = agg["total_plants"] - agg["alive_count"]
    return agg.sort_values("day")


def _get_season_boundaries(env_df: pd.DataFrame) -> list:
    """Return a list of (day, season_number) for each season > 1.

    For each unique season number > 1, find the FIRST day that season
    appears in the env log. These are the x-axis positions where we draw
    the vertical boundary line annotated 'Season N Starts'.

    Returns an empty list for single-season or legacy logs.
    """
    if env_df is None or env_df.empty:
        return []
    if "season" not in env_df.columns:
        return []
    seasons = sorted(env_df["season"].unique())
    if len(seasons) <= 1:
        return []
    boundaries = []
    for s in seasons:
        if s <= 1:
            continue
        first_day = int(env_df[env_df["season"] == s]["day"].min())
        boundaries.append((first_day, int(s)))
    return boundaries


def _build_daily_soil_metrics(soil_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate topsoil (layer=0) to per-day field means."""
    topsoil = soil_df[soil_df["layer"] == 0]
    agg = (
        topsoil
        .groupby(["day", "field_name"], observed=True)
        .agg(
            mean_moisture_pct= ("moisture_pct",  "mean"),
            mean_nitrogen_kg_ha=("nitrogen_kg_ha", "mean"),
        )
        .reset_index()
    )
    return agg.sort_values("day")


def build_csv_export(
    daily_metrics: pd.DataFrame,
    daily_soil: pd.DataFrame,
    env_df: Optional[pd.DataFrame],
    session_name: str,
) -> Optional[dict]:
    """Build a CSV export dict compatible with Dash dcc.Download data format.

    Module-level helper so tests can call it without Dash's callback context.
    Returns None if daily_metrics is empty.

    Parameters
    ----------
    daily_metrics:
        Aggregated plant metrics (day Ã— field_name).
    daily_soil:
        Aggregated topsoil metrics (day Ã— field_name), or empty DataFrame.
    env_df:
        Full environment Parquet table (for 'season' column), or None.
    session_name:
        Session identifier used in the filename.

    Returns
    -------
    dict or None
        ``{"content": csv_str, "filename": str, "type": "text/csv"}`` or None.
    """
    from datetime import datetime, timezone as _tz

    if daily_metrics is None or daily_metrics.empty:
        return None

    export_df = daily_metrics.copy()

    # Merge soil metrics so the researcher gets the full picture
    if daily_soil is not None and not daily_soil.empty:
        export_df = export_df.merge(daily_soil, on=["day", "field_name"], how="left")

    # Attach season column from env table (if available)
    if env_df is not None and not env_df.empty and "season" in env_df.columns:
        season_map = (
            env_df[["day", "field_name", "season"]]
            .drop_duplicates(subset=["day", "field_name"])
        )
        export_df = export_df.merge(season_map, on=["day", "field_name"], how="left")
        cols = list(export_df.columns)
        if "season" in cols:
            cols.remove("season")
            cols.insert(2, "season")
            export_df = export_df[cols]

    date_str = datetime.now(_tz.utc).strftime("%Y%m%d")
    filename = f"cropforge_timeseries_{session_name}_{date_str}.csv"
    csv_str = export_df.to_csv(index=False)
    return dict(content=csv_str, filename=filename, type="text/csv")


def build_yield_summary(plants_df: Optional[pd.DataFrame], terrain_meta: Optional[dict]) -> dict:
    """Compute final-day yield metrics from the logged representative plants."""
    if plants_df is None or plants_df.empty:
        return {
            "total_yield_kg": 0.0,
            "yield_kg_per_ha": 0.0,
            "yield_t_per_ha": 0.0,
            "fields": {},
        }

    fields = {}
    total_yield_kg = 0.0
    total_area_m2 = 0.0

    for field_name in sorted(plants_df["field_name"].astype(str).unique()):
        field_df = plants_df[plants_df["field_name"].astype(str) == field_name]
        if field_df.empty:
            continue

        final_day = int(field_df["day"].astype(int).max())
        final_df = field_df[field_df["day"].astype(int) == final_day].copy()
        if final_df.empty:
            continue

        rows = int(final_df["row"].astype(int).max()) + 1 if "row" in final_df else 0
        cols = int(final_df["col"].astype(int).max()) + 1 if "col" in final_df else 0
        resolution_m = 1.0
        if terrain_meta and field_name in terrain_meta:
            resolution_m = float(terrain_meta[field_name].get("resolution_m", 1.0) or 1.0)
            rows = int(terrain_meta[field_name].get("rows", rows) or rows)
            cols = int(terrain_meta[field_name].get("cols", cols) or cols)

        cell_area_m2 = resolution_m ** 2
        area_m2 = rows * cols * cell_area_m2

        if "sowing_density_plants_per_m2" in final_df.columns:
            density = final_df["sowing_density_plants_per_m2"].fillna(1.0).astype(float)
        else:
            density = pd.Series([1.0] * len(final_df), index=final_df.index)

        grain = pd.Series([0.0] * len(final_df), index=final_df.index, dtype=float)
        if "custom_json" in final_df.columns:
            def _grain_from_custom(raw):
                try:
                    return float(json.loads(raw or "{}").get("grain_biomass_g", 0.0) or 0.0)
                except Exception:
                    return 0.0
            grain = final_df["custom_json"].apply(_grain_from_custom).astype(float)

        biomass = final_df["biomass_g"].fillna(0.0).astype(float)
        yield_source = grain if float(grain.max() or 0.0) > 0.0 else biomass
        total_g = float((yield_source * density * cell_area_m2).sum())
        total_kg = total_g / 1000.0
        area_ha = area_m2 / 10000.0
        kg_per_ha = total_kg / area_ha if area_ha > 0 else 0.0

        fields[field_name] = {
            "total_yield_kg": total_kg,
            "yield_kg_per_ha": kg_per_ha,
            "yield_t_per_ha": kg_per_ha / 1000.0,
            "density_plants_per_m2": float(density.mean()) if len(density) else 1.0,
            "area_m2": area_m2,
            "final_day": final_day,
        }
        total_yield_kg += total_kg
        total_area_m2 += area_m2

    total_area_ha = total_area_m2 / 10000.0
    total_kg_per_ha = total_yield_kg / total_area_ha if total_area_ha > 0 else 0.0
    return {
        "total_yield_kg": total_yield_kg,
        "yield_kg_per_ha": total_kg_per_ha,
        "yield_t_per_ha": total_kg_per_ha / 1000.0,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Dash app factory
# ---------------------------------------------------------------------------

def create_dash_app(log_path: str):
    """Build and return the configured Plotly Dash application.

    Parameters
    ----------
    log_path:
        Absolute path to the Parquet session directory.

    Returns
    -------
    dash.Dash
        Fully configured application.  Caller uses ``app.server`` to mount
        into FastAPI.
    """
    import dash
    from dash import Input, Output, State, dcc, html

    # Load data once (idempotent â€” skip if already cached by boot())
    if _DATA["plants"] is None:
        _load_parquet(log_path)

    plants_df = _DATA["plants"]
    soil_df   = _DATA["soil"]
    env_df    = _DATA["env"]

    # ---- Pre-compute aggregates ----------------------------------------
    if plants_df is not None and not plants_df.empty:
        daily_metrics = _build_daily_metrics(plants_df)
    else:
        daily_metrics = pd.DataFrame()

    if soil_df is not None and not soil_df.empty:
        daily_soil = _build_daily_soil_metrics(soil_df)
    else:
        daily_soil = pd.DataFrame()

    # ---- Variable options for Panel 2 dropdowns -----------------------
    plant_metric_options = [
        {"label": "Mean Biomass (g/plant)", "value": "mean_biomass_g"},
        {"label": "Mean LAI (m2/m2)",        "value": "mean_lai"},
        {"label": "Mean Height (cm)",         "value": "mean_height_cm"},
        {"label": "Mean Root Depth (cm)",     "value": "mean_root_depth_cm"},
        {"label": "Mean Stress Index",        "value": "mean_stress_index"},
        {"label": "Alive Plant Count",        "value": "alive_count"},
        {"label": "Dead Plant Count",         "value": "dead_count"},
    ]

    spatial_options = [
        {"label": "Biomass (g/plant)",   "value": "biomass_g"},
        {"label": "LAI (m2/m2)",          "value": "lai"},
        {"label": "Weed LAI",             "value": "weed_lai"},
        {"label": "Height (cm)",          "value": "height_cm"},
        {"label": "Stress Index",         "value": "stress_index"},
    ]

    # v0.6.0 â€” Terrain surface overlay: soil/env variables draped over elevation
    terrain_overlay_options = [
        {"label": "Elevation only",              "value": "__elevation__"},
        {"label": "Soil Moisture (%)",           "value": "moisture_pct"},
        {"label": "Nitrogen (kg/ha)",            "value": "nitrogen_kg_ha"},
        {"label": "Biomass (g/plant)",           "value": "biomass_g"},
        {"label": "Stress Index",                "value": "stress_index"},
        {"label": "Weed LAI",                    "value": "weed_lai"},
        # v0.7.0 Phase 6 observables
        {"label": "Surface Runoff (mm)",         "value": "surface_runoff_mm_today"},
        {"label": "Cumulative Erosion Index",    "value": "cumulative_erosion_index"},
    ]

    # ---- Day range for scrubber ----------------------------------------
    if plants_df is not None and not plants_df.empty:
        day_min = int(plants_df["day"].min())
        day_max = int(plants_df["day"].max())
        day_marks = {
            d: str(d)
            for d in range(day_min, day_max + 1, max(1, (day_max - day_min) // 10))
        }
    else:
        day_min, day_max = 1, 1
        day_marks = {1: "1"}

    # ---- Event log text ------------------------------------------------
    event_lines: List[str] = []
    if env_df is not None and not env_df.empty:
        for _, row in env_df.sort_values("day").iterrows():
            try:
                events = json.loads(row["events_fired"])
            except Exception:
                events = []
            if events:
                for ev in events:
                    event_lines.append(f"Day {int(row['day'])}: {ev}")

    if not event_lines:
        event_lines = ["No management events recorded in this simulation."]

    # ---- Session info --------------------------------------------------
    session_name = Path(log_path).name
    n_days = day_max - day_min + 1
    n_plants = int(plants_df["plant_id"].nunique()) if plants_df is not None else 0
    n_fields = int(plants_df["field_name"].nunique()) if plants_df is not None else 0
    yield_summary_data = build_yield_summary(plants_df, _DATA.get("terrain"))

    # ==================================================================
    # Layout
    # ==================================================================

    app = dash.Dash(
        __name__,
        title="CropForge Dashboard",
        update_title=None,
        suppress_callback_exceptions=True,
        external_stylesheets=[],
    )

    # ---- Brand palette (v0.5.0 â€” MINIMALIST THEME) -------------------------
    # PRD Â§4.4: Minimalist UI, strict utilitarian document-style aesthetic.
    ACCENT   = "#111111"   # solid black (buttons, primary states)
    ACCENT_D = "#333333"   # hover states
    ACCENT_L = "#EAEAEA"   # light borders
    ACCENT_XL= "#EDF3EC"   # muted pastel green for tags
    ACCENT_TEXT = "#346538" # text for muted pastel tags

    BG_APP   = "#FBFBFA"   # warm off-white app shell
    BG_PANEL = "#FFFFFF"   # white panels
    BG_SIDEBAR="#FFFFFF"   # pure white sidebar
    BG_DARK  = "#9c9c9c"   # darkened viewport background to isolate 3D scene

    BORDER   = "rgba(0,0,0,0.06)"   # ultra-light border
    BORDER_STRONG = "#EAEAEA"       # primary structural border
    SHADOW_SM= "none"
    SHADOW_MD= "0 2px 8px rgba(0,0,0,0.04)"
    SHADOW   = "none"

    TEXT_PRI = "#111111"   # off-black â€” primary text
    TEXT_SEC = "#787774"   # muted grey â€” secondary / labels
    TEXT_DIM = "#9CA3AF"   # light grey â€” placeholders
    TEXT_ACC = "#111111"

    _ROOT_CSS = f"""
        /* ================================================================
           CropForge v0.5.0 â€” Premium Utilitarian Minimalism
        ================================================================ */
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'SF Pro Display', 'Geist Sans', 'Helvetica Neue', 'Switzer', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            background: {BG_APP};
            color: {TEXT_PRI};
            overflow: hidden;
            line-height: 1.6;
            font-variant-numeric: tabular-nums;
        }}

        /* ---- Loading overlay ---------------------------------------- */
        #cf-loading-overlay {{
            position: fixed; inset: 0; z-index: 9999;
            background: {BG_APP};
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            transition: opacity 0.5s cubic-bezier(0.32,0.72,0,1),
                        visibility 0.5s cubic-bezier(0.32,0.72,0,1);
        }}
        #cf-loading-overlay.hidden {{
            opacity: 0; visibility: hidden; pointer-events: none;
        }}
        .cf-logo-pulse {{
            width: 72px; height: 72px; border-radius: 18px;
            object-fit: contain;
            background: #FFFFFF;
            padding: 8px;
            margin-bottom: 22px;
            box-shadow: 0 4px 24px rgba(17,17,17,0.18);
            animation: cfPulse 2s cubic-bezier(0.4,0,0.6,1) infinite;
        }}
        @keyframes cfPulse {{
            0%, 100% {{ box-shadow: 0 4px 24px rgba(17,17,17,0.18); transform: scale(1); }}
            50%       {{ box-shadow: 0 4px 38px rgba(26,143,92,0.32); transform: scale(1.04); }}
        }}
        .cf-loading-title {{
            font-size: 14px; font-weight: 700; letter-spacing: 0.16em;
            text-transform: uppercase; color: {ACCENT}; margin-bottom: 4px;
        }}
        .cf-loading-sub {{
            font-size: 12px; color: {TEXT_SEC}; margin-bottom: 24px;
        }}
        .cf-progress-track {{
            width: 200px; height: 3px; background: #E5E7EB;
            border-radius: 99px; overflow: hidden;
        }}
        .cf-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {ACCENT_D}, {ACCENT_L});
            border-radius: 99px;
            animation: cfSlide 1.6s cubic-bezier(0.4,0,0.6,1) infinite;
        }}
        @keyframes cfSlide {{
            0%   {{ margin-left: -40%; width: 40%; }}
            60%  {{ margin-left: 60%; width: 50%; }}
            100% {{ margin-left: 130%; width: 40%; }}
        }}

        /* ---- Top bar ----------------------------------------------- */
        #cf-topbar {{
            height: 52px; flex-shrink: 0;
            background: {BG_PANEL};
            border-bottom: 1px solid #EAEAEA;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 24px;
            box-shadow: none;
            animation: cfFadeDown 0.35s cubic-bezier(0.32,0.72,0,1) both;
            position: relative; z-index: 10;
        }}
        @keyframes cfFadeDown {{
            from {{ opacity: 0; transform: translateY(-8px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .cf-wordmark {{
            font-size: 15px; font-weight: 800; letter-spacing: -0.01em;
            color: {ACCENT};
            display: flex; align-items: center; gap: 10px;
        }}
        .cf-wordmark-logo {{
            width: 26px; height: 26px; object-fit: contain;
            border-radius: 7px; background: #FFFFFF;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .cf-wordmark-divider {{
            width: 1px; height: 14px; background: #D1D5DB;
        }}
        .cf-wordmark-subtitle {{
            font-size: 12px; font-weight: 500; color: {TEXT_SEC};
            letter-spacing: 0;
        }}
        .cf-badge {{
            display: inline-flex; align-items: center;
            background: {ACCENT_XL};
            color: {ACCENT_TEXT};
            border: none;
            border-radius: 9999px;
            padding: 4px 10px; font-size: 11px; font-weight: 600;
            margin-left: 8px; font-variant-numeric: tabular-nums;
            letter-spacing: 0.05em; text-transform: uppercase;
        }}
        .cf-btn {{
            display: inline-flex; align-items: center; gap: 6px;
            background: {ACCENT};
            color: #FFFFFF;
            border: none; border-radius: 4px;
            padding: 6px 16px; font-size: 12px; font-weight: 600;
            cursor: pointer; margin-left: 12px;
            transition: background 0.15s cubic-bezier(0.32,0.72,0,1),
                        transform 0.1s cubic-bezier(0.32,0.72,0,1);
            min-height: 34px;
        }}
        .cf-btn:hover {{
            background: {ACCENT_D};
            transform: scale(0.98);
        }}
        .cf-btn:active {{ transform: scale(0.96); }}

        /* ---- Three-column shell ------------------------------------ */
        #cf-shell {{
            display: grid;
            grid-template-columns: 280px 1fr 380px;
            grid-template-rows: 1fr;
            height: calc(100vh - 52px);
            animation: cfFadeIn 0.5s 0.08s cubic-bezier(0.32,0.72,0,1) both;
            transition: grid-template-columns 0.3s cubic-bezier(0.32,0.72,0,1);
        }}
        @keyframes cfFadeIn {{
            from {{ opacity: 0; }}
            to   {{ opacity: 1; }}
        }}

        /* ---- Left sidebar ------------------------------------------ */
        #cf-sidebar {{
            background: {BG_SIDEBAR};
            border-right: 1px solid #EAEAEA;
            padding: 24px 20px;
            overflow: hidden;     /* grid-stretch sets exact height; this clips overflow */
            display: flex; flex-direction: column; gap: 0;
            /* ponytail: no height:100% â€” grid align-items:stretch already fills the row */
            box-sizing: border-box;
        }}
        /* Inner scroller â€” scrolls sidebar content without clipping dropdowns */
        #cf-sidebar-inner {{
            overflow-y: auto;
            flex: 1;
            min-height: 0;
            padding-right: 2px;  /* prevent scrollbar from clipping content */
        }}
        #cf-sidebar-logo {{
            display: flex; align-items: center; gap: 10px;
            padding-bottom: 16px;
            border-bottom: 1px solid #E5E7EB;
            margin-bottom: 16px;
            flex-shrink: 0;  /* never let the logo compress; gives space to inner scroller */
        }}
        #cf-sidebar-logo-mark {{
            width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
            object-fit: contain;
            background: #FFFFFF;
            padding: 3px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }}
        #cf-sidebar-logo-text {{
            font-size: 14px; font-weight: 700; color: {TEXT_PRI};
            letter-spacing: -0.01em;
        }}
        #cf-sidebar-logo-version {{
            font-size: 10px; font-weight: 500; color: {TEXT_DIM};
            letter-spacing: 0.04em;
        }}

        .cf-section {{
            margin-bottom: 18px;
            animation: cfFadeUp 0.4s cubic-bezier(0.32,0.72,0,1) both;
            position: relative;
        }}
        .cf-sec-1 {{ z-index: 50 !important; }}
        .cf-sec-2 {{ z-index: 40 !important; }}
        .cf-sec-3 {{ z-index: 30 !important; }}
        .cf-sec-4 {{ z-index: 20 !important; }}
        .cf-sec-5 {{ z-index: 10 !important; }}
        .cf-sec-6 {{ z-index: 8  !important; }}  /* terrain toggle button */
        .cf-sec-7 {{ z-index: 6  !important; }}  /* terrain overlay dropdown */

        /* ---- Terrain map modal (Task 3) -------------------------------- */
        #terrain-modal-overlay {{
            display: none;
            position: fixed; inset: 0; z-index: 1000;
            background: rgba(0,0,0,0.55);
            align-items: center; justify-content: center;
        }}
        #terrain-modal-overlay.cf-modal-open {{
            display: flex;
        }}
        #terrain-modal-box {{
            background: {BG_PANEL};
            border-radius: 12px;
            width: 82vw; height: 82vh;
            display: flex; flex-direction: column;
            box-shadow: 0 24px 80px rgba(0,0,0,0.35);
            overflow: hidden;
            animation: cfFadeIn 0.2s cubic-bezier(0.32,0.72,0,1) both;
        }}
        #terrain-modal-header {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 16px 24px;
            border-bottom: 1px solid #EAEAEA;
            flex-shrink: 0;
        }}
        #terrain-modal-body {{
            flex: 1; min-height: 0; padding: 8px 16px 16px;
            display: flex; flex-direction: column; gap: 10px;
        }}
        @keyframes cfFadeUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        .cf-label {{
            font-size: 10px; font-weight: 700; letter-spacing: 0.10em;
            text-transform: uppercase; color: {TEXT_DIM};
            margin-bottom: 7px; display: block;
        }}
        .cf-divider {{
            height: 1px; background: #F3F4F6; margin: 4px 0 18px;
        }}

        /* ---- Dash Dropdown overrides (Dash 4.x / react-select 5) ---------- */
        /* Outer container: white background so it is never transparent */
        .cf-select {{
            background: #FFFFFF !important;
            border-radius: 4px !important;
        }}
        /* Control box â€” the visible input row */
        .cf-select .dash-dropdown,
        .cf-select .Select-control,
        .cf-select div[class$="-container"],
        .cf-select div[class*="-control"] {{
            background: #FFFFFF !important;
            border-color: #EAEAEA !important;
            border-radius: 4px !important;
            box-shadow: none !important;
            min-height: 36px !important;
            cursor: pointer !important;
        }}
        .cf-select .Select-control:hover,
        .cf-select div[class*="-control"]:hover {{
            border-color: #333333 !important;
        }}
        /* Selected value text */
        .cf-select .Select-value,
        .cf-select div[class*="-singleValue"] {{
            color: {TEXT_PRI} !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }}
        .cf-select .Select-placeholder,
        .cf-select div[class*="-placeholder"] {{
            color: {TEXT_DIM} !important;
            font-size: 13px !important;
        }}
        /* Dropdown menu panel */
        .cf-select .Select-menu-outer,
        .cf-select div[class*="-menu"],
        .cf-select div[class*="-MenuList"] {{
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #D1D1D1 !important;
            border-radius: 4px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            z-index: 9999 !important;
        }}
        /* Option rows */
        .cf-select .Select-option,
        .cf-select div[class*="-option"] {{
            font-size: 13px !important;
            color: {TEXT_PRI} !important;
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            padding: 8px 14px !important;
            cursor: pointer !important;
        }}
        .cf-select .Select-option:hover,
        .cf-select .Select-option.is-focused,
        .cf-select div[class*="-option"]:hover,
        .cf-select div[class*="-option"][class*="focused"] {{
            background-color: {ACCENT_XL} !important;
            background: {ACCENT_XL} !important;
            color: {ACCENT} !important;
        }}
        .cf-select .Select-option.is-selected,
        .cf-select div[class*="-option"][class*="selected"] {{
            background-color: {ACCENT} !important;
            background: {ACCENT} !important;
            color: #FFFFFF !important;
        }}
        .cf-select div[class*="-indicatorContainer"] svg {{
            fill: {TEXT_SEC} !important;
        }}
        .cf-select div[class*="-indicatorSeparator"] {{
            background-color: #EAEAEA !important;
        }}

        /* ---- Centre â€” 3D viewport ---------------------------------- */
        #cf-viewport {{
            position: relative;
            background: {BG_DARK};
            border: 1px solid #D0D0D0;
            margin: 16px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: {SHADOW_MD};
            display: flex;
            flex-direction: column;
            min-height: 0;
        }}
        #cf-viewport iframe {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            border: none;
            display: block;
        }}
        /* dcc.Loading injects a div wrapper â€” must also be flex so iframe fills height */
        #cf-viewport > div,
        #viewport-loading,
        #viewport-loading > div {{
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
            width: 100%;
            position: static; /* Ensure they don't trap absolute positioning */
        }}

        /* ---- Right column ------------------------------------------ */
        #cf-right {{
            background: {BG_APP};
            border-left: 1px solid #EAEAEA;
            display: flex; flex-direction: column;
            overflow-y: auto;
            padding: 12px;
        }}

        /* Panel card â€” minimalist bento card */
        .cf-card {{
            background: {BG_PANEL};
            border-radius: 8px;
            border: 1px solid #C0C0C0;
            margin-bottom: 24px;
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            overflow: visible;
        }}

        .cf-panel-header {{
            padding: 20px 24px 12px;
            border-bottom: 1px solid #C0C0C0;
        }}
        .cf-panel-title {{
            font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: {TEXT_PRI};
        }}
        .cf-chart-area {{
            padding: 16px 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        /* ---- Plant Inspector --------------------------------------- */
        #cf-inspector {{
            flex-shrink: 0;
        }}
        .cf-inspector-placeholder {{
            font-size: 13px; color: {TEXT_DIM};
            text-align: center; padding: 32px 12px;
            line-height: 1.7;
        }}
        .cf-stat-row {{
            display: flex; justify-content: space-between;
            align-items: center; padding: 8px 0;
            border-bottom: 1px solid #EAEAEA;
            font-size: 13px;
        }}
        .cf-stat-row:last-child {{ border-bottom: none; }}
        .cf-stat-label {{ color: {TEXT_SEC}; font-family: 'Geist Mono', 'SF Mono', monospace; font-size: 12px; }}
        .cf-stat-value {{
            color: {TEXT_PRI}; font-weight: 500;
            font-variant-numeric: tabular-nums; font-size: 13px;
        }}

        /* ---- Event log entries ------------------------------------ */
        .cf-event-line {{
            padding: 6px 12px;
            border-left: 2px solid #EAEAEA;
            margin-bottom: 4px; font-size: 12px;
            font-family: 'Geist Mono', 'SF Mono', monospace;
            color: {TEXT_SEC};
            background: #F9F9F8;
            border-radius: 0 4px 4px 0;
        }}

        /* ---- Scrollbar -------------------------------------------- */
        ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: #D1D5DB; border-radius: 99px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_L}; }}

        /* ---- Slider overrides ------------------------------------- */
        .rc-slider-track {{ background: {ACCENT} !important; }}
        .rc-slider-handle {{
            border-color: {ACCENT} !important;
            box-shadow: 0 0 0 2px {ACCENT_XL} !important;
        }}
        .rc-slider-rail {{ background: #E5E7EB !important; }}

        /* ---- Plotly chart text visibility ------------------------- */
        .js-plotly-plot .plotly .gtitle,
        .js-plotly-plot .plotly .xtitle,
        .js-plotly-plot .plotly .ytitle {{
            fill: {TEXT_SEC} !important;
        }}
    """

    _STYLE_MAIN      = {"margin": "0", "padding": "0", "height": "100vh",
                        "display": "flex", "flexDirection": "column",
                        "background": BG_APP, "color": TEXT_PRI,
                        "fontFamily": "'SF Pro Display', 'Geist Sans', sans-serif"}
    _STYLE_SELECT    = {"background": "#FFFFFF",
                        "color": TEXT_PRI, "border": "1px solid #EAEAEA",
                        "borderRadius": "4px", "width": "100%", "marginBottom": "12px"}
    _STYLE_BADGE     = {"display": "inline-flex", "alignItems": "center",
                        "background": ACCENT_XL,
                        "color": ACCENT_TEXT, "border": "none",
                        "borderRadius": "9999px", "padding": "4px 10px",
                        "fontSize": "11px", "fontWeight": "600",
                        "marginLeft": "8px", "textTransform": "uppercase",
                        "fontVariantNumeric": "tabular-nums"}
    _STYLE_EVENT_LINE = {"padding": "5px 10px",
                         "borderLeft": f"2px solid {ACCENT_L}",
                         "marginBottom": "4px", "fontSize": "11px",
                         "fontFamily": "ui-monospace, monospace",
                         "color": TEXT_SEC,
                         "background": ACCENT_XL,
                         "borderRadius": "0 6px 6px 0"}
    _STYLE_STAT_ROW   = {"display": "flex", "justifyContent": "space-between",
                         "alignItems": "center", "padding": "5px 0",
                         "borderBottom": "1px solid #F3F4F6",
                         "fontSize": "12px"}
    _STYLE_STAT_LABEL = {"color": TEXT_SEC, "fontFamily": "ui-monospace, monospace",
                         "fontSize": "11px"}
    _STYLE_STAT_VALUE = {"color": TEXT_PRI, "fontWeight": "600",
                         "fontVariantNumeric": "tabular-nums", "fontSize": "12px"}

    # ---- Field selector options ----------------------------------------
    field_names = []
    if plants_df is not None and not plants_df.empty:
        field_names = sorted(plants_df["field_name"].unique().tolist())
    default_field = field_names[0] if field_names else ""
    field_options = [{"label": fn, "value": fn} for fn in field_names]

    # Inject global CSS via app.index_string (html.Style is not available in all Dash versions)
    # This is the canonical Dash mechanism for injecting <style> into <head>.
    app.index_string = (
        "<!DOCTYPE html>"
        "<html>"
        "<head>"
        "{%metas%}"
        "<title>CropForge Dashboard</title>"
        '<link rel="icon" type="image/png" href="/viewport/favicon.png">'
        "{%css%}"
        f"<style>{_ROOT_CSS}</style>"
        "</head>"
        "<body>"
        "{%app_entry%}"
        "<footer>"
        "{%config%}"
        "{%scripts%}"
        "{%renderer%}"
        "</footer>"
        "</body>"
        "</html>"
    )

    app.layout = html.Div(
        style=_STYLE_MAIN,
        children=[

            # Hidden state stores
            dcc.Store(id="selected-plant-store", storage_type="memory"),
            dcc.Store(id="current-day-store",    storage_type="memory",
                      data=int(day_min)),
            dcc.Store(id="selected-field-store", storage_type="memory",
                      data=default_field),
            dcc.Store(id="terrain-modal-open",   storage_type="memory", data=False),
            # Polling interval: fires every 250 ms to relay postMessage â†’ Dash store
            dcc.Interval(id="inspector-poll", interval=250, n_intervals=0),
            html.Div(id="plant-msg-trigger", style={"display": "none"}),
            # dcc.Download â€” file-delivery component
            dcc.Download(id="download-csv"),

            # ================================================================
            # Dash-level loading overlay (Task 2: Preloader State)
            # Visible immediately; hidden via JS when viewport LOAD_COMPLETE
            # fires. Prevents blank-screen flash on first render.
            # ================================================================
            html.Div(
                id="cf-loading-overlay",
                children=[
                    html.Img(src="/viewport/CropForge_Logo.png", className="cf-logo-pulse", alt="CropForge logo"),
                    html.Div("CropForge", className="cf-loading-title"),
                    html.Div("Initialising workspace...", className="cf-loading-sub"),
                    html.Div(
                        html.Div(className="cf-progress-fill"),
                        className="cf-progress-track",
                    ),
                ],
            ),

            # ================================================================
            # Top bar â€” wordmark, session badges, export button
            # ================================================================
            html.Div(id="cf-topbar", children=[
                html.Div([
                    html.Div([html.Img(src="/viewport/CropForge_Logo.png", className="cf-wordmark-logo", alt="CropForge logo"), html.Span("CropForge")], className="cf-wordmark"),
                    html.Div(className="cf-wordmark-divider"),
                    html.Div(session_name, className="cf-wordmark-subtitle"),
                ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
                html.Div([
                    html.Span(f"{n_days}d", style=_STYLE_BADGE,
                              title="Total simulation days"),
                    html.Span(f"{n_plants:,} plants", style=_STYLE_BADGE),
                    html.Span(f"{n_fields} field{'s' if n_fields != 1 else ''}",
                              style=_STYLE_BADGE),
                    html.Button(
                        "â¬‡ Export CSV",
                        id="export-csv-btn",
                        n_clicks=0,
                        className="cf-btn",
                    ),
                    html.Button(
                        "â—€ L",
                        id="toggle-left-btn",
                        n_clicks=0,
                        className="cf-btn",
                    ),
                    html.Button(
                        "R â–¶",
                        id="toggle-right-btn",
                        n_clicks=0,
                        className="cf-btn",
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
            ]),

            # ================================================================
            # Main three-column shell
            # ================================================================
            html.Div(id="cf-shell", children=[

                # ============================================================
                # LEFT SIDEBAR â€” all controls (PRD Â§4.3: 18%)
                # ============================================================
                html.Div(id="cf-sidebar", children=[

                    # -- Sidebar logo/brand mark ---
                    html.Div(id="cf-sidebar-logo", children=[
                        html.Img(src="/viewport/CropForge_Logo.png", id="cf-sidebar-logo-mark", alt="CropForge logo"),
                        html.Div([
                            html.Div("CropForge", id="cf-sidebar-logo-text"),
                            html.Div("v1.0.0 · Research Dashboard",
                                     id="cf-sidebar-logo-version"),
                        ]),
                    ]),

                    # -- Field selector (multi-field only) ---
                    html.Div(id="cf-sidebar-inner", children=[

                    html.Div(className="cf-section cf-sec-1", children=[
                        html.Span("Active Field", className="cf-label"),
                        dcc.Dropdown(
                            id="field-selector",
                            options=field_options,
                            value=default_field,
                            clearable=False,
                            style={
                                **_STYLE_SELECT,
                                "display": "block" if len(field_names) > 1 else "none",
                            },
                            className="cf-select",
                        ),
                        html.Div(
                            style={
                                "display": "block" if len(field_names) <= 1 else "none",
                            },
                            children=html.Span(
                                default_field or "â€”",
                                style={**_STYLE_BADGE,
                                       "fontSize": "12px", "marginLeft": "0"},
                            ),
                        ),
                    ]),

                    html.Div(className="cf-divider"),

                    # -- Spatial View toggle ---
                    html.Div(className="cf-section cf-sec-2", children=[
                        html.Span("Spatial View", className="cf-label"),
                        dcc.RadioItems(
                            id="spatial-view-toggle",
                            options=[
                                {"label": "2D Heatmap", "value": "2d"},
                                {"label": "3D Terrain", "value": "3d"},
                            ],
                            value="2d",
                            className="cf-radio",
                            labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "12px", "color": TEXT_PRI},
                        ),
                    ]),

                    html.Div(className="cf-divider"),

                    # -- 2D Heatmap Variable (only visible when 2d is selected) ---
                    html.Div(id="heatmap-var-container", className="cf-section cf-sec-3", children=[
                        html.Span("Heatmap Variable", className="cf-label"),
                        dcc.Dropdown(
                            id="heatmap-variable-dropdown",
                            options=[
                                {"label": "Biomass (g/plant)",   "value": "biomass_g"},
                                {"label": "LAI (mÂ²/mÂ²)", "value": "lai"},
                                {"label": "Weed LAI",             "value": "weed_lai"},
                                {"label": "Height (cm)",          "value": "height_cm"},
                                {"label": "Stress Index",         "value": "stress_index"},
                                # v0.7.0 Phase 6 observables (reads from soil layer 0)
                                {"label": "Surface Runoff (mm)",        "value": "surface_runoff_mm_today"},
                                {"label": "Cumulative Erosion Index",   "value": "cumulative_erosion_index"},
                            ],
                            value="biomass_g",
                            clearable=False,
                            style=_STYLE_SELECT,
                            className="cf-select",
                        ),
                    ]),

                    # -- 3D Terrain Variable (only visible when 3d is selected) ---
                    html.Div(id="terrain-var-container", className="cf-section cf-sec-3", style={"display": "none"}, children=[
                        html.Span("Surface Overlay Variable", className="cf-label"),
                        dcc.Dropdown(
                            id="sidebar-terrain-overlay-dropdown",
                            options=terrain_overlay_options,
                            value="__elevation__",
                            clearable=False,
                            style=_STYLE_SELECT,
                            className="cf-select",
                        ),
                    ]),

                    html.Div(className="cf-divider"),
                    # -- Metrics variable ---
                    html.Div(className="cf-section cf-sec-4", children=[
                        html.Span("Time-Series Metric", className="cf-label"),
                        dcc.Dropdown(
                            id="ts-variable-dropdown",
                            options=[
                                {"label": "Mean Biomass (g/plant)", "value": "mean_biomass_g"},
                                {"label": "Mean LAI (mÂ²/mÂ²)", "value": "mean_lai"},
                                {"label": "Mean Height (cm)",        "value": "mean_height_cm"},
                                {"label": "Mean Root Depth (cm)",    "value": "mean_root_depth_cm"},
                                {"label": "Mean Stress Index",       "value": "mean_stress_index"},
                                {"label": "Alive Plant Count",       "value": "alive_count"},
                                {"label": "Dead Plant Count",        "value": "dead_count"},
                            ],
                            value="mean_root_depth_cm",
                            clearable=False,
                            style=_STYLE_SELECT,
                            className="cf-select",
                        ),
                    ]),

                    html.Div(className="cf-divider"),

                    # -- Day scrubber ---
                    html.Div(className="cf-section cf-sec-5", children=[
                        html.Span("Simulation Day", className="cf-label"),
                        dcc.Slider(
                            id="day-scrubber",
                            min=day_min,
                            max=day_max,
                            step=1,
                            value=day_max,
                            marks=day_marks,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),

                    html.Div(className="cf-divider"),

                    # -- Event log ---
                    html.Div(className="cf-section cf-sec-6",
                             children=[
                        html.Span("Event Log", className="cf-label"),
                        html.Div(
                            id="event-log-content",
                            children=[
                                html.Div(line, className="cf-event-line")
                                for line in event_lines
                            ],
                        ),
                    ]),

                    ]),  # end cf-sidebar-inner

                ]),  # end sidebar

                # ============================================================
                # TERRAIN MODAL OVERLAY (Task 3 â€” v0.6.0)
                # Pure CSS modal; opened/closed via clientside callback.
                # ============================================================
                html.Div(id="terrain-modal-overlay", children=[
                    html.Div(id="terrain-modal-box", children=[
                        html.Div(id="terrain-modal-header", children=[
                            html.Div([
                                html.Span("â›°ï¸ ", style={"marginRight": "6px"}),
                                html.Span("3D Terrain Map", style={
                                    "fontSize": "13px", "fontWeight": "700",
                                    "color": TEXT_PRI,
                                }),
                            ], style={"display": "flex", "alignItems": "center"}),
                            html.Div([
                                # Overlay variable dropdown inside modal
                                dcc.Dropdown(
                                    id="terrain-overlay-dropdown",
                                    options=terrain_overlay_options,
                                    value="__elevation__",
                                    clearable=False,
                                    style={**_STYLE_SELECT,
                                           "width": "220px", "marginBottom": "0"},
                                    className="cf-select",
                                    placeholder="Surface Variable",
                                ),
                                html.Button(
                                    "âœ• Close",
                                    id="close-terrain-modal-btn",
                                    n_clicks=0,
                                    className="cf-btn",
                                    style={"marginLeft": "12px", "fontSize": "12px"},
                                ),
                            ], style={"display": "flex", "alignItems": "center"}),
                        ]),
                        html.Div(id="terrain-modal-body", children=[
                            dcc.Graph(
                                id="terrain-modal-chart",
                                config={
                                    "displayModeBar": True,
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "cropforge_terrain_3d",
                                        "height": 900, "width": 1400,
                                    },
                                },
                                style={"height": "100%", "width": "100%", "flex": "1"},
                            ),
                        ]),
                    ]),
                ]),

                # ============================================================
                # CENTRE â€” 3D Field View (PRD Â§4.3: 52%)
                # ============================================================
                html.Div(id="cf-viewport", children=[
                    dcc.Loading(
                        id="viewport-loading",
                        type="circle",
                        color=ACCENT,
                        children=[
                            html.Iframe(
                                id="viewport-iframe",
                                src="/viewport/",
                                style={
                                    "width": "100%",
                                    "flex": "1",
                                    "border": "none",
                                    "display": "block",
                                },
                            ),
                        ],
                    ),
                ]),

                # ============================================================
                # RIGHT COLUMN â€” Metrics + Inspector (PRD Â§4.3: 30%)
                # ============================================================
                html.Div(id="cf-right", children=[

                    # -- Metrics card (time-series + field heatmap) --
                    html.Div(className="cf-card",
                             style={"flexShrink": "0"}, children=[

                        html.Div(className="cf-panel-header", children=[
                            html.Div("Metrics", className="cf-panel-title"),
                        ]),

                        html.Div(className="cf-chart-area",
                                 children=[
                            dcc.Graph(
                                id="timeseries-chart",
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d"],
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "cropforge_timeseries",
                                    },
                                },
                                style={"height": "240px", "width": "100%"},
                            ),

                            html.Hr(style={"border": "0",
                                           "borderTop": "1px solid #EAEAEA",
                                           "margin": "12px 0"}),

                            dcc.Graph(
                                id="heatmap-chart",
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "cropforge_heatmap",
                                        "height": 600, "width": 900,
                                    },
                                },
                                style={"height": "220px", "width": "100%"},
                            ),

                            # v0.6.0 â€” Expand to full 3D Terrain modal
                            html.Button(
                                "â›°ï¸ Open 3D Terrain Map",
                                id="open-terrain-modal-btn",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "background": "#F9F9F8",
                                    "border": "1px solid #EAEAEA",
                                    "borderRadius": "4px",
                                    "padding": "8px",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "color": TEXT_SEC,
                                    "cursor": "pointer",
                                    "letterSpacing": "0.04em",
                                    "transition": "background 0.15s",
                                },
                            ),
                        ]),
                    ]),  # end metrics card

                    # -- Yield Metrics card (v1.0.0 planting density) --
                    html.Div(className="cf-card",
                             style={"flexShrink": "0"}, children=[
                        html.Div(className="cf-panel-header", children=[
                            html.Div("Yield Metrics", className="cf-panel-title"),
                            html.Button(
                                "Calculate Yield",
                                id="calculate-yield-btn",
                                n_clicks=0,
                                style={
                                    "background": "#F9F9F8",
                                    "border": "1px solid #C0C0C0",
                                    "borderRadius": "6px",
                                    "padding": "6px 10px",
                                    "fontSize": "12px",
                                    "cursor": "pointer",
                                },
                            ),
                        ]),
                        html.Div(
                            id="yield-summary-panel",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "10px",
                                "padding": "2px 0",
                            },
                        ),
                    ]),

                    # -- Plant Inspector card (always-visible) --
                    html.Div(
                        className="cf-card",
                        id="inspector-panel",
                        style={
                            "flexShrink": "0",
                        },
                        children=[
                            html.Div(className="cf-panel-header", style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                            }, children=[
                                html.Div("Plant Inspector", className="cf-panel-title",
                                         style={"paddingBottom": "0", "borderBottom": "none"}),
                                html.Button(
                                    "âœ•",
                                    id="inspector-close-btn",
                                    n_clicks=0,
                                    style={
                                        "background": "transparent",
                                        "border": "none", "color": TEXT_DIM,
                                        "fontSize": "13px", "cursor": "pointer",
                                        "padding": "4px 6px", "lineHeight": "1",
                                        "borderRadius": "4px",
                                        "transition": "color 0.15s cubic-bezier(0.32,0.72,0,1)",
                                        "minWidth": "24px", "minHeight": "24px",
                                    },
                                ),
                            ]),

                            html.Div(style={"padding": "10px 16px 12px"}, children=[

                                html.Div(id="inspector-plant-id", style={
                                    "fontSize": "13px", "fontWeight": "700",
                                    "color": TEXT_PRI, "marginBottom": "10px",
                                    "fontFamily": "ui-monospace, monospace",
                                    "fontVariantNumeric": "tabular-nums",
                                }),

                                # Placeholder â€” shown when no plant is selected
                                html.Div(
                                    id="inspector-content",
                                    children=html.Div(
                                        "Click any plant in the 3D Field View to inspect its state.",
                                        className="cf-inspector-placeholder",
                                    ),
                                ),

                                html.Div(id="inspector-soil-chart"),
                            ]),
                        ],
                    ),  # end inspector card

                ]),  # end right column

            ]),  # end cf-shell

        ]
    )


    # ==================================================================
    # Callbacks
    # ==================================================================

    # ------------------------------------------------------------------
    # Clientside callback 0: Dismiss Dash-level loading overlay
    # Listens for LOAD_COMPLETE postMessage from the Three.js iframe.
    # Fires once; adds 'hidden' class to #cf-loading-overlay via CSS
    # transition (opacity 0 + visibility hidden, 500ms ease).
    # ------------------------------------------------------------------
    app.clientside_callback(
        """
        function(iframeId) {
            if (window._cf_load_listener_registered) return window.dash_clientside.no_update;
            window._cf_load_listener_registered = true;
            window.addEventListener('message', function(evt) {
                if (evt.data && evt.data.type === 'LOAD_COMPLETE') {
                    var overlay = document.getElementById('cf-loading-overlay');
                    if (overlay) {
                        overlay.classList.add('hidden');
                    }
                }
            });
            /* Also hide overlay after 8 s max (fallback for slow networks) */
            setTimeout(function() {
                var overlay = document.getElementById('cf-loading-overlay');
                if (overlay) overlay.classList.add('hidden');
            }, 8000);
            return window.dash_clientside.no_update;
        }
        """,
        Output("cf-loading-overlay", "id"),  # dummy â€” id never changes
        Input("viewport-iframe", "id"),
        prevent_initial_call=False,
    )

    @app.callback(
        Output("timeseries-chart", "figure"),
        Input("ts-variable-dropdown", "value"),
    )
    def update_timeseries(variable: str):
        """Panel 2: Multi-field time-series with season boundary markers.

        Always plots ALL fields on the same chart with distinct colours.
        v0.4.0: draws a vertical dashed line at the first day of each
        season > 1 (annotated 'Season N Starts') so multi-season runs are
        visually clear.  Single-season logs see no change.
        """
        if daily_metrics.empty or variable not in daily_metrics.columns:
            return _empty_figure("No plant data available")

        label_map = {
            "mean_biomass_g":     "Mean Biomass (g/plant)",
            "mean_lai":           "Mean LAI (m\u00b2/m\u00b2)",
            "mean_height_cm":     "Mean Height (cm)",
            "mean_root_depth_cm": "Mean Root Depth (cm)",
            "mean_stress_index":  "Mean Stress Index",
            "alive_count":        "Alive Plants",
            "dead_count":         "Dead Plants",
        }
        y_label = label_map.get(variable, variable)

        # Merge soil metrics if available
        plot_df = daily_metrics.copy()
        if not daily_soil.empty:
            plot_df = plot_df.merge(daily_soil, on=["day", "field_name"], how="left")

        # Sort fields so order is deterministic
        fields = sorted(plot_df["field_name"].unique())
        fig = go.Figure()

        # Brand palette per PRD Â§4.4: primary green, secondary greens, red scale for stress
        colors = ["#4CAF7D", "#81C784", "#A5D6A7", "#E57373", "#EF9A9A"]
        for i, field in enumerate(fields):
            field_data = plot_df[plot_df["field_name"] == field].sort_values("day")
            fig.add_trace(go.Scatter(
                x=field_data["day"],
                y=field_data[variable],
                mode="lines+markers",
                name=field,
                line={"color": colors[i % len(colors)], "width": 2.5},
                marker={"size": 4},
            ))

        # v0.4.0 â€” Season boundary vertical lines (PRD Â§7.5)
        # Computed from the env table so the boundary is the actual
        # first Parquet day tagged season > 1, not a synthetic estimate.
        season_boundaries = _get_season_boundaries(env_df)
        for boundary_day, season_num in season_boundaries:
            fig.add_vline(
                x=boundary_day,
                line_color="#f59e0b",
                line_dash="dash",
                line_width=1.5,
                opacity=0.7,
                annotation_text=f"Season {season_num} Starts",
                annotation_position="top right",
                annotation_font={"size": 10, "color": "#f59e0b"},
            )

        fig.update_layout(
            **_chart_layout(),
            xaxis_title="Simulation Day",
            yaxis_title=y_label,
            showlegend=True,
            legend={
                "font": {"size": 10, "color": "#78a88a"},
                "bgcolor": "rgba(11,15,13,0.8)",
                "bordercolor": "rgba(76,175,125,0.15)",
                "borderwidth": 1,
                "x": 0.01, "y": 0.99,
                "xanchor": "left", "yanchor": "top",
            },
            margin={"l": 48, "r": 12, "t": 12, "b": 32},
        )
        return fig

    # ------------------------------------------------------------------
    # v0.6.0 â€” Terrain modal: toggle open/close via dcc.Store
    # Two callbacks: (1) button clicks â†’ store, (2) store â†’ overlay style.
    # ------------------------------------------------------------------
    @app.callback(
        Output("terrain-modal-open", "data"),
        Input("open-terrain-modal-btn",  "n_clicks"),
        Input("close-terrain-modal-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_terrain_modal(open_clicks, close_clicks):
        """Open modal on open-btn click; close on close-btn click."""
        from dash import ctx
        if not ctx.triggered_id:
            return False
        return ctx.triggered_id == "open-terrain-modal-btn"

    @app.callback(
        Output("terrain-modal-overlay", "style"),
        Input("terrain-modal-open", "data"),
        prevent_initial_call=False,
    )
    def show_terrain_modal(is_open):
        """Show/hide the terrain modal overlay div."""
        if is_open:
            return {
                "display": "flex",
                "position": "fixed", "inset": "0", "zIndex": "1000",
                "background": "rgba(0,0,0,0.55)",
                "alignItems": "center", "justifyContent": "center",
            }
        return {"display": "none"}

    def _build_terrain_surface(overlay_var: str, day: int, selected_field: str):
        """v0.6.0 â€” Helper to render go.Surface for the Terrain Map.
        z = elevation_grid (physical metres), surfacecolor = overlay variable.
        Upsampled 4Ã— with scipy.ndimage.zoom for high-res visual quality (PRD v0.9.0 Â§5).
        """
        import numpy as np
        from scipy.ndimage import zoom as _zoom

        terrain_info = None
        if _DATA["terrain"] and selected_field and selected_field in _DATA["terrain"]:
            terrain_info = _DATA["terrain"][selected_field]

        if terrain_info is None:
            return _empty_figure("No terrain data. Run simulation with set_terrain().")

        rows  = terrain_info["rows"]
        cols  = terrain_info["cols"]
        res   = terrain_info.get("resolution_m", 1.0)
        elev_grid = np.array(terrain_info["elevation_flat"]).reshape(rows, cols)
        x_m = [c * res for c in range(cols)]
        y_m = [r * res for r in range(rows)]

        # Determine surface colour data and scale
        if not overlay_var or overlay_var == "__elevation__":
            surface_color = elev_grid
            color_label   = "Elevation (m)"
            colorscale    = [[0.0, "#3b5ea6"], [0.25, "#5da832"],
                              [0.55, "#c8a45a"], [0.80, "#8b5e3c"], [1.0, "#f5f5f5"]]
        else:
            label_map = {
                "moisture_pct":             "Soil Moisture (%)",
                "nitrogen_kg_ha":           "Nitrogen (kg/ha)",
                "biomass_g":                "Biomass (g/plant)",
                "stress_index":             "Stress Index",
                "weed_lai":                 "Weed LAI",
                "weed_density_m2":          "Weed Density (m2)",
                # v0.7.0
                "surface_runoff_mm_today":  "Surface Runoff (mm)",
                "cumulative_erosion_index": "Cumulative Erosion Index",
            }
            colorscale_map_overlay = {
                "moisture_pct":             "RdYlGn",
                "nitrogen_kg_ha":           "RdYlGn",
                "biomass_g":                "RdYlGn",
                "stress_index":             "RdYlGn_r",
                "weed_lai":                 "YlGn",
                "weed_density_m2":          "YlOrBr",
                "surface_runoff_mm_today":  "Blues",
                "cumulative_erosion_index": "Reds",
            }
            color_label = label_map.get(overlay_var, overlay_var)
            colorscale  = colorscale_map_overlay.get(overlay_var, "RdYlGn")
            surface_color = elev_grid  # fallback

            # Try plants table
            overlay_src = None
            if plants_df is not None and not plants_df.empty:
                day_field = plants_df[
                    (plants_df["day"] == int(day)) &
                    (plants_df["field_name"] == selected_field)
                ]
                if overlay_var in day_field.columns and not day_field.empty:
                    overlay_src = day_field
                elif overlay_var in {"weed_lai", "weed_density_m2"} and not day_field.empty:
                    import json as _json

                    overlay_src = day_field.copy()
                    overlay_src[overlay_var] = overlay_src["custom_json"].apply(
                        lambda raw: float((_json.loads(raw or "{}")).get(overlay_var, 0.0))
                    )

            # Fall back to soil table
            if overlay_src is None and soil_df is not None and not soil_df.empty:
                soil_day = soil_df[
                    (soil_df["day"] == int(day)) &
                    (soil_df["field_name"] == selected_field) &
                    (soil_df["layer"] == 0)
                ]
                if overlay_var in soil_day.columns and not soil_day.empty:
                    overlay_src = soil_day

            if overlay_src is not None and not overlay_src.empty:
                try:
                    pivot = overlay_src.pivot_table(
                        index="row", columns="col", values=overlay_var, aggfunc="mean"
                    )
                    sc = np.full((rows, cols), float(pivot.values.mean()))
                    for ri in pivot.index:
                        for ci in pivot.columns:
                            if 0 <= int(ri) < rows and 0 <= int(ci) < cols:
                                sc[int(ri), int(ci)] = pivot.at[ri, ci]
                    surface_color = sc
                except Exception:
                    pass  # keep elevation fallback

        # Upsample both grids 4Ã— â€” z (bicubic) and surfacecolor (linear) must match shape.
        # ponytail: zoom both here, once, so every caller gets the same treatment.
        ZOOM = 4.0
        elev_up   = _zoom(elev_grid,    ZOOM, order=3)
        color_up  = _zoom(surface_color, ZOOM, order=1)

        # Regenerate x/y coords from upsampled shape (stays in physical metres)
        rows_up, cols_up = elev_up.shape
        x_up = [c * res / ZOOM for c in range(cols_up)]
        y_up = [r * res / ZOOM for r in range(rows_up)]

        fig = go.Figure(go.Surface(
            z=elev_up.tolist(),
            x=x_up,
            y=y_up,
            surfacecolor=color_up.tolist(),
            colorscale=colorscale,
            colorbar=dict(title=color_label, thickness=14, len=0.55),
            lighting=dict(roughness=0.9, specular=0.1, ambient=0.7),
            hovertemplate="X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Elev: %{z:.2f} m<extra></extra>",
        ))
        fig.update_layout(
            **_chart_layout(),
            margin={"l": 0, "r": 0, "t": 32, "b": 0},
            scene=dict(
                xaxis=dict(
                    title="East (m)",
                    showbackground=False, showgrid=False,
                    zeroline=False, showticklabels=False,
                ),
                yaxis=dict(
                    title="North (m)",
                    showbackground=False, showgrid=False,
                    zeroline=False, showticklabels=False,
                ),
                zaxis=dict(
                    title="Elevation (m)",
                    showbackground=False, showgrid=False,
                    zeroline=False, showticklabels=False,
                ),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.2),
                bgcolor="#F9F9F8",
            ),
            annotations=[{
                "text": f"Day {day}  \u2022  {selected_field or ''}",
                "x": 0.01, "y": 1.0, "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"size": 11, "color": "#64748b"},
                "xanchor": "left",
            }],
        )
        return fig

    @app.callback(
        Output("heatmap-var-container", "style"),
        Output("terrain-var-container", "style"),
        Output("open-terrain-modal-btn", "style"),
        Input("spatial-view-toggle", "value"),
    )
    def toggle_spatial_controls(view_mode):
        btn_style = {
            "width": "100%", "background": "#F9F9F8", "border": "1px solid #EAEAEA",
            "borderRadius": "4px", "padding": "8px", "fontSize": "12px",
            "fontWeight": "600", "color": TEXT_SEC, "cursor": "pointer",
            "letterSpacing": "0.04em", "transition": "background 0.15s",
        }
        if view_mode == "2d":
            return {"display": "block"}, {"display": "none"}, {"display": "none", **btn_style}
        else:
            return {"display": "none"}, {"display": "block"}, {"display": "block", **btn_style}

    @app.callback(
        Output("heatmap-chart", "figure"),
        Input("spatial-view-toggle", "value"),
        Input("heatmap-variable-dropdown", "value"),
        Input("sidebar-terrain-overlay-dropdown", "value"),
        Input("day-scrubber", "value"),
        Input("field-selector", "value"),
    )
    def update_heatmap(view_mode: str, variable: str, overlay_var: str, day: int, selected_field: str):
        """Panel 2: Renders either the 2D Heatmap or the 3D Terrain 'mini view'."""
        if view_mode == "3d":
            return _build_terrain_surface(overlay_var, day, selected_field)
            
        # --- 2D Heatmap Logic ---
        if plants_df is None or plants_df.empty:
            return _empty_figure("No plant data available")

        field_df = plants_df
        if selected_field:
            field_df = plants_df[plants_df["field_name"] == selected_field]
        if field_df.empty:
            return _empty_figure(f"No data for field '{selected_field}'")

        day_data = field_df[field_df["day"] == int(day)]
        if day_data.empty:
            return _empty_figure(f"No data for day {day}")

        # For soil-sourced variables (runoff, erosion), fall back to soil_df
        # ponytail: simple column-existence check; same fallback pattern as _build_terrain_surface
        _soil_vars = {"surface_runoff_mm_today", "cumulative_erosion_index"}
        if variable in _soil_vars and soil_df is not None and not soil_df.empty:
            day_data = soil_df[
                (soil_df["day"] == int(day)) &
                (soil_df["field_name"] == (selected_field or "")) &
                (soil_df["layer"] == 0)
            ]
            if day_data.empty:
                return _empty_figure(f"No soil data for day {day} (enable erosion=True)")
        elif variable in {"weed_lai", "weed_density_m2"} and variable not in day_data.columns:
            import json as _json

            day_data = day_data.copy()
            day_data[variable] = day_data["custom_json"].apply(
                lambda raw: float((_json.loads(raw or "{}")).get(variable, 0.0))
            )

        try:
            pivot = day_data.pivot_table(
                index="row", columns="col", values=variable, aggfunc="mean"
            )
        except Exception:
            return _empty_figure(f"Cannot build heatmap for '{variable}'")

        label_map = {
            "biomass_g":                 "Biomass (g/plant)",
            "lai":                       "LAI (m\u00b2/m\u00b2)",
            "weed_lai":                  "Weed LAI",
            "weed_density_m2":           "Weed Density (m\u00b2)",
            "height_cm":                 "Height (cm)",
            "stress_index":              "Stress Index",
            "surface_runoff_mm_today":   "Surface Runoff (mm)",
            "cumulative_erosion_index":  "Cumulative Erosion Index",
        }
        colorscale_map = {
            "biomass_g":                 "Viridis",
            "lai":                       "YlGn",
            "weed_lai":                  "YlGn",
            "weed_density_m2":           "YlOrBr",
            "height_cm":                 "Blues",
            "stress_index":              "RdYlGn_r",
            "surface_runoff_mm_today":   "Blues",
            "cumulative_erosion_index":  "Reds",
        }
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=colorscale_map.get(variable, "Viridis"),
            showscale=True,
            hoverongaps=False,
            hovertemplate=(
                f"Row: %{{y}}<br>Col: %{{x}}<br>"
                f"{label_map.get(variable, variable)}: %{{z:.3f}}<extra></extra>"
            ),
        ))
        fig.update_layout(
            **_chart_layout(),
            xaxis_title="Column",
            yaxis_title="Row",
            margin={"l": 48, "r": 12, "t": 12, "b": 32},
            annotations=[
                {"text": f"Day {day}", "x": 0.98, "y": 0.98,
                 "xref": "paper", "yref": "paper", "showarrow": False,
                 "font": {"size": 12, "color": "#4a9eff"}, "xanchor": "right"},
                {"text": selected_field or "", "x": 0.01, "y": 0.98,
                 "xref": "paper", "yref": "paper", "showarrow": False,
                 "font": {"size": 11, "color": "#64748b"}, "xanchor": "left"},
            ],
        )
        return fig

    @app.callback(
        Output("yield-summary-panel", "children"),
        Input("calculate-yield-btn", "n_clicks"),
        prevent_initial_call=False,
    )
    def update_yield_summary(_n_clicks):
        """Display final-day yield metrics computed from logged plant density."""
        def metric_tile(label: str, value: str, span: bool = False):
            return html.Div(
                [
                    html.Div(label, style={
                        "fontSize": "11px",
                        "textTransform": "uppercase",
                        "color": "#64748B",
                        "letterSpacing": "0.04em",
                    }),
                    html.Div(value, style={
                        "fontSize": "20px",
                        "fontWeight": "700",
                        "fontVariantNumeric": "tabular-nums",
                        "marginTop": "4px",
                    }),
                ],
                style={
                    "border": "1px solid #EAEAEA",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "gridColumn": "1 / -1" if span else "auto",
                },
            )

        summary = yield_summary_data
        fields = summary.get("fields", {})
        if not fields:
            return [metric_tile("Yield", "No harvest data", span=True)]

        children = [
            metric_tile("Yield kg/ha", f"{summary['yield_kg_per_ha']:,.1f}"),
            metric_tile("Total yield kg", f"{summary['total_yield_kg']:,.2f}"),
        ]
        breakdown = []
        for field_name, data in fields.items():
            breakdown.append(
                html.Div(
                    f"{field_name}: {data['yield_kg_per_ha']:,.1f} kg/ha, "
                    f"{data['total_yield_kg']:,.2f} kg total",
                    style={
                        "fontSize": "12px",
                        "fontVariantNumeric": "tabular-nums",
                        "padding": "2px 0",
                    },
                )
            )
        children.append(html.Div(
            [html.Div("Fields", style={
                "fontSize": "11px",
                "textTransform": "uppercase",
                "color": "#64748B",
                "letterSpacing": "0.04em",
                "marginBottom": "4px",
            })] + breakdown,
            style={
                "gridColumn": "1 / -1",
                "borderTop": "1px solid #EAEAEA",
                "paddingTop": "8px",
            },
        ))
        return children

    @app.callback(
        Output("terrain-modal-chart", "figure"),
        Input("open-terrain-modal-btn", "n_clicks"),
        Input("terrain-overlay-dropdown", "value"),
        Input("day-scrubber", "value"),
        Input("field-selector", "value"),
        prevent_initial_call=False,
    )
    def update_terrain_modal(n_clicks, overlay_var: str, day: int, selected_field: str):
        """v0.6.0 â€” Render go.Surface for the Terrain Map modal."""
        return _build_terrain_surface(overlay_var, day, selected_field)

    # ------------------------------------------------------------------
    # Clientside callback 1: Dash slider â†’ Three.js iframe (PRD Â§7.3)
    # ------------------------------------------------------------------
    app.clientside_callback(
        """
        function(day) {
            var iframe = document.getElementById('viewport-iframe');
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage(
                    { type: 'cf_set_day', day: day },
                    '*'
                );
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("viewport-iframe", "id"),
        Input("day-scrubber", "value"),
        prevent_initial_call=True,
    )

    # ponytail: Collapsible sidebars minimal clientside callback
    app.clientside_callback(
        """
        function(l_clicks, r_clicks) {
            var l_open = (l_clicks % 2 === 0);
            var r_open = (r_clicks % 2 === 0);
            
            var l_w = l_open ? "280px" : "0px";
            var r_w = r_open ? "380px" : "0px";
            
            var l_style = l_open ? {transition: "opacity 0.2s"} : {opacity: 0, pointerEvents: "none", overflow: "hidden", transition: "opacity 0.2s"};
            var r_style = r_open ? {transition: "opacity 0.2s"} : {opacity: 0, pointerEvents: "none", overflow: "hidden", transition: "opacity 0.2s"};
            
            return [{"gridTemplateColumns": l_w + " 1fr " + r_w}, l_style, r_style];
        }
        """,
        Output("cf-shell", "style"),
        Output("cf-sidebar", "style"),
        Output("cf-right", "style"),
        Input("toggle-left-btn", "n_clicks"),
        Input("toggle-right-btn", "n_clicks"),
        prevent_initial_call=False,
    )

    # ------------------------------------------------------------------
    # Clientside callback 1b: Field Selector â†’ Three.js iframe re-bootstrap
    # PRD v0.2.0 Â§8 (Multi-Field Frontend)
    # When the user picks a different field, post cf_set_field to the
    # iframe so it re-fetches /api/buffer/meta?field=<name> and reloads.
    # ------------------------------------------------------------------
    app.clientside_callback(
        """
        function(fieldName) {
            var iframe = document.getElementById('viewport-iframe');
            if (iframe && iframe.contentWindow && fieldName) {
                iframe.contentWindow.postMessage(
                    { type: 'cf_set_field', field: fieldName },
                    '*'
                );
            }
            /* Also store in selected-field-store for Python callbacks */
            return fieldName;
        }
        """,
        Output("selected-field-store", "data"),
        Input("field-selector", "value"),
        prevent_initial_call=False,
    )

    # ------------------------------------------------------------------
    # Clientside callback 2: Listen for PLANT_CLICKED postMessage
    # Writes plant identity into selected-plant-store so Python can react
    # ------------------------------------------------------------------
    app.clientside_callback(
        """
        function(_trigger) {
            /* Install listener once; guard with a flag on window */
            if (!window._cf_msg_listener) {
                window._cf_msg_listener = true;
                window.addEventListener('message', function(evt) {
                    if (!evt.data || typeof evt.data !== 'object') return;

                    if (evt.data.type === 'PLANT_CLICKED') {
                        var store = {
                            plant_id: evt.data.plant_id,
                            row:      evt.data.row,
                            col:      evt.data.col,
                            day:      evt.data.day,
                            ts:       Date.now()
                        };
                        /* Write to Dash store via a hidden element trick */
                        window._cf_pending_plant = store;
                        /* Trigger the polling interval */
                        var el = document.getElementById('plant-msg-trigger');
                        if (el) {
                            /* Nudge the interval by updating its data attribute */
                            el.setAttribute('data-plant', JSON.stringify(store));
                            el.dispatchEvent(new Event('change'));
                        }
                    }

                    if (evt.data.type === 'PLANT_DESELECTED') {
                        window._cf_pending_plant = null;
                        var el = document.getElementById('plant-msg-trigger');
                        if (el) {
                            el.setAttribute('data-plant', 'null');
                            el.dispatchEvent(new Event('change'));
                        }
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("selected-plant-store", "id"),  # dummy â€” id never changes
        Input("viewport-iframe",       "id"),
        prevent_initial_call=False,
    )

    # ------------------------------------------------------------------
    # Clientside callback 3: Polling interval â†’ selected-plant-store
    # Reads window._cf_pending_plant (set by the message listener above)
    # and writes it into the Dash store so Python callbacks can fire.
    # ------------------------------------------------------------------
    app.clientside_callback(
        """
        function(n) {
            var p = window._cf_pending_plant;
            if (p === undefined) return window.dash_clientside.no_update;
            /* Consume: reset so we don't re-fire on next tick */
            window._cf_pending_plant = undefined;
            if (p === null) return null;  /* deselect */
            return p;
        }
        """,
        Output("selected-plant-store", "data"),
        Input("inspector-poll",         "n_intervals"),
    )

    # ------------------------------------------------------------------
    # Python callback: selected-plant-store â†’ Panel 4 inspector UI
    # ------------------------------------------------------------------
    @app.callback(
        Output("inspector-panel",    "style"),
        Output("inspector-plant-id", "children"),
        Output("inspector-content",  "children"),
        Output("inspector-soil-chart", "children"),
        Input("selected-plant-store", "data"),
        Input("inspector-close-btn",  "n_clicks"),
        State("day-scrubber",         "value"),
        State("selected-field-store", "data"),
        prevent_initial_call=True,
    )
    def update_inspector(plant_data, close_clicks, current_day, selected_field):
        """Panel 4: render plant history and soil cross-section."""
        import json as _json
        from dash import ctx

        # Determine which input fired
        trigger_id = ctx.triggered_id if ctx.triggered_id else ""

        # Close button or deselect â†’ show placeholder
        if trigger_id == "inspector-close-btn" or plant_data is None:
            return {}, "", html.Div(
                "Click any plant in the Field View to inspect it.",
                style={"fontSize": "11px", "color": "#3d5c47",
                       "marginTop": "12px", "textAlign": "center",
                       "lineHeight": "1.6"},
            ), html.Div()

        # Extract identity
        if isinstance(plant_data, str):
            try:
                plant_data = _json.loads(plant_data)
            except Exception:
                return {}, "", html.Div(), html.Div()

        plant_id = plant_data.get("plant_id", "")
        p_row    = int(plant_data.get("row", 0))
        p_col    = int(plant_data.get("col", 0))
        day      = int(current_day or plant_data.get("day", day_min))

        # Query plant data for this cell across ALL days, filtered by field
        plant_history = pd.DataFrame()
        day_slice     = pd.Series(dtype=object)
        if plants_df is not None and not plants_df.empty:
            src = plants_df
            if selected_field:
                src = plants_df[plants_df["field_name"] == selected_field]
            mask = (
                (src["row"].astype(int) == p_row) &
                (src["col"].astype(int) == p_col)
            )
            plant_history = src[mask].sort_values("day")
            today = plant_history[plant_history["day"] == day]
            if not today.empty:
                day_slice = today.iloc[0]

        # ---- Stat rows for current day --------------------------------
        def _fmt(v, decimals=3):
            try:
                return f"{float(v):.{decimals}f}"
            except Exception:
                return str(v)

        stat_pairs = [
            ("Field",        selected_field or day_slice.get("field_name", "â€”")),
            ("Row",          p_row),
            ("Col",          p_col),
            ("Alive",        bool(day_slice.get("alive", True)) if len(day_slice) else "â€”"),
            ("Biomass (g)",  _fmt(day_slice.get("biomass_g",  0))),
            ("LAI (mÂ²/mÂ²)", _fmt(day_slice.get("lai",         0))),
            ("Height (cm)",  _fmt(day_slice.get("height_cm",  0), 1)),
            ("Root depth (cm)", _fmt(day_slice.get("root_depth_cm", 0), 1)),
            ("Stress index", _fmt(day_slice.get("stress_index", 0))),
            ("Stage",        day_slice.get("phenological_stage", "â€”")),
        ]

        stat_rows = []
        for label, val in stat_pairs:
            stat_rows.append(html.Div(style=_STYLE_STAT_ROW, children=[
                html.Span(label, style=_STYLE_STAT_LABEL),
                html.Span(str(val), style=_STYLE_STAT_VALUE),
            ]))

        # ---- Biomass time-series mini chart ---------------------------
        ts_chart = html.Div()
        if not plant_history.empty and "biomass_g" in plant_history.columns:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=plant_history["day"],
                y=plant_history["biomass_g"],
                mode="lines",
                name="Biomass",
                line={"color": "#4a9eff", "width": 1.5},
                fill="tozeroy",
                fillcolor="rgba(74,158,255,0.08)",
            ))
            # Vertical marker at current day
            fig_ts.add_vline(
                x=day,
                line_color="#f59e0b",
                line_dash="dash",
                line_width=1,
            )
            fig_ts.update_layout(
                **_chart_layout(),
                height=120,
                margin={"l": 40, "r": 8, "t": 8, "b": 24},
                xaxis_title="Day",
                yaxis_title="g",
                showlegend=False,
            )
            ts_chart = html.Div([
                html.Div("Biomass History", style={
                    "fontSize": "10px", "color": "#64748b",
                    "marginTop": "12px", "marginBottom": "4px",
                    "textTransform": "uppercase", "letterSpacing": "0.08em",
                }),
                dcc.Graph(figure=fig_ts, config={"displayModeBar": False},
                          style={"height": "120px"}),
            ])

        inspector_children = html.Div([
            html.Div(f"Day {day} snapshot", style={
                "fontSize": "10px", "color": "#64748b",
                "marginBottom": "8px",
                "textTransform": "uppercase", "letterSpacing": "0.08em",
            }),
            *stat_rows,
            ts_chart,
        ])

        # ---- Soil vertical cross-section chart (PRD Â§7.2) -------------
        soil_chart = html.Div()
        if soil_df is not None and not soil_df.empty:
            src_soil = soil_df
            if selected_field:
                src_soil = soil_df[soil_df["field_name"] == selected_field]
            soil_mask = (
                (src_soil["row"].astype(int) == p_row) &
                (src_soil["col"].astype(int) == p_col) &
                (src_soil["day"].astype(int) == day)
            )
            cell_soil = src_soil[soil_mask].sort_values("layer")

            if not cell_soil.empty:
                depths     = cell_soil["depth_bottom_cm"].tolist()
                moistures  = cell_soil["moisture_pct"].tolist()
                nitrogens  = cell_soil["nitrogen_kg_ha"].tolist() \
                             if "nitrogen_kg_ha" in cell_soil.columns else []

                fig_soil = go.Figure()
                fig_soil.add_trace(go.Bar(
                    x=moistures,
                    y=depths,
                    orientation="h",
                    name="Moisture %",
                    marker_color="#3b82f6",
                    width=[abs(d - cell_soil["depth_top_cm"].tolist()[i])
                           for i, d in enumerate(depths)],
                ))
                if nitrogens:
                    fig_soil.add_trace(go.Bar(
                        x=nitrogens,
                        y=depths,
                        orientation="h",
                        name="N (kg/ha)",
                        marker_color="#22c55e",
                        width=[abs(d - cell_soil["depth_top_cm"].tolist()[i])
                               for i, d in enumerate(depths)],
                    ))
                fig_soil.update_layout(
                    **_chart_layout(),
                    height=180,
                    margin={"l": 40, "r": 8, "t": 8, "b": 24},
                    xaxis_title="Value",
                    yaxis_title="Depth (cm)",
                    yaxis_autorange="reversed",
                    barmode="group",
                    legend={"font": {"size": 9}},
                )
                soil_chart = html.Div([
                    html.Div("Soil Cross-Section", style={
                        "fontSize": "10px", "color": "#64748b",
                        "marginTop": "12px", "marginBottom": "4px",
                        "textTransform": "uppercase", "letterSpacing": "0.08em",
                    }),
                    dcc.Graph(figure=fig_soil, config={"displayModeBar": False},
                              style={"height": "180px"}),
                ])

        return {}, plant_id, inspector_children, soil_chart

    # ------------------------------------------------------------------
    # CSV Download callback (PRD v0.4.0 Â§8.2)
    # Exports the aggregated daily_metrics DataFrame (all fields, all days)
    # to a browser-downloadable CSV.  Named following the PRD convention:
    #   cropforge_timeseries_{session}_{YYYYMMDD}.csv
    # ------------------------------------------------------------------
    @app.callback(
        Output("download-csv", "data"),
        Input("export-csv-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks):
        """Serialise the aggregated time-series data to a CSV download.

        Delegates to the module-level ``build_csv_export()`` helper so that
        tests can exercise the CSV logic without Dash's callback context.
        """
        return build_csv_export(daily_metrics, daily_soil, env_df, session_name)

    return app



# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _chart_layout() -> dict:
    """Common layout properties for all Plotly figures (v0.5.0 Minimalist Theme).

    v0.5.0: Updated to premium utilitarian palette (PRD Â§4.4).
    """
    return {
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor":  "#FFFFFF",
        "font":          {"color": "#787774", "size": 11, "family": "'SF Pro Display', 'Geist Sans', sans-serif"},
        "xaxis": {
            "gridcolor": "#EAEAEA", "zerolinecolor": "#EAEAEA",
            "tickfont": {"color": "#9CA3AF", "size": 10},
        },
        "yaxis": {
            "gridcolor": "#EAEAEA", "zerolinecolor": "#EAEAEA",
            "tickfont": {"color": "#9CA3AF", "size": 10},
        },
    }


def _empty_figure(message: str = "No data") -> go.Figure:
    """Return an empty figure with a centred message."""
    fig = go.Figure()
    fig.update_layout(
        **_chart_layout(),
        annotations=[{
            "text": message,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 13, "color": "#9CA3AF"},
        }],
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
    )
    return fig


