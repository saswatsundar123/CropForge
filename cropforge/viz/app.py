"""
cropforge/viz/app.py
====================
Plotly Dash application — CropForge Phase 2+3+4 Dashboard Frontend.

PRD References:
    Section 7.1 — Served by FastAPI on port 7860
    Section 7.2 — Four-panel layout
    Section 7.3 — Panel 1: Three.js iframe at /viewport/;
                  raycasting postMessage PLANT_CLICKED → Panel 4
    Section 7.2 — Panel 4: Farm Inspector sidebar (collapsed by default)
    Section 16  — Parquet schema driving the data layer

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

    v0.4.0 — Season column:
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
        Aggregated plant metrics (day × field_name).
    daily_soil:
        Aggregated topsoil metrics (day × field_name), or empty DataFrame.
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

    # Load data once (idempotent — skip if already cached by boot())
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
        {"label": "Height (cm)",          "value": "height_cm"},
        {"label": "Stress Index",         "value": "stress_index"},
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

    # ==================================================================
    # Layout
    # ==================================================================

    app = dash.Dash(
        __name__,
        title="CropForge Dashboard",
        suppress_callback_exceptions=True,
        external_stylesheets=[],
    )

    # ---- Inline CSS (no CDN — fully offline per PRD Section 7.1) -----
    _STYLE_MAIN = {
        "fontFamily": "'Segoe UI', Arial, sans-serif",
        "background": "#0f1117",
        "color": "#e2e8f0",
        "minHeight": "100vh",
        "margin": "0",
        "padding": "0",
    }
    _STYLE_HEADER = {
        "background": "linear-gradient(90deg, #1a2332 0%, #162032 100%)",
        "borderBottom": "1px solid #2d3748",
        "padding": "12px 24px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
    }
    _STYLE_CONTENT = {
        "display": "grid",
        "gridTemplateColumns": "60fr 40fr",
        "gridTemplateRows": "auto",
        "gap": "0",
        "height": "calc(100vh - 56px)",
    }
    _STYLE_PANEL = {
        "background": "#111827",
        "border": "1px solid #1e2a3a",
        "borderRadius": "4px",
        "margin": "8px",
        "padding": "16px",
        "overflow": "auto",
    }
    _STYLE_PANEL_TITLE = {
        "fontSize": "11px",
        "fontWeight": "700",
        "letterSpacing": "0.12em",
        "textTransform": "uppercase",
        "color": "#4a9eff",
        "marginBottom": "12px",
        "paddingBottom": "8px",
        "borderBottom": "1px solid #1e2a3a",
    }
    _STYLE_3D_PLACEHOLDER = {
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "center",
        "height": "calc(100% - 48px)",
        "background": "#0d1520",
        "border": "2px dashed #2d3748",
        "borderRadius": "6px",
        "color": "#4a5568",
    }
    _STYLE_RIGHT_COL = {
        "display": "flex",
        "flexDirection": "column",
        "gap": "0",
    }
    _STYLE_SELECT = {
        "background": "#1a2332",
        "color": "#e2e8f0",
        "border": "1px solid #2d3748",
        "borderRadius": "4px",
        "padding": "4px 8px",
        "fontSize": "13px",
        "width": "100%",
        "marginBottom": "8px",
    }
    _STYLE_BADGE = {
        "display": "inline-block",
        "background": "#1e3a5f",
        "color": "#4a9eff",
        "borderRadius": "4px",
        "padding": "2px 8px",
        "fontSize": "12px",
        "marginRight": "8px",
    }
    _STYLE_EVENT_LINE = {
        "padding": "4px 8px",
        "borderLeft": "2px solid #4a9eff",
        "marginBottom": "4px",
        "fontSize": "12px",
        "fontFamily": "monospace",
        "color": "#94a3b8",
        "background": "#0d1520",
        "borderRadius": "0 4px 4px 0",
    }
    _STYLE_INSPECTOR = {
        "position": "fixed",
        "right": "0",
        "top": "56px",
        "width": "340px",
        "height": "calc(100vh - 56px)",
        "background": "#111827",
        "borderLeft": "1px solid #2d3748",
        "padding": "16px",
        "overflowY": "auto",
        "transform": "translateX(340px)",
        "transition": "transform 0.3s ease",
        "zIndex": "100",
        "boxShadow": "-4px 0 24px rgba(0,0,0,0.5)",
    }
    _STYLE_INSPECTOR_OPEN = {
        **_STYLE_INSPECTOR,
        "transform": "translateX(0px)",
    }
    _STYLE_STAT_ROW = {
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "5px 0",
        "borderBottom": "1px solid #1e2a3a",
        "fontSize": "12px",
    }
    _STYLE_STAT_LABEL = {
        "color": "#64748b",
        "fontFamily": "monospace",
    }
    _STYLE_STAT_VALUE = {
        "color": "#e2e8f0",
        "fontWeight": "600",
    }

    # ---- Field selector options ----------------------------------------
    field_names = []
    if plants_df is not None and not plants_df.empty:
        field_names = sorted(plants_df["field_name"].unique().tolist())
    default_field = field_names[0] if field_names else ""
    field_options = [{"label": fn, "value": fn} for fn in field_names]

    app.layout = html.Div(
        style=_STYLE_MAIN,
        children=[

            # Hidden state stores
            dcc.Store(id="selected-plant-store", storage_type="memory"),
            dcc.Store(id="current-day-store",    storage_type="memory",
                      data=int(day_min)),
            dcc.Store(id="selected-field-store", storage_type="memory",
                      data=default_field),
            # Polling interval: fires every 250 ms to relay postMessage → Dash store
            dcc.Interval(id="inspector-poll", interval=250, n_intervals=0),
            # Hidden div: JS message listener writes plant data here as data attribute
            html.Div(id="plant-msg-trigger", style={"display": "none"}),


            # ---- Header bar ------------------------------------------
            html.Div(style=_STYLE_HEADER, children=[
                html.Div([
                    html.Span("CropForge", style={
                        "fontWeight": "800", "fontSize": "18px", "color": "#4a9eff",
                        "marginRight": "16px",
                    }),
                    html.Span("Dashboard", style={
                        "fontSize": "14px", "color": "#64748b",
                    }),
                ]),
                html.Div([
                    html.Span(f"Session: {session_name}", style=_STYLE_BADGE),
                    html.Span(f"{n_days} days", style=_STYLE_BADGE),
                    html.Span(f"{n_plants} plants", style=_STYLE_BADGE),
                    html.Span(f"{n_fields} field(s)", style=_STYLE_BADGE),
                    # v0.4.0 — CSV export button in header (PRD §8.2)
                    html.Button(
                        "⬇ Export CSV",
                        id="export-csv-btn",
                        n_clicks=0,
                        style={
                            "background": "#1e3a5f",
                            "color": "#4a9eff",
                            "border": "1px solid #2d5a8e",
                            "borderRadius": "4px",
                            "padding": "4px 12px",
                            "fontSize": "12px",
                            "cursor": "pointer",
                            "fontWeight": "600",
                            "marginLeft": "8px",
                            "transition": "background 0.2s",
                        },
                    ),
                    # dcc.Download — the actual file-delivery component
                    dcc.Download(id="download-csv"),
                ]),
            ]),

            # ---- Main content grid -----------------------------------
            html.Div(style=_STYLE_CONTENT, children=[

                # ========================================================
                # Panel 1: 3D Farm View — Three.js iframe (Phase 3)
                # ========================================================
                html.Div(style={**_STYLE_PANEL, "gridRow": "1 / 3",
                                "padding": "0", "overflow": "hidden"}, children=[
                    html.Iframe(
                        id="viewport-iframe",
                        src="/viewport/",
                        style={
                            "width": "100%",
                            "height": "100%",
                            "border": "none",
                            "display": "block",
                            "minHeight": "480px",
                        },
                    ),
                ]),

                # ========================================================
                # Right column wrapper
                # ========================================================
                html.Div(style=_STYLE_RIGHT_COL, children=[

                    # ====================================================
                    # Panel 2: Metrics Dashboard (upper right)
                    # ====================================================
                    html.Div(style={**_STYLE_PANEL, "flex": "1", "minHeight": "0"}, children=[
                        html.Div("Panel 2: Metrics Dashboard", style=_STYLE_PANEL_TITLE),
                        # ---- Field Selector (v0.2.0 Multi-Field) -------
                        # Single unconditional dropdown — always rendered so
                        # callbacks wire correctly. Wrapper is hidden for
                        # single-field sessions; the badge shows the field name.
                        html.Div(style={
                            "display": "flex" if len(field_names) > 1 else "none",
                            "gap": "12px",
                            "alignItems": "center",
                            "marginBottom": "8px",
                        }, children=[
                            html.Div("Active Field (3D + Heatmap)", style={
                                "fontSize": "11px", "color": "#64748b",
                                "fontWeight": "600", "whiteSpace": "nowrap",
                                "flexShrink": "0",
                            }),
                        ]),
                        # Field badge for single-field sessions (cosmetic only)
                        html.Div(style={
                            "display": "block" if len(field_names) <= 1 else "none",
                            "marginBottom": "6px",
                        }, children=[
                            html.Span(
                                default_field or "—",
                                style={**_STYLE_BADGE, "display": "inline-block"},
                            ),
                        ]),
                        # The one and only field-selector dropdown
                        dcc.Dropdown(
                            id="field-selector",
                            options=field_options,
                            value=default_field,
                            clearable=False,
                            style={
                                **_STYLE_SELECT,
                                "marginBottom": "8px",
                                "display": "block" if len(field_names) > 1 else "none",
                            },
                        ),


                        # ---- Time-series chart -------------------------
                        html.Div("Time-Series Variable", style={
                            "fontSize": "11px", "color": "#64748b",
                            "marginBottom": "4px", "fontWeight": "600",
                        }),
                        dcc.Dropdown(
                            id="ts-variable-dropdown",
                            options=plant_metric_options,
                            value="mean_root_depth_cm",
                            clearable=False,
                            style=_STYLE_SELECT,
                        ),
                        dcc.Graph(
                            id="timeseries-chart",
                            config={"displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d"],
                                    "toImageButtonOptions": {
                                        "format": "png", "filename": "cropforge_timeseries"
                                    }},
                            style={"height": "220px"},
                        ),

                        html.Hr(style={"border": "0", "borderTop": "1px solid #1e2a3a",
                                       "margin": "12px 0"}),

                        # ---- Spatial heatmap + scrubber ----------------
                        html.Div("Spatial Heatmap — Variable", style={
                            "fontSize": "11px", "color": "#64748b",
                            "marginBottom": "4px", "fontWeight": "600",
                        }),
                        dcc.Dropdown(
                            id="heatmap-variable-dropdown",
                            options=spatial_options,
                            value="biomass_g",
                            clearable=False,
                            style=_STYLE_SELECT,
                        ),
                        html.Div("Simulation Day", style={
                            "fontSize": "11px", "color": "#64748b",
                            "marginBottom": "4px", "marginTop": "8px", "fontWeight": "600",
                        }),
                        dcc.Slider(
                            id="day-scrubber",
                            min=day_min,
                            max=day_max,
                            step=1,
                            value=day_max,
                            marks=day_marks,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        dcc.Graph(
                            id="heatmap-chart",
                            config={"displayModeBar": False},
                            style={"height": "240px"},
                        ),
                    ]),

                    # ====================================================
                    # Panel 3: Event Log (lower right)
                    # ====================================================
                    html.Div(style={**_STYLE_PANEL, "flex": "0 0 180px",
                                    "overflowY": "auto"}, children=[
                        html.Div("Panel 3: Event Log", style=_STYLE_PANEL_TITLE),
                        html.Div(
                            id="event-log-content",
                            children=[
                                html.Div(line, style=_STYLE_EVENT_LINE)
                                for line in event_lines
                            ],
                        ),
                    ]),

                ]),  # end right column

            ]),  # end main content grid

            # ========================================================
            # Panel 4: Farm Inspector sidebar (collapsed by default)
            # PRD Section 7.2: opens when a plant is clicked
            # ========================================================
            html.Div(
                id="inspector-panel",
                style=_STYLE_INSPECTOR,
                children=[
                    # Header row with close button
                    html.Div(style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "12px",
                        "paddingBottom": "8px",
                        "borderBottom": "1px solid #1e2a3a",
                    }, children=[
                        html.Span("Farm Inspector", style={
                            "fontSize": "11px",
                            "fontWeight": "700",
                            "letterSpacing": "0.12em",
                            "textTransform": "uppercase",
                            "color": "#4a9eff",
                        }),
                        html.Button(
                            "✕",
                            id="inspector-close-btn",
                            n_clicks=0,
                            style={
                                "background": "transparent",
                                "border": "none",
                                "color": "#64748b",
                                "fontSize": "14px",
                                "cursor": "pointer",
                                "padding": "0 4px",
                                "lineHeight": "1",
                            },
                        ),
                    ]),

                    # Plant ID badge
                    html.Div(id="inspector-plant-id", style={
                        "fontSize": "16px",
                        "fontWeight": "700",
                        "color": "#e2e8f0",
                        "marginBottom": "12px",
                        "fontFamily": "monospace",
                    }),

                    # Placeholder until a plant is clicked
                    html.Div(
                        id="inspector-content",
                        children=html.Div(
                            "Click a plant in the 3D view to inspect it.",
                            style={"fontSize": "12px", "color": "#64748b",
                                   "marginTop": "24px", "textAlign": "center"},
                        ),
                    ),

                    # Soil cross-section chart (empty until plant clicked)
                    html.Div(id="inspector-soil-chart"),
                ],
            ),


        ]
    )

    # ==================================================================
    # Callbacks
    # ==================================================================

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

        # Vibrant, clearly distinct colours per field
        colors = ["#4a9eff", "#34d399", "#f59e0b", "#f87171", "#a78bfa"]
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

        # v0.4.0 — Season boundary vertical lines (PRD §7.5)
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
            **_dark_layout(),
            xaxis_title="Simulation Day",
            yaxis_title=y_label,
            showlegend=True,
            legend={
                "font": {"size": 10, "color": "#94a3b8"},
                "bgcolor": "rgba(13,21,32,0.7)",
                "bordercolor": "#1e2a3a",
                "borderwidth": 1,
                "x": 0.01, "y": 0.99,
                "xanchor": "left", "yanchor": "top",
            },
            margin={"l": 48, "r": 12, "t": 12, "b": 32},
        )
        return fig

    @app.callback(
        Output("heatmap-chart", "figure"),
        Input("heatmap-variable-dropdown", "value"),
        Input("day-scrubber", "value"),
        Input("field-selector", "value"),
    )
    def update_heatmap(variable: str, day: int, selected_field: str):
        """Panel 2: Update the 2D spatial heatmap for a chosen day and field.

        Filters to the currently selected field so the heatmap always
        shows the spatial layout of exactly one field at a time.
        """
        if plants_df is None or plants_df.empty:
            return _empty_figure("No plant data available")

        # Filter by selected field
        field_df = plants_df
        if selected_field:
            field_df = plants_df[plants_df["field_name"] == selected_field]
        if field_df.empty:
            return _empty_figure(f"No data for field '{selected_field}'")

        day_data = field_df[field_df["day"] == int(day)]
        if day_data.empty:
            return _empty_figure(f"No data for day {day}")

        # Build pivot grid
        try:
            pivot = day_data.pivot_table(
                index="row", columns="col", values=variable, aggfunc="mean"
            )
        except Exception:
            return _empty_figure(f"Cannot build heatmap for '{variable}'")

        label_map = {
            "biomass_g": "Biomass (g/plant)",
            "lai":       "LAI (m\u00b2/m\u00b2)",
            "height_cm": "Height (cm)",
            "stress_index": "Stress Index",
        }
        colorscale_map = {
            "biomass_g":   "Viridis",
            "lai":         "YlGn",
            "height_cm":   "Blues",
            "stress_index":"RdYlGn_r",
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
            **_dark_layout(),
            xaxis_title="Column",
            yaxis_title="Row",
            margin={"l": 48, "r": 12, "t": 12, "b": 32},
            annotations=[
                {
                    "text": f"Day {day}",
                    "x": 0.98, "y": 0.98,
                    "xref": "paper", "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 12, "color": "#4a9eff"},
                    "xanchor": "right",
                },
                {
                    "text": selected_field or "",
                    "x": 0.01, "y": 0.98,
                    "xref": "paper", "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 11, "color": "#64748b"},
                    "xanchor": "left",
                },
            ],
        )
        return fig

    # ------------------------------------------------------------------
    # Clientside callback 1: Dash slider → Three.js iframe (PRD §7.3)
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

    # ------------------------------------------------------------------
    # Clientside callback 1b: Field Selector → Three.js iframe re-bootstrap
    # PRD v0.2.0 §8 (Multi-Field Frontend)
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
        Output("selected-plant-store", "id"),  # dummy — id never changes
        Input("viewport-iframe",       "id"),
        prevent_initial_call=False,
    )

    # ------------------------------------------------------------------
    # Clientside callback 3: Polling interval → selected-plant-store
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
    # Python callback: selected-plant-store → Panel 4 inspector UI
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

        # Close button → collapse panel
        if trigger_id == "inspector-close-btn" or plant_data is None:
            return _STYLE_INSPECTOR, "", html.Div(
                "Click a plant in the 3D view to inspect it.",
                style={"fontSize": "12px", "color": "#64748b",
                       "marginTop": "24px", "textAlign": "center"},
            ), html.Div()

        # Extract identity
        if isinstance(plant_data, str):
            try:
                plant_data = _json.loads(plant_data)
            except Exception:
                return _STYLE_INSPECTOR, "", html.Div(), html.Div()

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
            ("Field",        selected_field or day_slice.get("field_name", "—")),
            ("Row",          p_row),
            ("Col",          p_col),
            ("Alive",        bool(day_slice.get("alive", True)) if len(day_slice) else "—"),
            ("Biomass (g)",  _fmt(day_slice.get("biomass_g",  0))),
            ("LAI (m²/m²)", _fmt(day_slice.get("lai",         0))),
            ("Height (cm)",  _fmt(day_slice.get("height_cm",  0), 1)),
            ("Root depth (cm)", _fmt(day_slice.get("root_depth_cm", 0), 1)),
            ("Stress index", _fmt(day_slice.get("stress_index", 0))),
            ("Stage",        day_slice.get("phenological_stage", "—")),
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
                **_dark_layout(),
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

        # ---- Soil vertical cross-section chart (PRD §7.2) -------------
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
                    **_dark_layout(),
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

        return _STYLE_INSPECTOR_OPEN, plant_id, inspector_children, soil_chart

    # ------------------------------------------------------------------
    # CSV Download callback (PRD v0.4.0 §8.2)
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

def _dark_layout() -> dict:
    """Common dark-theme layout properties for all Plotly figures."""
    return {
        "paper_bgcolor": "#0d1520",
        "plot_bgcolor":  "#0d1520",
        "font":          {"color": "#94a3b8", "size": 11},
        "xaxis": {
            "gridcolor": "#1e2a3a", "zerolinecolor": "#1e2a3a",
            "tickfont": {"color": "#64748b", "size": 10},
        },
        "yaxis": {
            "gridcolor": "#1e2a3a", "zerolinecolor": "#1e2a3a",
            "tickfont": {"color": "#64748b", "size": 10},
        },
    }


def _empty_figure(message: str = "No data") -> go.Figure:
    """Return a dark-themed empty figure with a centred message."""
    fig = go.Figure()
    fig.update_layout(
        **_dark_layout(),
        annotations=[{
            "text": message,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 13, "color": "#4a5568"},
        }],
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
    )
    return fig
