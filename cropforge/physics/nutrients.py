"""
cropforge/physics/nutrients.py
================================
Pure-math nitrogen transport functions for CropForge v0.3.0.

No CropForge simulation state imports — all inputs are plain Python
dicts and scalars. This guarantees complete unit-test isolation from
the engine, consistent with hydrology.py.

Scientific Model
----------------
Nitrate (NO3-) is a soluble anion that moves passively with water.
Two transport pathways are implemented:

1. **Vertical leaching** (downward with drainage):
       leached_kg_ha = drainage_mm × leaching_fraction × N_available_kg_ha
   The fraction `leaching_fraction` is a soil texture parameter.
   Leached N is deducted from the source layer and added to the layer
   below. N leaching past the bottom layer is lost from the model domain.

2. **Lateral runoff transport** (horizontal with surface runoff):
       N_moved_kg_ha = runoff_mm × runoff_n_fraction × N_layer0_kg_ha
   Surface runoff carries a fraction of layer-0 nitrogen to
   downslope cells. This is computed field-wide in `calculate_lateral_runoff`.

These two processes together allow the slope leaching trial (PRD §7.3)
to show measurable nitrogen accumulation at low-elevation cells.

Layer Descriptor Keys (same contract as hydrology.py)
------------------------------------------------------
Required keys (matching hydrology.py contract):
    moisture_pct, field_capacity_pct, wilting_point_pct,
    saturation_pct, depth_top_cm, depth_bottom_cm, drainage_coefficient

Additional keys read by this module:
    nitrogen_kg_ha        – mineral N in this layer (kg/ha)
    leaching_fraction     – fraction of mineral N leached per mm drainage (default 0.01)

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_LEACHING_FRACTION:  float = 0.01   # kg N leached per kg N per mm drainage
DEFAULT_RUNOFF_N_FRACTION:  float = 0.05   # fraction of layer-0 N moved with lateral runoff


# ---------------------------------------------------------------------------
# calculate_nitrogen_transport — vertical leaching
# ---------------------------------------------------------------------------

def calculate_nitrogen_transport(
    layers: List[Dict],
    vertical_drainage_fluxes: List[float],
    lateral_runoff_mm: float = 0.0,
    leaching_fraction: float = DEFAULT_LEACHING_FRACTION,
    runoff_n_fraction: float = DEFAULT_RUNOFF_N_FRACTION,
) -> Dict:
    """Transport soluble nitrogen downward with drainage and laterally with runoff.

    Implements two coupled transport mechanisms:

    **Vertical leaching** (FAO-56 / DSSAT simplified approach):
        For each layer i:
            N_leached[i] = drainage_flux[i] × leaching_fraction × N[i]
        N_leached[i] is removed from layer i and added to layer i+1.
        N leached past the bottom layer is lost (deep percolation N loss).

    **Lateral runoff N export** (from layer 0 only):
        N_exported = lateral_runoff_mm × runoff_n_fraction × N[layer_0]
        This N is removed from layer 0 and returned as a scalar
        `lateral_n_export_kg_ha` for the calling hook to distribute to
        downslope cells.

    Parameters
    ----------
    layers:
        List of layer descriptor dicts (top → bottom), each with at minimum:
        ``nitrogen_kg_ha``, ``depth_top_cm``, ``depth_bottom_cm``.
        Must not be mutated — function returns new dicts.
    vertical_drainage_fluxes:
        List of drainage fluxes (mm/day) for each layer. Must have the
        same length as ``layers``. Use 0.0 for layers with no drainage
        (e.g. layers above field capacity). Obtained from
        ``calculate_tipping_bucket`` results via the ``drainage_mm_today``
        key.
    lateral_runoff_mm:
        Surface runoff leaving this cell (mm). Only affects layer 0.
        Obtained from ``calculate_lateral_runoff``.
    leaching_fraction:
        Fraction of mineral N leached per mm of drainage (dimensionless).
        Default 0.01. Override via ``field.set_nitrogen_params()``.
    runoff_n_fraction:
        Fraction of layer-0 N exported with lateral runoff per mm.
        Default 0.05. Override via ``field.set_nitrogen_params()``.

    Returns
    -------
    dict with keys:
        ``layers``                 – Updated layer dicts with new nitrogen_kg_ha.
        ``leached_kg_ha``          – List of N leached from each layer (mm).
        ``lateral_n_export_kg_ha`` – N exported laterally from layer 0 (kg/ha).
        ``total_n_lost_kg_ha``     – Total N lost from the system (deep perc + lateral).

    Examples
    --------
    >>> layers = [{"nitrogen_kg_ha": 100.0, "depth_top_cm": 0.0,
    ...            "depth_bottom_cm": 20.0}]
    >>> result = calculate_nitrogen_transport(layers, [5.0], leaching_fraction=0.01)
    >>> round(result["leached_kg_ha"][0], 4)
    5.0
    """
    if not layers:
        return {
            "layers": [],
            "leached_kg_ha": [],
            "lateral_n_export_kg_ha": 0.0,
            "total_n_lost_kg_ha": 0.0,
        }

    n_layers = len(layers)
    # Pad drainage fluxes to match layer count
    if len(vertical_drainage_fluxes) < n_layers:
        vertical_drainage_fluxes = list(vertical_drainage_fluxes) + [0.0] * (
            n_layers - len(vertical_drainage_fluxes)
        )

    # Work on copies
    updated = [dict(L) for L in layers]
    leached: List[float] = [0.0] * n_layers

    # ---- Step 1: Lateral runoff N export from layer 0 ----
    lateral_n_export = 0.0
    if lateral_runoff_mm > 0.0:
        n0 = updated[0].get("nitrogen_kg_ha", 0.0)
        lateral_n_export = lateral_runoff_mm * runoff_n_fraction * n0 / 100.0
        lateral_n_export = min(lateral_n_export, n0)   # cannot export more than exists
        updated[0]["nitrogen_kg_ha"] = max(0.0, n0 - lateral_n_export)

    # ---- Step 2: Vertical leaching cascade ----
    n_deep_perc = 0.0  # N lost via deep percolation from the bottom layer
    for i in range(n_layers):
        drainage_mm = vertical_drainage_fluxes[i]
        if drainage_mm <= 0.0:
            continue
        n_available = updated[i].get("nitrogen_kg_ha", 0.0)
        if n_available <= 0.0:
            continue

        # N leached from layer i (kg/ha)
        n_leach = drainage_mm * leaching_fraction * n_available
        n_leach = min(n_leach, n_available)   # cannot leach more than exists

        updated[i]["nitrogen_kg_ha"] = max(0.0, n_available - n_leach)
        leached[i] = n_leach

        # Add to layer below, or discard if bottom layer
        if i + 1 < n_layers:
            updated[i + 1]["nitrogen_kg_ha"] = (
                updated[i + 1].get("nitrogen_kg_ha", 0.0) + n_leach
            )
        else:
            n_deep_perc += n_leach  # lost to groundwater

    total_n_lost = n_deep_perc + lateral_n_export

    return {
        "layers": updated,
        "leached_kg_ha": leached,
        "lateral_n_export_kg_ha": lateral_n_export,
        "total_n_lost_kg_ha": total_n_lost,
    }


# ---------------------------------------------------------------------------
# calculate_lateral_runoff — spatial surface flow on a DEM
# ---------------------------------------------------------------------------

def calculate_lateral_runoff(
    moisture_grid: List[List[float]],
    saturation_grid: List[List[float]],
    elevation_grid: List[List[float]],
    drainage_coefficient: float = 0.5,
) -> List[List[float]]:
    """Compute surface lateral runoff from each cell using a simplified D8 approach.

    Cells whose top-layer moisture exceeds saturation generate surface runoff.
    Runoff from each cell flows to the single steepest downslope neighbour
    (D8 routing). The runoff volume is:

        runoff_mm[i,j] = max(0, moisture[i,j] - saturation[i,j])
                         × drainage_coefficient × layer_depth_factor

    This is a simplified kinematic-wave approximation sufficient for the
    slope leaching trial scenario. It does not implement full Saint-Venant
    equations or routing delays.

    Parameters
    ----------
    moisture_grid:
        2-D list [row][col] of top-layer volumetric moisture (%).
    saturation_grid:
        2-D list [row][col] of top-layer saturation threshold (%).
    elevation_grid:
        2-D list [row][col] of cell elevation (m). Higher value = uphill.
    drainage_coefficient:
        Fraction of excess-above-saturation that runs off (default 0.5).

    Returns
    -------
    List[List[float]]
        2-D list [row][col] of net runoff LEAVING each cell (mm, ≥ 0).
        Note: inflow from upslope cells is NOT added here — the calling
        hook handles the net exchange by computing both outflow and inflow
        for each cell pair.

    Notes
    -----
    - Boundary cells (edge of grid) that have no lower neighbour retain
      their excess as standing water (no outflow).
    - The function returns the *outflow* only. The calling hook must
      subtract outflow from source cells and add it to sink cells.

    Examples
    --------
    >>> moisture = [[35.0, 35.0], [20.0, 20.0]]
    >>> saturation = [[30.0, 30.0], [30.0, 30.0]]
    >>> elevation = [[1.0, 1.0], [0.0, 0.0]]
    >>> runoff = calculate_lateral_runoff(moisture, saturation, elevation)
    >>> runoff[0][0] > 0   # top cells have excess → runoff
    True
    >>> runoff[1][0] == 0.0  # bottom cells are below sat → no runoff
    True
    """
    rows = len(moisture_grid)
    if rows == 0:
        return []
    cols = len(moisture_grid[0])

    # Directions: (delta_row, delta_col) for 8 neighbours
    _NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
                   (0,  -1),           (0,  1),
                   (1,  -1),  (1,  0), (1,  1)]

    # Build outflow grid (mm)
    outflow = [[0.0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            excess = moisture_grid[r][c] - saturation_grid[r][c]
            if excess <= 0.0:
                continue  # cell is below saturation — no runoff

            runoff_pct = excess * drainage_coefficient  # fraction of excess that runs off

            # Find the steepest downslope neighbour (D8 routing)
            elev_self = elevation_grid[r][c]
            best_drop = 0.0
            best_nr, best_nc = -1, -1

            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    drop = elev_self - elevation_grid[nr][nc]
                    if drop > best_drop:
                        best_drop = drop
                        best_nr, best_nc = nr, nc

            if best_nr == -1:
                # No lower neighbour — water ponds in place; no outflow
                continue

            outflow[r][c] = runoff_pct  # record outflow leaving this cell (in % units)

    return outflow


# ---------------------------------------------------------------------------
# apply_lateral_n_exchange — post-process runoff into N fluxes on a grid
# ---------------------------------------------------------------------------

def apply_lateral_n_exchange(
    nitrogen_grid: List[List[float]],
    moisture_grid: List[List[float]],
    saturation_grid: List[List[float]],
    elevation_grid: List[List[float]],
    runoff_n_fraction: float = DEFAULT_RUNOFF_N_FRACTION,
    drainage_coefficient: float = 0.5,
) -> List[List[float]]:
    """Compute the net nitrogen change at each cell due to lateral runoff.

    Uses ``calculate_lateral_runoff`` internally. For each cell pair
    (source → sink), N carried with runoff is:

        N_transported = runoff_fraction × N_source × runoff_n_fraction

    Returns a 2-D grid of net N change (kg/ha) for each cell.
    Positive = N gained; negative = N lost.

    Parameters
    ----------
    nitrogen_grid:
        2-D list [row][col] of top-layer nitrogen (kg/ha).
    moisture_grid:
        2-D list [row][col] of top-layer moisture (%).
    saturation_grid:
        2-D list [row][col] of top-layer saturation (%).
    elevation_grid:
        2-D list [row][col] of cell elevation (m).
    runoff_n_fraction:
        Fraction of N exported per unit of runoff. Default 0.05.
    drainage_coefficient:
        Fraction of excess-above-sat that becomes runoff.

    Returns
    -------
    List[List[float]]
        2-D grid of net nitrogen delta (kg/ha) for each cell.
        Sum across all cells approximates zero (conservation, minus boundary losses).
    """
    rows = len(nitrogen_grid)
    if rows == 0:
        return []
    cols = len(nitrogen_grid[0])

    _NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
                   (0,  -1),           (0,  1),
                   (1,  -1),  (1,  0), (1,  1)]

    delta_n = [[0.0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            excess = moisture_grid[r][c] - saturation_grid[r][c]
            if excess <= 0.0:
                continue

            runoff_pct = excess * drainage_coefficient
            if runoff_pct <= 0.0:
                continue

            # Find steepest downslope neighbour
            elev_self = elevation_grid[r][c]
            best_drop = 0.0
            best_nr, best_nc = -1, -1

            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    drop = elev_self - elevation_grid[nr][nc]
                    if drop > best_drop:
                        best_drop = drop
                        best_nr, best_nc = nr, nc

            if best_nr == -1:
                continue

            # N transported from (r,c) to (best_nr, best_nc)
            n_source = max(0.0, nitrogen_grid[r][c])
            n_transported = runoff_pct * runoff_n_fraction * n_source
            n_transported = min(n_transported, n_source)  # cap at available N

            delta_n[r][c]               -= n_transported
            delta_n[best_nr][best_nc]   += n_transported

    return delta_n
