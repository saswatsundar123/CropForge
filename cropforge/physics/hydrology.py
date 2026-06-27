"""
cropforge/physics/hydrology.py
================================
Pure-math soil water balance functions for CropForge v0.3.0.

This module is a physics-only library. It contains no CropForge
simulation state imports — inputs are plain Python scalars and lists,
outputs are plain Python dicts. This guarantees the functions are
unit-testable in complete isolation from the engine.

Scientific Basis (PRD v0.3.0 Section 5):
    FAO-56 Daily Soil Water Balance (Chapter 8, Allen et al. 1998).

    The model treats each soil layer as a "bucket":
        - Inflow:  rainfall + irrigation (added to layer 0 only)
        - Outflow: evapotranspiration demand from root zone layers
        - Gravity: excess above field_capacity drains to next layer at
                   rate `drainage_coefficient` per day (tipping-bucket)

    The stress coefficient Ks (0.0 → 1.0) is computed for each plant
    from the mean moisture of the root zone:

        Ks = (θ - θ_wp) / (θ_fc - θ_wp)    clipped to [0, 1]

    where θ is volumetric moisture (%), θ_fc is field capacity,
    and θ_wp is permanent wilting point.

Functions
---------
calculate_tipping_bucket   -- Apply rainfall/irrigation and gravity drainage
calculate_water_extraction -- Deduct ETc from root zone; return Ks stress scalar

Both functions take lightweight layer descriptors (dicts) rather than
SoilVoxelState objects so they remain import-free and trivially testable.

Layer descriptor format (used by both functions):
    {
        "moisture_pct":       float,   # current volumetric water content (%)
        "field_capacity_pct": float,   # FC threshold (%)
        "wilting_point_pct":  float,   # WP threshold (%)
        "saturation_pct":     float,   # SAT ceiling (%)
        "depth_top_cm":       float,   # layer top boundary (cm)
        "depth_bottom_cm":    float,   # layer bottom boundary (cm)
        "drainage_coefficient": float, # fraction of excess drained per day [0, 1]
    }

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Public constants (defaults referenced in builtin_hooks.py)
# ---------------------------------------------------------------------------

DEFAULT_DRAINAGE_COEFFICIENT: float = 0.5    # fraction of excess drained per day
DEFAULT_CROP_COEFFICIENT:     float = 1.0    # Kc (dimensionless)
DEFAULT_STRESS_INCREMENT:     float = 0.05   # stress_index increase per full-stress day
DEFAULT_FIELD_CAPACITY_PCT:   float = 32.0   # %
DEFAULT_WILTING_POINT_PCT:    float = 14.0   # %
DEFAULT_SATURATION_PCT:       float = 48.0   # %


# ---------------------------------------------------------------------------
# Helper: unit conversion
# ---------------------------------------------------------------------------

def _mm_to_pct(mm: float, depth_cm: float) -> float:
    """Convert a water depth (mm) to a volumetric % change for one layer.

    Relationship:
        1 mm of water over 1 cm depth = 10 % volumetric water content
        1 mm over *depth_cm* cm       = (10 / depth_cm) %

    This is the standard agronomic unit conversion used throughout FAO-56.

    Parameters
    ----------
    mm:
        Water depth in millimetres.
    depth_cm:
        Layer thickness in centimetres.

    Returns
    -------
    float
        Equivalent volumetric water content change (%).
    """
    if depth_cm <= 0:
        return 0.0
    return (mm / (depth_cm * 10.0)) * 100.0


# ---------------------------------------------------------------------------
# Task 1a: calculate_tipping_bucket
# ---------------------------------------------------------------------------

def calculate_tipping_bucket(
    layers: List[Dict],
    precipitation_mm: float,
    irrigation_mm: float,
) -> List[Dict]:
    """Apply inflows and gravity drainage across a soil profile.

    Implements the FAO-56 "tipping bucket" (cascade) drainage model:

    1. Convert (rainfall + irrigation) from mm to % volumetric for layer 0.
    2. Add inflow to layer 0, capped at saturation.
    3. Calculate excess above field_capacity for each layer.
    4. Drain excess × drainage_coefficient down to the next layer.
    5. Any drainage past the bottom layer is lost (deep percolation).
    6. Moisture floors at 0.0 and ceilings at saturation_pct.

    Parameters
    ----------
    layers:
        List of layer descriptor dicts (ordered top → bottom).
        Each dict **must** contain the keys defined in the module docstring.
        The function returns *new* dicts; inputs are not mutated.
    precipitation_mm:
        Today's rainfall (mm). Added to layer 0.
    irrigation_mm:
        Irrigation water applied today (mm). Added to layer 0.

    Returns
    -------
    List[Dict]
        Updated layer descriptors with modified ``moisture_pct`` and a new
        ``drainage_mm_today`` key recording how much drained from each layer.

    Notes
    -----
    - If ``layers`` is empty, returns an empty list immediately.
    - All moisture values are clipped to [0, saturation_pct] after each step.
    - deep percolation from the bottom layer is intentionally discarded
      (water lost from the model domain — PRD Section 5.3).

    Examples
    --------
    >>> layers = [{
    ...     "moisture_pct": 20.0, "field_capacity_pct": 30.0,
    ...     "wilting_point_pct": 14.0, "saturation_pct": 45.0,
    ...     "depth_top_cm": 0.0, "depth_bottom_cm": 20.0,
    ...     "drainage_coefficient": 0.5,
    ... }]
    >>> result = calculate_tipping_bucket(layers, precipitation_mm=10.0, irrigation_mm=0.0)
    >>> round(result[0]["moisture_pct"], 2)
    25.0
    """
    if not layers:
        return []

    # Work on a deep copy of layer data so inputs are not mutated
    updated: List[Dict] = [dict(layer) for layer in layers]
    for layer in updated:
        layer["drainage_mm_today"] = 0.0

    # ---- Step 1 & 2: Add inflow to layer 0, track surface runoff ----
    inflow_mm = precipitation_mm + irrigation_mm
    if inflow_mm > 0.0 and updated:
        top = updated[0]
        layer_depth_cm = top["depth_bottom_cm"] - top["depth_top_cm"]
        inflow_pct = _mm_to_pct(inflow_mm, layer_depth_cm)
        new_moisture = top["moisture_pct"] + inflow_pct
        if new_moisture > top["saturation_pct"]:
            # Water above saturation runs off as surface runoff (mm)
            excess_pct = new_moisture - top["saturation_pct"]
            excess_mm  = excess_pct * layer_depth_cm * 10.0 / 100.0
            top["surface_runoff_mm_today"] = excess_mm
            top["moisture_pct"] = top["saturation_pct"]
        else:
            top["surface_runoff_mm_today"] = 0.0
            top["moisture_pct"] = new_moisture
    elif updated:
        updated[0]["surface_runoff_mm_today"] = 0.0

    # ---- Steps 3–5: Cascade drainage downward ----
    for i, layer in enumerate(updated):
        sat  = layer["saturation_pct"]
        fc   = layer["field_capacity_pct"]
        coef = layer.get("drainage_coefficient", DEFAULT_DRAINAGE_COEFFICIENT)
        depth_cm = layer["depth_bottom_cm"] - layer["depth_top_cm"]

        # Clip moisture to [0, saturation] first
        layer["moisture_pct"] = max(0.0, min(sat, layer["moisture_pct"]))

        # Excess above field capacity drains
        excess_pct = max(0.0, layer["moisture_pct"] - fc)
        drainage_pct = excess_pct * coef
        layer["moisture_pct"] -= drainage_pct

        # Convert drainage from % back to mm for the log and for the next layer
        drainage_mm = drainage_pct * (depth_cm * 10.0) / 100.0
        layer["drainage_mm_today"] = drainage_mm

        # Add drained water as inflow to the layer below
        if i + 1 < len(updated) and drainage_mm > 0.0:
            next_layer = updated[i + 1]
            next_depth_cm = next_layer["depth_bottom_cm"] - next_layer["depth_top_cm"]
            next_inflow_pct = _mm_to_pct(drainage_mm, next_depth_cm)
            next_layer["moisture_pct"] = min(
                next_layer["saturation_pct"],
                next_layer["moisture_pct"] + next_inflow_pct,
            )
        # If i is the last layer: drainage_mm is lost to deep percolation (discarded)

        # Final floor / ceiling clip
        layer["moisture_pct"] = max(0.0, min(sat, layer["moisture_pct"]))

    return updated


# ---------------------------------------------------------------------------
# Task 1b: calculate_water_extraction
# ---------------------------------------------------------------------------

def calculate_water_extraction(
    layers: List[Dict],
    root_depth_cm: float,
    et0_demand: float,
    crop_coefficient: float = DEFAULT_CROP_COEFFICIENT,
) -> Dict:
    """Deduct crop water demand from root-zone layers; return stress scalar Ks.

    Implements FAO-56 actual evapotranspiration extraction and the stress
    coefficient calculation (Allen et al. 1998, Chapter 8):

        ETc    = ET0 x Kc                    [mm/day]
        Ks     = (theta_mean - theta_wp) / (theta_fc - theta_wp)  clipped to [0, 1]

    Extraction algorithm:
        1. Identify active root-zone layers (layers with any depth <= root_depth_cm).
        2. Distribute ETc uniformly across all active root-zone layers.
        3. Convert ETc per layer from mm to % volumetric.
        4. Deduct from each layer; floor at 0.0 (cannot go below zero).
        5. Compute Ks from the mean moisture of root-zone layers.

    Parameters
    ----------
    layers:
        List of layer descriptor dicts (ordered top -> bottom).
        Each dict must contain the standard keys plus ``depth_top_cm``
        and ``depth_bottom_cm``. The function returns *new* dicts.
    root_depth_cm:
        Plant rooting depth (cm). Only layers whose top boundary is
        *below* this depth are in the active root zone. If 0.0 or the
        root does not reach any layer, uses layer 0 exclusively.
    et0_demand:
        Daily reference ET0 (mm). Multiplied by ``crop_coefficient`` to
        get the actual crop demand ETc.
    crop_coefficient:
        Kc value (dimensionless). Default 1.0. Researcher-settable via
        ``field.set_water_params(crop_coefficient=Kc)``.

    Returns
    -------
    dict with keys:
        ``layers``      - Updated layer descriptors with new moisture values.
        ``ks``          - Water stress coefficient [0.0, 1.0].
                          1.0 = no stress (moisture at FC), 0.0 = full stress (at WP).
        ``etc_mm``      - Actual ETc demanded (mm, before stress reduction).
        ``extracted_mm``- Water actually removed (mm, may be < ETc if soil is dry).

    Examples
    --------
    >>> layers = [{
    ...     "moisture_pct": 30.0, "field_capacity_pct": 30.0,
    ...     "wilting_point_pct": 14.0, "saturation_pct": 45.0,
    ...     "depth_top_cm": 0.0, "depth_bottom_cm": 20.0,
    ...     "drainage_coefficient": 0.5,
    ... }]
    >>> result = calculate_water_extraction(layers, root_depth_cm=20.0, et0_demand=5.0)
    >>> result["ks"]  # at FC before extraction
    1.0
    """
    if not layers:
        return {"layers": [], "ks": 1.0, "etc_mm": 0.0, "extracted_mm": 0.0}

    # Work on a copy to avoid mutating inputs
    updated = [dict(layer) for layer in layers]

    # ---- Step 1: Identify root zone layers ----
    root_layers = [
        i for i, L in enumerate(updated)
        if L["depth_top_cm"] < max(root_depth_cm, 1e-9)
    ]
    if not root_layers:
        # Root not deep enough to reach any layer -- use layer 0 as fallback
        root_layers = [0]

    # ---- Step 2: Compute ETc and per-layer extraction demand ----
    etc_mm = et0_demand * crop_coefficient
    if etc_mm <= 0.0 or not root_layers:
        # No demand -- compute Ks only
        pass
    else:
        extraction_per_layer_mm = etc_mm / len(root_layers)

        # ---- Steps 3-4: Deduct from each root zone layer ----
        for i in root_layers:
            layer = updated[i]
            depth_cm = layer["depth_bottom_cm"] - layer["depth_top_cm"]
            extraction_pct = _mm_to_pct(extraction_per_layer_mm, depth_cm)
            layer["moisture_pct"] = max(0.0, layer["moisture_pct"] - extraction_pct)

    # ---- Step 5: Compute Ks from mean root-zone moisture ----
    moisture_vals = [updated[i]["moisture_pct"] for i in root_layers]
    fc_vals       = [updated[i]["field_capacity_pct"] for i in root_layers]
    wp_vals       = [updated[i]["wilting_point_pct"] for i in root_layers]

    theta_mean = sum(moisture_vals) / len(moisture_vals)
    theta_fc   = sum(fc_vals)       / len(fc_vals)
    theta_wp   = sum(wp_vals)       / len(wp_vals)

    fc_wp_range = theta_fc - theta_wp
    if fc_wp_range <= 0.0:
        ks = 1.0  # Degenerate soil params -- assume no stress
    else:
        ks = (theta_mean - theta_wp) / fc_wp_range
        ks = max(0.0, min(1.0, ks))

    # ---- Compute actual extracted mm ----
    extracted_mm = 0.0
    if etc_mm > 0.0 and root_layers:
        for i in root_layers:
            orig = layers[i]["moisture_pct"]
            new  = updated[i]["moisture_pct"]
            depth_cm = layers[i]["depth_bottom_cm"] - layers[i]["depth_top_cm"]
            removed_pct = orig - new
            extracted_mm += removed_pct * (depth_cm * 10.0) / 100.0

    return {
        "layers": updated,
        "ks": ks,
        "etc_mm": etc_mm,
        "extracted_mm": extracted_mm,
    }


# ---------------------------------------------------------------------------
# route_surface_water -- D8 lateral inflow accumulation (v0.4.0 Phase 2)
# ---------------------------------------------------------------------------

def route_surface_water(
    surface_runoff_grid: List[List[float]],
    elevation_grid: List[List[float]],
) -> List[List[float]]:
    """Route surface runoff downslope and return the lateral inflow each cell receives.

    Uses the D8 (deterministic-8) steepest-descent routing algorithm:
    each cell's surface runoff is directed to the single neighbour with the
    largest elevation drop. The function returns a ``lateral_inflow_grid``
    representing the millimetres of water flowing *into* each cell from its
    uphill neighbours.

    This is the water-mass counterpart to the existing nitrogen lateral transport
    in ``nutrients.py``. Both functions read the same ``surface_runoff_mm_today``
    values written by ``calculate_tipping_bucket`` but operate independently
    so that lateral water accumulation can be enabled without nitrogen transport.

    Scientific Basis
    ----------------
    Simplified kinematic-wave lateral redistribution (O'Callaghan & Mark 1984
    D8 single-flow-direction routing). One-timestep lag: runoff generated
    *today* is added to *tomorrow's* precipitation before the tipping-bucket
    runs. This matches the v0.3.0 nitrogen lag convention (nutrients hook at
    phase=-4 reads yesterday's drainage).

    Parameters
    ----------
    surface_runoff_grid:
        2-D list [row][col] of surface runoff leaving each cell (mm). These
        are the ``surface_runoff_mm_today`` values written by
        ``calculate_tipping_bucket``. Values must be >= 0.
    elevation_grid:
        2-D list [row][col] of cell elevation (m). Higher value = uphill.
        Must have the same dimensions as ``surface_runoff_grid``.

    Returns
    -------
    List[List[float]]
        2-D list [row][col] of lateral inflow received by each cell (mm).
        Each entry is the sum of all runoff routed into that cell from
        upslope neighbours. The sum over all cells equals the sum of all
        runoff that successfully routed to a lower neighbour (runoff that
        has no downslope neighbour is lost to the boundary).

    Notes
    -----
    - Boundary accumulation: if a cell has no lower neighbour (local sink or
      edge cell on a flat grid), its runoff is NOT added to any inflow cell.
    - Conservation: sum(inflow) <= sum(runoff). Equality holds when every
      cell has at least one lower neighbour.
    - Units are mm of water depth, consistent with ``calculate_tipping_bucket``
      inputs and outputs.

    Examples
    --------
    >>> runoff = [[5.0, 5.0], [0.0, 0.0]]
    >>> elev   = [[1.0, 1.0], [0.0, 0.0]]
    >>> inflow = route_surface_water(runoff, elev)
    >>> inflow[0][0]   # top cell: no uphill neighbour, receives 0
    0.0
    >>> inflow[1][0] > 0.0   # bottom-left receives runoff from top-left
    True
    """
    rows = len(surface_runoff_grid)
    if rows == 0:
        return []
    cols = len(surface_runoff_grid[0])

    # 8 cardinal + diagonal neighbours (D8)
    _NEIGHBOURS = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    # Accumulate inflow received by each cell
    lateral_inflow: List[List[float]] = [[0.0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            runoff_mm = surface_runoff_grid[r][c]
            if runoff_mm <= 0.0:
                continue  # no runoff from this cell

            # Find the steepest downslope neighbour (D8)
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
                # No lower neighbour -- water ponds at boundary; no inflow credited
                continue

            # Route all runoff from (r,c) to (best_nr, best_nc)
            lateral_inflow[best_nr][best_nc] += runoff_mm

    return lateral_inflow

    """Deduct crop water demand from root-zone layers; return stress scalar Ks.

    Implements FAO-56 actual evapotranspiration extraction and the stress
    coefficient calculation (Allen et al. 1998, Chapter 8):

        ETc    = ET0 × Kc                    [mm/day]
        Ks     = (θ_mean - θ_wp) / (θ_fc - θ_wp)  clipped to [0, 1]

    Extraction algorithm:
        1. Identify active root-zone layers (layers with any depth ≤ root_depth_cm).
        2. Distribute ETc uniformly across all active root-zone layers.
        3. Convert ETc per layer from mm to % volumetric.
        4. Deduct from each layer; floor at 0.0 (cannot go below zero).
        5. Compute Ks from the mean moisture of root-zone layers.

    Parameters
    ----------
    layers:
        List of layer descriptor dicts (ordered top → bottom).
        Each dict must contain the standard keys plus ``depth_top_cm``
        and ``depth_bottom_cm``. The function returns *new* dicts.
    root_depth_cm:
        Plant rooting depth (cm). Only layers whose top boundary is
        *below* this depth are in the active root zone. If 0.0 or the
        root does not reach any layer, uses layer 0 exclusively.
    et0_demand:
        Daily reference ET0 (mm). Multiplied by ``crop_coefficient`` to
        get the actual crop demand ETc.
    crop_coefficient:
        Kc value (dimensionless). Default 1.0. Researcher-settable via
        ``field.set_water_params(crop_coefficient=Kc)``.

    Returns
    -------
    dict with keys:
        ``layers``      – Updated layer descriptors with new moisture values.
        ``ks``          – Water stress coefficient [0.0, 1.0].
                          1.0 = no stress (moisture at FC), 0.0 = full stress (at WP).
        ``etc_mm``      – Actual ETc demanded (mm, before stress reduction).
        ``extracted_mm``– Water actually removed (mm, may be < ETc if soil is dry).

    Examples
    --------
    >>> layers = [{
    ...     "moisture_pct": 30.0, "field_capacity_pct": 30.0,
    ...     "wilting_point_pct": 14.0, "saturation_pct": 45.0,
    ...     "depth_top_cm": 0.0, "depth_bottom_cm": 20.0,
    ...     "drainage_coefficient": 0.5,
    ... }]
    >>> result = calculate_water_extraction(layers, root_depth_cm=20.0, et0_demand=5.0)
    >>> result["ks"]  # at FC before extraction
    1.0
    """
    if not layers:
        return {"layers": [], "ks": 1.0, "etc_mm": 0.0, "extracted_mm": 0.0}

    # Work on a copy to avoid mutating inputs
    updated = [dict(layer) for layer in layers]

    # ---- Step 1: Identify root zone layers ----
    root_layers = [
        i for i, L in enumerate(updated)
        if L["depth_top_cm"] < max(root_depth_cm, 1e-9)
    ]
    if not root_layers:
        # Root not deep enough to reach any layer — use layer 0 as fallback
        root_layers = [0]

    # ---- Step 2: Compute ETc and per-layer extraction demand ----
    etc_mm = et0_demand * crop_coefficient
    if etc_mm <= 0.0 or not root_layers:
        # No demand — compute Ks only
        pass
    else:
        extraction_per_layer_mm = etc_mm / len(root_layers)

        # ---- Steps 3–4: Deduct from each root zone layer ----
        for i in root_layers:
            layer = updated[i]
            depth_cm = layer["depth_bottom_cm"] - layer["depth_top_cm"]
            extraction_pct = _mm_to_pct(extraction_per_layer_mm, depth_cm)
            layer["moisture_pct"] = max(0.0, layer["moisture_pct"] - extraction_pct)

    # ---- Step 5: Compute Ks from mean root-zone moisture ----
    moisture_vals = [updated[i]["moisture_pct"] for i in root_layers]
    fc_vals       = [updated[i]["field_capacity_pct"] for i in root_layers]
    wp_vals       = [updated[i]["wilting_point_pct"] for i in root_layers]

    theta_mean = sum(moisture_vals) / len(moisture_vals)
    theta_fc   = sum(fc_vals)       / len(fc_vals)
    theta_wp   = sum(wp_vals)       / len(wp_vals)

    fc_wp_range = theta_fc - theta_wp
    if fc_wp_range <= 0.0:
        ks = 1.0  # Degenerate soil params — assume no stress
    else:
        ks = (theta_mean - theta_wp) / fc_wp_range
        ks = max(0.0, min(1.0, ks))

    # ---- Compute actual extracted mm ----
    extracted_mm = 0.0
    if etc_mm > 0.0 and root_layers:
        extraction_per_layer_mm = etc_mm / len(root_layers)
        for i in root_layers:
            orig = layers[i]["moisture_pct"]
            new  = updated[i]["moisture_pct"]
            depth_cm = layers[i]["depth_bottom_cm"] - layers[i]["depth_top_cm"]
            removed_pct = orig - new
            extracted_mm += removed_pct * (depth_cm * 10.0) / 100.0

    return {
        "layers": updated,
        "ks": ks,
        "etc_mm": etc_mm,
        "extracted_mm": extracted_mm,
    }
