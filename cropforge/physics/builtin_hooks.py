"""
cropforge/physics/builtin_hooks.py
====================================
Built-in physics engine hooks for CropForge v0.2.0 / v0.3.0.

These are step functions that the engine registers at negative phases
when the researcher calls farm.use_physics().
Because their phase values are negative, they are guaranteed to execute
BEFORE any researcher-registered @farm.step (which must be >= 0).

Phase assignment (v0.3.0 PRD Section 5.3):
    phase=-3  Soil Water Balance (hydrology)   -- runs first, updates moisture
    phase=-2  ET0 engine (Penman-Monteith)     -- reads env, writes et0_mm
    phase=-1  Root impedance engine            -- reads root_depth, writes multiplier
    phase=0+  Researcher @farm.step functions

Each hook follows the standard step function signature:
    fn(state: FieldState, env: EnvironmentState) -> FieldState

They are pure callables -- they carry no state of their own. All
configuration is captured at registration time via closures.

PRD References:
    v0.2.0 Section 4   -- Penman-Monteith ET0 Engine
    v0.2.0 Section 5   -- Root Growth Engine
    v0.3.0 Section 5   -- Soil Water Balance
    v0.3.0 Section 5.3 -- Daily Execution Order

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cropforge.state import EnvironmentState, FieldState

logger = logging.getLogger(__name__)

# Phase constants -- must be negative so they precede all researcher steps
# PRD v0.3.0 / v0.5.0 execution order:
PHASE_NUTRIENTS_ENGINE  = -4   # Lateral flow + N transport (runs before hydrology)
PHASE_HYDROLOGY_ENGINE  = -3   # Soil water balance (updates moisture)
PHASE_ET0_ENGINE        = -2   # Penman-Monteith ET0
PHASE_ROOT_ENGINE       = -1   # Root impedance multiplier
# v0.5.0 new phases:
PHASE_RADIATION_ENGINE  = -2   # Radiation interception (same phase as ET0; runs per-plant)
PHASE_WIND_ENGINE       = -2   # Topographical wind field (same phase; writes env.custom grid)
PHASE_DISEASE_ENGINE    = -1   # Spatial SIR disease spread (same phase as root; per-plant)
PHASE_WEED_ENGINE       = -1   # Weed competition (opt-in; water + PAR suppression)
PHASE_ROOT_CLAMP        = +1   # Root depth clamping (PRD v0.7.0 §7.3) — runs AFTER all plugins
PHASE_CLOD_ENGINE       = -4   # Clod dynamics / roughness decay (PRD v0.7.0 §7.4)
PHASE_EROSION_ENGINE    = -2   # Soil erosion index (PRD v0.7.0 §7.5); runs after hydrology (-3)
PHASE_SEDIMENT_ENGINE   = +2   # Sediment transport (PRD v0.8.0 §5.2); day-end, reads erosion grid


# ---------------------------------------------------------------------------
# Hook 1: ET0 Engine  (PRD Section 4, Execution Order step 2)
# ---------------------------------------------------------------------------

def make_et0_hook(
    latitude_deg: float,
    elevation_m: float,
    anemometer_height_m: float = 2.0,
):
    """Return a built-in step function that computes FAO-56 ET0 each day.

    The returned function is registered at phase=PHASE_ET0_ENGINE (-2).
    It reads raw meteorological data from EnvironmentState, calls
    calculate_fao56_et0(), and writes et0_mm plus the four intermediate
    fields (vp_kpa, psychrometric_kpa, slope_svp, net_radiation_mj) back
    into the EnvironmentState instance.

    Parameters
    ----------
    latitude_deg:
        Site latitude (decimal degrees, positive = N). Sourced from
        Farm.location[0] when use_physics() is called.
    elevation_m:
        Site elevation above mean sea level (m). Sourced from
        Farm.elevation_m when use_physics() is called (default 0.0).
    anemometer_height_m:
        Height of wind speed measurement (m). Default 2.0.
        Pass the Weather object's anemometer_height_m if the weather
        station is at a non-standard height.
    """
    from cropforge.physics.environment import calculate_fao56_et0

    def _et0_engine_step(state: "FieldState", env: "EnvironmentState") -> "FieldState":
        """Built-in ET0 engine -- phase=-2 (runs before all @farm.step)."""
        try:
            result = calculate_fao56_et0(
                temp_max_c=env.temp_max_c,
                temp_min_c=env.temp_min_c,
                humidity_pct=env.humidity_pct,
                wind_speed_ms=env.wind_speed_ms,
                radiation_mj_m2=env.radiation_mj_m2,
                elevation_m=elevation_m,
                latitude_deg=latitude_deg,
                doy=env.doy,
                anemometer_height_m=anemometer_height_m,
            )
            # Write all outputs back into the EnvironmentState
            env.et0_mm            = result["et0_mm"]
            env.vp_kpa            = result["vp_kpa"]
            env.psychrometric_kpa = result["psychrometric_kpa"]
            env.slope_svp         = result["slope_svp"]
            env.net_radiation_mj  = result["net_radiation_mj"]
        except Exception:
            logger.exception(
                "ET0 engine failed on day %d. EnvironmentState.et0_mm "
                "left at its previous value (%s).",
                env.day,
                env.et0_mm,
            )
        return state

    _et0_engine_step.__name__ = "_et0_engine_step"
    return _et0_engine_step


# ---------------------------------------------------------------------------
# Hook 2: Root Impedance Engine  (PRD Section 5, Execution Order step 3)
# ---------------------------------------------------------------------------

def make_root_impedance_hook():
    """Return a built-in step function that updates root_growth_multiplier.

    For every living plant the hook:
      1. Finds the SoilVoxelState layer at the plant's current root_depth_cm.
      2. Calls calculate_root_impedance(layer.penetration_resistance).
      3. Writes the result into plant.root_growth_multiplier.

    If the root front is at or below the deepest layer boundary, the last
    layer's resistance is used (conservative: assume hard pan continues).

    The returned function is registered at phase=PHASE_ROOT_ENGINE (-1),
    so it runs AFTER the ET0 hook and BEFORE researcher @farm.step.

    Notes
    -----
    The hook does NOT update root_depth_cm itself -- that is the
    thermal-time root extension engine (Phase 3 build item in PRD Section
    13). This hook ONLY computes the impedance multiplier that a researcher
    (or the thermal-time engine, once implemented) can read.

    PRD v0.2.0 Section 10 backward compatibility:
        PlantState.root_growth_multiplier defaults to 1.0. When this hook
        is not registered (use_physics not called), it stays 1.0 --
        identical to unrestricted growth, preserving v0.1.0 behaviour.
    """
    from cropforge.physics.soil import calculate_root_impedance

    def _root_impedance_step(
        state: "FieldState", env: "EnvironmentState"
    ) -> "FieldState":
        """Built-in root impedance engine -- phase=-1."""
        for plant in state.plants:
            if not plant.alive:
                continue
            soil_column = state.soil[plant.row][plant.col]  # list of SoilVoxelState
            multiplier = _get_impedance_at_depth(
                soil_column, plant.root_depth_cm, calculate_root_impedance
            )
            plant.root_growth_multiplier = multiplier
        return state

    _root_impedance_step.__name__ = "_root_impedance_step"
    return _root_impedance_step


def _get_impedance_at_depth(soil_column, depth_cm: float, impedance_fn) -> float:
    """Return the root impedance multiplier for the layer at *depth_cm*.

    Iterates the soil column (list of SoilVoxelState sorted by layer index)
    and returns the impedance for the first layer whose depth range contains
    *depth_cm*.  If *depth_cm* is below all layers, uses the deepest layer.

    Parameters
    ----------
    soil_column:
        ``state.soil[row][col]`` -- list of SoilVoxelState for one grid cell.
    depth_cm:
        The plant's current root_depth_cm.
    impedance_fn:
        calculate_root_impedance function.

    Returns
    -------
    float
        Root growth multiplier in [0.0, 1.0].
    """
    if not soil_column:
        return 1.0  # no soil data -> assume unrestricted

    # Find the layer containing depth_cm
    for voxel in soil_column:
        if voxel.depth_top_cm <= depth_cm < voxel.depth_bottom_cm:
            return impedance_fn(voxel.penetration_resistance)

    # depth_cm is at or below the bottom of the last layer -> use deepest
    deepest = soil_column[-1]
    return impedance_fn(deepest.penetration_resistance)


# ---------------------------------------------------------------------------
# Hook 3: Soil Water Balance  (PRD v0.3.0 Section 5, phase=-3)
# ---------------------------------------------------------------------------

def make_hydrology_hook(
    stress_increment_per_day: float = 0.05,
    lateral_flow: bool = False,
    use_roughness: bool = False,
):
    """Return a built-in step function that runs the daily soil water balance.

    Implements the FAO-56 tipping-bucket water balance and closes the
    ET0 -> soil moisture -> plant stress loop that v0.2.0 left open.

    Phase: ``PHASE_HYDROLOGY_ENGINE`` (-3) -- runs before ET0 (-2) and
    root impedance (-1).  This guarantees that moisture is updated for
    today's rainfall before ET0 reads it and before user steps run.

    Per-day execution (PRD v0.3.0 / v0.4.0 Section 5.3):
        1. Read today's rainfall from ``env.rainfall_mm``.
        2. [v0.4.0] If lateral_flow=True: read yesterday's
           ``surface_runoff_mm_today`` from each top-layer voxel, route
           it downslope via D8, and add the lateral inflow to each cell's
           effective precipitation before the tipping-bucket runs.
        3. Apply tipping-bucket drainage cascade (rainfall + lateral inflow).
        4. Deduct ETc (= ET0 x Kc) from root zone layers.
        5. Compute Ks for each plant.
        6. Write Ks to ``plant.custom['water_stress_ks']``.
        7. Accumulate ``plant.stress_index += (1 - Ks) x stress_increment``.
        8. Update voxel moisture_pct from the returned layer dicts.
        9. Write drainage_mm_today and surface_runoff_mm_today to voxel.custom.

    Closure captures
    ----------------
    stress_increment_per_day:
        How much ``stress_index`` increases per day at full stress (Ks=0.0).
    lateral_flow:
        When True, route yesterday's surface runoff downslope before running
        the tipping-bucket, so downslope cells receive additional inflow from
        uphill neighbours (v0.4.0 Phase 2 -- Lateral Water Accumulation).
        Gated behind ``farm.use_physics(lateral_flow=True)``.

    PRD v0.3.0/v0.4.0 backward compatibility:
        This hook is only registered when ``use_physics(water_balance=True)``.
        Scripts that do not call ``use_physics()`` at all are completely
        unaffected.
    """
    from cropforge.physics.hydrology import (
        calculate_tipping_bucket,
        calculate_water_extraction,
        route_surface_water,
        DEFAULT_CROP_COEFFICIENT,
        DEFAULT_STRESS_INCREMENT,
        DEFAULT_FIELD_CAPACITY_PCT,
        DEFAULT_WILTING_POINT_PCT,
        DEFAULT_SATURATION_PCT,
        DEFAULT_DRAINAGE_COEFFICIENT,
    )

    _slope_cache: list = [None]  # ponytail: lazy one-time slope grid compute

    def _hydrology_step(
        state: "FieldState", env: "EnvironmentState"
    ) -> "FieldState":
        """Built-in soil water balance engine -- phase=-3."""

        rows = len(state.soil)
        if rows == 0:
            return state
        cols = len(state.soil[0]) if rows > 0 else 0

        # ---- Step 1: Read today's direct rainfall ----
        rain_mm = max(0.0, env.rainfall_mm)

        # ---- Step 2 (v0.4.0): Compute lateral inflow from yesterday's runoff ----
        # Read surface_runoff_mm_today written by yesterday's tipping-bucket
        # and route it downslope via D8. The result is added to each cell's
        # effective precipitation before today's tipping-bucket runs.
        lateral_inflow_grid: list = [[0.0] * cols for _ in range(rows)]
        if lateral_flow:
            elev = state.elevation_grid.tolist()
            runoff_grid = [
                [
                    state.soil[r][c][0].custom.get("surface_runoff_mm_today", 0.0)
                    if state.soil[r][c] else 0.0
                    for c in range(cols)
                ]
                for r in range(rows)
            ]
            lateral_inflow_grid = route_surface_water(runoff_grid, elev)

            # Log the total lateral redistribution for debugging
            total_lateral = sum(
                lateral_inflow_grid[r][c]
                for r in range(rows) for c in range(cols)
            )
            if total_lateral > 0.0:
                logger.debug(
                    "Hydrology lateral flow: day=%d total_inflow=%.2f mm across %dx%d grid.",
                    env.day, total_lateral, rows, cols,
                )

        # ---- Step 3: Apply tipping-bucket per soil column ----
        # Each cell gets rain_mm + lateral_inflow from uphill neighbours
        processed_cells: set = set()
        for row_idx, row_soils in enumerate(state.soil):
            for col_idx, cell_soils in enumerate(row_soils):
                if not cell_soils or (row_idx, col_idx) in processed_cells:
                    continue
                processed_cells.add((row_idx, col_idx))

                # Build layer descriptor list
                layer_dicts = _voxels_to_dicts(cell_soils)

                # Effective precipitation = direct rain + lateral inflow from uphill
                effective_precip_mm = rain_mm + lateral_inflow_grid[row_idx][col_idx]

                # v0.7.0 -- slope/roughness direct runoff (only when clod_dynamics enabled)
                # ponytail: bilinear coupling; ceiling: 1.0, floor: 0.0
                direct_runoff_mm = 0.0
                if use_roughness and cell_soils:
                    if _slope_cache[0] is None:
                        _slope_cache[0] = _compute_slope_normalized(state.elevation_grid)
                    slope_frac = _slope_cache[0][row_idx][col_idx]
                    if slope_frac > 0.0:
                        roughness = cell_soils[0].surface_roughness_index
                        frac = slope_frac * max(0.0, 1.0 - roughness)
                        direct_runoff_mm = frac * effective_precip_mm
                        effective_precip_mm = max(0.0, effective_precip_mm - direct_runoff_mm)

                # Apply tipping-bucket (lateral inflow treated as additional precipitation)
                updated_layers = calculate_tipping_bucket(
                    layer_dicts,
                    precipitation_mm=effective_precip_mm,
                    irrigation_mm=0.0,   # already applied by Event
                )

                # Write drainage + surface runoff back to voxels
                _dicts_to_voxels(updated_layers, cell_soils)

                # Add slope/roughness direct runoff on top of saturation runoff
                if direct_runoff_mm > 0.0:
                    cell_soils[0].custom["surface_runoff_mm_today"] = (
                        cell_soils[0].custom.get("surface_runoff_mm_today", 0.0)
                        + direct_runoff_mm
                    )

                # Store lateral inflow received today for diagnostics / reporting
                if lateral_flow:
                    cell_soils[0].custom["lateral_inflow_mm_today"] = (
                        lateral_inflow_grid[row_idx][col_idx]
                    )

        # ---- Steps 4-7: Water extraction per plant (root-zone aware) ----
        for plant in state.plants:
            if not plant.alive:
                continue

            cell_soils = state.soil[plant.row][plant.col]
            if not cell_soils:
                continue

            layer_dicts = _voxels_to_dicts(cell_soils)

            # Crop coefficient and stress increment may be field-level params
            kc = cell_soils[0].custom.get("crop_coefficient", DEFAULT_CROP_COEFFICIENT)
            s_inc = cell_soils[0].custom.get(
                "stress_increment_per_day", stress_increment_per_day
            )

            result = calculate_water_extraction(
                layer_dicts,
                root_depth_cm=plant.root_depth_cm,
                et0_demand=env.et0_mm,
                crop_coefficient=kc,
            )

            # Write updated moisture back to voxels
            _dicts_to_voxels(result["layers"], cell_soils)

            # Write Ks to plant state
            ks = result["ks"]
            plant.custom["water_stress_ks"] = ks

            # Accumulate stress index
            stress_delta = (1.0 - ks) * s_inc
            plant.stress_index = min(1.0, plant.stress_index + stress_delta)

            logger.debug(
                "Hydrology: day=%d plant=%s ks=%.3f stress_index=%.3f",
                env.day, plant.plant_id, ks, plant.stress_index,
            )

        return state

    _hydrology_step.__name__ = "_hydrology_step"
    return _hydrology_step


    """Return a built-in step function that runs the daily soil water balance.

    Implements the FAO-56 tipping-bucket water balance and closes the
    ET0 → soil moisture → plant stress loop that v0.2.0 left open.

    Phase: ``PHASE_HYDROLOGY_ENGINE`` (-3) -- runs before ET0 (-2) and
    root impedance (-1).  This guarantees that moisture is updated for
    today's rainfall before ET0 reads it and before user steps run.

    Per-day execution (PRD v0.3.0 Section 5.3 steps 1–8):
        1. Read today's rainfall from ``env.rainfall_mm``.
        2. Read today's irrigation from ``env.et0_mm`` context
           (irrigation was already added to moisture by Event.irrigation
           at the END OF YESTERDAY -- events fire after steps).  The hook
           reads ``field.rainfall_mm`` only; irrigation was pre-applied.
           NOTE: The tipping-bucket is called with irrigation_mm=0.0 here
           because Event.irrigation already modified voxel moisture before
           the logger recorded yesterday. Today the bucket only sees rain.
        3. Apply tipping-bucket drainage cascade.
        4. Deduct ETc (= ET0 × Kc) from root zone layers using
           ``calculate_water_extraction``.
           IMPORTANT: On phase=-3, ``env.et0_mm`` is still at yesterday's
           value (ET0 engine runs at phase=-2, after us). We use
           ``env.et0_mm`` from the environment as provided by the weather
           source (or last-computed value). This is the correct FAO-56
           approach: use the previous day's ET0 as today's demand estimate
           since we don't yet have today's computed ET0. Researchers can
           override by running at phase=-4 or higher.
        5. Compute Ks for each plant.
        6. Write Ks to ``plant.custom['water_stress_ks']``.
        7. Accumulate ``plant.stress_index += (1 - Ks) × stress_increment``.
        8. Update voxel moisture_pct from the returned layer dicts.
        9. Write drainage_mm_today to ``voxel.custom['drainage_mm_today']``.

    Closure captures
    ----------------
    stress_increment_per_day:
        How much ``stress_index`` increases per day at full stress (Ks=0.0).
        Default 0.05 (full stress for 20 days kills a plant: index → 1.0).
        Override via ``field.set_water_params(stress_increment_per_day=...)``,
        but that requires the field to store this per-field, so the hook
        reads it from ``field._water_params`` if present.

    PRD v0.3.0 backward compatibility:
        This hook is only registered when ``use_physics(soil_water_balance=True)``
        is called. Scripts that do not call ``use_physics()`` at all are
        completely unaffected -- ``SoilVoxelState.moisture_pct`` stays at
        whatever value the Soil loader initialised it to.
    """
    from cropforge.physics.hydrology import (
        calculate_tipping_bucket,
        calculate_water_extraction,
        DEFAULT_CROP_COEFFICIENT,
        DEFAULT_STRESS_INCREMENT,
        DEFAULT_FIELD_CAPACITY_PCT,
        DEFAULT_WILTING_POINT_PCT,
        DEFAULT_SATURATION_PCT,
        DEFAULT_DRAINAGE_COEFFICIENT,
    )

    def _hydrology_step(
        state: "FieldState", env: "EnvironmentState"
    ) -> "FieldState":
        """Built-in soil water balance engine -- phase=-3."""

        # ---- Build layer descriptor dicts from SoilVoxelState objects ----
        # We work column-by-column (one plant's soil stack at a time)
        # to keep memory usage low.

        # First pass: apply tipping-bucket to every soil column (shared across
        # plants at the same row/col). We process each (row, col) pair once.
        processed_cells: set = set()
        rain_mm = env.rainfall_mm if env.rainfall_mm > 0.0 else 0.0

        for row_idx, row_soils in enumerate(state.soil):
            for col_idx, cell_soils in enumerate(row_soils):
                if not cell_soils or (row_idx, col_idx) in processed_cells:
                    continue
                processed_cells.add((row_idx, col_idx))

                # Build layer descriptor list
                layer_dicts = _voxels_to_dicts(cell_soils)

                # Apply tipping-bucket (rainfall only -- irrigation was applied
                # by Event.irrigation at end-of-yesterday)
                updated_layers = calculate_tipping_bucket(
                    layer_dicts,
                    precipitation_mm=rain_mm,
                    irrigation_mm=0.0,   # already applied by Event
                )

                # Write drainage back to voxels
                _dicts_to_voxels(updated_layers, cell_soils)

        # Second pass: water extraction per plant (root-zone aware)
        for plant in state.plants:
            if not plant.alive:
                continue

            cell_soils = state.soil[plant.row][plant.col]
            if not cell_soils:
                continue

            layer_dicts = _voxels_to_dicts(cell_soils)

            # Crop coefficient and stress increment may be field-level params
            # (stored in voxel.custom by set_water_params)
            kc = cell_soils[0].custom.get("crop_coefficient", DEFAULT_CROP_COEFFICIENT)
            s_inc = cell_soils[0].custom.get(
                "stress_increment_per_day", stress_increment_per_day
            )

            result = calculate_water_extraction(
                layer_dicts,
                root_depth_cm=plant.root_depth_cm,
                et0_demand=env.et0_mm,
                crop_coefficient=kc,
            )

            # Write updated moisture back to voxels
            _dicts_to_voxels(result["layers"], cell_soils)

            # Write Ks to plant state
            ks = result["ks"]
            plant.custom["water_stress_ks"] = ks

            # Accumulate stress index
            stress_delta = (1.0 - ks) * s_inc
            plant.stress_index = min(1.0, plant.stress_index + stress_delta)

            logger.debug(
                "Hydrology: day=%d plant=%s ks=%.3f stress_index=%.3f",
                env.day, plant.plant_id, ks, plant.stress_index,
            )

        return state

    _hydrology_step.__name__ = "_hydrology_step"
    return _hydrology_step


# ---------------------------------------------------------------------------
# Helper: convert between SoilVoxelState and layer descriptor dicts
# ---------------------------------------------------------------------------

def _voxels_to_dicts(voxels) -> list:
    """Convert a list of SoilVoxelState to hydrology layer descriptor dicts.

    Uses the PRD-specified water parameter names from ``voxel.custom``
    (populated by ``field.set_water_params()``) with sensible defaults.
    """
    from cropforge.physics.hydrology import (
        DEFAULT_FIELD_CAPACITY_PCT,
        DEFAULT_WILTING_POINT_PCT,
        DEFAULT_SATURATION_PCT,
        DEFAULT_DRAINAGE_COEFFICIENT,
    )
    result = []
    for v in voxels:
        result.append({
            "moisture_pct":       v.moisture_pct,
            "field_capacity_pct": v.custom.get("field_capacity_pct",  DEFAULT_FIELD_CAPACITY_PCT),
            "wilting_point_pct":  v.custom.get("wilting_point_pct",   DEFAULT_WILTING_POINT_PCT),
            "saturation_pct":     v.custom.get("saturation_pct",      DEFAULT_SATURATION_PCT),
            "depth_top_cm":       v.depth_top_cm,
            "depth_bottom_cm":    v.depth_bottom_cm,
            "drainage_coefficient": v.custom.get("drainage_coefficient", DEFAULT_DRAINAGE_COEFFICIENT),
        })
    return result


def _dicts_to_voxels(dicts: list, voxels) -> None:
    """Write updated moisture, drainage, and surface-runoff values from layer dicts back to voxels."""
    for layer_dict, voxel in zip(dicts, voxels):
        voxel.moisture_pct = layer_dict["moisture_pct"]
        if "drainage_mm_today" in layer_dict:
            voxel.custom["drainage_mm_today"] = layer_dict["drainage_mm_today"]
        if "surface_runoff_mm_today" in layer_dict:
            voxel.custom["surface_runoff_mm_today"] = layer_dict["surface_runoff_mm_today"]



# ---------------------------------------------------------------------------
# Helper: per-cell slope (for clod dynamics runoff coupling)
# ---------------------------------------------------------------------------

def _compute_slope_normalized(elevation_grid) -> list:
    """Return a 2-D list of per-cell normalized slope fractions (0.0 -- 1.0).

    Each cell's value = (max elevation drop to any 4-neighbour) / (grid max drop).
    Cells with no lower neighbour get 0.0. Used by the hydrology hook when
    clod_dynamics=True to compute slope-driven direct runoff.
    """
    import numpy as np  # already a project dependency
    elev = np.asarray(elevation_grid, dtype=float)
    rows, cols = elev.shape
    raw = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            nbrs = []
            if r > 0:       nbrs.append(elev[r - 1, c])
            if r < rows - 1: nbrs.append(elev[r + 1, c])
            if c > 0:       nbrs.append(elev[r, c - 1])
            if c < cols - 1: nbrs.append(elev[r, c + 1])
            if nbrs:
                raw[r][c] = max(0.0, float(elev[r, c]) - float(min(nbrs)))
    max_drop = max(raw[r][c] for r in range(rows) for c in range(cols))
    if max_drop > 0.0:
        return [[raw[r][c] / max_drop for c in range(cols)] for r in range(rows)]
    return raw  # flat field: all zeros


# ---------------------------------------------------------------------------
# Hook 5: Clod Dynamics  (phase=-4, PRD v0.7.0 §7.4)
# ---------------------------------------------------------------------------

def make_clod_dynamics_hook(
    decay_factor: float = 0.05,
    min_roughness: float = 0.1,
):
    """Return a step function that decays surface_roughness_index each day.

    Simulates physical clod breakdown: rainfall progressively reduces the
    surface micro-relief until a sealed-surface floor is reached.

    Phase: PHASE_CLOD_ENGINE (-4) -- runs before hydrology (-3) so the
    freshly-decayed roughness is used immediately in the tipping-bucket
    runoff calculation.

    Only the TOP layer (layer 0) of each soil column carries
    ``surface_roughness_index``; this hook only touches that layer.

    Closure captures
    ----------------
    decay_factor:
        Fraction of (roughness - min_roughness) lost per 10 mm rainfall.
    min_roughness:
        Floor; roughness stays >= this even after heavy rains.
    """
    from cropforge.physics.soil import calculate_roughness_decay

    def _clod_step(state: "FieldState", env: "EnvironmentState") -> "FieldState":
        """Decay clod roughness -- phase=-4."""
        rain = max(0.0, env.rainfall_mm)
        for row_soils in state.soil:
            for cell_soils in row_soils:
                if cell_soils:
                    v = cell_soils[0]
                    v.surface_roughness_index = calculate_roughness_decay(
                        v.surface_roughness_index, rain, decay_factor, min_roughness
                    )
        return state

    _clod_step.__name__ = "_clod_step"
    return _clod_step


# ---------------------------------------------------------------------------
# Hook 4: Lateral Flow + Nitrogen Transport  (phase=-4)
# ---------------------------------------------------------------------------

def make_nutrients_hook(
    leaching_fraction: float = 0.01,
    runoff_n_fraction: float = 0.05,
):
    """Return a step function that applies lateral water + nitrogen transport.

    Runs at ``PHASE_NUTRIENTS_ENGINE`` (-4), before the vertical hydrology
    hook (-3). This ensures that yesterday's drainage-driven N fluxes are
    applied before today's vertical balance recalculates moisture.

    Per-day execution:
        1. Read ``drainage_mm_today`` from each voxel's ``custom`` dict
           (written by yesterday's hydrology hook).
        2. Apply ``calculate_nitrogen_transport`` per soil column for
           vertical leaching driven by those drainage values.
        3. Compute top-layer excess moisture for every cell to determine
           lateral runoff using the field's ``elevation_grid``.
        4. Apply ``apply_lateral_n_exchange`` to compute net N delta
           for each cell and update ``voxel.nitrogen_kg_ha``.

    PRD v0.3.0 backward compatibility:
        Only registered when ``use_physics(nutrients=True, lateral_flow=True)``.
        Unregistered scripts see no change to ``nitrogen_kg_ha``.
    """
    def _nutrients_step(
        state: "FieldState", env: "EnvironmentState"
    ) -> "FieldState":
        """Built-in lateral flow + nutrients engine -- phase=-4."""
        from cropforge.physics.nutrients import (
            calculate_nitrogen_transport,
            apply_lateral_n_exchange,
        )
        from cropforge.physics.hydrology import (
            DEFAULT_SATURATION_PCT,
            DEFAULT_FIELD_CAPACITY_PCT,
        )

        rows = len(state.soil)
        if rows == 0:
            return state
        cols = len(state.soil[0]) if rows > 0 else 0

        # ---- Step 1 & 2: Vertical N leaching per soil column ----
        for r in range(rows):
            for c in range(cols):
                cell_soils = state.soil[r][c]
                if not cell_soils:
                    continue

                # Collect drainage fluxes from yesterday's hydrology run
                drainage_fluxes = [
                    v.custom.get("drainage_mm_today", 0.0) for v in cell_soils
                ]

                # Build N layer dicts
                n_layers = [
                    {
                        "nitrogen_kg_ha": v.nitrogen_kg_ha,
                        "depth_top_cm":   v.depth_top_cm,
                        "depth_bottom_cm": v.depth_bottom_cm,
                    }
                    for v in cell_soils
                ]

                result = calculate_nitrogen_transport(
                    n_layers,
                    drainage_fluxes,
                    lateral_runoff_mm=0.0,    # lateral handled below
                    leaching_fraction=cell_soils[0].custom.get(
                        "leaching_fraction", leaching_fraction
                    ),
                    runoff_n_fraction=runoff_n_fraction,
                )

                # Write updated N back to voxels
                for i, (layer_out, voxel) in enumerate(zip(result["layers"], cell_soils)):
                    voxel.nitrogen_kg_ha = max(0.0, layer_out.get("nitrogen_kg_ha", 0.0))
                    voxel.custom["n_leached_today_kg_ha"] = result["leached_kg_ha"][i]

        # ---- Steps 3 & 4: Lateral N exchange using surface runoff from yesterday ----
        # The hydrology hook (phase=-3) stores 'surface_runoff_mm_today' in each voxel's
        # custom dict. We read yesterday's value here (nutrients runs at phase=-4, one
        # step before hydrology, so this day's value was written on day-1).
        elev = state.elevation_grid.tolist()

        # Build surface runoff grid from yesterday's tipping-bucket overflow
        # (positive = water that couldn't infiltrate due to saturation cap)
        runoff_grid = [
            [
                state.soil[r][c][0].custom.get("surface_runoff_mm_today", 0.0)
                if state.soil[r][c] else 0.0
                for c in range(cols)
            ]
            for r in range(rows)
        ]

        nitrogen_top = [
            [
                state.soil[r][c][0].nitrogen_kg_ha if state.soil[r][c] else 0.0
                for c in range(cols)
            ]
            for r in range(rows)
        ]

        # D8 routing using surface runoff as the water flux driver
        _NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
                       (0,  -1),           (0,  1),
                       (1,  -1),  (1,  0), (1,  1)]

        delta_n = [[0.0] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                runoff_mm = runoff_grid[r][c]
                if runoff_mm <= 0.0:
                    continue

                n_source = max(0.0, nitrogen_top[r][c])
                if n_source <= 0.0:
                    continue

                # Find steepest downslope neighbour
                elev_self = elev[r][c]
                best_drop = 0.0
                best_nr, best_nc = -1, -1
                for dr, dc in _NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        drop = elev_self - elev[nr][nc]
                        if drop > best_drop:
                            best_drop = drop
                            best_nr, best_nc = nr, nc

                if best_nr == -1:
                    continue  # No lower neighbour — water ponds

                per_cell_rnf = state.soil[r][c][0].custom.get(
                    "runoff_n_fraction", runoff_n_fraction
                )
                # N transported: fraction of available N proportional to runoff volume
                n_moved = min(n_source, runoff_mm * per_cell_rnf * n_source / 100.0)
                delta_n[r][c]               -= n_moved
                delta_n[best_nr][best_nc]   += n_moved

        # Apply N deltas to top-layer voxels
        for r in range(rows):
            for c in range(cols):
                if state.soil[r][c]:
                    dn = delta_n[r][c]
                    voxel = state.soil[r][c][0]
                    voxel.nitrogen_kg_ha = max(0.0, voxel.nitrogen_kg_ha + dn)
                    voxel.custom["lateral_n_delta_kg_ha"] = dn


        logger.debug(
            "Nutrients hook: day=%d vertical leaching + lateral N exchange applied.",
            env.day,
        )
        return state

    _nutrients_step.__name__ = "_nutrients_step"
    return _nutrients_step


# ---------------------------------------------------------------------------
# Hook 5: Radiation Interception Engine  (PRD v0.5.0 §7)
# ---------------------------------------------------------------------------

def make_radiation_hook(
    k_extinction: float = 0.45,
    slope_radiation_correction: bool = False,
    latitude_deg: float = 0.0,
):
    """Return a per-plant Beer-Lambert radiation interception hook.

    Runs at ``PHASE_RADIATION_ENGINE`` (-2), alongside ET0. For each living
    plant it calls ``calculate_intercepted_par()`` using:
        - ``env.radiation_mj_m2`` as incoming solar radiation (possibly
          corrected per-cell by terrain slope/aspect when
          ``slope_radiation_correction=True``)
        - ``plant.lai``
        - The closure ``k_extinction`` parameter

    Writes the result to ``plant.custom['intercepted_par_mj']``.
    When slope correction is active also writes per-cell corrected radiation
    grid to ``env.custom['solar_rad_corrected_mj']`` (2-D list, rows×cols).

    Parameters
    ----------
    k_extinction:
        Beer-Lambert extinction coefficient. Default 0.45 (C3 wheat/rice).
    slope_radiation_correction:
        When True, reads terrain slope/aspect grids and computes per-cell
        solar radiation multipliers. Fields with no terrain use factor=1.0
        (no behaviour change from v0.6.0).
    latitude_deg:
        Site latitude for solar position calculation. Only used when
        ``slope_radiation_correction=True``.
    """
    from cropforge.physics.radiation import (
        calculate_intercepted_par,
        calculate_solar_position,
        calculate_slope_radiation_factor,
    )

    def _radiation_step(
        state: "FieldState", env: "EnvironmentState"
    ) -> "FieldState":
        """Built-in radiation interception engine -- phase=-2."""
        solar_rad_base = max(0.0, env.radiation_mj_m2)

        # --- Build per-cell radiation grid (slope correction or flat 1.0) ---
        if slope_radiation_correction and getattr(state, "terrain", None) is not None:
            terrain = state.terrain
            # Solar noon (hour=12) — daily integral approximation
            # ponytail: solar-noon factor as daily representative; sub-daily not needed at field scale
            alt_deg, az_deg = calculate_solar_position(env.doy, 12.0, latitude_deg)
            rows_n = terrain.slope_grid.shape[0]
            cols_n = terrain.slope_grid.shape[1]
            rad_grid = []
            for r in range(rows_n):
                row_rads = []
                for c in range(cols_n):
                    factor = calculate_slope_radiation_factor(
                        slope_deg=float(terrain.slope_grid[r, c]),
                        aspect_deg=float(terrain.aspect_grid[r, c]),
                        solar_altitude_deg=alt_deg,
                        solar_azimuth_deg=az_deg,
                    )
                    row_rads.append(solar_rad_base * factor)
                rad_grid.append(row_rads)
            env.custom["solar_rad_corrected_mj"] = rad_grid
        else:
            rad_grid = None  # use scalar for all plants

        for plant in state.plants:
            if not plant.alive:
                plant.custom["intercepted_par_mj"] = 0.0
                continue
            if rad_grid is not None:
                cell_rad = rad_grid[plant.row][plant.col]
            else:
                cell_rad = solar_rad_base
            par = calculate_intercepted_par(
                solar_rad_mj=cell_rad,
                lai=plant.lai,
                k_extinction=k_extinction,
            )
            plant.custom["intercepted_par_mj"] = par

        logger.debug(
            "Radiation hook: day=%d solar_rad=%.2f MJ/m2 k=%.3f correction=%s (%d plants).",
            env.day, solar_rad_base, k_extinction, slope_radiation_correction, len(state.plants),
        )
        return state

    _radiation_step.__name__ = "_radiation_step"
    return _radiation_step


# ---------------------------------------------------------------------------
# Hook 6b: Weed Competition Engine  (PRD v1.0.0 §3)
# ---------------------------------------------------------------------------

def make_weed_hook(params):
    """Return the opt-in weed competition hook."""
    import numpy as _np

    from cropforge.physics.weeds import (
        compute_weed_radiation_suppression,
        step_weeds,
    )

    def _weed_step(state: "FieldState", env: "EnvironmentState") -> "FieldState":
        if not getattr(state, "weed_grid", None):
            return state
        active_params = state.custom.get("_weed_params", params)

        rng = getattr(state, "_weed_rng", None)
        if rng is None:
            seed = int(state.custom.get("_weed_seed", 1009))
            rng = _np.random.default_rng(seed)
            setattr(state, "_weed_rng", rng)

        step_weeds(
            weed_grid=state.weed_grid,
            soil_grid=state.soil,
            plant_grid=state.plants,
            env=env,
            params=active_params,
            doy=env.doy,
            rng=rng,
        )

        if env.doy >= active_params.emergence_doy:
            suppression = compute_weed_radiation_suppression(
                state.weed_grid,
                state.plants,
                active_params.competitive_index,
            )
            state.custom["weed_radiation_suppression"] = suppression.tolist()
            for plant in state.plants:
                factor = float(suppression[plant.row, plant.col])
                plant.custom["weed_radiation_suppression"] = factor
                if "intercepted_par_mj" in plant.custom:
                    plant.custom["intercepted_par_mj"] *= factor
                weed = state.weed_grid[plant.row][plant.col]
                plant.custom["weed_lai"] = float(weed.lai) if weed and weed.alive else 0.0
                plant.custom["weed_density_m2"] = (
                    float(active_params.initial_density_m2) if weed and weed.alive else 0.0
                )

        return state

    _weed_step.__name__ = "_weed_step"
    return _weed_step


# ---------------------------------------------------------------------------
# Hook 7: Topographical Wind Field Engine  (PRD v0.7.0 §6)
# ---------------------------------------------------------------------------

def make_wind_hook(
    wind_direction_deg: float = 270.0,
    fetch_cells: int = 3,
    sensitivity: float = 0.3,
):
    """Return a hook that computes a per-cell wind speed grid from terrain.

    Runs at ``PHASE_WIND_ENGINE`` (-2). Reads terrain slope from
    ``state.terrain``. If terrain is absent, writes a flat 1.0 grid so
    downstream consumers (disease engine, ET0) always find the key.

    Writes ``env.custom['wind_speed_ms_grid']`` -- a 2-D list (rows × cols)
    of corrected wind speeds in m/s.

    Parameters
    ----------
    wind_direction_deg:
        Meteorological bearing the wind blows FROM. Default 270 (from West).
    fetch_cells:
        Upwind sampling depth. Default 3 cells.
    sensitivity:
        Multiplier sensitivity to shelter index. Default 0.3.

    Backward compatibility:
        Only registered when ``use_physics(terrain_wind=True)``. Scripts
        without it never see 'wind_speed_ms_grid' in env.custom.
    """
    from cropforge.physics.wind import calculate_wind_multiplier

    def _wind_step(
        state: "FieldState", env: "EnvironmentState"
    ) -> "FieldState":
        """Built-in topographical wind field engine -- phase=-2."""
        import numpy as np
        base_wind = max(0.0, env.wind_speed_ms)
        terrain = getattr(state, "terrain", None)

        if terrain is not None:
            mult = calculate_wind_multiplier(
                terrain.elevation_grid,
                wind_direction_deg=wind_direction_deg,
                fetch_cells=fetch_cells,
                sensitivity=sensitivity,
            )
        else:
            # No terrain → flat 1.0 grid; disease engine still gets the key
            rows = len(state.soil)
            cols = len(state.soil[0]) if rows > 0 else 0
            mult = np.ones((rows, cols), dtype=float)

        # Write 2-D list of corrected wind speeds
        env.custom["wind_speed_ms_grid"] = (
            (mult * base_wind).tolist()
        )
        env.custom["wind_multiplier_grid"] = mult.tolist()

        logger.debug(
            "Wind hook: day=%d base_wind=%.2f m/s direction=%.0f° terrain=%s.",
            env.day, base_wind, wind_direction_deg, terrain is not None,
        )
        return state

    _wind_step.__name__ = "_wind_step"
    return _wind_step

def make_disease_hook(
    initial_foci: list[tuple[int, int]] | None = None,
    spread_rate: float = 0.15,
    latency_days: int = 5,
    stress_increment: float = 0.04,
    wind_direction_deg: float = 270.0,
    anisotropy: float = 0.80,
    seed: int | None = None,
):
    """Return a spatially explicit SIR disease spread hook.

    Registered at ``PHASE_DISEASE_ENGINE`` (-1). On each simulation day:
      1. On day 1 only: seeds the initial infection foci specified by
         ``initial_foci``.
      2. Calls ``calculate_disease_spread()`` with the closure parameters
         and the current day's wind speed (from ``env.wind_speed_ms``).

    Parameters
    ----------
    initial_foci:
        List of (row, col) tuples to infect on the first simulation day.
        If None, no automatic seeding occurs (manual Event seeding still
        works via plant.custom['disease_state'] = 'I').
    spread_rate:
        Base daily infection probability from each infected cell to each
        susceptible neighbour (isotropic baseline). Range [0, 1].
        Default 0.15.
    latency_days:
        Days from infection to when the plant becomes infectious (contagious).
        Default 5 (5-day latency period).
    stress_increment:
        Daily disease_stress increment for each infected plant. Default 0.04.
    wind_direction_deg:
        Prevailing wind direction in met bearing (0°=N, 90°=E, 180°=S, 270°=W).
        Direction FROM which wind blows. Default 270.0 (wind from West → blows East).
    anisotropy:
        Strength of directional bias. 0=isotropic, 1=fully directional.
        Default 0.80.
    seed:
        Random seed for reproducible spread. If None, non-deterministic.

    Backward compatibility:
        When this hook is NOT registered (disease=False), all plant.custom
        dicts have no 'disease_state' key. Code reading disease_state must
        use: plant.custom.get('disease_state', 'S').
    """
    from cropforge.physics.pathology import (
        calculate_disease_spread,
        seed_initial_foci,
    )

    rng = random.Random(seed) if seed is not None else random.Random()
    foci = list(initial_foci) if initial_foci else []
    _seeded = [False]  # mutable closure flag

    def _disease_step(
        state: "FieldState", env: "EnvironmentState"
    ) -> "FieldState":
        """Built-in spatial disease spread engine -- phase=-1."""
        # Rebuild plant_grid from flat list for 2D neighbourhood access
        rows = len(state.soil)
        cols = len(state.soil[0]) if rows > 0 else 0
        plant_grid = [
            state.plants[r * cols: (r + 1) * cols]
            for r in range(rows)
        ]

        # Seed initial foci exactly once (day 1 equivalent)
        if not _seeded[0] and foci:
            seed_initial_foci(plant_grid, foci)
            _seeded[0] = True
            logger.info(
                "Disease hook: initial foci seeded at %s.", foci
            )

        # Use spatially corrected wind speed if terrain_wind is active,
        # falling back to the scalar for full backward compatibility.
        wind_grid = env.custom.get("wind_speed_ms_grid")
        if wind_grid is not None:
            # Representative field-mean wind for the scalar spread probability.
            # ponytail: mean scalar; per-cell spread needs redesign of calculate_disease_spread.
            total = sum(v for row in wind_grid for v in row)
            n = sum(len(row) for row in wind_grid)
            effective_wind = total / n if n else env.wind_speed_ms
        else:
            effective_wind = env.wind_speed_ms

        calculate_disease_spread(
            plant_grid=plant_grid,
            wind_speed_ms=effective_wind,
            wind_direction_deg=wind_direction_deg,
            base_infection_rate=spread_rate,
            latency_days=latency_days,
            stress_increment=stress_increment,
            anisotropy=anisotropy,
            rng=rng,
        )

        logger.debug(
            "Disease hook: day=%d wind=%.1f° spread=%.2f",
            env.day, wind_direction_deg, spread_rate,
        )
        return state

    _disease_step.__name__ = "_disease_step"
    return _disease_step


# ---------------------------------------------------------------------------
# Hook 8: Root Effective Depth Clamp  (PRD v0.7.0 §7.3)
# ---------------------------------------------------------------------------

def make_root_clamp_hook():
    """Return a hook that clamps plant root depth to the effective soil depth.

    Runs at ``PHASE_ROOT_CLAMP`` (+1), i.e. after all plugin step functions
    (phase 0) have grown the roots for the day.

    Reads ``state.custom['effective_soil_depth_cm_grid']`` — a 2-D list of
    per-cell caps in centimetres — computed by ``Field._initialize_field``
    from base soil profile depth + land-preparation elevation delta.

    If the grid is absent (e.g. old serialised states), the hook is a no-op
    and backward compatibility is fully preserved.

    Backward compatibility:
        Only registered when ``use_physics(root_clamping=True)``.  Without
        it, plant.root_depth_cm is unconstrained by this engine.
    """
    def _root_clamp_step(
        state: "FieldState", env: "EnvironmentState"
    ) -> "FieldState":
        """Built-in root depth clamp engine -- phase=+1."""
        grid = state.custom.get("effective_soil_depth_cm_grid")
        if grid is None:
            return state  # no-op; grid absent
        n_clamped = 0
        for plant in state.plants:
            if not plant.alive:
                continue
            cap = grid[plant.row][plant.col]
            if plant.root_depth_cm > cap:
                plant.root_depth_cm = cap
                n_clamped += 1
        if n_clamped:
            logger.debug(
                "Root clamp hook: day=%d clamped %d plants to effective soil depth.",
                env.day, n_clamped,
            )
        return state

    _root_clamp_step.__name__ = "_root_clamp_step"
    return _root_clamp_step


# ---------------------------------------------------------------------------
# Hook 8: Soil Erosion Engine  (phase=-2, PRD v0.7.0 §7.5)
# ---------------------------------------------------------------------------

def make_erosion_hook() -> object:
    """Return a step function that computes daily soil erosion index per cell.

    Implements a simplified RUSLE model:

        erosion_index = runoff_mm × slope_frac × (1 - roughness) × (1 - veg_cover)

    Phase: ``PHASE_EROSION_ENGINE`` (-2) -- runs after the hydrology hook (-3)
    so ``surface_runoff_mm_today`` is already populated.  When ``water_balance``
    is not enabled, falls back to ``env.rainfall_mm`` as the runoff proxy.

    Accumulates daily values into
    ``state.custom['cumulative_erosion_index_grid']`` (2-D list, rows × cols).
    Per-day values are also written to
    ``state.custom['daily_erosion_index_grid']`` for diagnostics.

    Vegetation cover per cell is derived from the maximum LAI among all
    plants at that (row, col): ``cover = min(1.0, max_lai / 3.0)``.
    Cells with no plants use cover = 0.0 (bare ground).
    """
    from cropforge.physics.soil import calculate_erosion_index

    _slope_cache: list = [None]  # ponytail: lazy one-time compute

    def _erosion_step(state: "FieldState", env: "EnvironmentState") -> "FieldState":
        """Compute + accumulate daily erosion index -- phase=-2."""
        # Prefer slope grid written by sediment hook (geomorphological feedback, PRD v0.8.0 §6.1).
        # Falls back to closure cache (first day, or sediment transport disabled).
        slopes = state.custom.get("slope_grid") or _slope_cache[0]
        if slopes is None:
            _slope_cache[0] = _compute_slope_normalized(state.elevation_grid)
            state.custom["slope_grid"] = _slope_cache[0]
            slopes = _slope_cache[0]

        rows = len(state.soil)
        if not rows:
            return state
        cols = len(state.soil[0])

        # Build per-cell vegetation cover from LAI (Beer-Lambert proxy)
        # ponytail: max_lai / 3.0; ceiling 1.0
        cell_lai: dict = {}
        for plant in state.plants:
            if plant.alive:
                key = (plant.row, plant.col)
                cell_lai[key] = max(cell_lai.get(key, 0.0), plant.lai)

        # Initialise accumulator grids if first day
        cum_key = "cumulative_erosion_index_grid"
        day_key = "daily_erosion_index_grid"
        if cum_key not in state.custom:
            state.custom[cum_key] = [[0.0] * cols for _ in range(rows)]

        daily_grid = [[0.0] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                cell_soils = state.soil[r][c]
                if not cell_soils:
                    continue
                vox = cell_soils[0]  # top layer only
                runoff = vox.custom.get("surface_runoff_mm_today", env.rainfall_mm)
                slope_frac = slopes[r][c]
                roughness = vox.surface_roughness_index
                veg_cover = min(1.0, cell_lai.get((r, c), 0.0) / 3.0)

                idx = calculate_erosion_index(
                    runoff, slope_frac, roughness, veg_cover
                )
                daily_grid[r][c] = idx
                state.custom[cum_key][r][c] += idx

        state.custom[day_key] = daily_grid
        return state

    _erosion_step.__name__ = "_erosion_step"
    return _erosion_step


# ---------------------------------------------------------------------------
# Hook 9: Sediment Transport  (phase=+2, PRD v0.8.0 §5.2)
# ---------------------------------------------------------------------------

def make_sediment_hook(
    k_erodibility: float = 0.005,
    k_transport: float = 0.02,
    terrain_feedback: bool = True,
) -> object:
    """Return a step function that routes detached soil via D8 each day.

    Algorithm per day:
      1. Read ``daily_erosion_index_grid`` written by the erosion hook.
      2. Read ``surface_runoff_mm_today`` per cell from soil voxels.
      3. Call ``calculate_sediment_transport`` per cell → eroded_mm grid.
      4. Route eroded_mm downslope via the existing ``route_surface_water``
         D8 function (identical routing, different payload).
      5. Update ``state.custom['effective_soil_depth_cm_grid']``:
           - donor cell  : depth -= eroded_mm / 10  (cm)
           - receiver cell: depth += deposited_mm / 10 (cm)
      6. Accumulate ``cumulative_sediment_*`` grids for observability.

    Mass conservation guarantee:
        sum(eroded) == sum(deposited) + boundary_escaped
        where boundary_escaped is sediment from edge cells with no lower
        neighbour (exits field domain, discarded identically to D8 water).

    Dependencies:
        Requires ``erosion=True`` (writes daily_erosion_index_grid) and
        ``water_balance=True`` (writes surface_runoff_mm_today).
    """
    from cropforge.physics.soil import calculate_sediment_transport
    from cropforge.physics.hydrology import route_surface_water

    _slope_cache: list = [None]  # ponytail: lazy one-time compute, same pattern as erosion hook

    def _sediment_step(state: "FieldState", env: "EnvironmentState") -> "FieldState":
        """D8 sediment routing engine -- phase=+2 (day-end)."""
        # Use slope from state.custom if available (set by erosion hook this day)
        slopes = state.custom.get("slope_grid") or _slope_cache[0]
        if slopes is None:
            _slope_cache[0] = _compute_slope_normalized(state.elevation_grid)
            state.custom["slope_grid"] = _slope_cache[0]
            slopes = _slope_cache[0]

        rows = len(state.soil)
        if not rows:
            return state
        cols = len(state.soil[0])

        # ---- Step 1-3: Compute eroded_mm per cell ----
        daily_erosion = state.custom.get("daily_erosion_index_grid") or [
            [0.0] * cols for _ in range(rows)
        ]

        eroded_grid = [[0.0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                cell_soils = state.soil[r][c]
                if not cell_soils:
                    continue
                runoff = cell_soils[0].custom.get("surface_runoff_mm_today", env.rainfall_mm)
                idx = daily_erosion[r][c]
                slope = slopes[r][c]
                eroded, _ = calculate_sediment_transport(
                    idx, runoff, slope, k_erodibility, k_transport
                )
                # Cap erosion by available layer-0 thickness (safety floor)
                if cell_soils:
                    available_mm = max(0.0, cell_soils[0].depth_bottom_cm - cell_soils[0].depth_top_cm) * 10.0
                    eroded = min(eroded, max(0.0, available_mm - 1.0))  # keep at least 1mm of topsoil
                eroded_grid[r][c] = eroded

        # ---- Step 4: D8 route eroded soil downslope ----
        # ponytail: reuse route_surface_water — same D8 algorithm, different payload
        elev = state.elevation_grid.tolist()
        deposit_grid = route_surface_water(eroded_grid, elev)

        # ---- Step 5: Update effective soil depth ----
        depth_grid = state.custom.get("effective_soil_depth_cm_grid")
        if depth_grid is None:
            # Fallback: build from bottom of soil profile (should always exist post-init)
            depth_grid = [
                [
                    (state.soil[r][c][-1].depth_bottom_cm if state.soil[r][c] else 30.0)
                    for c in range(cols)
                ]
                for r in range(rows)
            ]
            state.custom["effective_soil_depth_cm_grid"] = depth_grid

        for r in range(rows):
            for c in range(cols):
                loss_cm = eroded_grid[r][c] / 10.0    # mm → cm
                gain_cm = deposit_grid[r][c] / 10.0
                depth_grid[r][c] = max(0.0, depth_grid[r][c] - loss_cm + gain_cm)

                # ---- Task 2: Expand / contract Layer-0 thickness (PRD v0.8.0 §6.2) ----
                cell_soils = state.soil[r][c]
                if cell_soils:
                    vox0 = cell_soils[0]
                    net_cm = gain_cm - loss_cm
                    new_bottom = vox0.depth_bottom_cm + net_cm
                    # Floor: 0.1cm so layer never vanishes; ponytail: expose layer-1 in v0.9.0
                    vox0.depth_bottom_cm = max(vox0.depth_top_cm + 0.1, new_bottom)

        # ---- Task 3: Burial penalty (PRD v0.8.0 §6.3) ----
        _BURIAL_THRESHOLD_MM = 50.0   # ponytail: constant; make configurable when user requests
        for plant in state.plants:
            if not plant.alive:
                continue
            dep = deposit_grid[plant.row][plant.col]
            if dep > _BURIAL_THRESHOLD_MM and plant.height_cm < dep:
                # Rapid deep deposition buries small plants
                plant.stress_index = min(1.0, plant.stress_index + 0.5)

        # ---- Task 1: Update elevation grid + refresh slope for next day's erosion ----
        # (PRD v0.8.0 §6.1 — geomorphological feedback)
        for r in range(rows):
            for c in range(cols):
                delta_m = (deposit_grid[r][c] - eroded_grid[r][c]) / 1000.0  # mm → m
                if terrain_feedback:
                    state.elevation_grid[r, c] += delta_m

        # Recompute slope from updated DEM; stored in state.custom for next day's erosion hook.
        # terrain_feedback=False freezes terrain geometry after init (PRD v0.9.0 §4.3).
        if terrain_feedback:
            state.custom["slope_grid"] = _compute_slope_normalized(state.elevation_grid)

        # ---- Step 6: Accumulate season grids for observability ----
        for key, grid in (
            ("cumulative_sediment_eroded_mm_grid",   eroded_grid),
            ("cumulative_sediment_deposited_mm_grid", deposit_grid),
        ):
            if key not in state.custom:
                state.custom[key] = [[0.0] * cols for _ in range(rows)]
            cum = state.custom[key]
            for r in range(rows):
                for c in range(cols):
                    cum[r][c] += grid[r][c]

        state.custom["daily_sediment_eroded_mm_grid"]    = eroded_grid
        state.custom["daily_sediment_deposited_mm_grid"] = deposit_grid

        logger.debug(
            "Sediment hook: day=%d total_eroded=%.4f mm total_deposited=%.4f mm.",
            env.day,
            sum(eroded_grid[r][c] for r in range(rows) for c in range(cols)),
            sum(deposit_grid[r][c] for r in range(rows) for c in range(cols)),
        )
        return state

    _sediment_step.__name__ = "_sediment_step"
    return _sediment_step
