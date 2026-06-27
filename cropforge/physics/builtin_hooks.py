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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cropforge.state import EnvironmentState, FieldState

logger = logging.getLogger(__name__)

# Phase constants -- must be negative so they precede all researcher steps
# PRD v0.3.0 execution order (extended for lateral flow + nutrients):
PHASE_NUTRIENTS_ENGINE  = -4   # Lateral flow + N transport (runs before hydrology)
PHASE_HYDROLOGY_ENGINE  = -3   # Soil water balance (updates moisture)
PHASE_ET0_ENGINE        = -2   # Penman-Monteith ET0
PHASE_ROOT_ENGINE       = -1   # Root impedance multiplier


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

                # Apply tipping-bucket (lateral inflow treated as additional precipitation)
                updated_layers = calculate_tipping_bucket(
                    layer_dicts,
                    precipitation_mm=effective_precip_mm,
                    irrigation_mm=0.0,   # already applied by Event
                )

                # Write drainage + surface runoff back to voxels
                _dicts_to_voxels(updated_layers, cell_soils)

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
