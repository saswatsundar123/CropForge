"""
cropforge/physics/builtin_hooks.py
====================================
Built-in physics engine hooks for CropForge v0.2.0.

These are step functions that the engine registers at phase=-2 (ET0) and
phase=-1 (root impedance) when the researcher calls farm.use_physics().
Because their phase values are negative, they are guaranteed to execute
BEFORE any researcher-registered @farm.step (which must be >= 0).

Each hook follows the standard step function signature:
    fn(state: FieldState, env: EnvironmentState) -> FieldState

They are pure callables -- they carry no state of their own. All
configuration is captured at registration time via closures.

PRD v0.2.0 References:
    Section 4   -- Penman-Monteith ET0 Engine
    Section 5   -- Root Growth Engine
    Section 9   -- Daily Timestep Execution Order (steps 2 and 3)
    Section 10  -- Backward Compatibility Requirements

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
PHASE_ET0_ENGINE    = -2   # PRD Section 9, step 2: ET0 before root extension
PHASE_ROOT_ENGINE   = -1   # PRD Section 9, step 3: root depth after ET0


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
