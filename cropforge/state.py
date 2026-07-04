"""
cropforge/state.py
==================
SimulationState schema -- the central data contract for CropForge.

Every object in the runtime engine, state logger, and visual frontend
reads from or writes to the structures defined here.

Design Rule (PRD Section 5):
    Every class carries a ``custom: Dict[str, Any]`` field. This allows
    researchers to attach arbitrary variables without modifying the core
    schema. CropForge logs all custom fields automatically.

v0.2.0 additions (PRD v0.2.0 Section 11):
    PlantState.root_growth_multiplier -- engine-computed impedance multiplier.
    EnvironmentState: four FAO-56 intermediate computed fields
        (vp_kpa, psychrometric_kpa, slope_svp, net_radiation_mj).
    All new fields default to non-breaking values so v0.1.0 scripts run
    without modification.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from cropforge.terrain import Terrain


# ---------------------------------------------------------------------------
# 5.1  PlantState
# ---------------------------------------------------------------------------

@dataclass
class PlantState:
    """State of a single plant at one simulation timestep (PRD Section 5.1).

    v0.2.0 addition:
        root_growth_multiplier: float = 1.0
            Set by the root-impedance engine (when enabled) to the
            calculate_root_impedance() value for the layer at the current
            root front. Defaults to 1.0 (unrestricted) so v0.1.0 scripts
            that manage root_depth_cm themselves are unaffected.
    """

    plant_id: str
    row: int
    col: int

    # Physiological state
    age_days: int = 0
    lai: float = 0.0
    biomass_g: float = 0.0
    height_cm: float = 0.0
    root_depth_cm: float = 0.0
    stress_index: float = 0.0
    alive: bool = True
    phenological_stage: str = "germination"

    # v0.2.0 -- root impedance multiplier (PRD v0.2.0 Section 5.3)
    # 1.0 = unrestricted, 0.0 = hard pan block.
    # Engine-set when use_physics(root_impedance=True); stays 1.0 otherwise.
    root_growth_multiplier: float = 1.0

    custom: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 5.2  SoilVoxelState
# ---------------------------------------------------------------------------

@dataclass
class SoilVoxelState:
    """State of one soil voxel at one timestep (PRD Section 5.2)."""

    row: int
    col: int
    layer: int
    depth_top_cm: float
    depth_bottom_cm: float
    moisture_pct: float
    nitrogen_kg_ha: float
    bulk_density: float
    penetration_resistance: float

    # v0.6.0 -- land preparation soil-property deltas (PRD v0.6.0 §6.6)
    # Computed once at sim start by LandPrep.apply(); default 0.0 preserves v0.5.0 compatibility.
    porosity_delta: float = 0.0
    bulk_density_delta: float = 0.0
    surface_roughness_index: float = 0.0  # ponytail: computed but not consumed by physics until v0.7.0

    custom: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 5.3  FieldState
# ---------------------------------------------------------------------------

@dataclass
class FieldState:
    """Complete state of one field at one simulation day (PRD Section 5.3)."""

    day: int
    plants: List[PlantState]
    soil: List[List[List[SoilVoxelState]]]   # [row][col][layer]
    elevation_grid: np.ndarray               # shape (n_rows, n_cols), float64
    events_fired: List[str]

    # v0.6.0 -- terrain geometry (PRD v0.6.0 §5.6)
    # None means flat field -- zero behaviour change from v0.5.0.
    terrain: Optional["Terrain"] = None

    custom: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 5.4  EnvironmentState
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentState:
    """Meteorological conditions for one field on one day (PRD Section 5.4).

    v0.2.0 additions (PRD v0.2.0 Section 11.2):
        vp_kpa, psychrometric_kpa, slope_svp, net_radiation_mj are FAO-56
        intermediate values populated by the Penman-Monteith engine when
        enabled. All default to 0.0 -- v0.1.0 scripts run unchanged.
    """

    # Core meteorological fields -- v0.1.0 (schema frozen)
    day: int
    doy: int
    temp_max_c: float
    temp_min_c: float
    temp_mean_c: float
    radiation_mj_m2: float
    rainfall_mm: float
    et0_mm: float
    wind_speed_ms: float
    humidity_pct: float

    co2_ppm: float = 415.0

    # v0.2.0 additions -- FAO-56 computed intermediates (PRD v0.2.0 Section 11)
    # Default 0.0 preserves full backward compatibility with v0.1.0.
    vp_kpa: float = 0.0              # Actual vapour pressure (kPa)
    psychrometric_kpa: float = 0.0   # Psychrometric constant gamma (kPa/degC)
    slope_svp: float = 0.0           # Slope of SVP curve Delta (kPa/degC)
    net_radiation_mj: float = 0.0    # Net radiation Rn (MJ m-2 day-1)

    # v0.4.0 -- multi-season tracking (PRD v0.4.0 Section 7)
    # Defaults to 1 so all v0.1.0-v0.3.0 scripts run identically.
    # Set automatically by farm.run() from farm._current_season.
    season: int = 1

    custom: Dict[str, Any] = field(default_factory=dict)
