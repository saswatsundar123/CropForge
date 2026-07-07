"""
cropforge/plugins/maize.py
===========================
StandardMaize — First-party maize crop plugin for CropForge v0.5.0.

Ports the root-impedance / hard-pan clamping and simplified water-stress
logic from examples/maize_dual_plot.py directly into a CropPlugin, proving
that plugins can interact safely with the engine's phase=-1 physics.

C4 photosynthesis parameters (higher RUE than wheat, higher base temperature).

PRD References: §6.2, §6.3 (v0.5.0)

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import math

from cropforge._plugins_base import CropPlugin, register_crop


# ---------------------------------------------------------------------------
# Phenological stage thermal-time thresholds — CERES-Maize inspired
# Base temp = 8 °C (C4 pathway)
# ---------------------------------------------------------------------------

_STAGE_TT = {
    "germination":    0.0,
    "emergence":     80.0,
    "vegetative":   250.0,
    "tasseling":    600.0,
    "silking":      700.0,
    "grain_fill":   850.0,
    "maturity":    1200.0,
}

_STAGE_ORDER = [
    "germination", "emergence", "vegetative",
    "tasseling", "silking", "grain_fill", "maturity",
]


def _get_stage(thermal_time: float) -> str:
    stage = "germination"
    for name in _STAGE_ORDER:
        if thermal_time >= _STAGE_TT[name]:
            stage = name
    return stage


def _get_stage_progress(thermal_time: float, stage: str) -> float:
    """Fractional progress within *stage* [0.0, 1.0] — ponytail: same pattern as wheat."""
    idx = _STAGE_ORDER.index(stage)
    start_tt = _STAGE_TT[stage]
    end_tt = _STAGE_TT[_STAGE_ORDER[idx + 1]] if idx + 1 < len(_STAGE_ORDER) else start_tt + 400.0
    if end_tt <= start_tt:
        return 1.0
    return min(1.0, max(0.0, (thermal_time - start_tt) / (end_tt - start_tt)))


@register_crop("maize")
class StandardMaize(CropPlugin):
    """CERES-Maize inspired crop plugin for CropForge.

    Combines:
    - C4 thermal-time phenology (base_temp=8°C, RUE=1.70)
    - Root-impedance / hard-pan clamping (ported from maize_dual_plot.py)
    - Simplified water-stress accumulation and plant death

    Parameters
    ----------
    base_temp_c:
        Base temperature for thermal time (°C). Maize default = 8.0.
    rue:
        Radiation Use Efficiency (g DM / MJ PAR). C4 default = 1.70.
    k_extinction:
        Beer-Lambert extinction coefficient. Default = 0.50.
    base_root_rate_cm:
        Daily root extension rate under optimal soil (cm/day). Default = 0.35.
    pwp_pct:
        Permanent wilting point for stress check (% VWC). Default = 13.0.
    stress_death_days:
        Consecutive days below PWP before plant death. Default = 5.
    grain_partition_fraction:
        Fraction of daily biomass added to grain during grain_fill. Default = 0.40.
    k_height:
        Height allometry constant (cm / √g). Default = 0.9.
    sowing_doy:
        Default Kharif sowing DOY. Default = 120.

    Examples
    --------
    >>> from cropforge.plugins import StandardMaize
    >>> plugin = StandardMaize(base_root_rate_cm=0.40)
    >>> plugin.base_root_rate_cm
    0.4
    """

    species = "Zea mays"

    def __init__(
        self,
        base_temp_c: float = 8.0,
        rue: float = 1.70,
        k_extinction: float = 0.50,
        base_root_rate_cm: float = 0.35,
        pwp_pct: float = 13.0,
        stress_death_days: int = 5,
        grain_partition_fraction: float = 0.40,
        k_height: float = 0.9,
        sowing_doy: int = 120,
    ):
        self.base_temp_c = base_temp_c
        self.rue = rue
        self.k_extinction = k_extinction
        self.base_root_rate_cm = base_root_rate_cm
        self.pwp_pct = pwp_pct
        self.stress_death_days = stress_death_days
        self.grain_partition_fraction = grain_partition_fraction
        self.k_height = k_height
        self.sowing_doy = sowing_doy

    # ------------------------------------------------------------------
    # CropPlugin interface
    # ------------------------------------------------------------------

    def step(self, state, env):
        """Daily maize growth step: phenology → roots → biomass → water stress."""
        self._update_soil_moisture(state, env)

        for plant in state.plants:
            if not plant.alive:
                continue
            self._update_phenology(plant, env)
            self._update_lai(plant, env)
            self._grow_roots(plant, state)
            self._accumulate_biomass(plant, env)
            self._partition_grain(plant)
            self._update_height(plant)
            self._check_water_stress(plant, state)
        return state

    # ------------------------------------------------------------------
    # Internal sub-models
    # ------------------------------------------------------------------

    def _update_phenology(self, plant, env) -> None:
        tt_today = max(0.0, env.temp_mean_c - self.base_temp_c)
        plant.custom["thermal_time"] = plant.custom.get("thermal_time", 0.0) + tt_today
        stage = _get_stage(plant.custom["thermal_time"])
        plant.custom["phenological_stage"] = stage
        plant.phenological_stage = stage
        plant.stage_progress = _get_stage_progress(plant.custom["thermal_time"], stage)

    def _update_lai(self, plant, env) -> None:
        stage = plant.custom.get("phenological_stage", "germination")
        tt_today = max(0.0, env.temp_mean_c - self.base_temp_c)
        lai_slope = 0.003  # C4 higher peak LAI

        if stage in ("germination", "emergence"):
            plant.lai = max(0.0, plant.lai + tt_today * lai_slope * 0.2)
        elif stage in ("vegetative", "tasseling", "silking"):
            plant.lai = max(0.0, plant.lai + tt_today * lai_slope)
        elif stage in ("grain_fill", "maturity"):
            plant.lai = max(0.0, plant.lai * 0.985)  # senescence

    def _grow_roots(self, plant, state) -> None:
        """Root extension with hard-pan clamping (ported from maize_dual_plot.py).

        Reads plant.root_growth_multiplier set by the impedance engine at
        phase=-1 (when enabled). Falls back to 1.0 if engine is not active.
        """
        from cropforge.physics.soil import calculate_root_impedance

        if plant.root_depth_cm == 0.0:
            plant.root_depth_cm = 2.0  # germination: 2 cm root at sowing

        # root_growth_multiplier is 1.0 by default (no physics engine needed)
        daily_extension = self.base_root_rate_cm * plant.root_growth_multiplier
        if daily_extension <= 0.0:
            return

        new_depth = plant.root_depth_cm + daily_extension

        # Safety clamp: stop at first blocked soil layer (exact boundary)
        soil_col = state.soil[plant.row][plant.col]
        for voxel in soil_col:
            mult = calculate_root_impedance(voxel.penetration_resistance)
            if mult <= 0.0:
                if new_depth >= voxel.depth_top_cm:
                    new_depth = voxel.depth_top_cm
                    break

        plant.root_depth_cm = new_depth

    def _accumulate_biomass(self, plant, env) -> None:
        ks = plant.custom.get("water_stress_ks", 1.0)
        kn = plant.custom.get("n_stress_kn", 1.0)
        intercepted_par = (
            env.radiation_mj_m2
            * 0.5
            * (1.0 - math.exp(-self.k_extinction * max(0.0, plant.lai)))
        )
        plant.biomass_g += max(0.0, self.rue * intercepted_par * ks * kn)

    def _partition_grain(self, plant) -> None:
        stage = plant.custom.get("phenological_stage", "germination")
        if stage == "grain_fill":
            prev = plant.custom.get("_prev_biomass_g", plant.biomass_g)
            delta = max(0.0, plant.biomass_g - prev)
            plant.custom["grain_biomass_g"] = (
                plant.custom.get("grain_biomass_g", 0.0)
                + delta * self.grain_partition_fraction
            )
        plant.custom["_prev_biomass_g"] = plant.biomass_g

    def _update_height(self, plant) -> None:
        plant.height_cm = self.k_height * math.sqrt(max(0.0, plant.biomass_g))

    def _update_soil_moisture(self, state, env) -> None:
        """Simplified ET draw-down and rainfall refill (from maize_dual_plot.py)."""
        et0_fraction = 0.55
        daily_depletion = env.et0_mm * et0_fraction * 0.08

        for row_list in state.soil:
            for col_list in row_list:
                if col_list:
                    col_list[0].moisture_pct = max(
                        0.0,
                        col_list[0].moisture_pct - daily_depletion,
                    )
                    col_list[0].moisture_pct = min(
                        32.0,
                        col_list[0].moisture_pct + env.rainfall_mm * 0.06,
                    )

    def _check_water_stress(self, plant, state) -> None:
        """PWP-based stress accumulation and plant death."""
        soil_col = state.soil[plant.row][plant.col]
        surface_moisture = (
            soil_col[0].moisture_pct if soil_col else self.pwp_pct + 1.0
        )

        if surface_moisture < self.pwp_pct:
            plant.stress_index = min(1.0, plant.stress_index + 0.2)
            plant.custom.setdefault("stress_days", 0)
            plant.custom["stress_days"] += 1
            if plant.custom["stress_days"] >= self.stress_death_days:
                plant.alive = False
        else:
            plant.stress_index = max(0.0, plant.stress_index - 0.05)
            plant.custom["stress_days"] = 0

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def default_crop(cls):
        """Return a Crop pre-configured for DH-Maize Kharif variety."""
        from cropforge import Crop
        return Crop(species=cls.species, variety="DH-Maize", sowing_doy=120)
