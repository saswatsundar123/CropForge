"""
cropforge/plugins/wheat.py
===========================
StandardWheat — First-party wheat crop plugin for CropForge v0.5.0.

Implements a simplified CERES-Wheat style model:
  - Thermal-time driven phenology (6 stages)
  - RUE × intercepted PAR biomass accumulation
  - Grain biomass partitioning during grain_fill stage
  - LAI development and senescence

Default parameters are calibrated for HD-2967 (Indian subcontinent, Rabi).
All parameters are researcher-overridable at construction time.

PRD References: §5.2, §5.3 (v0.5.0)

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import math
from typing import Optional

from cropforge._plugins_base import CropPlugin, register_crop

# Auto-register first-party GLTF stage models when this plugin is imported.
# Researchers never need to call AssetRegistry.register() manually.
try:
    from cropforge.viz.registry import initialize_first_party_bundles as _init_bundles
    _init_bundles()
except Exception:  # ponytail: never crash on missing visual assets
    pass


# ---------------------------------------------------------------------------
# Phenological stage thermal-time thresholds (degree-days above base temp)
# Calibrated for HD-2967 (ICAR default variety)
# ---------------------------------------------------------------------------

_STAGE_TT = {
    "germination":    0.0,
    "emergence":     60.0,
    "tillering":    200.0,
    "stem_extension": 500.0,
    "anthesis":     800.0,
    "grain_fill":  1000.0,
    "maturity":    1400.0,
}

# Ordered list for transition checks
_STAGE_ORDER = [
    "germination", "emergence", "tillering",
    "stem_extension", "anthesis", "grain_fill", "maturity",
]


def _get_stage(thermal_time: float) -> str:
    """Return phenological stage name for accumulated thermal time."""
    stage = "germination"
    for name in _STAGE_ORDER:
        if thermal_time >= _STAGE_TT[name]:
            stage = name
    return stage


def _get_stage_progress(thermal_time: float, stage: str) -> float:
    """Fractional progress within *stage* [0.0, 1.0].

    ponytail: uses existing _STAGE_TT/_STAGE_ORDER — no new data structures.
    """
    idx = _STAGE_ORDER.index(stage)
    start_tt = _STAGE_TT[stage]
    # End TT = next stage's threshold (or +400 for final stage as a sentinel)
    end_tt = _STAGE_TT[_STAGE_ORDER[idx + 1]] if idx + 1 < len(_STAGE_ORDER) else start_tt + 400.0
    if end_tt <= start_tt:
        return 1.0
    return min(1.0, max(0.0, (thermal_time - start_tt) / (end_tt - start_tt)))


@register_crop("wheat")
class StandardWheat(CropPlugin):
    """CERES-Wheat style crop plugin for CropForge.

    Parameters
    ----------
    base_temp_c:
        Base temperature for thermal time accumulation (°C). HD-2967 default = 0.0.
    rue:
        Radiation Use Efficiency (g DM / MJ PAR). HD-2967 default = 1.24.
    k_extinction:
        Beer-Lambert extinction coefficient (dimensionless). Default = 0.45.
    grain_partition_fraction:
        Fraction of daily biomass added to grain_biomass_g during grain_fill.
        Default = 0.35 (35 % harvest-index rate per day during fill).
    lai_slope:
        LAI increment per °C·day above base temp (m²/m²/°C·day). Default = 0.002.
    lai_senescence_rate:
        Daily LAI loss fraction during grain_fill and maturity. Default = 0.015.
    k_height:
        Height allometry constant (cm / √g). Default = 0.8.
    sowing_doy:
        Day-of-year for HD-2967 Rabi sowing. Informational; used by default_crop().

    Examples
    --------
    >>> from cropforge.plugins import StandardWheat
    >>> plugin = StandardWheat(rue=1.30)
    >>> plugin.rue
    1.3
    """

    species = "Triticum aestivum"

    def __init__(
        self,
        base_temp_c: float = 0.0,
        rue: float = 1.24,
        k_extinction: float = 0.45,
        grain_partition_fraction: float = 0.35,
        lai_slope: float = 0.002,
        lai_senescence_rate: float = 0.015,
        k_height: float = 0.8,
        sowing_doy: int = 320,
    ):
        self.base_temp_c = base_temp_c
        self.rue = rue
        self.k_extinction = k_extinction
        self.grain_partition_fraction = grain_partition_fraction
        self.lai_slope = lai_slope
        self.lai_senescence_rate = lai_senescence_rate
        self.k_height = k_height
        self.sowing_doy = sowing_doy

    # ------------------------------------------------------------------
    # CropPlugin interface
    # ------------------------------------------------------------------

    def step(self, state, env):
        """Daily wheat growth step — phenology → biomass → grain partition."""
        for plant in state.plants:
            if not plant.alive:
                continue
            self._update_phenology(plant, env)
            self._update_lai(plant, env)
            self._accumulate_biomass(plant, env)
            self._partition_grain(plant)
            self._update_height(plant)
        return state

    # ------------------------------------------------------------------
    # Internal sub-models
    # ------------------------------------------------------------------

    def _update_phenology(self, plant, env) -> None:
        """Accumulate thermal time and set phenological_stage."""
        tt_today = max(0.0, env.temp_mean_c - self.base_temp_c)
        plant.custom["thermal_time"] = plant.custom.get("thermal_time", 0.0) + tt_today
        stage = _get_stage(plant.custom["thermal_time"])
        plant.custom["phenological_stage"] = stage
        # Keep PlantState fields in sync for the visualiser and buffer
        plant.phenological_stage = stage
        plant.stage_progress = _get_stage_progress(plant.custom["thermal_time"], stage)

    def _update_lai(self, plant, env) -> None:
        """LAI development driven by thermal time; senescence during fill."""
        stage = plant.custom.get("phenological_stage", "germination")
        tt_today = max(0.0, env.temp_mean_c - self.base_temp_c)

        if stage in ("germination", "emergence"):
            # Very slow canopy development before emergence
            plant.lai = max(0.0, plant.lai + tt_today * self.lai_slope * 0.3)
        elif stage in ("tillering", "stem_extension", "anthesis"):
            plant.lai = max(0.0, plant.lai + tt_today * self.lai_slope)
        elif stage in ("grain_fill", "maturity"):
            # Senescence: LAI declines daily
            plant.lai = max(0.0, plant.lai * (1.0 - self.lai_senescence_rate))

    def _accumulate_biomass(self, plant, env) -> None:
        """RUE × intercepted PAR × water stress → daily biomass increment."""
        ks = plant.custom.get("water_stress_ks", 1.0)
        kn = plant.custom.get("n_stress_kn", 1.0)

        intercepted_par = plant.custom.get("intercepted_par_mj")
        if intercepted_par is None:
            intercepted_par = (
                env.radiation_mj_m2
                * 0.5
                * (1.0 - math.exp(-self.k_extinction * max(0.0, plant.lai)))
            )
        delta_biomass = self.rue * intercepted_par * ks * kn
        plant.biomass_g += max(0.0, delta_biomass)

    def _partition_grain(self, plant) -> None:
        """Partition a fraction of today's biomass increment into grain."""
        stage = plant.custom.get("phenological_stage", "germination")
        if stage == "grain_fill":
            # Use today's biomass increment as basis
            # We track yesterday's biomass in extra to compute delta
            prev = plant.custom.get("_prev_biomass_g", plant.biomass_g)
            delta = max(0.0, plant.biomass_g - prev)
            plant.custom["grain_biomass_g"] = (
                plant.custom.get("grain_biomass_g", 0.0)
                + delta * self.grain_partition_fraction
            )
        plant.custom["_prev_biomass_g"] = plant.biomass_g

    def _update_height(self, plant) -> None:
        plant.height_cm = self.k_height * math.sqrt(max(0.0, plant.biomass_g))

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def default_crop(cls):
        """Return a Crop pre-configured for HD-2967 Rabi wheat."""
        from cropforge import Crop
        return Crop(species=cls.species, variety="HD-2967", sowing_doy=320)
