"""
examples/digital_twin_full_lifecycle.py
=======================================
CropForge v0.9.5 capstone: full digital-twin lifecycle showcase.

This script exercises the complete visual architecture arc:
procedural terrain, TiedRidges land preparation, StandardWheat first-party
assets, full opt-in physics, machinery path logging, disease stress visuals,
rain particle metadata, GLB export, and the enhanced WebGL dashboard.

Run:
    python examples/digital_twin_full_lifecycle.py
"""
from __future__ import annotations

import os

from cropforge import Event, Farm, Field, Terrain, TiedRidges
from cropforge.plugins import StandardWheat
from cropforge.state import EnvironmentState


class LifecycleWeather:
    """Synthetic 90-day season with a heavy storm on day 15."""

    def get_day(self, day: int) -> EnvironmentState:
        rain = 28.0 if day == 15 else (4.0 if day in {22, 41, 63} else 0.0)
        return EnvironmentState(
            day=day,
            doy=day,
            temp_max_c=29.0,
            temp_min_c=17.0,
            temp_mean_c=23.0,
            radiation_mj_m2=21.0,
            rainfall_mm=rain,
            et0_mm=4.8,
            wind_speed_ms=3.2,
            humidity_pct=78.0 if rain > 0.0 else 58.0,
        )


def build_farm() -> Farm:
    rows, cols = 18, 24

    farm = Farm("CropForge_v095_Digital_Twin", location=(23.4, 85.3))
    field = Field("Lifecycle_Plot", rows=rows, cols=cols, area_ha=0.6)

    terrain = Terrain.procedural(
        rows=rows,
        cols=cols,
        generator="undulating",
        resolution_m=1.0,
        seed=95,
        amplitude_m=1.2,
        frequency=0.045,
        base_elevation_m=410.0,
    )
    field.set_terrain(terrain)
    field.set_land_prep(TiedRidges(
        ridge_height_m=0.22,
        ridge_spacing_m=1.5,
        tie_spacing_m=4.0,
        tie_height_m=0.12,
    ))
    field.set_weather(LifecycleWeather())

    # StandardWheat auto-registers bundled first-party stage assets.
    field.use_plugin(StandardWheat)
    farm.add_field(field)

    farm.use_physics(
        et0=True,
        root_impedance=True,
        water_balance=True,
        nutrients=True,
        lateral_flow=True,
        radiation=True,
        slope_radiation_correction=True,
        clod_dynamics=True,
        erosion=True,
        sediment_transport=True,
        terrain_wind=True,
        wind_direction_deg=270.0,
        disease=True,
        disease_foci=[(rows // 2, cols // 2)],
        disease_spread_rate=0.18,
        disease_latency_days=3,
        disease_stress_increment=0.035,
        disease_wind_direction_deg=270.0,
        disease_anisotropy=0.80,
        disease_seed=95,
    )

    farm.add_event(Event.tillage(field="Lifecycle_Plot", day=2))
    farm.add_event(Event.harvest(field="Lifecycle_Plot", day=90))
    return farm


def main() -> Farm:
    farm = build_farm()
    farm.run(days=90)
    farm.export_scene(day=45, filepath="mid_season_export.glb")

    if os.environ.get("CROPFORGE_SKIP_DASHBOARD") == "1":
        print("Dashboard launch skipped by CROPFORGE_SKIP_DASHBOARD=1.")
        return farm

    farm.visualize(quality="enhanced")
    return farm


if __name__ == "__main__":
    main()
