# Quickstart

This example runs a tiny wheat simulation without external data files. It is
the fastest way to confirm that CropForge is installed and that the dashboard
can read a simulation log.

```python
from cropforge import Crop, Farm, Field
from cropforge.plugins import StandardWheat
from cropforge.state import EnvironmentState


class WarmWeather:
    def get_day(self, day: int):
        return EnvironmentState(
            day=day,
            doy=day,
            temp_max_c=30.0,
            temp_min_c=20.0,
            temp_mean_c=25.0,
            radiation_mj_m2=20.0,
            rainfall_mm=0.0,
            et0_mm=4.0,
            wind_speed_ms=1.0,
            humidity_pct=60.0,
        )


farm = Farm("Quickstart")
field = Field("Plot", rows=10, cols=10)
field.set_crop(Crop(species="wheat"), sowing_density_plants_per_m2=250.0)
field.set_weather(WarmWeather())
field.use_plugin(StandardWheat)
farm.add_field(field)

farm.use_physics(radiation=True)
farm.run(days=30)

print(farm.yield_summary())
farm.visualize()
```

## What To Try Next

- Add weed competition with `farm.use_physics(weed_pressure=True)` and
  `farm.set_weed_params(...)`.
- Change planting density with `field.set_crop(...,
  sowing_density_plants_per_m2=...)` or `field.set_planting_config(...)`.
- Schedule irrigation with `Event.irrigation(...)`; enhanced dashboard mode
  animates sprinkler water particles.
- Use `farm.visualize(quality="enhanced")` for PBR lighting, rain particles,
  disease stress coloration, machinery animation, and first-party crop assets.

## Where Output Goes

Each run writes a Parquet session under `cropforge_output/`. The dashboard,
comparison tools, and GLB exporter read from those logs.
