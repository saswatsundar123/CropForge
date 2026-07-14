# Weed Management

Weed competition is opt-in. A standard simulation remains unchanged until
`weed_pressure=True` is enabled.

```python
farm.set_weed_params(
    species="generic_grass",
    initial_density_m2=0.10,
    emergence_doy=1,
    spread_rate=0.05,
    competitive_index=1.0,
)

farm.use_physics(
    radiation=True,
    weed_pressure=True,
    weed_seed=7,
)
```

The weed engine:

- emerges after `emergence_doy`;
- spreads to neighbouring empty cells according to `spread_rate`;
- reduces topsoil moisture in occupied cells;
- suppresses intercepted crop radiation through `weed_radiation_suppression`;
- logs occupied weeds to `weed_states.parquet`;
- exposes `weed_lai` and `weed_density_m2` for dashboard overlays.

Use weed-free and weed-enabled runs side by side when estimating yield loss.
The disabled path is designed to preserve existing outputs.
