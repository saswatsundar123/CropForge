# Geomorphology Tutorial

> **v0.8.0** — How CropForge models erosion, sediment transport, and terrain feedback together.

The full runnable script is at `examples/geomorphology_tutorial.py`.

---

## What this covers

Three v0.8.0 subsystems working in sequence each day:

1. **Sub-metre resolution** — `resolution_m` scales all spatial fluxes (runoff, LS factor, D8 routing) to physical cell size.
2. **Sediment transport** — eroded soil is routed downslope via D8 flow and deposited in accumulation zones. Mass is conserved to floating-point precision.
3. **Geomorphological feedback** — the elevation grid updates daily from net sediment flux; slope and aspect recompute for the next timestep.

---

## Setup

```python
from cropforge import Farm, Field, Terrain, TiedRidges, ZeroTillage
from cropforge.plugins import StandardMaize

# 50×50 field at 0.5 m resolution (25 m × 25 m)
terrain = Terrain.from_array(elev, resolution_m=0.5)

field = Field(name="TiedRidges_Field", rows=50, cols=50)
field.set_terrain(terrain)
field.set_land_prep(TiedRidges(
    ridge_height_m=0.20, ridge_spacing_m=2.0,
    tie_spacing_m=4.0,   tie_height_m=0.10,
))
field.use_plugin(StandardMaize)
```

## Enable sediment physics

```python
farm.use_physics(
    et0=True,
    soil_water_balance=True,
    clod_dynamics=True,
    erosion=True,
    sediment_transport=True,   # routes eroded mass D8 downslope
)
farm.run(days=30)
```

## Reading the sediment budget

After the run, each `SoilState` cell (layer 0) carries:

| Field | Unit | Meaning |
|---|---|---|
| `cumulative_sediment_loss_kg_m2` | kg/m² | Total eroded and exported |
| `cumulative_deposition_kg_m2` | kg/m² | Total received from upslope |
| `sediment_flux_kg_m2` | kg/m² | Sediment passing through today |
| `sediment_deposited_kg_m2` | kg/m² | Deposited today |

```python
for r in range(rows):
    for c in range(cols):
        cell = field_state.soil[r][c][0]
        net = cell.cumulative_sediment_loss_kg_m2 - cell.cumulative_deposition_kg_m2
        # net > 0  → net erosion zone
        # net < 0  → net deposition zone (fan, channel bed)
```

## Expected output (TiedRidges vs. ZeroTillage)

Running the tutorial script prints a comparison like:

```
TiedRidges (treatment)
  Cells with net erosion    : 312 / 2500
  Cells with net deposition : 188 / 2500
  Total sediment loss       : 0.0841 kg

ZeroTillage (control)
  Cells with net erosion    : 478 / 2500
  Cells with net deposition : 188 / 2500
  Total sediment loss       : 0.1563 kg

TiedRidges reduced total sediment export by 46.2%
```

The tie-dams intercept D8 flow paths and create micro-deposition zones inside furrows, reducing net field export.

---

## See also

- `examples/conservation_ag_trial.py` — 60-day dual-field comparison with full physics
- `examples/submetre_performance_trial.py` — 500×500 LOD rendering stress test
- [API Reference — Sediment Transport](../reference/api.md#erosion--sediment-transport-cropforgephysicssoilpy)
