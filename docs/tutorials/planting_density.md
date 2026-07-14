# Planting Density And Yield Summary

CropForge represents one plant per grid cell internally. Planting density tells
the yield summary how many real plants that representative plant stands for.

```python
field.set_crop(
    Crop(species="wheat"),
    sowing_density_plants_per_m2=250.0,
)
```

You can also configure planting geometry directly:

```python
field.set_planting_config(
    pattern="rows",
    row_spacing_m=0.20,
    plant_spacing_m=0.08,
)
```

After the run:

```python
farm.run(days=90)
summary = farm.yield_summary()
print(summary["yield_kg_per_ha"])
print(summary["total_yield_kg"])
```

## Scaling

`yield_summary()` uses the physical grid resolution:

```text
cell_area_m2 = resolution_m ** 2
cell_yield_g = representative_yield_g * plants_per_m2 * cell_area_m2
total_yield_kg = sum(cell_yield_g) / 1000
yield_kg_per_ha = total_yield_kg / (field_area_m2 / 10000)
```

If grain biomass exists, grain is used. Otherwise CropForge falls back to total
biomass, so custom crop plugins can still produce a useful summary.
