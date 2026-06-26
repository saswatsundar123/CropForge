# Crop

`cropforge.Crop` — defines the genetic identity of a crop for a simulation.

```python
from cropforge import Crop

maize = Crop(species="maize", variety="Kharif-1", sowing_doy=150)
```

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `species` | `str` | Crop species name (e.g. `"wheat"`, `"maize"`, `"rice"`). |
| `variety` | `str` | Variety or cultivar name (e.g. `"HD-2967"`). |
| `sowing_doy` | `int` | Day of year on which the crop is sown (1–365). |

## Notes

`Crop` is a plain dataclass. It carries genetic identity metadata only. All growth model logic is written by the researcher in `@farm.step` functions. CropForge does not interpret the species or variety strings — they are passed through to the Parquet log for metadata traceability.
