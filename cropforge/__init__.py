"""
CropForge — Virtual Farm Runtime for Agricultural Researchers.

Open-source, code-first simulation engine. Researchers write the model;
CropForge faithfully executes it and visualises the result.

Public API (PRD Section 6.1):
    from cropforge import Farm, Field, Crop

Maintainer : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence    : MIT
"""

__version__ = "0.1.0"

# Core public API — matches `from cropforge import Farm, Field, Crop, Weather, Soil, ...`
from cropforge.crop import Crop
from cropforge.farm import Farm, Field
from cropforge.loaders import Soil, Weather
from cropforge.runtime import CropForgeStepError, CropForgeVisualizeError

__all__ = [
    "Farm",
    "Field",
    "Crop",
    "Weather",
    "Soil",
    "CropForgeStepError",
    "CropForgeVisualizeError",
    "__version__",
]
