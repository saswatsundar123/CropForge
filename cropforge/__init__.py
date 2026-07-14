"""
CropForge — Virtual Farm Runtime for Agricultural Researchers.

Open-source, code-first simulation engine. Researchers write the model;
CropForge faithfully executes it and visualises the result.

Public API (PRD Section 6.1):
    from cropforge import Farm, Field, Crop, Event, Terrain, LandPrep

Maintainer : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence    : MIT
"""

__version__ = "1.0.0"

# Core public API — matches `from cropforge import Farm, Field, Crop, Weather, Soil, Event, ...`
from cropforge.crop import Crop
from cropforge.farm import Farm, Field
from cropforge.loaders import Soil, Weather
from cropforge.events import Event, CropForgeEventError
from cropforge.runtime import CropForgeStepError, CropForgeVisualizeError, CropForgeConfigError, CropForgeStateError
from cropforge._plugins_base import (
    CropPlugin,
    CropForgePluginError,
    register_crop,
    get_plugin,
    list_plugins,
)
from cropforge.compare import compare
from cropforge.terrain import Terrain
from cropforge.land_prep import (
    LandPrep,
    RidgeFurrow,
    ContourBund,
    Terrace,
    ZeroTillage,
    ConventionalTill,
    TiedRidges,
    VegetativeFilterStrip,
    Mulching,
    BroadBedFurrow,
)
from cropforge.viz.registry import AssetRegistry
from cropforge.models import ModelRegistry  # compat alias (PRD v0.9.5 §2.3)

__all__ = [
    "Farm",
    "Field",
    "Crop",
    "Weather",
    "Soil",
    "Event",
    "CropPlugin",
    "CropForgeEventError",
    "CropForgePluginError",
    "CropForgeConfigError",
    "CropForgeStepError",
    "CropForgeVisualizeError",
    "CropForgeStateError",
    "register_crop",
    "get_plugin",
    "list_plugins",
    "compare",
    "Terrain",
    "LandPrep",
    "RidgeFurrow",
    "ContourBund",
    "Terrace",
    "ZeroTillage",
    "ConventionalTill",
    "TiedRidges",
    "VegetativeFilterStrip",
    "Mulching",
    "BroadBedFurrow",
    "AssetRegistry",
    "ModelRegistry",
    "__version__",
]
