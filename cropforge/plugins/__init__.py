"""
cropforge/plugins/__init__.py
==============================
CropForge plugin system — v0.5.0.

This package merges two concerns:
1. The v0.4.0 plugin API (CropPlugin base class, registry, register_crop, etc.)
   — imported from cropforge._plugins_base so existing code is unaffected.
2. First-party official crop plugins (StandardWheat, StandardMaize) — new in v0.5.0.

All existing imports remain valid::

    from cropforge.plugins import CropPlugin, register_crop
    from cropforge.plugins import StandardWheat, StandardMaize

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

# Re-export the full v0.4.0 plugin API so cropforge.__init__.py continues to
# import CropPlugin, register_crop, etc. from cropforge.plugins unchanged.
from cropforge._plugins_base import (
    CropPlugin,
    CropForgePluginError,
    register_crop,
    get_plugin,
    list_plugins,
    _REGISTRY,
)

# First-party official crop plugins (v0.5.0)
from cropforge.plugins.wheat import StandardWheat
from cropforge.plugins.maize import StandardMaize

__all__ = [
    # v0.4.0 plugin API (unchanged)
    "CropPlugin",
    "CropForgePluginError",
    "register_crop",
    "get_plugin",
    "list_plugins",
    "_REGISTRY",
    # v0.5.0 official plugins
    "StandardWheat",
    "StandardMaize",
]
