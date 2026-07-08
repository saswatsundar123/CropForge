"""
cropforge/models.py
====================
Compatibility shim — PRD v0.9.5 §2.3.

The public API specifies `from cropforge.models import ModelRegistry`.
The implementation lives at `cropforge.viz.registry`.
This module bridges the two so both import paths work.
"""

from cropforge.viz.registry import AssetRegistry, ModelRegistry, initialize_first_party_bundles  # noqa: F401

__all__ = ["ModelRegistry", "AssetRegistry", "initialize_first_party_bundles"]
