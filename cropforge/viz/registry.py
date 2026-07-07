"""
cropforge/viz/registry.py
=========================
Asset registry mapping (species, stage_index) → GLTF model URI.

Two distribution patterns are supported (PRD v0.9.0 §8):
  Option C — standalone model package (e.g. cropforge-models-wheat)
  Option B — bundled inside an agronomic plugin package

Both call ``AssetRegistry.register()`` at import time. The renderer reads
``AssetRegistry.get_model_path()`` during the buffer build phase.

Cylinder fallback: if no model is registered for a (species, stage), the
frontend falls back to the existing THREE.CylinderGeometry automatically.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

from typing import Optional


class AssetRegistry:
    """Central registry for crop-stage GLTF model paths (PRD v0.9.0 §8.2).

    All methods are class-methods — the registry is module-level global state,
    shared across all Farm instances in a session. This is intentional: model
    packages register once at import time.

    Example
    -------
    >>> AssetRegistry.register("StandardWheat", stage=4, uri="assets/wheat_anthesis.gltf")
    >>> AssetRegistry.get_model_path("StandardWheat", stage=4)
    'assets/wheat_anthesis.gltf'
    >>> AssetRegistry.get_model_path("StandardMaize", stage=4)  # not registered
    """

    # ponytail: plain dict, no class hierarchy. Add TTL/reload if needed later.
    _registry: dict[str, dict[int, str]] = {}

    @classmethod
    def register(cls, crop: str, stage: int, uri: str) -> None:
        """Register a GLTF model URI for a crop species at a growth stage index.

        Parameters
        ----------
        crop:
            Crop / species name matching ``PlantState.custom['crop_name']``
            or the plugin class name (e.g. ``"StandardWheat"``).
        stage:
            Stage index (0–6) as defined in ``STAGE_INDEX`` mapping.
        uri:
            Path or URI to a GLTF/GLB file. May be relative or absolute.
            For bundled models use ``str(Path(__file__).parent / "models/x.gltf")``.

        Example
        -------
        >>> AssetRegistry.register("StandardWheat", stage=4,
        ...     uri="assets/wheat_anthesis.gltf")
        """
        cls._registry.setdefault(crop, {})[stage] = uri

    @classmethod
    def get_model_path(cls, crop: str, stage: int) -> Optional[str]:
        """Return the registered GLTF URI, or None if not registered.

        None is the cylinder-fallback trigger — the JS renderer uses
        ``THREE.CylinderGeometry`` for any plant whose model_id is empty.

        Parameters
        ----------
        crop:
            Crop name (same as used in ``register()``).
        stage:
            Stage index (0–6).
        """
        return cls._registry.get(crop, {}).get(stage)

    @classmethod
    def list_registered(cls) -> dict[str, list[int]]:
        """Return all registered crops and their available stage indices."""
        return {crop: sorted(stages.keys()) for crop, stages in cls._registry.items()}

    @classmethod
    def clear(cls) -> None:
        """Reset registry — useful in tests to avoid cross-test state leakage."""
        cls._registry.clear()
