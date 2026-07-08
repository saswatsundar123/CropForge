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

from importlib.resources import files as _resource_files
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
    def register(
        cls,
        crop: Optional[str] = None,
        stage: Optional[int] = None,
        uri: Optional[str] = None,
        *,
        species: Optional[str] = None,
        gltf_path: Optional[str] = None,
    ) -> None:
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
        species, gltf_path:
            Public aliases for ``crop`` and ``uri``. These match the README
            and PRD examples while preserving older AssetRegistry calls.

        Example
        -------
        >>> AssetRegistry.register("StandardWheat", stage=4,
        ...     uri="assets/wheat_anthesis.gltf")
        """
        crop_key = species if species is not None else crop
        model_uri = gltf_path if gltf_path is not None else uri
        if not crop_key:
            raise ValueError("ModelRegistry.register() requires crop or species.")
        if stage is None:
            raise ValueError("ModelRegistry.register() requires stage.")
        if not model_uri:
            raise ValueError("ModelRegistry.register() requires uri or gltf_path.")
        cls._registry.setdefault(str(crop_key), {})[int(stage)] = str(model_uri)

    @classmethod
    def get_model_path(
        cls,
        crop: Optional[str] = None,
        stage: int = 0,
        *,
        species: Optional[str] = None,
    ) -> Optional[str]:
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
        crop_key = species if species is not None else crop
        if not crop_key:
            return None
        return cls._registry.get(str(crop_key), {}).get(int(stage))

    @classmethod
    def list_registered(cls) -> dict[str, list[int]]:
        """Return all registered crops and their available stage indices."""
        return {crop: sorted(stages.keys()) for crop, stages in cls._registry.items()}

    @classmethod
    def clear(cls) -> None:
        """Reset registry — useful in tests to avoid cross-test state leakage."""
        cls._registry.clear()


# ---------------------------------------------------------------------------
# First-party bundle boot loader (PRD v0.9.5 §4.5)
# ---------------------------------------------------------------------------


# stage index → filename stem (same order as _STAGE_ORDER in each plugin)
_WHEAT_FILENAMES = [
    "stage_0_germination",
    "stage_1_emergence",
    "stage_2_tillering",
    "stage_3_stem_ext",
    "stage_4_anthesis",
    "stage_5_grain_fill",
    "stage_6_senescence",
]
_MAIZE_FILENAMES = [
    "stage_0_germination",
    "stage_1_emergence",
    "stage_2_veg_early",
    "stage_3_veg_late",
    "stage_4_anthesis",
    "stage_5_grain_fill",
    "stage_6_senescence",
]


def _register_bundle(crop_key: str, subdir: str, filenames: list) -> None:
    """Register all stage GLTF files for *crop_key* from *subdir*.

    Silently skips missing files — cylinder fallback stays active.
    """
    try:
        bundle_dir = _resource_files("cropforge").joinpath("viz", "assets", subdir)
        for stage_idx, stem in enumerate(filenames):
            path = bundle_dir.joinpath(f"{stem}.gltf")
            if path.is_file():
                AssetRegistry.register(crop_key, stage_idx, str(path))
    except Exception:
        return


def initialize_first_party_bundles() -> None:
    """Register all built-in crop stage GLTF models.

    Called automatically on import of StandardWheat / StandardMaize.
    Researchers never need to call this manually.
    """
    _register_bundle("StandardWheat", "standard_wheat", _WHEAT_FILENAMES)
    _register_bundle("StandardMaize", "standard_maize", _MAIZE_FILENAMES)


# ---------------------------------------------------------------------------
# PRD v0.9.5 §2.3 compat alias — `from cropforge.models import ModelRegistry` works
# ---------------------------------------------------------------------------
ModelRegistry = AssetRegistry
