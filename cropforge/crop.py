"""
cropforge/crop.py
=================
Crop metadata class.

The ``Crop`` object carries species identity, variety, and sowing calendar.
It is **not** a model — it holds no physiological equations. The researcher's
``@farm.step`` functions are the model.  ``Crop`` is attached to a ``Field``
via ``Field.set_crop()`` and made available to step functions through the
field's state so they can make phenology decisions (e.g. days since sowing).

PRD Reference: Section 6.1 — Basic Usage Example.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Crop:
    """Metadata describing a crop planted in a field.

    Parameters
    ----------
    species:
        Common species name, e.g. ``"wheat"``, ``"maize"``, ``"chickpea"``.
        CropForge imposes no controlled vocabulary — the string is passed
        through to the state log and is entirely researcher-defined.
    variety:
        Cultivar or variety identifier, e.g. ``"HD-2967"``, ``"custom"``.
        Defaults to ``"generic"`` when the researcher does not distinguish
        varieties.
    sowing_doy:
        Calendar day-of-year (1–366) on which sowing occurs.  The runtime
        engine uses this to determine when to set ``PlantState.age_days = 0``
        for each plant in the field.  Plants are placed in the field on the
        first simulation day whose ``EnvironmentState.doy`` matches or
        exceeds ``sowing_doy``.
    custom:
        Arbitrary researcher-defined crop metadata (e.g. base temperature,
        photoperiod sensitivity).  Not used by the engine directly; accessible
        inside step functions via ``field.crop.custom``.

    Examples
    --------
    >>> crop = Crop(species="wheat", variety="HD-2967", sowing_doy=290)
    >>> crop.species
    'wheat'
    >>> crop.sowing_doy
    290
    """

    species: str
    variety: str = "generic"
    sowing_doy: int = 1

    # Extensibility hook — keeps the same pattern as SimulationState objects
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.species, str) or not self.species.strip():
            raise ValueError("Crop.species must be a non-empty string.")
        if not (1 <= self.sowing_doy <= 366):
            raise ValueError(
                f"Crop.sowing_doy must be between 1 and 366, got {self.sowing_doy}."
            )

    def __repr__(self) -> str:
        return (
            f"Crop(species={self.species!r}, variety={self.variety!r}, "
            f"sowing_doy={self.sowing_doy})"
        )
