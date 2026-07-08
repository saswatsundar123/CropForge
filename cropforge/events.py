"""
cropforge/events.py
===================
Farm Event System — discrete management actions scheduled by day or interval.

Events fire at the END of each simulation day, after all physics hooks and
all @farm.step functions have executed. State modifications are visible to
model logic from day+1 onwards.

Execution order per day (PRD v0.3.0 Section 4.3):
    phase=-2   ET0 physics hook
    phase=-1   Root impedance hook
    phase=0+   Researcher @farm.step functions
    END OF DAY Events fire here
    LOGGER     Records completed timestep

Built-in event types:
    Event.irrigation(field, interval_days, amount_mm, start_day, end_day)
    Event.fertiliser(field, day=None, days=None, n_kg_ha, apply_to_layer)
    Event.custom(field, day)   — decorator factory for arbitrary functions

PRD References:
    Section 4 — Farm Event System (full specification)
    Section 4.2 — API Design
    Section 4.3 — Execution Contract
    Section 4.5 — Tests Required

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

if TYPE_CHECKING:
    from cropforge.state import EnvironmentState, FieldState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception (PRD Section 4.3)
# ---------------------------------------------------------------------------

class CropForgeEventError(ValueError):
    """Raised when an Event is configured with invalid parameters.

    Raised at ``farm.run()`` validation time, not at event registration time.
    This matches the PRD contract: researchers see the error before any
    simulation work is wasted.

    Examples
    --------
    >>> Event.irrigation(field="Plot_A", interval_days=0, amount_mm=30)
    # Raises CropForgeEventError at farm.run() with a clear message.
    """


# ---------------------------------------------------------------------------
# Base Event class (internal)
# ---------------------------------------------------------------------------

class _BaseEvent:
    """Common interface required by the runtime's ``_fire_events()`` function.

    Subclasses must implement:
        should_fire(field_name: str, day: int) -> bool
        apply(field_state: FieldState, env_state: EnvironmentState, day: int) -> None
        validate() -> None   (raises CropForgeEventError if misconfigured)

    The ``name`` attribute is written to ``FieldState.events_fired`` and
    to the Parquet ``events_fired`` JSON column.
    """

    name: str = "base_event"
    field: str = ""

    def should_fire(self, field_name: str, day: int) -> bool:  # pragma: no cover
        raise NotImplementedError

    def apply(
        self,
        field_state: "FieldState",
        env_state: "EnvironmentState",
        day: int,
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def validate(self) -> None:
        """Called at farm.run() time. Raise CropForgeEventError if invalid."""


# ---------------------------------------------------------------------------
# IrrigationEvent
# ---------------------------------------------------------------------------

class _IrrigationEvent(_BaseEvent):
    """Interval-based irrigation. Adds water to layer 0 of the target field.

    Fires on days: start_day, start_day + interval_days, start_day + 2*interval, …
    Stops on the first day > end_day.

    Water is added to ``SoilVoxelState.moisture_pct`` for all soil voxels
    in layer 0. Moisture is capped at the voxel's ``saturation_pct``
    (stored in ``SoilVoxelState.custom.get('saturation_pct', 100.0)``).

    Parameters
    ----------
    field:
        Name of the target field. Must exactly match ``Field.name``.
    interval_days:
        Days between irrigations. Must be >= 1 (validated at run time).
    amount_mm:
        Water depth added per irrigation event (mm).
        Internally converted to a fractional increase in ``moisture_pct``
        using a simple 1 mm ≈ 0.1 pct conversion (placeholder until the
        full water balance in v0.3.0 provides volumetric capacity data).
    start_day:
        First day on which the event fires. Default 1.
    end_day:
        Last day (inclusive) on which the event may fire.
        ``None`` means no upper bound (fires until end of simulation).
    """

    name = "irrigation"

    def __init__(
        self,
        field: str,
        interval_days: int,
        amount_mm: float,
        start_day: int = 1,
        end_day: Optional[int] = None,
    ) -> None:
        self.field = field
        self.interval_days = interval_days
        self.amount_mm = amount_mm
        self.start_day = start_day
        self.end_day = end_day

    def validate(self) -> None:
        if self.interval_days < 1:
            raise CropForgeEventError(
                f"Event.irrigation(field={self.field!r}) has interval_days="
                f"{self.interval_days}. interval_days must be >= 1."
            )
        if self.amount_mm < 0:
            raise CropForgeEventError(
                f"Event.irrigation(field={self.field!r}) has amount_mm="
                f"{self.amount_mm}. amount_mm must be >= 0."
            )

    def should_fire(self, field_name: str, day: int) -> bool:
        if field_name != self.field:
            return False
        if day < self.start_day:
            return False
        if self.end_day is not None and day > self.end_day:
            return False
        return (day - self.start_day) % self.interval_days == 0

    def apply(
        self,
        field_state: "FieldState",
        env_state: "EnvironmentState",
        day: int,
    ) -> None:
        # Convert mm to moisture_pct increase.
        # Placeholder conversion: 1 mm ≈ 0.1 % volumetric.
        # The full soil water balance (v0.3.0 Phase 2) will replace this.
        moisture_increase = self.amount_mm * 0.1

        n_cells = 0
        for row_soils in field_state.soil:
            for cell_soils in row_soils:
                if not cell_soils:
                    continue
                voxel = cell_soils[0]  # layer 0 only
                sat = voxel.custom.get("saturation_pct", 100.0)
                voxel.moisture_pct = min(sat, voxel.moisture_pct + moisture_increase)
                n_cells += 1

        logger.debug(
            "IrrigationEvent: day=%d field=%r +%.1f mm → %d cells layer-0 moisture updated.",
            day, self.field, self.amount_mm, n_cells,
        )

    def __repr__(self) -> str:
        return (
            f"IrrigationEvent(field={self.field!r}, interval_days={self.interval_days}, "
            f"amount_mm={self.amount_mm}, start_day={self.start_day}, "
            f"end_day={self.end_day})"
        )


# ---------------------------------------------------------------------------
# FertiliserEvent
# ---------------------------------------------------------------------------

class _FertiliserEvent(_BaseEvent):
    """One-time or multi-day nitrogen fertiliser application.

    Adds nitrogen to a specified soil layer in all grid cells of the
    target field on the configured day(s).

    Parameters
    ----------
    field:
        Name of the target field.
    fire_days:
        Sorted list of simulation days on which this event fires.
    n_kg_ha:
        Nitrogen amount to add (kg ha⁻¹). Applied uniformly across
        all grid cells in the target layer.
    apply_to_layer:
        Layer index (0 = topsoil) that receives the nitrogen.
    """

    name = "fertiliser"

    def __init__(
        self,
        field: str,
        fire_days: List[int],
        n_kg_ha: float,
        apply_to_layer: int = 0,
    ) -> None:
        self.field = field
        self.fire_days = sorted(fire_days)
        self.n_kg_ha = n_kg_ha
        self.apply_to_layer = apply_to_layer

    def validate(self) -> None:
        if self.n_kg_ha < 0:
            raise CropForgeEventError(
                f"Event.fertiliser(field={self.field!r}) has n_kg_ha="
                f"{self.n_kg_ha}. n_kg_ha must be >= 0."
            )
        if self.apply_to_layer < 0:
            raise CropForgeEventError(
                f"Event.fertiliser(field={self.field!r}) has apply_to_layer="
                f"{self.apply_to_layer}. Layer index must be >= 0."
            )
        if not self.fire_days:
            raise CropForgeEventError(
                f"Event.fertiliser(field={self.field!r}) has no fire_days configured."
            )

    def should_fire(self, field_name: str, day: int) -> bool:
        return field_name == self.field and day in self.fire_days

    def apply(
        self,
        field_state: "FieldState",
        env_state: "EnvironmentState",
        day: int,
    ) -> None:
        n_cells = 0
        for row_soils in field_state.soil:
            for cell_soils in row_soils:
                if self.apply_to_layer >= len(cell_soils):
                    continue
                cell_soils[self.apply_to_layer].nitrogen_kg_ha += self.n_kg_ha
                n_cells += 1

        logger.debug(
            "FertiliserEvent: day=%d field=%r +%.1f kg/ha N → %d cells layer-%d updated.",
            day, self.field, self.n_kg_ha, n_cells, self.apply_to_layer,
        )

    def __repr__(self) -> str:
        return (
            f"FertiliserEvent(field={self.field!r}, days={self.fire_days}, "
            f"n_kg_ha={self.n_kg_ha}, layer={self.apply_to_layer})"
        )


# ---------------------------------------------------------------------------
# Machinery path events
# ---------------------------------------------------------------------------

def _boustrophedon_path(field_state: "FieldState") -> list[list[float]]:
    """Return a simple row-by-row machinery path inside field bounds."""
    rows = len(field_state.soil)
    cols = len(field_state.soil[0]) if rows else 0
    if rows <= 0 or cols <= 0:
        return []

    path: list[list[float]] = []
    max_col = max(0, cols - 1)
    for row in range(rows):
        if row % 2 == 0:
            path.append([0.0, float(row)])
            path.append([float(max_col), float(row)])
        else:
            path.append([float(max_col), float(row)])
            path.append([0.0, float(row)])
    return path


class _MachineryPathEvent(_BaseEvent):
    """Base class for visual machinery events that log frontend paths."""

    name = "machinery"
    machine_type = "machine"

    def __init__(self, field: str, day: int) -> None:
        self.field = field
        self.fire_day = day

    def validate(self) -> None:
        if self.fire_day < 1:
            raise CropForgeEventError(
                f"{self.__class__.__name__}(field={self.field!r}) has day="
                f"{self.fire_day}. day must be >= 1."
            )

    def should_fire(self, field_name: str, day: int) -> bool:
        return field_name == self.field and day == self.fire_day

    def apply(
        self,
        field_state: "FieldState",
        env_state: "EnvironmentState",
        day: int,
    ) -> None:
        path = _boustrophedon_path(field_state)
        event = {
            "day": day,
            "field_name": self.field,
            "event_name": self.name,
            "machine_type": self.machine_type,
            "path": path,
        }
        field_state.custom.setdefault("machinery_events", []).append(event)
        logger.debug(
            "%s: day=%d field=%r machine=%s waypoints=%d",
            self.__class__.__name__, day, self.field, self.machine_type, len(path),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(field={self.field!r}, day={self.fire_day}, "
            f"machine_type={self.machine_type!r})"
        )


class _TillageEvent(_MachineryPathEvent):
    name = "tillage"
    machine_type = "tractor"


class _HarvestEvent(_MachineryPathEvent):
    name = "harvest"
    machine_type = "harvester"


class _SprayEvent(_MachineryPathEvent):
    name = "spray"
    machine_type = "sprayer"


TillageEvent = _TillageEvent
HarvestEvent = _HarvestEvent
SprayEvent = _SprayEvent


# ---------------------------------------------------------------------------
# CustomEvent
# ---------------------------------------------------------------------------

class _CustomEvent(_BaseEvent):
    """Arbitrary researcher function called on a specific day.

    The decorated function receives ``(field_state, env_state)`` and may
    return None or a modified ``field_state``.  If a modified state is
    returned, it replaces the current field state.

    Exceptions raised by the function are caught, logged to the event
    log as an error entry, and the simulation continues (PRD Section 4.3).

    Parameters
    ----------
    field:
        Name of the target field.
    day:
        Simulation day on which the function fires.
    fn:
        The researcher's function. Attached via ``Event.custom()`` used
        as a decorator factory.
    """

    name = "custom"

    def __init__(self, field: str, day: int, fn: Optional[Callable] = None) -> None:
        self.field = field
        self.fire_day = day
        self.fn = fn
        if fn is not None:
            self.name = f"custom:{fn.__name__}"

    def attach(self, fn: Callable) -> "_CustomEvent":
        """Attach a callable to this event. Returns self for chaining."""
        self.fn = fn
        self.name = f"custom:{fn.__name__}"
        return self

    def validate(self) -> None:
        if self.fn is None:
            raise CropForgeEventError(
                f"Event.custom(field={self.field!r}, day={self.fire_day}) "
                "has no function attached. Use it as a decorator: "
                "@farm.add_event(Event.custom(...))."
            )

    def should_fire(self, field_name: str, day: int) -> bool:
        return field_name == self.field and day == self.fire_day

    def apply(
        self,
        field_state: "FieldState",
        env_state: "EnvironmentState",
        day: int,
    ) -> None:
        if self.fn is None:
            return
        try:
            result = self.fn(field_state, env_state)
            # If the function returns a modified state, update in place
            # by copying the returned state's attributes to field_state.
            # We cannot just replace the object here (the caller holds the
            # original reference via field._field_state).  Instead, we
            # propagate the returned value through the return value of apply(),
            # which _fire_events() checks and replaces on the field.
            #
            # To allow the runtime to detect a returned state, we store it
            # as an instance attribute that _fire_events() checks.
            self._last_result = result
        except Exception:
            logger.exception(
                "CustomEvent %r raised on day %d for field %r. "
                "Simulation continues.",
                self.name, day, self.field,
            )
            self._last_result = None

    def __repr__(self) -> str:
        fn_name = self.fn.__name__ if self.fn else "<no function>"
        return f"CustomEvent(field={self.field!r}, day={self.fire_day}, fn={fn_name})"


# ---------------------------------------------------------------------------
# Public Event factory (the user-facing API)
# ---------------------------------------------------------------------------

class Event:
    """Factory class for creating management events.

    All event types are constructed via class methods:

    Examples
    --------
    >>> farm.add_event(Event.irrigation(
    ...     field="Plot_A", interval_days=15, amount_mm=30,
    ... ))
    >>> farm.add_event(Event.fertiliser(
    ...     field="Plot_A", day=20, n_kg_ha=40.0,
    ... ))
    >>> @farm.add_event(Event.custom(field="Plot_A", day=50))
    ... def stress_test(field_state, env_state):
    ...     for plant in field_state.plants:
    ...         plant.custom["drought_stressed"] = True
    """

    @staticmethod
    def irrigation(
        field: str,
        interval_days: int,
        amount_mm: float,
        start_day: int = 1,
        end_day: Optional[int] = None,
    ) -> _IrrigationEvent:
        """Create an interval-based irrigation event.

        Parameters
        ----------
        field:
            Target field name.
        interval_days:
            Days between irrigations (must be >= 1).
        amount_mm:
            Water added per irrigation (mm).
        start_day:
            First day the event fires. Default 1.
        end_day:
            Last day (inclusive). ``None`` = end of simulation.
        """
        return _IrrigationEvent(
            field=field,
            interval_days=interval_days,
            amount_mm=amount_mm,
            start_day=start_day,
            end_day=end_day,
        )

    @staticmethod
    def fertiliser(
        field: str,
        n_kg_ha: float,
        apply_to_layer: int = 0,
        day: Optional[int] = None,
        days: Optional[List[int]] = None,
    ) -> _FertiliserEvent:
        """Create a fertiliser application event.

        Parameters
        ----------
        field:
            Target field name.
        n_kg_ha:
            Nitrogen added (kg ha⁻¹) at each application.
        apply_to_layer:
            Soil layer index to receive nitrogen. Default 0 (topsoil).
        day:
            Single day on which to fire (mutually exclusive with ``days``).
        days:
            List of days on which to fire (mutually exclusive with ``day``).

        Raises
        ------
        ValueError
            If neither ``day`` nor ``days`` is provided, or if both are provided.
        """
        if day is not None and days is not None:
            raise ValueError(
                "Event.fertiliser(): provide either 'day' or 'days', not both."
            )
        if day is None and days is None:
            raise ValueError(
                "Event.fertiliser(): must provide either 'day' or 'days'."
            )
        fire_days: List[int] = [day] if day is not None else list(days)  # type: ignore[arg-type]
        return _FertiliserEvent(
            field=field,
            fire_days=fire_days,
            n_kg_ha=n_kg_ha,
            apply_to_layer=apply_to_layer,
        )

    @staticmethod
    def tillage(field: str, day: int) -> _TillageEvent:
        """Create a tillage machinery pass logged for frontend animation."""
        return _TillageEvent(field=field, day=day)

    @staticmethod
    def harvest(field: str, day: int) -> _HarvestEvent:
        """Create a harvest machinery pass logged for frontend animation."""
        return _HarvestEvent(field=field, day=day)

    @staticmethod
    def spray(field: str, day: int) -> _SprayEvent:
        """Create a spray machinery pass logged for frontend animation."""
        return _SprayEvent(field=field, day=day)

    @staticmethod
    def custom(field: str, day: int) -> _CustomEvent:
        """Create a custom event placeholder to be used as a decorator.

        Usage::

            @farm.add_event(Event.custom(field="Plot_A", day=50))
            def my_event(field_state, env_state):
                ...

        Parameters
        ----------
        field:
            Target field name.
        day:
            Simulation day on which the function fires.
        """
        return _CustomEvent(field=field, day=day)
