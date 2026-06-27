"""
cropforge/runtime.py
====================
Time-stepping engine and error-handling contract.

This module contains:
  - ``CropForgeStepError``  -- raised when a step function crashes the run.
  - ``CropForgeVisualizeError`` -- raised by the pre-flight check (Phase 2).
  - ``_execute_run()`` -- the core daily loop called by ``Farm.run()``.

The engine is deliberately thin.  It knows about the step registry, the
``FieldState``/``EnvironmentState`` contract, and the error contract.  It has
no knowledge of the visualisation layer.

PRD References:
    Section 6.2 -- @farm.step decorator, phase rules
    Section 6.4 -- Error Handling Contract (full implementation)
    Section 4.2 -- Compute-First Principle
    Section 15  -- Phase 1 milestones 3, 4, 7

v0.4.0: Plugin API integration -- _execute_run now passes the current field
    to _sorted_steps() so per-field plugin steps are merged in, guaranteeing
    that Field_A plugins never execute for Field_B.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from cropforge.farm import Farm, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class CropForgeEventError(ValueError):
    """Re-exported from cropforge.events for convenience.

    Raised when an Event is misconfigured (e.g. interval_days=0).
    See :class:`cropforge.events.CropForgeEventError`.
    """


class CropForgeConfigError(ValueError):
    """Raised when conflicting or invalid physics configuration is detected.

    This error is raised at ``farm.run()`` time (or at ``use_physics()`` time
    for hard conflicts), never at registration time.

    Example
    -------
    Enabling ``soil_water_balance`` without ``et0`` raises this error because
    the water balance depends on ET0 for evapotranspiration demand.
    """


class CropForgeStepError(RuntimeError):
    """Raised when a researcher's step function crashes the simulation run.

    PRD Section 6.4, rule 4:
        "A CropForgeStepError is raised in the terminal with a summary
        message pointing the researcher to the crash log."

    Attributes
    ----------
    day:
        Simulation day on which the failure occurred.
    step_name:
        ``__name__`` of the step function that raised the exception.
    crash_log_path:
        Absolute path to the written crash log file.
    original_exception:
        The original exception that caused the failure.
    """

    def __init__(
        self,
        day: int,
        step_name: str,
        crash_log_path: str,
        original_exception: BaseException,
    ) -> None:
        self.day = day
        self.step_name = step_name
        self.crash_log_path = crash_log_path
        self.original_exception = original_exception
        super().__init__(
            f"\n"
            f"CropForge Simulation Failure\n"
            f"  Day        : {day}\n"
            f"  Step       : {step_name}()\n"
            f"  Error      : {type(original_exception).__name__}: {original_exception}\n"
            f"  Crash log  : {crash_log_path}\n"
            f"\n"
            f"Simulation halted. All completed timesteps have been flushed\n"
            f"to the partial Parquet log (if logging is enabled).\n"
        )


class CropForgeVisualizeError(RuntimeError):
    """Raised by the pre-flight check in ``farm.visualize()`` (PRD Section 6.5).

    This exception is defined here so it can be imported without importing
    the visualisation layer.
    """


class CropForgeStateError(ValueError):
    """Raised by ``farm.load_state()`` when a .cfstate file is incompatible.

    PRD v0.4.0 Section 7.6:
        - Wrong cropforge_version -> CropForgeStateError
        - Wrong field_name        -> CropForgeStateError

    Attributes
    ----------
    path:
        Path to the .cfstate file that caused the error.
    reason:
        Human-readable description of why the file was rejected.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(
            f"CropForge state file rejected: {path}\n"
            f"  Reason: {reason}\n"
            f"  Fix: ensure the .cfstate file was saved by the same CropForge version\n"
            f"  and for the same field name."
        )


# ---------------------------------------------------------------------------
# Crash log writer (PRD Section 6.4, rule 3)
# ---------------------------------------------------------------------------

def _write_crash_log(
    day: int,
    step_name: str,
    tb_str: str,
    output_dir: str = ".",
) -> str:
    """Write a crash log to *output_dir*/cropforge_crash.log.

    Parameters
    ----------
    day:
        The simulation day on which the exception occurred.
    step_name:
        Name of the step function that raised.
    tb_str:
        Full Python traceback as a string.
    output_dir:
        Directory in which to write the log.  Defaults to the current
        working directory.

    Returns
    -------
    str
        Absolute path to the written crash log file.
    """
    log_path = Path(output_dir) / "cropforge_crash.log"
    timestamp = datetime.now(timezone.utc).isoformat()

    content = (
        f"# CropForge Crash Log\n"
        f"# Generated : {timestamp}\n"
        f"# ================================================\n"
        f"\n"
        f"Simulation day : {day}\n"
        f"Step function  : {step_name}()\n"
        f"\n"
        f"Full traceback:\n"
        f"---------------\n"
        f"{tb_str}\n"
    )
    log_path.write_text(content, encoding="utf-8")
    return str(log_path.resolve())


# ---------------------------------------------------------------------------
# Environment resolver stub
# ---------------------------------------------------------------------------

def _resolve_environment(field: "Field", day: int) -> "Any":
    """Return an ``EnvironmentState`` for *field* on *day*.

    In Phase 1 this will delegate to ``field.weather.get_day(day)``.  For
    now it returns a minimal stub so the engine can run without a real
    weather source (enabling unit tests of the engine loop itself).

    The stub produces a constant environment: temperate, no rain, zero ET0.
    """
    from cropforge.state import EnvironmentState

    # If a real weather source is attached, use it
    if field.weather is not None and hasattr(field.weather, "get_day"):
        return field.weather.get_day(day)

    # Stub fallback -- used in tests and skeleton runs
    return EnvironmentState(
        day=day,
        doy=((day - 1) % 365) + 1,
        temp_max_c=25.0,
        temp_min_c=15.0,
        temp_mean_c=20.0,
        radiation_mj_m2=15.0,
        rainfall_mm=0.0,
        et0_mm=0.0,
        wind_speed_ms=2.0,
        humidity_pct=60.0,
    )


# ---------------------------------------------------------------------------
# Event validation (called once at the start of farm.run())
# ---------------------------------------------------------------------------

def _validate_events(farm: "Farm") -> None:
    """Validate all registered events before the simulation loop starts.

    Raises ``CropForgeEventError`` if any event has an invalid configuration
    (e.g. ``interval_days=0``).  This is called at ``farm.run()`` time so
    researchers see the error immediately, before any work is wasted.

    PRD Section 4.3:
        'interval_days=0 is invalid and raises ValueError at farm.run() time,
        not at event registration.'
    """
    from cropforge.events import CropForgeEventError as _EventError
    for event in farm._events:
        if hasattr(event, "validate"):
            try:
                event.validate()
            except _EventError:
                raise
            except Exception as exc:
                raise _EventError(
                    f"Event {event!r} failed validation: {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Event firing (PRD Section 4.3 -- events fire END OF DAY, after all steps)
# ---------------------------------------------------------------------------

def _fire_events(
    farm: "Farm",
    field: "Field",
    day: int,
    env: Any,
) -> List[str]:
    """Fire any events registered for *field* on *day*.

    Called AFTER all step functions for the day have completed.
    Returns a list of event name strings for the Parquet log.

    PRD Section 4.3 execution contract:
        Events fire at end of each day -- after phase=-2 (ET0),
        phase=-1 (root), phase=0 (user steps). State modifications
        are visible from day+1 onward.
    """
    fired: List[str] = []
    field_state = field._field_state

    for event in farm._events:
        if not (hasattr(event, "should_fire") and event.should_fire(field.name, day)):
            continue
        try:
            event.apply(field_state, env, day)
            event_name = getattr(event, "name", repr(event))
            fired.append(event_name)
            logger.debug(
                "Event fired: day=%d field=%r event=%r",
                day, field.name, event_name,
            )

            # Custom events may return a modified state object.
            # We detect this via the _last_result attribute.
            from cropforge.events import _CustomEvent
            if isinstance(event, _CustomEvent):
                result = getattr(event, "_last_result", None)
                if result is not None and hasattr(result, "plants"):
                    field._field_state = result
                    field_state = result

        except Exception:
            logger.exception(
                "Event %r raised an exception on day %d for field %r. "
                "Simulation continues.",
                event, day, field.name,
            )
            fired.append(f"{getattr(event, 'name', repr(event))}:ERROR")
    return fired


# ---------------------------------------------------------------------------
# Core execution loop (PRD Section 6.4)
# ---------------------------------------------------------------------------

def _execute_run(farm: "Farm", days: int) -> None:
    """Drive the simulation for *days* timesteps.

    This function is called by ``Farm.run()`` and is the single authoritative
    implementation of the time-stepping loop and the error-handling contract.

    Algorithm per day *d* (1-indexed):
        1. For each field:
            a. Resolve EnvironmentState for the field.
            b. Build the per-field sorted step list (physics hooks at phase<0,
               researcher steps + field plugin steps at phase>=0).
            c. Execute each step function in ascending phase order.
            d. Fire events (END OF DAY, after all steps).
            e. Log the completed FieldState to Parquet.
        2. Advance plant ages.

    Error contract (PRD Section 6.4):
        - Any uncaught exception from a step function halts immediately.
        - Write crash log (cropforge_crash.log).
        - Flush completed timesteps to partial Parquet log.
        - Raise CropForgeStepError.

    Parameters
    ----------
    farm:
        The ``Farm`` instance driving this run.
    days:
        Total number of daily timesteps.
    """
    if not farm._fields:
        logger.warning(
            "Farm %r has no fields attached. farm.run() completed with 0 work.",
            farm.name,
        )
        return

    if not farm._step_registry:
        # Check if any field has plugin steps (still valid to run without @farm.step)
        has_plugins = any(f._plugin_steps for f in farm._fields)
        if not has_plugins:
            logger.warning(
                "Farm %r has no step functions registered. "
                "The simulation will run but no model logic will execute.",
                farm.name,
            )

    # Validate all events before the loop starts (PRD Section 4.3)
    _validate_events(farm)

    # Set the running guard so field.use_plugin() raises CropForgePluginError if called now
    farm._is_running = True

    try:
        # Trigger phase-conflict warnings via global registry (before per-field merge)
        farm._sorted_steps()

        # Initialise FieldState for every field.
        # Season 2+: if _field_state already exists (set by prepare_next_season or
        # load_state), do NOT re-initialise -- soil must carry over exactly.
        for field in farm._fields:
            if field._field_state is None:
                field._init_field_state(day=1)
            else:
                # Season 2+: reset day counter to 1 (within-season)
                # but leave soil, events_fired (already cleared by prepare_next_season)
                field._field_state.day = 1

        # Create the Parquet state logger for this run (PRD Section 16)
        from cropforge.logger import StateLogger
        from cropforge import __version__
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_name = farm.name.replace(" ", "_").replace("/", "_")
        session_name = f"{safe_name}_{timestamp}"
        state_log = StateLogger(
            session_name=session_name,
            cropforge_version=__version__,
        )
        farm._last_log_path = state_log.log_path

        logger.info(
            "Starting simulation: farm=%r, fields=%d, days=%d, steps=%d",
            farm.name,
            len(farm._fields),
            days,
            len(farm._step_registry) + len(farm._physics_registry),
        )

        # ------------------------------------------------------------------
        # Main time-stepping loop
        # ------------------------------------------------------------------
        for day in range(1, days + 1):
            for field in farm._fields:
                state = field._field_state
                assert state is not None  # guaranteed by _init_field_state above

                # 1. Update day counter (within-season, 1-indexed)
                state.day = day

                # 2. Resolve environment (before steps so physics hooks can read it)
                env = _resolve_environment(field, day)

                # 2b. Stamp multi-season metadata onto env (v0.4.0)
                env.season = farm._current_season

                # 3. Build per-field sorted steps (merges field-specific plugin steps)
                #    This is the isolation mechanism: Field_A plugins never run for Field_B.
                field_steps = farm._sorted_steps(field=field)

                # 4. Execute step functions in phase order
                #    Physics hooks at phase<0, researcher + plugin steps at phase>=0
                for _phase_val, step_fn in field_steps:
                    try:
                        result = step_fn(state, env)
                        # Accept None for researcher/plugin convenience
                        if result is not None:
                            field._field_state = result
                            state = result

                    except Exception as exc:
                        # ---- Error Handling Contract (PRD Section 6.4) ----
                        tb_str = traceback.format_exc()
                        crash_path = _write_crash_log(
                            day=day,
                            step_name=step_fn.__name__,
                            tb_str=tb_str,
                        )
                        logger.error(
                            "Step function %r raised on day %d. Run halted. "
                            "Crash log: %s",
                            step_fn.__name__,
                            day,
                            crash_path,
                        )
                        try:
                            state_log.flush()
                            logger.info(
                                "Partial Parquet log flushed to %s",
                                state_log.log_path,
                            )
                        except Exception:
                            logger.exception("Failed to flush partial state log.")

                        raise CropForgeStepError(
                            day=day,
                            step_name=step_fn.__name__,
                            crash_log_path=crash_path,
                            original_exception=exc,
                        ) from exc

                # 5. Fire events for this field on this day (END OF DAY).
                #    PRD Section 4.3: events fire AFTER all step functions.
                state.events_fired = _fire_events(farm, field, day, env)
                # Re-fetch state -- a custom event may have replaced the object.
                state = field._field_state

                # 6. Record completed timestep to Parquet logger
                state_log.record(field, state, env)

                # 7. Advance plant ages by one day
                for plant in state.plants:
                    if plant.alive:
                        plant.age_days += 1

        # Flush all collected data to Parquet
        state_log.flush()

        # Update _day_offset so Season 2 Parquet day numbers are continuous.
        # Season 1 runs 10 days → _day_offset becomes 10.
        # Season 2 runs 10 days → Parquet days 11-20 (_day_offset will become 20).
        farm._day_offset += days

        logger.info(
            "Simulation complete: farm=%r, %d days, %d fields. Log: %s",
            farm.name,
            days,
            len(farm._fields),
            state_log.log_path,
        )

    finally:
        # Always reset the running guard, even on exception
        farm._is_running = False
