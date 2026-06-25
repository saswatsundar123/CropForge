"""
cropforge/runtime.py
====================
Time-stepping engine and error-handling contract.

This module contains:
  - ``CropForgeStepError``  — raised when a step function crashes the run.
  - ``CropForgeVisualizeError`` — raised by the pre-flight check (Phase 2).
  - ``_execute_run()`` — the core daily loop called by ``Farm.run()``.

The engine is deliberately thin.  It knows about the step registry, the
``FieldState``/``EnvironmentState`` contract, and the error contract.  It has
no knowledge of the visualisation layer.

PRD References:
    Section 6.2 — @farm.step decorator, phase rules
    Section 6.4 — Error Handling Contract (full implementation)
    Section 4.2 — Compute-First Principle
    Section 15  — Phase 1 milestones 3, 4, 7

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
            f"╔══════════════════════════════════════════════════════════════╗\n"
            f"║               CropForge Simulation Failure                   ║\n"
            f"╚══════════════════════════════════════════════════════════════╝\n"
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

    Notes
    -----
    PRD Section 6.4, rule 3:
        "The crash log contains: the day number on which the exception
        occurred, the name of the step function that raised it, and the
        full Python traceback."
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

    # Stub fallback — used in tests and skeleton runs
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
# Event firing stub (Section 6.3 — full implementation deferred to loaders)
# ---------------------------------------------------------------------------

def _fire_events(farm: "Farm", field: "Field", day: int) -> List[str]:
    """Fire any events registered for *field* on *day*.

    Returns a list of event names that fired, to be stored in
    ``FieldState.events_fired``.

    Full implementation deferred to Phase 1 Event class.
    """
    fired: List[str] = []
    for event in farm._events:
        if hasattr(event, "should_fire") and event.should_fire(field.name, day):
            try:
                event.apply(field._field_state, day)
                fired.append(getattr(event, "name", repr(event)))
            except Exception:
                logger.exception(
                    "Event %r raised an exception on day %d for field %r.",
                    event,
                    day,
                    field.name,
                )
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
            a. Fire events → populate ``FieldState.events_fired``.
            b. Resolve ``EnvironmentState`` for the field.
            c. Execute each registered step function in ascending phase order,
               passing ``(FieldState, EnvironmentState)`` as arguments.
            d. Log the completed ``FieldState`` to Parquet (Phase 1).
        2. Advance ``FieldState.day``.

    Error contract (PRD Section 6.4):
        - Any uncaught exception from a step function → halt immediately.
        - Write crash log (``cropforge_crash.log``).
        - Flush completed timesteps to partial Parquet log (Phase 1).
        - Raise ``CropForgeStepError``.

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
        logger.warning(
            "Farm %r has no step functions registered. "
            "The simulation will run but no model logic will execute.",
            farm.name,
        )

    # Resolve and validate phase ordering once before the loop
    sorted_steps = farm._sorted_steps()

    # Initialise FieldState for every field
    for field in farm._fields:
        field._init_field_state(day=1)

    # Create the Parquet state logger for this run (PRD Section 16)
    from cropforge.logger import StateLogger
    from cropforge import __version__
    from datetime import datetime, timezone
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
        len(sorted_steps),
    )

    # ------------------------------------------------------------------
    # Main time-stepping loop
    # ------------------------------------------------------------------
    for day in range(1, days + 1):
        for field in farm._fields:
            state = field._field_state
            assert state is not None  # guaranteed by _init_field_state above

            # 1. Update day counter
            state.day = day

            # 2. Fire events for this field on this day
            state.events_fired = _fire_events(farm, field, day)

            # 3. Resolve environment
            env = _resolve_environment(field, day)

            # 4. Execute step functions in phase order
            for phase_val, step_fn in sorted_steps:
                try:
                    result = step_fn(state, env)
                    # Step functions SHOULD return state (PRD Section 6.2
                    # examples show `return state`). Accept None for
                    # researcher convenience but keep existing state.
                    if result is not None:
                        field._field_state = result
                        state = result

                except Exception as exc:
                    # ---- Error Handling Contract (PRD Section 6.4) ----
                    tb_str = traceback.format_exc()

                    # Rule 3: Write crash log
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

                    # Rule 2: Flush partial log so completed days survive crash
                    try:
                        state_log.flush()
                        logger.info(
                            "Partial Parquet log flushed to %s",
                            state_log.log_path,
                        )
                    except Exception:
                        logger.exception("Failed to flush partial state log.")

                    # Rule 4: Raise CropForgeStepError
                    raise CropForgeStepError(
                        day=day,
                        step_name=step_fn.__name__,
                        crash_log_path=crash_path,
                        original_exception=exc,
                    ) from exc

            # 5. Record completed timestep to Parquet logger
            state_log.record(field, state, env)

            # 6. Advance plant ages by one day
            for plant in state.plants:
                if plant.alive:
                    plant.age_days += 1

    # Flush all collected data to Parquet
    state_log.flush()

    logger.info(
        "Simulation complete: farm=%r, %d days, %d fields. Log: %s",
        farm.name,
        days,
        len(farm._fields),
        state_log.log_path,
    )
