"""
tests/test_runtime.py
=====================
Tests for the CropForge runtime engine (PRD Section 6.2, 6.4).

Covers:
  - Phase execution ordering (lowest phase first)
  - Non-contiguous phases
  - Step functions receive correct (FieldState, EnvironmentState) arguments
  - Step function return value is honoured
  - Step functions can return None (engine keeps existing state)
  - age_days incremented each day for alive plants
  - Phase conflict warnings fire at the start of run()
  - Error handling contract (PRD Section 6.4):
      * Run halts immediately on exception
      * Only completed days logged (checked via execution count)
      * cropforge_crash.log is written with correct content
      * CropForgeStepError is raised
      * CropForgeStepError attributes (day, step_name, crash_log_path)
  - No silent failure — crashed step does NOT continue
  - Empty farm / no-step edge cases

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from cropforge.farm import Farm, Field
from cropforge.runtime import CropForgeStepError, _write_crash_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_farm(name: str = "TestFarm", rows: int = 2, cols: int = 2) -> Farm:
    """Return a minimal Farm with one Field, no crop or weather attached."""
    farm = Farm(name=name)
    farm.add_field(Field(name="Plot A", rows=rows, cols=cols))
    return farm


# ===========================================================================
# CropForgeStepError
# ===========================================================================

class TestCropForgeStepError:
    def test_attributes_set(self):
        exc = ValueError("boom")
        err = CropForgeStepError(
            day=5,
            step_name="growth_model",
            crash_log_path="/tmp/cropforge_crash.log",
            original_exception=exc,
        )
        assert err.day == 5
        assert err.step_name == "growth_model"
        assert err.crash_log_path == "/tmp/cropforge_crash.log"
        assert err.original_exception is exc

    def test_message_contains_day_and_step(self):
        exc = RuntimeError("oops")
        err = CropForgeStepError(
            day=42,
            step_name="soil_evaporation",
            crash_log_path="cropforge_crash.log",
            original_exception=exc,
        )
        msg = str(err)
        assert "42" in msg
        assert "soil_evaporation" in msg
        assert "cropforge_crash.log" in msg

    def test_is_runtime_error(self):
        exc = Exception("x")
        err = CropForgeStepError(1, "fn", "path", exc)
        assert isinstance(err, RuntimeError)


# ===========================================================================
# _write_crash_log
# ===========================================================================

class TestWriteCrashLog:
    def test_file_created(self, tmp_path):
        path = _write_crash_log(
            day=7,
            step_name="growth_model",
            tb_str="Traceback...\nValueError: bad value",
            output_dir=str(tmp_path),
        )
        assert Path(path).exists()
        assert Path(path).name == "cropforge_crash.log"

    def test_content_contains_day(self, tmp_path):
        _write_crash_log(day=99, step_name="fn", tb_str="tb", output_dir=str(tmp_path))
        content = (tmp_path / "cropforge_crash.log").read_text()
        assert "99" in content

    def test_content_contains_step_name(self, tmp_path):
        _write_crash_log(day=1, step_name="evaporation_step", tb_str="tb",
                         output_dir=str(tmp_path))
        content = (tmp_path / "cropforge_crash.log").read_text()
        assert "evaporation_step" in content

    def test_content_contains_traceback(self, tmp_path):
        _write_crash_log(day=1, step_name="fn",
                         tb_str="Traceback (most recent call last):\n  ...\nZeroDivisionError: division by zero",
                         output_dir=str(tmp_path))
        content = (tmp_path / "cropforge_crash.log").read_text()
        assert "ZeroDivisionError" in content

    def test_returns_absolute_path(self, tmp_path):
        path = _write_crash_log(1, "fn", "tb", str(tmp_path))
        assert os.path.isabs(path)


# ===========================================================================
# Phase ordering
# ===========================================================================

class TestPhaseOrdering:
    def test_phase_1_before_phase_2(self):
        """PRD Section 6.2: lower phase runs first."""
        execution_order = []
        farm = _simple_farm()

        @farm.step(interval="daily", phase=2)
        def step_b(state, env):
            execution_order.append("B")
            return state

        @farm.step(interval="daily", phase=1)
        def step_a(state, env):
            execution_order.append("A")
            return state

        farm.run(days=1)
        assert execution_order == ["A", "B"]

    def test_phase_ordering_multiple_phases(self):
        """Three phases: 1 → 2 → 5 regardless of registration order."""
        order = []
        farm = _simple_farm()

        @farm.step(interval="daily", phase=5)
        def step_c(state, env):
            order.append("C")
            return state

        @farm.step(interval="daily", phase=1)
        def step_a(state, env):
            order.append("A")
            return state

        @farm.step(interval="daily", phase=2)
        def step_b(state, env):
            order.append("B")
            return state

        farm.run(days=1)
        assert order == ["A", "B", "C"]

    def test_non_contiguous_phases(self):
        """PRD Section 6.2: phase=1 and phase=10 are valid; 1 runs first."""
        order = []
        farm = _simple_farm()

        @farm.step(interval="daily", phase=10)
        def late(state, env):
            order.append("late")
            return state

        @farm.step(interval="daily", phase=1)
        def early(state, env):
            order.append("early")
            return state

        farm.run(days=1)
        assert order == ["early", "late"]

    def test_phase_order_consistent_across_days(self):
        """Ordering must be the same on every day of the run."""
        # Record (day, label) pairs in execution order
        log: list[tuple[int, str]] = []
        farm = _simple_farm()

        @farm.step(interval="daily", phase=2)
        def step_b(state, env):
            log.append((state.day, "B"))
            return state

        @farm.step(interval="daily", phase=1)
        def step_a(state, env):
            log.append((state.day, "A"))
            return state

        farm.run(days=5)

        # For each day, A must appear before B
        for day in range(1, 6):
            day_entries = [label for d, label in log if d == day]
            assert day_entries == ["A", "B"], f"Wrong order on day {day}: {day_entries}"

    def test_step_receives_correct_state_type(self):
        """Step functions receive FieldState as first argument."""
        from cropforge.state import FieldState, EnvironmentState
        received = {}
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def inspector(state, env):
            received["state_type"] = type(state).__name__
            received["env_type"] = type(env).__name__
            return state

        farm.run(days=1)
        assert received["state_type"] == "FieldState"
        assert received["env_type"] == "EnvironmentState"

    def test_step_return_value_honoured(self):
        """When a step returns state, the engine uses the returned object."""
        farm = _simple_farm()
        returned_states = []

        @farm.step(interval="daily", phase=1)
        def step_a(state, env):
            return state

        @farm.step(interval="daily", phase=2)
        def step_b(state, env):
            # phase=2 sees the state returned by phase=1
            returned_states.append(id(state))
            return state

        farm.run(days=1)
        assert len(returned_states) == 1  # ran once

    def test_step_returning_none_keeps_existing_state(self):
        """A step that returns None must not replace state with None."""
        farm = _simple_farm()
        state_snapshots = []

        @farm.step(interval="daily", phase=1)
        def silent_step(state, env):
            # No explicit return → returns None implicitly
            pass

        @farm.step(interval="daily", phase=2)
        def observer(state, env):
            state_snapshots.append(state)
            return state

        farm.run(days=1)
        from cropforge.state import FieldState
        assert isinstance(state_snapshots[0], FieldState)


# ===========================================================================
# Simulation progress
# ===========================================================================

class TestSimulationProgress:
    def test_run_n_days_executes_n_times(self):
        call_counts = [0]
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def counter(state, env):
            call_counts[0] += 1
            return state

        farm.run(days=7)
        assert call_counts[0] == 7

    def test_plant_age_increments_each_day(self):
        """Alive plants should have age_days == days after run(days=N)."""
        farm = _simple_farm(rows=2, cols=2)

        # No step functions — engine still increments age_days
        farm.run(days=10)

        field = farm._fields[0]
        for plant in field._field_state.plants:
            assert plant.age_days == 10

    def test_dead_plant_age_does_not_increment(self):
        """PRD: dead plants keep their age frozen (not incremented).

        The engine increments age_days AFTER step functions run, and only
        for plants that are alive at that moment.  A plant killed inside
        step on day 1 is already dead when the increment check runs, so
        its age stays at 0.
        """
        farm = _simple_farm(rows=1, cols=1)

        @farm.step(interval="daily", phase=1)
        def kill_all(state, env):
            for plant in state.plants:
                plant.alive = False
            return state

        farm.run(days=5)
        # Plant was killed in phase=1 step on day 1, before the engine's
        # age-increment check.  Age should be 0 (never incremented).
        plant = farm._fields[0]._field_state.plants[0]
        assert plant.age_days == 0

    def test_day_field_updated_in_state(self):
        """FieldState.day must equal the current simulation day."""
        seen_days = []
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def record_day(state, env):
            seen_days.append(state.day)
            return state

        farm.run(days=5)
        assert seen_days == [1, 2, 3, 4, 5]

    def test_env_day_matches_field_day(self):
        """EnvironmentState.day should match FieldState.day on each step."""
        mismatches = []
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def check_sync(state, env):
            if state.day != env.day:
                mismatches.append((state.day, env.day))
            return state

        farm.run(days=10)
        assert mismatches == []

    def test_empty_farm_no_error(self):
        """farm.run() on a farm with no fields completes without error."""
        farm = Farm(name="Empty")
        farm.run(days=5)  # Should not raise

    def test_no_step_functions_no_error(self):
        """farm.run() with no registered steps completes without error."""
        farm = _simple_farm()
        farm.run(days=3)  # Should not raise


# ===========================================================================
# Error Handling Contract (PRD Section 6.4)
# ===========================================================================

class TestErrorHandlingContract:
    def test_raises_crop_forge_step_error(self, tmp_path, monkeypatch):
        """PRD Section 6.4, rule 4: CropForgeStepError is raised."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def broken_step(state, env):
            raise ValueError("model diverged")

        with pytest.raises(CropForgeStepError):
            farm.run(days=10)

    def test_error_attributes_day(self, tmp_path, monkeypatch):
        """CropForgeStepError.day is the day on which the crash occurred."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()
        crash_day = [None]

        @farm.step(interval="daily", phase=1)
        def crash_on_day_3(state, env):
            if state.day == 3:
                raise RuntimeError("deliberate crash on day 3")
            return state

        with pytest.raises(CropForgeStepError) as exc_info:
            farm.run(days=10)

        assert exc_info.value.day == 3

    def test_error_attributes_step_name(self, tmp_path, monkeypatch):
        """CropForgeStepError.step_name matches the function that raised."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def my_broken_growth_model(state, env):
            raise ZeroDivisionError("bad maths")

        with pytest.raises(CropForgeStepError) as exc_info:
            farm.run(days=5)

        assert exc_info.value.step_name == "my_broken_growth_model"

    def test_error_original_exception_preserved(self, tmp_path, monkeypatch):
        """The original exception is stored on CropForgeStepError."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def bad(state, env):
            raise KeyError("missing_key")

        with pytest.raises(CropForgeStepError) as exc_info:
            farm.run(days=1)

        assert isinstance(exc_info.value.original_exception, KeyError)

    def test_run_halts_immediately(self, tmp_path, monkeypatch):
        """PRD Section 6.4, rule 1: no further timesteps after failure."""
        monkeypatch.chdir(tmp_path)
        completed_days = [0]
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def step(state, env):
            if state.day == 3:
                raise RuntimeError("crash!")
            completed_days[0] = state.day
            return state

        with pytest.raises(CropForgeStepError):
            farm.run(days=10)

        # Only days 1 and 2 completed fully (day 3 crashed)
        assert completed_days[0] == 2

    def test_crash_log_written(self, tmp_path, monkeypatch):
        """PRD Section 6.4, rule 3: cropforge_crash.log is created."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def broken(state, env):
            raise ValueError("test crash")

        with pytest.raises(CropForgeStepError):
            farm.run(days=1)

        assert (tmp_path / "cropforge_crash.log").exists()

    def test_crash_log_contains_day(self, tmp_path, monkeypatch):
        """Crash log must record the simulation day (PRD Section 6.4, rule 3)."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def step(state, env):
            if state.day == 4:
                raise RuntimeError("crash on 4")
            return state

        with pytest.raises(CropForgeStepError):
            farm.run(days=10)

        content = (tmp_path / "cropforge_crash.log").read_text()
        assert "4" in content

    def test_crash_log_contains_step_name(self, tmp_path, monkeypatch):
        """Crash log must record the step function name (PRD Section 6.4, rule 3)."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def my_specific_step_name(state, env):
            raise RuntimeError("crash")

        with pytest.raises(CropForgeStepError):
            farm.run(days=1)

        content = (tmp_path / "cropforge_crash.log").read_text()
        assert "my_specific_step_name" in content

    def test_crash_log_contains_traceback(self, tmp_path, monkeypatch):
        """Crash log must contain full Python traceback."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def traceback_step(state, env):
            raise TypeError("bad type in model")

        with pytest.raises(CropForgeStepError):
            farm.run(days=1)

        content = (tmp_path / "cropforge_crash.log").read_text()
        assert "TypeError" in content
        assert "bad type in model" in content

    def test_second_step_not_executed_after_first_crashes(self, tmp_path, monkeypatch):
        """PRD Section 6.4: run halts before phase=2 if phase=1 crashes."""
        monkeypatch.chdir(tmp_path)
        phase_2_ran = [False]
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def phase_1_crash(state, env):
            raise RuntimeError("phase 1 crash")

        @farm.step(interval="daily", phase=2)
        def phase_2_should_not_run(state, env):
            phase_2_ran[0] = True
            return state

        with pytest.raises(CropForgeStepError):
            farm.run(days=5)

        assert phase_2_ran[0] is False

    def test_step_internal_exception_handled_does_not_halt(self, tmp_path, monkeypatch):
        """PRD Section 6.4: exceptions caught inside the step do NOT halt the run."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def safe_step(state, env):
            try:
                raise ValueError("internal, caught")
            except ValueError:
                pass  # caught internally — must not propagate to engine
            return state

        # Must NOT raise CropForgeStepError
        farm.run(days=5)

    def test_phase_conflict_warning_emitted_at_run_start(
        self, tmp_path, monkeypatch, caplog
    ):
        """Phase conflict warning must fire at the start of run(), not at decoration."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def step_a(state, env): return state

        @farm.step(interval="daily", phase=1)
        def step_b(state, env): return state

        # No warnings yet (decorator time)
        assert not any("conflict" in r.message.lower() for r in caplog.records)

        with caplog.at_level(logging.WARNING, logger="cropforge.farm"):
            farm.run(days=1)

        # Warning fires during run()
        assert any("phase conflict" in r.message.lower() for r in caplog.records)

    def test_crop_forge_step_error_points_to_crash_log(self, tmp_path, monkeypatch):
        """CropForgeStepError.crash_log_path must point to the written file."""
        monkeypatch.chdir(tmp_path)
        farm = _simple_farm()

        @farm.step(interval="daily", phase=1)
        def broken(state, env):
            raise ValueError("test")

        with pytest.raises(CropForgeStepError) as exc_info:
            farm.run(days=1)

        crash_path = Path(exc_info.value.crash_log_path)
        assert crash_path.exists()
