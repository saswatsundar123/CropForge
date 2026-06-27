"""
tests/test_plugins.py
=====================
Test suite for the CropForge v0.4.0 Plugin API.

PRD §5.8 Test Requirements:
    [✓] register_crop() decorator adds plugin to registry
    [✓] get_plugin() returns correct class by name; returns None for unknown name
    [✓] list_plugins() returns all registered names
    [✓] field.use_plugin() registers plugin's step function at phase=0
    [✓] Plugin step function receives correct (state, env) objects
    [✓] on_register() is called exactly once when field.use_plugin() is called
    [✓] Two plugins on two different fields do not interfere
    [✓] Plugin registered after farm.run() raises CropForgePluginError
    [✓] All 427 existing tests pass with plugins.py added

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import pytest

from cropforge import Farm, Field, Crop
from cropforge.plugins import (
    CropPlugin,
    CropForgePluginError,
    register_crop,
    get_plugin,
    list_plugins,
    _REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_farm_with_field(field_name: str = "TestField") -> tuple[Farm, Field]:
    """Return a minimal (Farm, Field) pair ready for use_plugin() and run()."""
    farm = Farm(name="TestFarm")
    field = Field(name=field_name, rows=2, cols=2, area_ha=0.5)
    field.set_crop(Crop(species="Mock species"))
    farm.add_field(field)
    return farm, field


# ---------------------------------------------------------------------------
# MockWheatPlugin — the canonical test plugin (PRD §5.8 integration test)
# ---------------------------------------------------------------------------

class MockWheatPlugin(CropPlugin):
    """Minimal plugin: adds exactly 1.5 g biomass per plant per day."""
    species = "Triticum aestivum (mock)"

    def step(self, state, env):
        for plant in state.plants:
            if plant.alive:
                plant.biomass_g += 1.5
        return state


# ---------------------------------------------------------------------------
# Task 1: Plugin Registry Tests
# ---------------------------------------------------------------------------

class TestRegisterCrop:
    """register_crop() decorator correctly manages the global registry."""

    def test_register_adds_to_registry(self):
        """@register_crop() stores the class under the given name."""
        @register_crop("_test_barley_pytest")
        class BarleyPlugin(CropPlugin):
            species = "Hordeum vulgare"
            def step(self, state, env): return state

        assert _REGISTRY.get("_test_barley_pytest") is BarleyPlugin
        # Cleanup
        _REGISTRY.pop("_test_barley_pytest", None)

    def test_register_returns_class_unchanged(self):
        """Decorator must return the original class (not a wrapper)."""
        @register_crop("_test_sorghum_pytest")
        class SorghumPlugin(CropPlugin):
            species = "Sorghum bicolor"
            def step(self, state, env): return state

        assert SorghumPlugin.__name__ == "SorghumPlugin"
        _REGISTRY.pop("_test_sorghum_pytest", None)

    def test_register_overwrites_on_duplicate_name(self):
        """A second @register_crop() with same name overwrites the first."""
        @register_crop("_test_dup_pytest")
        class FirstPlugin(CropPlugin):
            def step(self, state, env): return state

        @register_crop("_test_dup_pytest")
        class SecondPlugin(CropPlugin):
            def step(self, state, env): return state

        assert _REGISTRY["_test_dup_pytest"] is SecondPlugin
        _REGISTRY.pop("_test_dup_pytest", None)


class TestGetPlugin:
    """get_plugin() returns the right class or None."""

    def test_get_returns_registered_class(self):
        """get_plugin() should return the class registered under that name."""
        @register_crop("_test_get_lentil_pytest")
        class LentilPlugin(CropPlugin):
            def step(self, state, env): return state

        assert get_plugin("_test_get_lentil_pytest") is LentilPlugin
        _REGISTRY.pop("_test_get_lentil_pytest", None)

    def test_get_returns_none_for_unknown(self):
        """get_plugin() returns None for a name that was never registered."""
        assert get_plugin("no_such_crop_xyzzy") is None

    def test_get_with_empty_string(self):
        """get_plugin("") returns None when "" was never registered."""
        assert get_plugin("") is None


class TestListPlugins:
    """list_plugins() returns sorted names."""

    def test_list_is_sorted(self):
        """list_plugins() must return alphabetically sorted list."""
        names_before = set(list_plugins())
        result = list_plugins()
        assert result == sorted(result), "list_plugins() must return a sorted list"

    def test_list_contains_newly_registered(self):
        """A newly registered plugin appears in list_plugins()."""
        _REGISTRY.pop("_test_list_pulse_pytest", None)
        @register_crop("_test_list_pulse_pytest")
        class PulsePlugin(CropPlugin):
            def step(self, state, env): return state

        assert "_test_list_pulse_pytest" in list_plugins()
        _REGISTRY.pop("_test_list_pulse_pytest", None)


# ---------------------------------------------------------------------------
# Task 2: field.use_plugin() Tests
# ---------------------------------------------------------------------------

class TestUsePlugin:
    """field.use_plugin() correctly registers plugin steps."""

    def test_use_plugin_stores_step_in_field(self):
        """After use_plugin(), field._plugin_steps has one entry."""
        _, field = _make_farm_with_field()
        assert len(field._plugin_steps) == 0
        field.use_plugin(MockWheatPlugin)
        assert len(field._plugin_steps) == 1

    def test_use_plugin_step_at_default_phase_zero(self):
        """Plugin step is registered at phase=0 by default."""
        _, field = _make_farm_with_field()
        field.use_plugin(MockWheatPlugin)
        phase, _ = field._plugin_steps[0]
        assert phase == 0

    def test_use_plugin_custom_phase(self):
        """Plugin step respects a custom phase kwarg."""
        _, field = _make_farm_with_field()
        field.use_plugin(MockWheatPlugin, phase=5)
        phase, _ = field._plugin_steps[0]
        assert phase == 5

    def test_use_plugin_rejects_negative_phase(self):
        """Plugin phase must be non-negative (physics phases are reserved)."""
        _, field = _make_farm_with_field()
        with pytest.raises(ValueError, match="non-negative"):
            field.use_plugin(MockWheatPlugin, phase=-1)

    def test_use_plugin_rejects_non_plugin_class(self):
        """use_plugin() with a non-CropPlugin class raises CropForgePluginError."""
        _, field = _make_farm_with_field()

        class NotAPlugin:
            pass

        with pytest.raises(CropForgePluginError, match="not a subclass of CropPlugin"):
            field.use_plugin(NotAPlugin)

    def test_use_plugin_rejects_instance_not_class(self):
        """use_plugin() with an instance (not a class) raises TypeError."""
        _, field = _make_farm_with_field()
        instance = MockWheatPlugin()
        with pytest.raises(TypeError, match="expects a CropPlugin class"):
            field.use_plugin(instance)

    def test_use_plugin_multiple_plugins_on_same_field(self):
        """Multiple plugins can be stacked on the same field."""
        _, field = _make_farm_with_field()

        class PluginA(CropPlugin):
            def step(self, state, env): return state

        class PluginB(CropPlugin):
            def step(self, state, env): return state

        field.use_plugin(PluginA)
        field.use_plugin(PluginB)
        assert len(field._plugin_steps) == 2


class TestOnRegister:
    """on_register() is called exactly once at registration time."""

    def test_on_register_called_once(self):
        """on_register() fires exactly once when field.use_plugin() is called."""
        call_log: list[tuple] = []

        class TrackedPlugin(CropPlugin):
            species = "Tracked"
            def on_register(self, farm, field):
                call_log.append((farm, field))
            def step(self, state, env):
                return state

        farm, field = _make_farm_with_field()
        field.use_plugin(TrackedPlugin)

        assert len(call_log) == 1, "on_register() must be called exactly once"
        assert call_log[0][0] is farm, "on_register() must receive the Farm"
        assert call_log[0][1] is field, "on_register() must receive the Field"

    def test_on_register_receives_correct_objects(self):
        """on_register() receives the actual Farm and Field objects."""
        received = {}

        class CheckPlugin(CropPlugin):
            def on_register(self, farm, field):
                received["farm"] = farm
                received["field"] = field
            def step(self, state, env):
                return state

        farm, field = _make_farm_with_field("CheckField")
        field.use_plugin(CheckPlugin)

        assert received["farm"] is farm
        assert received["field"] is field
        assert received["field"].name == "CheckField"

    def test_on_register_not_called_again_on_run(self):
        """on_register() is NOT called again when farm.run() starts."""
        call_count = [0]

        class CountingPlugin(CropPlugin):
            def on_register(self, farm, field):
                call_count[0] += 1
            def step(self, state, env):
                return state

        farm, field = _make_farm_with_field()
        field.use_plugin(CountingPlugin)
        assert call_count[0] == 1

        farm.run(days=3)
        assert call_count[0] == 1  # Must NOT increment during run()


class TestPluginAfterRunGuard:
    """Registering a plugin after farm.run() raises CropForgePluginError."""

    def test_plugin_after_run_raises(self):
        """use_plugin() during run() raises CropForgePluginError.

        We simulate "during run" by manually setting _is_running.
        """
        farm, field = _make_farm_with_field()
        farm._is_running = True  # simulate mid-run state

        with pytest.raises(CropForgePluginError, match="after farm.run\\(\\)"):
            field.use_plugin(MockWheatPlugin)

        farm._is_running = False  # cleanup

    def test_plugin_before_run_is_allowed(self):
        """use_plugin() before farm.run() must succeed without error."""
        farm, field = _make_farm_with_field()
        field.use_plugin(MockWheatPlugin)  # should not raise
        assert len(field._plugin_steps) == 1


# ---------------------------------------------------------------------------
# Task 3: Integration Test — PRD §5.8 Criterion
# "Plugin step function receives correct (state, env) objects"
# "Two plugins on two different fields do not interfere"
# ---------------------------------------------------------------------------

class TestPluginIntegration:
    """Full integration: load plugin, run, verify outcome."""

    def test_mock_wheat_plugin_biomass_accumulation(self):
        """MockWheatPlugin: 10 days × 1.5 g/plant = 15.0 g per plant.

        This is the PRD §5.8 canonical integration test:
        - No researcher @farm.step is written
        - The plugin alone drives biomass accumulation
        - After 10 days, every plant must have gained exactly 15.0 g
        """
        farm, field = _make_farm_with_field("WheatPlot")
        # Attach the plugin — NO @farm.step written by researcher
        field.use_plugin(MockWheatPlugin)
        farm.run(days=10)

        state = field._field_state
        assert state is not None

        # All 4 plants (2×2 grid) must have gained exactly 15.0 g
        for plant in state.plants:
            assert plant.alive, f"Plant {plant.plant_id} unexpectedly died"
            assert plant.biomass_g == pytest.approx(15.0, abs=1e-9), (
                f"Plant {plant.plant_id}: expected 15.0 g, got {plant.biomass_g} g. "
                f"MockWheatPlugin should add 1.5 g/day × 10 days = 15.0 g."
            )

    def test_plugin_step_receives_correct_state_type(self):
        """Plugin step receives a FieldState object (not None, not a dict)."""
        from cropforge.state import FieldState, EnvironmentState
        received = {}

        class TypeCheckPlugin(CropPlugin):
            def step(self, state, env):
                received["state_type"] = type(state).__name__
                received["env_type"] = type(env).__name__
                return state

        farm, field = _make_farm_with_field()
        field.use_plugin(TypeCheckPlugin)
        farm.run(days=1)

        assert received["state_type"] == "FieldState"
        assert received["env_type"] == "EnvironmentState"

    def test_two_plugins_on_two_fields_do_not_interfere(self):
        """Plugin isolation: Field_A plugin must not affect Field_B plants.

        This is the PRD §5.8 isolation requirement.
        """
        # Field A: 1.5 g/day plugin
        class PluginA(CropPlugin):
            species = "Species A"
            def step(self, state, env):
                for plant in state.plants:
                    plant.biomass_g += 1.5
                return state

        # Field B: 3.0 g/day plugin (different rate)
        class PluginB(CropPlugin):
            species = "Species B"
            def step(self, state, env):
                for plant in state.plants:
                    plant.biomass_g += 3.0
                return state

        farm = Farm(name="IsolationTest")

        field_a = Field(name="Field_A", rows=2, cols=2, area_ha=0.5)
        field_a.set_crop(Crop(species="Species A"))
        farm.add_field(field_a)
        field_a.use_plugin(PluginA)

        field_b = Field(name="Field_B", rows=2, cols=2, area_ha=0.5)
        field_b.set_crop(Crop(species="Species B"))
        farm.add_field(field_b)
        field_b.use_plugin(PluginB)

        farm.run(days=10)

        # Field A: 1.5 × 10 = 15.0 g
        for plant in field_a._field_state.plants:
            assert plant.biomass_g == pytest.approx(15.0, abs=1e-9), (
                f"Field_A plant {plant.plant_id}: expected 15.0, got {plant.biomass_g}. "
                "Field_B's plugin may have bled into Field_A."
            )

        # Field B: 3.0 × 10 = 30.0 g
        for plant in field_b._field_state.plants:
            assert plant.biomass_g == pytest.approx(30.0, abs=1e-9), (
                f"Field_B plant {plant.plant_id}: expected 30.0, got {plant.biomass_g}. "
                "Field_A's plugin may have bled into Field_B."
            )

    def test_plugin_and_researcher_step_coexist(self):
        """A plugin and a researcher @farm.step can run on the same field.

        Both should accumulate without overwriting each other.
        Plugin: +1.5 g/day. Researcher: +2.0 g/day. Total: +3.5 g/day × 5d = 17.5g.
        """
        farm, field = _make_farm_with_field()
        field.use_plugin(MockWheatPlugin)  # +1.5 g/day

        @farm.step(phase=1)
        def researcher_growth(state, env):
            for plant in state.plants:
                plant.biomass_g += 2.0
            return state

        farm.run(days=5)

        for plant in field._field_state.plants:
            assert plant.biomass_g == pytest.approx(17.5, abs=1e-9), (
                f"Expected 17.5 g (plugin 1.5 + researcher 2.0) × 5 days, "
                f"got {plant.biomass_g} g."
            )

    def test_plugin_step_return_none_accepted(self):
        """Plugin step returning None keeps existing state (PRD convention)."""
        class NoneReturnPlugin(CropPlugin):
            def step(self, state, env):
                for plant in state.plants:
                    plant.biomass_g += 1.0
                return None  # Not returning state — must still work

        farm, field = _make_farm_with_field()
        field.use_plugin(NoneReturnPlugin)
        farm.run(days=3)

        for plant in field._field_state.plants:
            assert plant.biomass_g == pytest.approx(3.0, abs=1e-9)

    def test_plugin_on_register_can_set_field_defaults(self):
        """on_register() can call set_water_params() or similar setup."""
        setup_called = [False]

        class SetupPlugin(CropPlugin):
            def on_register(self, farm, field):
                setup_called[0] = True
                # Demonstrate that on_register can configure the field
                field.set_water_params(field_capacity_pct=35.0)

            def step(self, state, env):
                return state

        farm, field = _make_farm_with_field()
        field.use_plugin(SetupPlugin)

        assert setup_called[0] is True
        assert field._water_params["field_capacity_pct"] == 35.0


# ---------------------------------------------------------------------------
# CropPlugin base class tests
# ---------------------------------------------------------------------------

class TestCropPluginBase:
    """CropPlugin base class interface."""

    def test_base_step_raises_not_implemented(self):
        """Calling step() on the unsubclassed base raises NotImplementedError."""
        plugin = CropPlugin()
        with pytest.raises(NotImplementedError, match="must be implemented"):
            plugin.step(None, None)

    def test_base_on_register_is_noop(self):
        """on_register() on base class must not raise."""
        plugin = CropPlugin()
        plugin.on_register(None, None)  # should return None silently

    def test_default_crop_returns_crop_instance(self):
        """default_crop() returns a Crop with species set from the class attribute."""
        from cropforge import Crop as CropClass

        class MyCrop(CropPlugin):
            species = "Vigna radiata"
            def step(self, state, env): return state

        crop = MyCrop.default_crop()
        assert isinstance(crop, CropClass)
        assert crop.species == "Vigna radiata"

    def test_default_crop_with_empty_species(self):
        """default_crop() on base CropPlugin with empty species raises ValueError.

        Crop validates that species must be non-empty, so calling default_crop()
        on the unsubclassed base (species='') correctly raises ValueError.
        Plugin authors must set a non-empty species class attribute.
        """
        with pytest.raises(ValueError, match="non-empty"):
            CropPlugin.default_crop()
