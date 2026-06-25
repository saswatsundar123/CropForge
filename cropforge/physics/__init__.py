"""
cropforge/physics/__init__.py
==============================
CropForge v0.2.0 physics primitives -- opt-in solver functions.

These are pure mathematical functions with no side effects on state.
The engines (v0.2.0 build phases) call these functions and write results
into EnvironmentState / PlantState before researcher step functions run.

Public API:
    calculate_fao56_et0  -- Penman-Monteith ET0 (FAO-56)
    calculate_root_impedance  -- root growth multiplier from soil resistance
"""

from cropforge.physics.environment import calculate_fao56_et0
from cropforge.physics.soil import calculate_root_impedance

__all__ = [
    "calculate_fao56_et0",
    "calculate_root_impedance",
]
