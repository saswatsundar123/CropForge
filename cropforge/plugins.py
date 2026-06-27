"""
cropforge/plugins.py
====================
Plugin API for CropForge v0.4.0.

This module provides the infrastructure for third-party crop model packages
(e.g. ``cropforge-chickpea``) to integrate with CropForge via a stable,
versioned interface.

Architecture
------------
A plugin author subclasses :class:`CropPlugin`, overrides :meth:`CropPlugin.step`,
and optionally decorates their class with :func:`register_crop`.  A researcher
then calls ``field.use_plugin(MyPlugin)`` to attach the plugin to a field before
calling ``farm.run()``.

Plugin Lifecycle
----------------
1. Author publishes ``cropforge-chickpea`` containing a ``ChickpeaPlugin``
   subclass decorated with ``@register_crop("chickpea")``.
2. Researcher installs the package: ``pip install cropforge-chickpea``.
3. Researcher writes::

       from cropforge_chickpea import ChickpeaPlugin
       field.use_plugin(ChickpeaPlugin)
       farm.run(days=120)

4. At ``farm.run()`` time, each day the plugin's ``step(state, env)`` method
   is called for the field it is attached to — at ``phase=0``, after all
   built-in physics engines (which run at negative phases).
5. ``on_register(farm, field)`` is called exactly once when
   ``field.use_plugin()`` is invoked — not at run time.

Backward Compatibility
----------------------
- If no plugin is registered, behaviour is identical to v0.3.0.
- Plugins do not affect any field other than the one they are attached to.
- Registering a plugin after ``farm.run()`` has started raises
  :class:`CropForgePluginError`.

PRD References: §5.1 – §5.8 (v0.4.0)

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level plugin registry (name → class)
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type["CropPlugin"]] = {}


def register_crop(name: str):
    """Class decorator that registers a :class:`CropPlugin` under a species name.

    Parameters
    ----------
    name:
        Short species identifier (e.g. ``"chickpea"``, ``"wheat"``).  Used to
        look up the plugin via :func:`get_plugin` and :func:`list_plugins`.

    Returns
    -------
    callable
        The decorator function that stores the class in ``_REGISTRY`` and
        returns it unchanged.

    Examples
    --------
    >>> @register_crop("chickpea")
    ... class ChickpeaPlugin(CropPlugin):
    ...     species = "Cicer arietinum"
    ...     def step(self, state, env):
    ...         return state
    >>> get_plugin("chickpea") is ChickpeaPlugin
    True
    """
    def decorator(cls: Type["CropPlugin"]) -> Type["CropPlugin"]:
        _REGISTRY[name] = cls
        logger.debug("CropPlugin registered under name %r: %s", name, cls.__name__)
        return cls
    return decorator


def get_plugin(name: str) -> Optional[Type["CropPlugin"]]:
    """Return the :class:`CropPlugin` class registered under *name*, or ``None``.

    Parameters
    ----------
    name:
        Species name passed to :func:`register_crop` (e.g. ``"chickpea"``).

    Returns
    -------
    Type[CropPlugin] or None
        The registered class, or ``None`` if no plugin is registered under
        that name.

    Examples
    --------
    >>> get_plugin("unknown_crop") is None
    True
    """
    return _REGISTRY.get(name)


def list_plugins() -> List[str]:
    """Return a sorted list of all registered plugin species names.

    Returns
    -------
    List[str]
        Alphabetically sorted list of registered names.

    Examples
    --------
    >>> list_plugins()  # doctest: +SKIP
    ['chickpea', 'maize', 'wheat']
    """
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# CropPlugin base class
# ---------------------------------------------------------------------------

class CropPlugin:
    """Abstract base for third-party crop model plugins.

    Plugin authors subclass this class, override :meth:`step`, and optionally
    :meth:`on_register`.  The class is then attached to a field via
    ``field.use_plugin(MyPlugin)``.

    Class Attributes
    ----------------
    species:
        Canonical Latin species name (e.g. ``"Cicer arietinum"``).
        Informational; used by :meth:`default_crop`.

    Methods
    -------
    step(state, env) → state
        Core growth model — executed once per day per field.  Must be
        overridden.  Returning ``None`` is accepted but the existing state
        is retained (same convention as ``@farm.step``).
    on_register(farm, field)
        Called once when ``field.use_plugin(cls)`` is invoked.  Override
        to set default soil/weather parameters or do other setup.
    default_crop()
        Class method — returns a :class:`~cropforge.crop.Crop` instance
        using ``cls.species``.

    Examples
    --------
    >>> class MockPlugin(CropPlugin):
    ...     species = "Mock species"
    ...     def step(self, state, env):
    ...         for plant in state.plants:
    ...             plant.biomass_g += 1.5
    ...         return state
    """

    species: str = ""

    def step(self, state, env):
        """Core daily growth model.

        Called once per simulation day for the field this plugin is attached
        to.  The method signature is identical to ``@farm.step`` functions:
        receives ``(FieldState, EnvironmentState)`` and should return the
        (possibly modified) ``FieldState``.

        Parameters
        ----------
        state:
            :class:`~cropforge.state.FieldState` for the current field.
        env:
            :class:`~cropforge.state.EnvironmentState` for the current day.

        Returns
        -------
        FieldState
            The (possibly modified) field state.  Returning ``None`` is
            accepted; the existing state is retained.

        Raises
        ------
        NotImplementedError
            If the base class method is called directly (plugin author must
            override this method).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.step() must be implemented by the plugin author. "
            "Override this method in your CropPlugin subclass to provide crop-specific "
            "growth model logic."
        )

    def on_register(self, farm, field) -> None:
        """Hook called exactly once when the plugin is attached to a field.

        Override to set recommended soil/weather defaults or initialise any
        plugin-level state.  The default implementation is a no-op.

        Parameters
        ----------
        farm:
            The :class:`~cropforge.farm.Farm` instance this field belongs to.
            May be ``None`` if the plugin is registered before ``farm.add_field()``.
        field:
            The :class:`~cropforge.farm.Field` this plugin is being attached to.
        """
        pass  # No-op by default

    @classmethod
    def default_crop(cls):
        """Return a :class:`~cropforge.crop.Crop` using this plugin's species.

        Convenience factory so researchers can write::

            field.set_crop(MyPlugin.default_crop())

        Returns
        -------
        Crop
            A new :class:`~cropforge.crop.Crop` instance with ``species``
            set to ``cls.species``.
        """
        from cropforge import Crop
        return Crop(species=cls.species)


# ---------------------------------------------------------------------------
# CropForgePluginError
# ---------------------------------------------------------------------------

class CropForgePluginError(ValueError):
    """Raised when a plugin is misconfigured or used incorrectly.

    Examples
    --------
    - Calling ``field.use_plugin()`` after ``farm.run()`` has started.
    - Passing an object that is not a :class:`CropPlugin` subclass.
    """
