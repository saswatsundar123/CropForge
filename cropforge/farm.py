"""
cropforge/farm.py
=================
Farm and Field domain classes.

``Field`` represents one spatial plot: a rows×cols grid of plant positions and
soil voxels, with an attached ``Crop``, weather source, and soil profile.

``Farm`` is the top-level container that holds multiple ``Field`` objects and
owns the step-function registry.  The ``@farm.step`` decorator (defined here
for co-location with the registry) registers user model logic; ``Farm.run()``
drives the time-stepping loop.

PRD References:
    Section 4.3 — Directory Structure
    Section 6.1 — Basic Usage Example
    Section 6.2 — @farm.step decorator + phase rules
    Section 6.3 — Event system (skeleton)
    Section 6.4 — Error Handling Contract
    Section 6.5 — farm.visualize() Pre-flight Check (stub)

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from cropforge.crop import Crop
from cropforge.state import (
    EnvironmentState,
    FieldState,
    PlantState,
    SoilVoxelState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel elevation profiles (PRD Section 6.1 / 8.3)
# ---------------------------------------------------------------------------

_BUILTIN_ELEVATION_PROFILES: Dict[str, str] = {
    "flat": "flat",
    "slope_1pct_N": "slope_1pct_N",
    "slope_2pct_N": "slope_2pct_N",
}


def _build_elevation_grid(
    rows: int,
    cols: int,
    profile: Union[str, np.ndarray, None],
) -> np.ndarray:
    """Return a (rows, cols) float64 elevation array.

    Accepts either a string shorthand (e.g. ``"slope_1pct_N"``) or a
    pre-built NumPy array.  If *profile* is ``None`` the grid is flat zero.

    Parameters
    ----------
    rows, cols:
        Field grid dimensions.
    profile:
        ``None`` → flat zeros.
        ``"flat"`` → flat zeros.
        ``"slope_1pct_N"`` → 1 % northward slope (row index × 0.01 m).
        ``"slope_2pct_N"`` → 2 % northward slope (row index × 0.02 m).
        ``np.ndarray`` → used directly; must have shape ``(rows, cols)``.
    """
    # Check for ndarray FIRST — before any string equality test,
    # because numpy arrays raise ValueError on `array == "string"` comparisons.
    if isinstance(profile, np.ndarray):
        if profile.shape != (rows, cols):
            raise ValueError(
                f"elevation_profile array shape {profile.shape} does not match "
                f"field dimensions ({rows}, {cols})."
            )
        return profile.astype(np.float64)
    if profile is None or profile == "flat":
        return np.zeros((rows, cols), dtype=np.float64)
    if profile == "slope_1pct_N":
        elev = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            elev[r, :] = r * 0.01
        return elev
    if profile == "slope_2pct_N":
        elev = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            elev[r, :] = r * 0.02
        return elev
    raise ValueError(
        f"Unknown elevation_profile string {profile!r}. "
        f"Known profiles: {list(_BUILTIN_ELEVATION_PROFILES)}. "
        "Pass a NumPy ndarray for a custom DEM."
    )


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

class Field:
    """One spatial field (plot) within a farm.

    The field owns the PlantState grid and the SoilVoxelState grid for its
    spatial extent.  After construction, the researcher attaches a ``Crop``,
    a weather source, and a soil profile via the ``set_*`` methods.

    Parameters
    ----------
    name:
        Human-readable field identifier, e.g. ``"Plot A"``.  Used as the
        ``field_name`` key throughout the Parquet log.
    rows:
        Number of rows in the plant grid (north–south axis by convention).
    cols:
        Number of columns in the plant grid (east–west axis by convention).
    area_ha:
        Physical area of the field in hectares.  Used for per-hectare
        quantity calculations (e.g. N application in kg/ha).
    elevation_profile:
        Initial elevation surface.  See :func:`_build_elevation_grid` for
        accepted values.

    Examples
    --------
    >>> from cropforge.farm import Field
    >>> f = Field(name="Plot A", rows=20, cols=30, area_ha=1.0)
    >>> f.rows, f.cols
    (20, 30)
    """

    def __init__(
        self,
        name: str,
        rows: int,
        cols: int,
        area_ha: float = 1.0,
        elevation_profile: Union[str, np.ndarray, None] = None,
    ) -> None:
        if not name.strip():
            raise ValueError("Field.name must be a non-empty string.")
        if rows < 1 or cols < 1:
            raise ValueError(
                f"Field dimensions must be positive integers, got rows={rows}, cols={cols}."
            )
        if area_ha <= 0:
            raise ValueError(f"Field.area_ha must be positive, got {area_ha}.")

        self.name: str = name
        self.rows: int = rows
        self.cols: int = cols
        self.area_ha: float = area_ha

        # Build the elevation grid immediately so it is always available
        self.elevation_grid: np.ndarray = _build_elevation_grid(
            rows, cols, elevation_profile
        )

        # Attached data — set via set_* methods before farm.run()
        self.crop: Optional[Crop] = None
        self.weather: Optional[Any] = None   # Weather object — set in Phase 1 loaders
        self.soil_profile: Optional[Any] = None  # Soil object — set in Phase 1 loaders

        # Runtime state — populated by the engine at the start of farm.run()
        self._field_state: Optional[FieldState] = None

        # Plugin list: [(phase, bound_step_method)] — per-field, isolated from other fields
        self._plugin_steps: List[Tuple[int, Callable]] = []

        # Back-reference to the owning Farm — set by farm.add_field(), used by use_plugin()
        self._farm: Optional[Any] = None

        # v0.6.0 -- Terrain object (None = flat, backward compatible)
        self._terrain: Optional[Any] = None

        # v0.6.0 -- Land preparation modifier (None = no modification)
        self._land_prep: Optional[Any] = None

    # ------------------------------------------------------------------
    # Physical area properties (PRD v0.8.0 §4.2)
    # ------------------------------------------------------------------

    @property
    def cell_area_m2(self) -> float:
        """Physical area of one grid cell in m² (resolution_m²). Default 1.0."""
        res = self._terrain.resolution_m if self._terrain is not None else 1.0
        return res * res

    @property
    def field_area_m2(self) -> float:
        """Total physical field area in m² (rows × cols × cell_area_m2)."""
        return self.rows * self.cols * self.cell_area_m2

    # ------------------------------------------------------------------
    # Attachment methods (PRD Section 6.1)
    # ------------------------------------------------------------------

    def set_crop(self, crop: Crop) -> None:
        """Attach a :class:`~cropforge.crop.Crop` to this field."""
        if not isinstance(crop, Crop):
            raise TypeError(f"Expected a Crop instance, got {type(crop).__name__}.")
        self.crop = crop

    def set_weather(self, weather: Any) -> None:
        """Attach a Weather data source.

        The ``Weather`` class is implemented in :mod:`cropforge.loaders`
        (Phase 1). This method accepts any object so that the field class
        does not create a circular import.
        """
        self.weather = weather

    def set_soil(self, soil: Any) -> None:
        """Attach a Soil profile.

        The ``Soil`` class is implemented in :mod:`cropforge.loaders`
        (Phase 1). This method accepts any object for the same reason as
        :meth:`set_weather`.
        """
        self.soil_profile = soil

    def set_water_params(
        self,
        field_capacity_pct: float = 32.0,
        wilting_point_pct: float = 14.0,
        saturation_pct: float = 48.0,
        drainage_coefficient: float = 0.5,
        crop_coefficient: float = 1.0,
        stress_increment_per_day: float = 0.05,
    ) -> None:
        """Configure soil water balance parameters for this field (PRD v0.3.0 Section 5.2).

        These parameters are stamped into every ``SoilVoxelState.custom`` dict
        when the field state is initialised at ``farm.run()`` time.  The
        hydrology hook reads them from there each day.

        Call this method **before** ``farm.run()`` and **after** ``set_soil()``.
        If called after ``farm.run()`` has started, the parameters will take
        effect from the next day.

        Parameters
        ----------
        field_capacity_pct:
            Volumetric water content at field capacity (%).
            Default 32.0 (typical loam soil).
        wilting_point_pct:
            Permanent wilting point (%). Plants at this moisture have Ks=0.
            Default 14.0.
        saturation_pct:
            Saturation moisture content (%). Upper ceiling for any layer.
            Default 48.0.
        drainage_coefficient:
            Fraction of excess-above-FC water that drains per day. Default 0.5.
            Set to 1.0 for free-draining sandy soils, 0.1 for heavy clays.
        crop_coefficient:
            Kc value (dimensionless). Multiply ET0 by Kc to get ETc.
            Default 1.0. Can be a float for a constant Kc or a dict keyed
            by phenological stage (future extension).
        stress_increment_per_day:
            How much ``stress_index`` increases per day at Ks=0 (full stress).
            Default 0.05 (20 days at full stress drives stress_index to 1.0).

        Notes
        -----
        Parameters are stored in ``self._water_params`` and are applied to
        every voxel's ``custom`` dict whenever ``_init_field_state()`` is
        called.  They are also applied immediately if the field state has
        already been initialised (i.e., if called during a multi-phase run).
        """
        self._water_params: dict = {
            "field_capacity_pct":     float(field_capacity_pct),
            "wilting_point_pct":      float(wilting_point_pct),
            "saturation_pct":         float(saturation_pct),
            "drainage_coefficient":   float(drainage_coefficient),
            "crop_coefficient":       float(crop_coefficient),
            "stress_increment_per_day": float(stress_increment_per_day),
        }
        # If the field state is already initialised, propagate immediately
        if self._field_state is not None:
            self._apply_water_params_to_state(self._field_state)

    def _apply_water_params_to_state(self, state: "FieldState") -> None:
        """Stamp water params into every SoilVoxelState.custom dict."""
        params = getattr(self, "_water_params", None)
        if params is None:
            return
        for row_soils in state.soil:
            for cell_soils in row_soils:
                for voxel in cell_soils:
                    voxel.custom.update(params)

    def set_nitrogen_params(
        self,
        leaching_fraction: float = 0.01,
        runoff_n_fraction: float = 0.05,
    ) -> None:
        """Configure nitrogen transport parameters for this field (PRD v0.3.0 Phase 4).

        Parameters are stamped into every ``SoilVoxelState.custom`` dict at
        run time so the nutrients hook can read them per-column.

        Parameters
        ----------
        leaching_fraction:
            Fraction of mineral N leached per mm of drainage. Default 0.01.
            Higher values (0.05) for sandy soils, lower (0.002) for clay.
        runoff_n_fraction:
            Fraction of top-layer N exported per mm of lateral runoff. Default 0.05.
        """
        self._nitrogen_params: dict = {
            "leaching_fraction": float(leaching_fraction),
            "runoff_n_fraction": float(runoff_n_fraction),
        }
        if self._field_state is not None:
            self._apply_nitrogen_params_to_state(self._field_state)

    def _apply_nitrogen_params_to_state(self, state: "FieldState") -> None:
        """Stamp nitrogen params into every SoilVoxelState.custom dict."""
        params = getattr(self, "_nitrogen_params", None)
        if params is None:
            return
        for row_soils in state.soil:
            for cell_soils in row_soils:
                for voxel in cell_soils:
                    voxel.custom.update(params)

    def set_elevation(self, dem: np.ndarray) -> None:
        """Replace the elevation grid with a custom DEM array.

        Parameters
        ----------
        dem:
            NumPy array of shape ``(rows, cols)`` containing relative
            elevation in metres (PRD Section 8.3).
        """
        if not isinstance(dem, np.ndarray):
            raise TypeError("dem must be a NumPy ndarray.")
        if dem.shape != (self.rows, self.cols):
            raise ValueError(
                f"DEM shape {dem.shape} does not match field "
                f"dimensions ({self.rows}, {self.cols})."
            )
        self.elevation_grid = dem.astype(np.float64)
        # v0.6.0 -- also update the Terrain wrapper so FieldState.terrain stays consistent
        from cropforge.terrain import Terrain as _Terrain
        self._terrain = _Terrain.from_array(self.elevation_grid)

    @staticmethod
    def elevation_from_csv(path: str) -> np.ndarray:
        """Load a DEM from a CSV file (PRD Section 8.3).

        Implemented in Phase 1 loaders; this stub is provided so the
        method name is discoverable from the public API.

        Raises
        ------
        NotImplementedError
            Until the loaders module implements this.
        """
        raise NotImplementedError(
            "Field.elevation_from_csv() will be implemented in Phase 1 loaders."
        )

    def set_terrain(self, terrain: Any) -> None:
        """Attach a :class:`~cropforge.terrain.Terrain` object to this field (PRD v0.6.0 §5).

        The terrain's ``elevation_grid`` supersedes any previous
        ``set_elevation()`` call.  Also updates ``self.elevation_grid`` so
        the existing D8 lateral flow engine continues to work unchanged.

        Parameters
        ----------
        terrain:
            A :class:`cropforge.terrain.Terrain` instance.
        """
        from cropforge.terrain import Terrain as _Terrain
        if not isinstance(terrain, _Terrain):
            raise TypeError(f"Expected a Terrain instance, got {type(terrain).__name__}.")
        self._terrain = terrain
        self.elevation_grid = terrain.elevation_grid.astype(np.float64)

    def set_land_prep(self, modifier: Any) -> None:
        """Attach a :class:`~cropforge.land_prep.LandPrep` modifier to this field (PRD v0.6.0 §6).

        The modifier's ``apply()`` method is called once at ``farm.run()``
        start, transforming the base elevation grid. The base terrain object
        is preserved unchanged; the *modified* grid feeds the D8 engine.

        Parameters
        ----------
        modifier:
            A :class:`cropforge.land_prep.LandPrep` instance.
        """
        from cropforge.land_prep import LandPrep as _LandPrep
        if not isinstance(modifier, _LandPrep):
            raise TypeError(
                f"Expected a LandPrep instance, got {type(modifier).__name__}. "
                "Subclass cropforge.land_prep.LandPrep and override apply()."
            )
        self._land_prep = modifier

    def use_plugin(self, plugin_cls: type, phase: int = 0, **kwargs) -> None:
        """Attach a :class:`~cropforge.plugins.CropPlugin` to this field.

        The plugin's :meth:`~cropforge.plugins.CropPlugin.step` method will be
        called every simulation day for this field (and *only* this field),
        at the specified *phase* alongside researcher ``@farm.step`` functions.
        :meth:`~cropforge.plugins.CropPlugin.on_register` is called exactly
        once, immediately when this method is invoked.

        Parameters
        ----------
        plugin_cls:
            A subclass of :class:`~cropforge.plugins.CropPlugin` (not an
            instance — the class itself).  ``use_plugin()`` instantiates it
            internally so the plugin holds no state shared between fields.
        phase:
            Phase at which the plugin step runs.  Default ``0``, same as
            researcher ``@farm.step`` functions.  Must be a non-negative integer.
        **kwargs:
            Additional keyword arguments passed to the plugin_cls constructor.

        Raises
        ------
        CropForgePluginError
            - If *plugin_cls* is not a subclass of :class:`~cropforge.plugins.CropPlugin`.
            - If the owning farm's simulation is already running
              (``farm.run()`` has been called and is in progress).
        TypeError
            If *plugin_cls* is an instance rather than a class.

        Examples
        --------
        >>> class GrowPlugin(CropPlugin):
        ...     def step(self, state, env):
        ...         for plant in state.plants: plant.biomass_g += 1.0
        ...         return state
        >>> field.use_plugin(GrowPlugin)
        """
        from cropforge.plugins import CropPlugin, CropForgePluginError

        # Guard: must receive a class, not an instance
        if not isinstance(plugin_cls, type):
            raise TypeError(
                f"field.use_plugin() expects a CropPlugin class (not an instance). "
                f"Did you mean field.use_plugin({type(plugin_cls).__name__}) instead of "
                f"field.use_plugin({type(plugin_cls).__name__}())?"
            )

        # Guard: must be a CropPlugin subclass
        if not issubclass(plugin_cls, CropPlugin):
            raise CropForgePluginError(
                f"{plugin_cls.__name__} is not a subclass of CropPlugin. "
                "Plugin classes must inherit from cropforge.plugins.CropPlugin."
            )

        # Guard: cannot register a plugin after the simulation has started
        farm = self._farm
        if farm is not None and getattr(farm, "_is_running", False):
            raise CropForgePluginError(
                f"Cannot register plugin {plugin_cls.__name__!r} after farm.run() "
                f"has started. Register all plugins before calling farm.run()."
            )

        # Phase must be non-negative (plugins run after physics engines)
        if not isinstance(phase, int) or phase < 0:
            raise ValueError(
                f"Plugin phase must be a non-negative integer, got {phase!r}."
            )

        # Instantiate the plugin (each field gets its own instance → isolation)
        instance = plugin_cls(**kwargs)

        # Call on_register exactly once
        instance.on_register(farm, self)
        logger.info(
            "Plugin %r registered on field %r at phase=%d.",
            plugin_cls.__name__, self.name, phase,
        )

        # Store the bound step method tagged with this field's name so the
        # engine can guard execution to the correct field only.
        self._plugin_steps.append((phase, instance.step))

    # ------------------------------------------------------------------
    # State initialisation (called by the engine at run-time)
    # ------------------------------------------------------------------

    def _init_field_state(self, day: int = 1) -> FieldState:
        """Build and return the initial ``FieldState`` for this field.

        Called by :class:`Farm` at the start of ``run()``.  If no soil
        profile has been attached, a minimal default (1 layer, all zeros)
        is generated so the field can still be run in skeleton tests.

        Also applies ``_water_params`` to every voxel if ``set_water_params()``
        has been called before ``run()``.
        """
        plants: List[PlantState] = [
            PlantState(
                plant_id=f"r{r:02d}c{c:02d}",
                row=r,
                col=c,
            )
            for r in range(self.rows)
            for c in range(self.cols)
        ]

        # Build soil grid: [row][col][layer]
        # Real soil data is injected by the loaders (Phase 1).
        # When no soil is attached, create 1 default layer per cell.
        if self.soil_profile is not None and hasattr(self.soil_profile, "build_grid"):
            soil: List[List[List[SoilVoxelState]]] = self.soil_profile.build_grid(
                self.rows, self.cols
            )
        else:
            soil = [
                [
                    [
                        SoilVoxelState(
                            row=r,
                            col=c,
                            layer=0,
                            depth_top_cm=0.0,
                            depth_bottom_cm=20.0,
                            moisture_pct=0.0,
                            nitrogen_kg_ha=0.0,
                            bulk_density=1.3,
                            penetration_resistance=0.5,
                        )
                    ]
                    for c in range(self.cols)
                ]
                for r in range(self.rows)
            ]

        # v0.6.0 -- Apply land preparation modifier if set.
        # Modifies the elevation grid used by D8; base terrain.elevation_grid is unchanged.
        resolution_m = self._terrain.resolution_m if self._terrain is not None else 1.0
        effective_elevation = self.elevation_grid.copy()
        _soil_deltas: dict = {}
        _per_cell_mods: dict = {}
        if self._land_prep is not None:
            _result = self._land_prep.apply(effective_elevation, resolution_m)
            if len(_result) == 3:
                effective_elevation, _soil_deltas, _per_cell_mods = _result
            else:
                effective_elevation, _soil_deltas = _result

        self._field_state = FieldState(
            day=day,
            plants=plants,
            soil=soil,
            elevation_grid=effective_elevation,
            events_fired=[],
            terrain=self._terrain,
        )
        # Apply water balance parameters to all voxels if configured
        self._apply_water_params_to_state(self._field_state)
        self._apply_nitrogen_params_to_state(self._field_state)
        # Stamp land prep soil property deltas onto every voxel (PRD v0.6.0 §6.5)
        if _soil_deltas:
            self._apply_land_prep_deltas(self._field_state, _soil_deltas)
        # Per-cell overrides from LandPrep subclasses that return a 3-tuple
        # (e.g. VegetativeFilterStrip). Applied after global deltas so strip
        # cells get their specific roughness even if global delta differs.
        for (r, c), cell_mods in _per_cell_mods.items():
            if 0 <= r < self.rows and 0 <= c < self.cols:
                for voxel in self._field_state.soil[r][c]:
                    for key, val in cell_mods.items():
                        setattr(voxel, key, float(val))


        # v0.7.0 -- Effective soil depth per cell for root clamping (PRD v0.7.0 §7.3)
        # delta = (post-land-prep elevation) - (base elevation), in metres.
        # Flat field / no land prep → delta == 0 → effective depth == profile depth.
        _elev_delta = effective_elevation - self.elevation_grid  # metres (rows, cols)
        self._field_state.custom["effective_soil_depth_cm_grid"] = [
            [
                max(0.0, self._field_state.soil[r][c][-1].depth_bottom_cm
                    + _elev_delta[r, c] * 100.0)
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]

        return self._field_state

    def _apply_land_prep_deltas(self, state: "FieldState", deltas: dict) -> None:
        """Stamp land-prep soil property deltas into every SoilVoxelState."""
        for row_soils in state.soil:
            for cell_soils in row_soils:
                for voxel in cell_soils:
                    for key in ("porosity_delta", "bulk_density_delta", "surface_roughness_index"):
                        if key in deltas:
                            setattr(voxel, key, float(deltas[key]))

    def __repr__(self) -> str:
        crop_str = repr(self.crop) if self.crop else "None"
        return (
            f"Field(name={self.name!r}, rows={self.rows}, cols={self.cols}, "
            f"area_ha={self.area_ha}, crop={crop_str})"
        )


# ---------------------------------------------------------------------------
# Farm
# ---------------------------------------------------------------------------

class Farm:
    """Top-level simulation container.

    The researcher creates one ``Farm``, attaches ``Field`` objects, registers
    step functions with ``@farm.step``, and then calls ``farm.run(days=N)``.

    Parameters
    ----------
    name:
        Human-readable farm / trial identifier, e.g. ``"Trial 2026-A"``.
    location:
        (latitude, longitude) tuple in decimal degrees.  Stored for metadata
        purposes; not used in v0.1 computations.

    Examples
    --------
    >>> from cropforge.farm import Farm
    >>> farm = Farm(name="Trial 2026-A", location=(23.4, 85.3))
    >>> farm.name
    'Trial 2026-A'
    """

    def __init__(
        self,
        name: str,
        location: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if not name.strip():
            raise ValueError("Farm.name must be a non-empty string.")

        self.name: str = name
        self.location: Tuple[float, float] = location

        # Ordered list of attached fields
        self._fields: List[Field] = []

        # Step function registry: list of (phase, fn) tuples, unsorted until run()
        self._step_registry: List[Tuple[int, Callable]] = []

        # Physics engine registry (v0.2.0) -- built-in hooks at negative phases.
        # Populated only when use_physics() is called.  Never contains
        # researcher-registered functions.  Kept separate so _sorted_steps()
        # can merge and sort both registries cleanly.
        self._physics_registry: List[Tuple[int, Callable]] = []

        # Event registry (Phase 1 Event system; stored here as Any for now)
        self._events: List[Any] = []

        # Path to the most recent Parquet log (set by logger at end of run)
        self._last_log_path: Optional[str] = None

        # Physics configuration snapshot (for introspection / documentation)
        self._physics_config: Dict[str, Any] = {}

        # Runtime guard: True while farm.run() is executing
        # Used by field.use_plugin() to reject late registrations.
        self._is_running: bool = False

        # Multi-season tracking (v0.4.0 -- PRD Section 7)
        # _current_season: stamped onto every EnvironmentState.season during run().
        # _day_offset: added to per-season day so Parquet days are continuous
        #   (Season 1 day 1..N → offset=0; Season 2 day 1..M → offset=N → Parquet day N+1..N+M).
        self._current_season: int = 1
        self._day_offset: int = 0

    # ------------------------------------------------------------------
    # Field management
    # ------------------------------------------------------------------

    def add_field(self, field: Field) -> None:
        """Attach a configured :class:`Field` to this farm.

        Parameters
        ----------
        field:
            A ``Field`` instance.  Field names must be unique within a farm.

        Raises
        ------
        TypeError
            If *field* is not a ``Field`` instance.
        ValueError
            If a field with the same name already exists in this farm.
        """
        if not isinstance(field, Field):
            raise TypeError(f"Expected a Field instance, got {type(field).__name__}.")
        existing_names = [f.name for f in self._fields]
        if field.name in existing_names:
            raise ValueError(
                f"A field named {field.name!r} already exists in farm {self.name!r}. "
                "Field names must be unique."
            )
        # Give the field a back-reference to this farm (needed by use_plugin)
        field._farm = self
        self._fields.append(field)

    @property
    def fields(self) -> List[Field]:
        """Read-only view of attached fields."""
        return list(self._fields)

    # ------------------------------------------------------------------
    # Event management (PRD v0.3.0 Section 4)
    # ------------------------------------------------------------------

    def add_event(self, event: Any) -> Any:
        """Register a management event or use as a decorator factory.

        This method works in two modes:

        **Direct registration** (irrigation, fertiliser):
            ``farm.add_event(Event.irrigation(...))``
            Appends the event to the registry and returns ``None``.

        **Decorator factory** (custom events):
            Used as ``@farm.add_event(Event.custom(...))`` — in this case
            the return value is a decorator that attaches the decorated
            function to the event and then registers it.

            .. code-block:: python

                @farm.add_event(Event.custom(field="Plot_A", day=50))
                def stress_test(field_state, env_state):
                    for plant in field_state.plants:
                        plant.custom["drought_stressed"] = True

        PRD v0.3.0 Section 4.2.
        """
        from cropforge.events import _CustomEvent

        if isinstance(event, _CustomEvent):
            # Decorator factory mode: return a function that will attach the
            # researcher's function to the event and then register the event.
            def _decorator(fn: Callable) -> Callable:
                event.attach(fn)
                self._events.append(event)
                return fn
            return _decorator

        # Direct registration mode (irrigation, fertiliser, etc.)
        self._events.append(event)
        return None

    # ------------------------------------------------------------------
    # @farm.step decorator (Section 6.2)
    # ------------------------------------------------------------------

    def step(
        self,
        interval: str = "daily",
        phase: int = 0,
    ) -> Callable:
        """Decorator that registers a step function with the farm engine.

        Parameters
        ----------
        interval:
            Execution frequency.  Only ``"daily"`` is supported in v0.1.
        phase:
            Execution order integer.  Lower values run first.  Must be a
            non-negative integer (PRD Section 6.2 phase rules).
            Defaults to 0 when omitted.

        Returns
        -------
        Callable
            The original function, unmodified.  The decorator is purely
            registrative — it does not wrap the function.

        Examples
        --------
        >>> @farm.step(interval="daily", phase=1)
        ... def soil_evaporation(state, env):
        ...     return state

        Phase rules (PRD Section 6.2):
            - ``phase`` must be a non-negative integer.
            - Multiple steps with the same phase → undefined order within
              that phase.  A warning is logged at the start of ``run()``.
            - Steps with no ``phase`` argument default to 0; multiple
              unphased steps trigger the same warning.
        """
        if not isinstance(phase, int) or phase < 0:
            raise ValueError(
                f"@farm.step phase must be a non-negative integer, got {phase!r}."
            )
        if interval != "daily":
            raise ValueError(
                f"@farm.step interval must be 'daily' in v0.1, got {interval!r}."
            )

        def decorator(fn: Callable) -> Callable:
            self._step_registry.append((phase, fn))
            return fn

        return decorator

    def _sorted_steps(self, field: "Optional[Field]" = None) -> "List[Tuple[int, Callable]]":
        """Return all step functions sorted by phase (ascending).

        Merges researcher-registered steps (_step_registry, phase >= 0)
        with built-in physics hooks (_physics_registry, phase < 0) and,
        optionally, field-specific plugin steps (_plugin_steps, phase >= 0).

        Plugin steps are included only when *field* is provided, ensuring
        that a plugin attached to Field_A never executes for Field_B.

        Also emits phase-conflict warnings per PRD Section 6.2.
        Must be called at the start of ``run()``.
        """
        from collections import Counter

        # Researcher steps only -- check for conflicts within these
        phase_counts = Counter(phase for phase, _ in self._step_registry)
        for phase_val, count in phase_counts.items():
            if count > 1:
                fn_names = [
                    fn.__name__
                    for p, fn in self._step_registry
                    if p == phase_val
                ]
                logger.warning(
                    "CropForge phase conflict: %d step functions share phase=%d "
                    "(%s). Their execution order within this phase is "
                    "non-deterministic. Assign unique phase values to enforce "
                    "deterministic ordering.",
                    count,
                    phase_val,
                    ", ".join(fn_names),
                )

        # Merge physics hooks (negative phases) + researcher steps (non-negative)
        # + field plugin steps (non-negative, field-scoped)
        plugin_steps = field._plugin_steps if field is not None else []
        all_steps = self._physics_registry + self._step_registry + plugin_steps
        return sorted(all_steps, key=lambda t: t[0])

    # ------------------------------------------------------------------
    # use_physics() -- Opt-In Physics API (PRD v0.2.0 Section 4 / 5 / 9)
    # ------------------------------------------------------------------

    def use_physics(
        self,
        et0: bool = False,
        root_impedance: bool = False,
        water_balance: bool = False,
        nutrients: bool = False,
        lateral_flow: bool = False,
        radiation: bool = False,
        disease: bool = False,
        elevation_m: float = 0.0,
        anemometer_height_m: float = 2.0,
        # Radiation engine parameters
        k_extinction: float = 0.45,
        slope_radiation_correction: bool = False,
        # Wind field parameters
        terrain_wind: bool = False,
        wind_direction_deg: float = 270.0,
        # Root constraint parameters
        root_clamping: bool = False,
        # Clod dynamics parameters
        clod_dynamics: bool = False,
        clod_decay_factor: float = 0.05,
        # Erosion engine parameters
        erosion: bool = False,
        # Sediment transport parameters (PRD v0.8.0 §5.2)
        sediment_transport: bool = False,
        k_erodibility: float = 0.005,
        k_transport: float = 0.02,
        # Terrain feedback control (PRD v0.9.0 §4.3)
        # True (default) = recompute slope/aspect daily after sediment update.
        # False = freeze terrain geometry at init (faster; use for non-erosion studies).
        terrain_feedback: bool = True,

        # Disease engine parameters
        disease_foci: list | None = None,
        disease_spread_rate: float = 0.15,
        disease_latency_days: int = 5,
        disease_stress_increment: float = 0.04,
        disease_wind_direction_deg: float = 270.0,
        disease_anisotropy: float = 0.80,
        disease_seed: int | None = None,
    ) -> None:
        """Enable opt-in physics engines for this farm (PRD v0.2.0 / v0.3.0 / v0.5.0).

        Execution order when all engines are enabled:
            phase=-4  Lateral flow + nitrogen transport
            phase=-3  Soil water balance engine
            phase=-2  ET0 engine (Penman-Monteith)
            phase=-2  Radiation interception engine (same phase, per-plant)
            phase=-1  Root impedance engine
            phase=-1  Spatial disease spread engine (same phase, per-plant)
            phase= 0  Researcher @farm.step (default phase)

        Parameters
        ----------
        et0:
            Enable the FAO-56 Penman-Monteith ET0 engine.
        root_impedance:
            Enable the root impedance engine.
        water_balance:
            Enable the FAO-56 daily soil water balance engine (v0.3.0).
            **Requires ``et0=True``.**
        nutrients:
            Enable vertical N leaching (driven by drainage fluxes). Runs
            at phase=-4. **Requires ``water_balance=True``** (needs drainage
            fluxes computed by the water balance hook).
        lateral_flow:
            Enable lateral surface runoff N transport between cells.
            Uses the field ``elevation_grid`` for D8 routing. Activates
            automatically when ``nutrients=True``; can also be enabled
            independently to suppress vertical leaching but allow lateral flow.
            **Requires ``water_balance=True``.**
        radiation:
            Enable the Beer-Lambert radiation interception engine (v0.5.0).
            Writes ``plant.custom['intercepted_par_mj']`` for every plant
            each day. Gated: no effect when False.
        disease:
            Enable the spatially explicit SIR disease spread engine (v0.5.0).
            See ``disease_*`` parameters for configuration.
        elevation_m:
            Site elevation above mean sea level (m). Used by ET0 engine.
        anemometer_height_m:
            Anemometer height (m) for wind height correction. Default 2.0 m.
        k_extinction:
            Beer-Lambert extinction coefficient for the radiation engine.
            Default 0.45 (C3 crops). Use 0.50 for C4 (maize).
        disease_foci:
            List of (row, col) tuples to infect on day 1. Default None.
        disease_spread_rate:
            Base daily infection probability (isotropic baseline). Default 0.15.
        disease_latency_days:
            Days from infection to when a plant becomes infectious. Default 5.
        disease_stress_increment:
            Daily disease_stress increment for infected plants. Default 0.04.
        disease_wind_direction_deg:
            Prevailing wind direction (met bearing, 0=N, 90=E, 180=S, 270=W).
            Direction FROM which wind blows. Default 270 (wind from West).
        disease_anisotropy:
            Directional bias strength [0=isotropic, 1=fully directional].
            Default 0.80.
        disease_seed:
            Random seed for reproducible spread. Default None.

        Raises
        ------
        CropForgeConfigError
            - If ``water_balance=True`` and ``et0=False``.
            - If ``nutrients=True`` or ``lateral_flow=True`` and
              ``water_balance=False``.

        Notes
        -----
        PRD v0.2.0 / v0.3.0 / v0.5.0 backward compatibility:
            If ``use_physics()`` is never called, simulation behaves identically
            to v0.1.0 -- no engine hooks are registered.
        """
        from cropforge.physics.builtin_hooks import (
            PHASE_ET0_ENGINE,
            PHASE_ROOT_ENGINE,
            PHASE_ROOT_CLAMP,
            PHASE_HYDROLOGY_ENGINE,
            PHASE_NUTRIENTS_ENGINE,
            PHASE_RADIATION_ENGINE,
            PHASE_WIND_ENGINE,
            PHASE_DISEASE_ENGINE,
            PHASE_CLOD_ENGINE,
            PHASE_EROSION_ENGINE,
            PHASE_SEDIMENT_ENGINE,
            make_et0_hook,
            make_root_impedance_hook,
            make_hydrology_hook,
            make_nutrients_hook,
            make_radiation_hook,
            make_wind_hook,
            make_disease_hook,
            make_root_clamp_hook,
            make_clod_dynamics_hook,
            make_erosion_hook,
            make_sediment_hook,
        )
        from cropforge.runtime import CropForgeConfigError

        # --- Validate dependency constraints ---
        if water_balance and not et0:
            raise CropForgeConfigError(
                "use_physics(water_balance=True) requires et0=True. "
                "Enable ET0 alongside water_balance: "
                "farm.use_physics(et0=True, water_balance=True)."
            )
        if (nutrients or lateral_flow) and not water_balance:
            raise CropForgeConfigError(
                "use_physics(nutrients=True) and use_physics(lateral_flow=True) both "
                "require water_balance=True. The nutrient transport engine reads "
                "drainage_mm_today which is written by the water balance hook. "
                "Enable: farm.use_physics(et0=True, water_balance=True, nutrients=True)."
            )
        if clod_dynamics and not water_balance:
            raise CropForgeConfigError(
                "use_physics(clod_dynamics=True) requires water_balance=True. "
                "Enable: farm.use_physics(et0=True, water_balance=True, clod_dynamics=True)."
            )

        # Record configuration for introspection
        self._physics_config.update({
            "et0": et0,
            "root_impedance": root_impedance,
            "water_balance": water_balance,
            "nutrients": nutrients,
            "lateral_flow": lateral_flow,
            "radiation": radiation,
            "slope_radiation_correction": slope_radiation_correction,
            "terrain_wind": terrain_wind,
            "wind_direction_deg": wind_direction_deg,
            "root_clamping": root_clamping,
            "clod_dynamics": clod_dynamics,
            "erosion": erosion,
            "sediment_transport": sediment_transport,
            "terrain_feedback": terrain_feedback,
            "disease": disease,
            "elevation_m": elevation_m,
            "anemometer_height_m": anemometer_height_m,
        })

        if nutrients or lateral_flow:
            hook = make_nutrients_hook()
            self._physics_registry.append((PHASE_NUTRIENTS_ENGINE, hook))
            logger.info(
                "Farm %r: Nutrients+lateral-flow engine enabled (phase=%d).",
                self.name, PHASE_NUTRIENTS_ENGINE,
            )

        if water_balance:
            hook = make_hydrology_hook(lateral_flow=lateral_flow, use_roughness=clod_dynamics)
            self._physics_registry.append((PHASE_HYDROLOGY_ENGINE, hook))
            logger.info(
                "Farm %r: Soil water balance engine enabled (phase=%d).",
                self.name, PHASE_HYDROLOGY_ENGINE,
            )

        if et0:
            latitude_deg = self.location[0]
            hook = make_et0_hook(
                latitude_deg=latitude_deg,
                elevation_m=elevation_m,
                anemometer_height_m=anemometer_height_m,
            )
            self._physics_registry.append((PHASE_ET0_ENGINE, hook))
            logger.info(
                "Farm %r: ET0 engine enabled (lat=%.3f, elev=%.1f m, anem=%.1f m).",
                self.name, latitude_deg, elevation_m, anemometer_height_m,
            )

        if root_impedance:
            hook = make_root_impedance_hook()
            self._physics_registry.append((PHASE_ROOT_ENGINE, hook))
            logger.info(
                "Farm %r: Root impedance engine enabled.",
                self.name,
            )

        if radiation:
            hook = make_radiation_hook(
                k_extinction=k_extinction,
                slope_radiation_correction=slope_radiation_correction,
                latitude_deg=self.location[0] if slope_radiation_correction else 0.0,
            )
            self._physics_registry.append((PHASE_RADIATION_ENGINE, hook))
            logger.info(
                "Farm %r: Radiation interception engine enabled (k=%.3f).",
                self.name, k_extinction,
            )

        if terrain_wind:
            hook = make_wind_hook(
                wind_direction_deg=wind_direction_deg,
            )
            self._physics_registry.append((PHASE_WIND_ENGINE, hook))
            logger.info(
                "Farm %r: Topographical wind field enabled (direction=%.0f°).",
                self.name, wind_direction_deg,
            )

        if disease:
            hook = make_disease_hook(
                initial_foci=disease_foci,
                spread_rate=disease_spread_rate,
                latency_days=disease_latency_days,
                stress_increment=disease_stress_increment,
                wind_direction_deg=disease_wind_direction_deg,
                anisotropy=disease_anisotropy,
                seed=disease_seed,
            )
            self._physics_registry.append((PHASE_DISEASE_ENGINE, hook))
            logger.info(
                "Farm %r: Spatial disease engine enabled "
                "(spread=%.2f, wind=%.1f°, latency=%d days).",
                self.name, disease_spread_rate,
                disease_wind_direction_deg, disease_latency_days,
            )

        if root_clamping:
            hook = make_root_clamp_hook()
            self._physics_registry.append((PHASE_ROOT_CLAMP, hook))
            logger.info(
                "Farm %r: Root effective-depth clamp enabled (phase=%d).",
                self.name, PHASE_ROOT_CLAMP,
            )

        if clod_dynamics:
            hook = make_clod_dynamics_hook(decay_factor=clod_decay_factor)
            self._physics_registry.append((PHASE_CLOD_ENGINE, hook))
            logger.info(
                "Farm %r: Clod dynamics engine enabled (decay_factor=%.3f, phase=%d).",
                self.name, clod_decay_factor, PHASE_CLOD_ENGINE,
            )

        if erosion:
            hook = make_erosion_hook()
            self._physics_registry.append((PHASE_EROSION_ENGINE, hook))
            logger.info(
                "Farm %r: Soil erosion engine enabled (phase=%d).",
                self.name, PHASE_EROSION_ENGINE,
            )

        if sediment_transport:
            if not erosion:
                from cropforge.runtime import CropForgeConfigError
                raise CropForgeConfigError(
                    "use_physics(sediment_transport=True) requires erosion=True. "
                    "The sediment hook reads daily_erosion_index_grid written by the "
                    "erosion engine. Enable: use_physics(erosion=True, sediment_transport=True)."
                )
            hook = make_sediment_hook(k_erodibility=k_erodibility, k_transport=k_transport, terrain_feedback=terrain_feedback)
            self._physics_registry.append((PHASE_SEDIMENT_ENGINE, hook))
            logger.info(
                "Farm %r: Sediment transport engine enabled (phase=%d).",
                self.name, PHASE_SEDIMENT_ENGINE,
            )

        if not any([et0, root_impedance, water_balance,
                     nutrients, lateral_flow, radiation,
                     disease, terrain_wind, root_clamping, clod_dynamics, erosion,
                     sediment_transport]):
            logger.debug(
                "Farm %r: use_physics() called with all engines disabled -- no-op.",
                self.name,
            )

    # ------------------------------------------------------------------
    # farm.run() — Time-stepping loop (Section 6.4)
    # ------------------------------------------------------------------

    def run(self, days: int) -> None:
        """Execute the simulation for *days* timesteps.

        For each day:
          1. Fire any registered events for that day.
          2. Execute each registered step function in ascending phase order.
          3. Pass the (possibly modified) ``FieldState`` to the Parquet
             logger (Phase 1 — logger stub called here).

        Error handling contract (PRD Section 6.4):
          - If any step function raises an unhandled exception the run halts.
          - All completed timesteps are flushed to the Parquet log.
          - A crash log is written to ``cropforge_crash.log``.
          - ``CropForgeStepError`` is raised in the terminal.

        Parameters
        ----------
        days:
            Number of daily timesteps to simulate (1-indexed days 1…N).
        """
        from cropforge.runtime import CropForgeStepError, _execute_run

        _execute_run(self, days)

    # ------------------------------------------------------------------
    # Multi-season API (v0.4.0 -- PRD Section 7)
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Serialise the final SoilState of every field to a .cfstate JSON file.

        PRD v0.4.0 Section 7.4 -- .cfstate file format.

        The file records the soil moisture, nitrogen, and all other SoilVoxelState
        fields for every cell and layer in every field attached to this farm.
        Plant state, events, and environment state are NOT saved -- they reset
        on the next season. Physics engine parameters are NOT saved -- they are
        re-registered by the researcher's script.

        Parameters
        ----------
        path:
            Destination file path (should end in ``.cfstate``).  Any parent
            directories will be created automatically.

        Raises
        ------
        RuntimeError
            If ``farm.run()`` has never been called (no soil state exists).

        Examples
        --------
        >>> farm.run(days=120)
        >>> farm.save_state("trial_2026_s1.cfstate")
        """
        import json as _json
        import cropforge as _cf
        from pathlib import Path as _Path

        _Path(path).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "cropforge_version": _cf.__version__,
            "season": self._current_season,
            "final_day": self._day_offset,  # total days simulated so far
            "fields": [],
        }

        for f in self._fields:
            if f._field_state is None:
                raise CropForgeStateError(
                    f"Field {f.name!r} has no simulation state. "
                    "Call farm.run() before farm.save_state().",
                    path="(none)",
                    reason="run() not yet called",
                )


            field_entry = {
                "field_name": f.name,
                "soil": [],
            }

            for row_list in f._field_state.soil:
                for col_list in row_list:
                    for voxel in col_list:
                        field_entry["soil"].append({
                            "row":                    voxel.row,
                            "col":                    voxel.col,
                            "layer":                  voxel.layer,
                            "depth_top_cm":           voxel.depth_top_cm,
                            "depth_bottom_cm":        voxel.depth_bottom_cm,
                            "moisture_pct":           voxel.moisture_pct,
                            "nitrogen_kg_ha":         voxel.nitrogen_kg_ha,
                            "bulk_density":           voxel.bulk_density,
                            "penetration_resistance": voxel.penetration_resistance,
                            "custom":                 voxel.custom,
                        })

            payload["fields"].append(field_entry)

        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2)

        logger.info(
            "Farm %r: soil state saved to %s (season=%d, final_day=%d, fields=%d).",
            self.name, path, self._current_season, self._day_offset, len(self._fields),
        )

    def load_state(self, path: str) -> None:
        """Restore SoilState from a .cfstate JSON file produced by ``save_state()``.

        PRD v0.4.0 Section 7.6:
          - Restores moisture, nitrogen, and all SoilVoxelState fields for every
            cell and layer exactly (float precision preserved as JSON).
          - Does NOT restore PlantState, Events, or physics config.
          - Raises ``CropForgeStateError`` if:
              * ``cropforge_version`` in the file does not match the running version, OR
              * any ``field_name`` in the file does not match a field attached to this farm.

        Parameters
        ----------
        path:
            Path to a ``.cfstate`` file previously written by ``save_state()``.

        Raises
        ------
        CropForgeStateError
            Version mismatch or field_name mismatch.
        FileNotFoundError
            If *path* does not exist.
        """
        import json as _json
        import cropforge as _cf
        from cropforge.runtime import CropForgeStateError
        from cropforge.state import SoilVoxelState as _SVS

        with open(path, "r", encoding="utf-8") as fh:
            data = _json.load(fh)

        # ---- Version check (PRD §7.6) ---------------------------------
        saved_version = data.get("cropforge_version", "")
        if saved_version != _cf.__version__:
            raise CropForgeStateError(
                path=path,
                reason=(
                    f"Version mismatch: file was saved by CropForge {saved_version!r}, "
                    f"but running CropForge {_cf.__version__!r}."
                ),
            )

        # ---- Build lookup: field_name → Field -------------------------
        field_by_name = {f.name: f for f in self._fields}

        for field_data in data.get("fields", []):
            fname = field_data["field_name"]
            if fname not in field_by_name:
                raise CropForgeStateError(
                    path=path,
                    reason=(
                        f"Field name mismatch: file contains field {fname!r}, "
                        f"but this farm has fields: {list(field_by_name.keys())}."
                    ),
                )

            target_field = field_by_name[fname]

            # If field has no state yet, initialise it now so we can write into it
            if target_field._field_state is None:
                target_field._init_field_state(day=1)

            # Build a lookup: (row, col, layer) → voxel object
            voxel_by_key = {
                (v.row, v.col, v.layer): v
                for row_list in target_field._field_state.soil
                for col_list in row_list
                for v in col_list
            }

            for entry in field_data["soil"]:
                key = (entry["row"], entry["col"], entry["layer"])
                voxel = voxel_by_key.get(key)
                if voxel is None:
                    logger.warning(
                        "load_state: voxel (%d,%d,%d) not found in field %r -- skipping.",
                        *key, fname,
                    )
                    continue

                # Restore all SoilVoxelState fields exactly
                voxel.depth_top_cm           = entry["depth_top_cm"]
                voxel.depth_bottom_cm        = entry["depth_bottom_cm"]
                voxel.moisture_pct           = entry["moisture_pct"]
                voxel.nitrogen_kg_ha         = entry["nitrogen_kg_ha"]
                voxel.bulk_density           = entry["bulk_density"]
                voxel.penetration_resistance = entry["penetration_resistance"]
                voxel.custom                 = dict(entry.get("custom", {}))

        # Update season / day tracking from file metadata
        self._current_season = data.get("season", self._current_season)
        self._day_offset     = data.get("final_day", self._day_offset)

        logger.info(
            "Farm %r: soil state loaded from %s (season=%d, day_offset=%d).",
            self.name, path, self._current_season, self._day_offset,
        )

    def prepare_next_season(self) -> None:
        """Advance to the next growing season: reset plants, preserve soil.

        PRD v0.4.0 Section 7.3 -- Multi-Season Carry-over.

        Exactly what carries over vs resets:

        CARRIES OVER (soil physical state):
            - ``SoilVoxelState.moisture_pct``
            - ``SoilVoxelState.nitrogen_kg_ha``
            - ``SoilVoxelState.bulk_density``
            - ``SoilVoxelState.penetration_resistance``
            - ``SoilVoxelState.custom`` (all entries)
            - Physics engine parameters (field_capacity_pct, wilting_point_pct, etc.)

        RESETS (plant / season state):
            - All ``PlantState`` objects (new sowing — biomass=0, lai=0, stress=0).
            - ``FieldState.day`` → reset to 1 for the new season's internal counter.
            - Events: NOT automatically re-registered. The researcher re-registers
              events for Season 2 using the same @farm.event syntax.

        Day numbering design choice:
            The new season uses *continuous* day numbers in the Parquet log
            (if Season 1 ran for 120 days, Season 2 Parquet rows start at day 121).
            This is implemented via ``_day_offset``.  The ``FieldState.day`` that
            step functions see is the within-season day (starting at 1), while
            the Parquet ``day`` column is ``FieldState.day + _day_offset``.
            This makes multi-season Parquet datasets trivially joinable on a
            continuous time axis.

        Usage
        -----
        >>> farm.run(days=120)           # Season 1
        >>> farm.save_state("s1.cfstate")
        >>> farm.prepare_next_season()   # Plants reset, soil carries over
        >>> farm.run(days=120)           # Season 2 (Parquet days 121-240)
        """
        if self._is_running:
            raise CropForgeConfigError(
                "prepare_next_season() cannot be called while farm.run() is executing. "
                "Wait for the current run to complete before starting the next season."
            )


        # Advance season counter and save the total days run so far as the offset
        self._current_season += 1

        # _day_offset is updated at the END of the previous run() by _execute_run.
        # If the user calls prepare_next_season() without running first, the offset
        # stays at 0 (harmless -- Season 2 day 1 is still Parquet day 1).

        # Reset plant grids for all attached fields -- NEW SOWING
        # Soil is left completely untouched (the voxel objects are reused as-is).
        from cropforge.state import PlantState as _PS

        for f in self._fields:
            if f._field_state is None:
                # Field never ran -- nothing to reset
                continue

            # Rebuild plant list from scratch (same grid positions, all state = default)
            f._field_state.plants = [
                _PS(
                    plant_id=f"r{r:02d}c{c:02d}",
                    row=r,
                    col=c,
                    # All other fields take dataclass defaults:
                    #   age_days=0, lai=0.0, biomass_g=0.0, height_cm=0.0,
                    #   root_depth_cm=0.0, stress_index=0.0, alive=True,
                    #   phenological_stage="germination", root_growth_multiplier=1.0
                )
                for r in range(f.rows)
                for c in range(f.cols)
            ]

            # Reset the within-season day counter
            f._field_state.day = 1

            # Clear events_fired from the previous season
            f._field_state.events_fired = []

        # Clear the farm-level event list so the researcher re-registers events
        self._events = []

        logger.info(
            "Farm %r: prepare_next_season() → season=%d, day_offset=%d. "
            "Plants reset. Soil state preserved across all %d fields.",
            self.name, self._current_season, self._day_offset, len(self._fields),
        )


    # ------------------------------------------------------------------
    # farm.visualize() (Section 6.5 — pre-flight stub)
    # ------------------------------------------------------------------

    def visualize(self, log: Optional[str] = None, quality: str = "standard") -> None:
        """Launch the visual frontend.

        Parameters
        ----------
        log:
            Explicit path to a Parquet session directory.  If ``None``,
            the most recent ``farm.run()`` log is used.
        quality:
            ``"standard"`` (default) — identical rendering to v0.8.0, no shadows.
            ``"enhanced"`` — PBR shadows on terrain and plants (higher GPU cost).

        Raises
        ------
        CropForgeVisualizeError
            If no valid Parquet log is found.
        """
        import json
        from pathlib import Path

        from cropforge.runtime import CropForgeVisualizeError

        # ---- Rule 1: Resolve log path ---------------------------------
        resolved_log: Optional[str] = log or self._last_log_path

        # ---- Rule 2: Validate the path --------------------------------
        _NO_LOG_MSG = (
            "No valid simulation log found. Run farm.run() before calling "
            "farm.visualize(), or pass an explicit log path via "
            "farm.visualize(log=path)."
        )

        if not resolved_log:
            raise CropForgeVisualizeError(_NO_LOG_MSG)

        log_dir = Path(resolved_log)
        if not log_dir.exists():
            raise CropForgeVisualizeError(
                f"{_NO_LOG_MSG}\n  (Path checked: {resolved_log})"
            )

        # Log directory must contain at least one Parquet file
        parquet_files = list(log_dir.rglob("*.parquet"))
        if not parquet_files:
            raise CropForgeVisualizeError(
                f"{_NO_LOG_MSG}\n  (Directory exists but contains no Parquet files: "
                f"{resolved_log})"
            )

        # ---- Rule 3: Version mismatch check ---------------------------
        from cropforge import __version__
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(parquet_files[0]).metadata
            file_version = meta.get(b"cropforge_version", b"unknown").decode()
            if file_version != __version__ and file_version != "unknown":
                import warnings
                warnings.warn(
                    f"[CropForge] Version mismatch: this log was produced by "
                    f"CropForge {file_version}, but you are running "
                    f"CropForge {__version__}. Visualisation will proceed but "
                    "results may differ. Re-run farm.run() to update the log.",
                    UserWarning,
                    stacklevel=2,
                )
                logger.warning(
                    "Parquet log version mismatch: log=%s, runtime=%s. "
                    "Proceeding with visualisation.",
                    file_version,
                    __version__,
                )
        except Exception:
            # Non-fatal: version check is best-effort
            pass

        # ---- Launch the server ----------------------------------------
        from cropforge.viz.server import boot
        boot(log_path=str(log_dir.resolve()), cropforge_version=__version__, quality=quality)

    # ------------------------------------------------------------------
    # farm.export_scene() — headless GLTF/GLB export (PRD v0.9.0 §5)
    # ------------------------------------------------------------------

    def export_scene(
        self,
        day: int,
        filepath: str = "scene.glb",
        field: Optional[str] = None,
    ) -> "Path":
        """Export the farm scene for *day* to a .glb file.

        Parameters
        ----------
        day:
            Simulation day to export (must exist in the run log).
        filepath:
            Destination path for the .glb file. Defaults to ``scene.glb``
            in the current working directory.
        field:
            Field name to export. Defaults to the first field.

        Returns
        -------
        pathlib.Path
            Resolved path of the written file.

        Raises
        ------
        CropForgeVisualizeError
            If no valid Parquet log is found.
        ImportError
            If ``pygltflib`` is not installed.

        Example
        -------
        >>> farm.run(days=30)
        >>> farm.export_scene(day=15, filepath="out/day15.glb")
        """
        from pathlib import Path as _Path
        from cropforge.runtime import CropForgeVisualizeError
        from cropforge.export_gltf import export_scene as _export

        resolved_log = self._last_log_path
        if not resolved_log or not _Path(resolved_log).exists():
            raise CropForgeVisualizeError(
                "No valid simulation log found. Run farm.run() before export_scene()."
            )

        return _export(log_path=str(resolved_log), day=day, filepath=filepath, field=field)



    def __repr__(self) -> str:
        return (
            f"Farm(name={self.name!r}, location={self.location}, "
            f"fields={[f.name for f in self._fields]})"
        )
