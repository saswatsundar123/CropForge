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

        self._field_state = FieldState(
            day=day,
            plants=plants,
            soil=soil,
            elevation_grid=self.elevation_grid.copy(),
            events_fired=[],
        )
        # Apply water balance parameters to all voxels if configured
        self._apply_water_params_to_state(self._field_state)
        self._apply_nitrogen_params_to_state(self._field_state)
        return self._field_state

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

    def _sorted_steps(self) -> List[Tuple[int, Callable]]:
        """Return all step functions sorted by phase (ascending).

        Merges researcher-registered steps (_step_registry, phase >= 0)
        with built-in physics hooks (_physics_registry, phase < 0).
        The negative-phase physics hooks are always guaranteed to appear
        before any researcher steps in the sorted output.

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
        all_steps = self._physics_registry + self._step_registry
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
        elevation_m: float = 0.0,
        anemometer_height_m: float = 2.0,
    ) -> None:
        """Enable opt-in physics engines for this farm (PRD v0.2.0 / v0.3.0).

        Execution order when all engines are enabled:
            phase=-4  Lateral flow + nitrogen transport
            phase=-3  Soil water balance engine
            phase=-2  ET0 engine (Penman-Monteith)
            phase=-1  Root impedance engine
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
        elevation_m:
            Site elevation above mean sea level (m). Used by ET0 engine.
        anemometer_height_m:
            Anemometer height (m) for wind height correction. Default 2.0 m.

        Raises
        ------
        CropForgeConfigError
            - If ``water_balance=True`` and ``et0=False``.
            - If ``nutrients=True`` or ``lateral_flow=True`` and
              ``water_balance=False``.

        Notes
        -----
        PRD v0.2.0 / v0.3.0 backward compatibility:
            If ``use_physics()`` is never called, simulation behaves identically
            to v0.1.0 -- no engine hooks are registered.
        """
        from cropforge.physics.builtin_hooks import (
            PHASE_ET0_ENGINE,
            PHASE_ROOT_ENGINE,
            PHASE_HYDROLOGY_ENGINE,
            PHASE_NUTRIENTS_ENGINE,
            make_et0_hook,
            make_root_impedance_hook,
            make_hydrology_hook,
            make_nutrients_hook,
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

        # Record configuration for introspection
        self._physics_config.update({
            "et0": et0,
            "root_impedance": root_impedance,
            "water_balance": water_balance,
            "nutrients": nutrients,
            "lateral_flow": lateral_flow,
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
            hook = make_hydrology_hook()
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

        if not et0 and not root_impedance and not water_balance and not nutrients and not lateral_flow:
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
    # farm.visualize() (Section 6.5 — pre-flight stub)
    # ------------------------------------------------------------------

    def visualize(self, log: Optional[str] = None) -> None:
        """Launch the visual frontend (Phase 2).

        Performs the pre-flight check (PRD Section 6.5), then starts the
        FastAPI + Dash server and opens the default browser to
        ``http://localhost:7860``.

        Parameters
        ----------
        log:
            Explicit path to a Parquet session directory.  If ``None``,
            the log from the most recent ``farm.run()`` call in this session
            is used (stored in ``self._last_log_path``).

        Raises
        ------
        CropForgeVisualizeError
            If no valid Parquet log is found (PRD Section 6.5, rule 2).

        Notes
        -----
        PRD Section 6.5 pre-flight check:
            1. Locate the log: explicit path > ``_last_log_path``.
            2. If not found or empty → ``CropForgeVisualizeError``.
            3. If version mismatch → warning printed, visualisation proceeds.
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
        boot(log_path=str(log_dir.resolve()), cropforge_version=__version__)


    def __repr__(self) -> str:
        return (
            f"Farm(name={self.name!r}, location={self.location}, "
            f"fields={[f.name for f in self._fields]})"
        )
