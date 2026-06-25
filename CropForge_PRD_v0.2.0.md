

**CROPFORGE**

*Virtual Farm Runtime for Agricultural Researchers*

**PRODUCT REQUIREMENTS DOCUMENT**

Version 0.2.0 — Soil Physics, Environmental Engine & Multi-Field

Builds on: CropForge v0.1.0 PRD | Status: Planned

Prepared by: Saswat Sundar Rath

ICAR-IARI Jharkhand  |  June 2026

---

# **1. Purpose and Scope of This Document**

This document specifies the requirements for CropForge v0.2.0. It is written assuming full completion of v0.1.0 as described in the v0.1.0 PRD. All architectural decisions, data schemas, and API conventions established in v0.1.0 remain binding unless explicitly superseded here.

v0.2.0 has a single governing objective: make the maize dual-plot scenario (undulating terrain with nitrogen deficit and wind vs. flat land with hard pan and soil clods) fully simulable without the researcher coding the soil physics themselves. In v0.1.0, a researcher could model this scenario only if they implemented lateral water flow, root impedance, and Penman-Monteith ET₀ in their own step function. v0.2.0 makes these built-in primitives available as optional engine-level services.

> ****Design Principle****: *v0.2.0 adds built-in physics primitives. These are opt-in. Researchers who prefer to code their own soil physics continue to use @farm.step exactly as in v0.1.0. Nothing breaks. Nothing is removed.*

| **Attribute** | **Value** |
| --- | --- |
| Version | 0.2.0 |
| Builds on | CropForge v0.1.0 (all v0.1 tests must still pass) |
| API contract | Fully backward-compatible. No breaking changes. |
| Target release | Q4 2026 |
| Test requirement | 230 existing tests pass + all new tests for v0.2 subsystems |

---

# **2. v0.1.0 Pre-Release Items (Must Ship Before v0.2 Work Begins)**

> [!NOTE]
> ****⚠  NOTE****: *These three items were identified in the v0.1.0 audit as incomplete. v0.2.0 work must not begin until all three are delivered and v0.1.0 is tagged and published on PyPI.*

## **2.1 maize_dual_plot.py Example**

This is the flagship scientific example and must ship with v0.1.0. It is the proof of concept that will be shown to the agricultural scientists who reviewed the project.

The example must demonstrate:

* Two Field objects instantiated on the same Farm — Plot A (undulating, N-deficit, 3.8 m/s wind) and Plot B (flat, hard pan at 19 cm, high soil clods, 0.93 m/s wind)
* Irrigation scheduled at 15-day intervals on both plots using the built-in Event.irrigation()
* A researcher-coded @farm.step function that implements basic biomass accumulation, water stress, and a PWP death check
* A clearly visible divergence in plant survival between the two plots visible in the 3D view and Plotly dashboard
* Inline comments explaining every model decision, so it functions as a teaching document

> [!NOTE]
> ****⚠  NOTE****: *In v0.1.0, the researcher must code lateral water flow and hard-pan root impedance themselves if they want the full physical accuracy. The maize_dual_plot.py example must acknowledge this in comments and show a simplified version that still demonstrates meaningful plot divergence through the PWP death logic alone.*

## **2.2 API Reference Documentation (7 Pages)**

All seven API reference pages must be written with actual content before v0.1.0 is tagged. Placeholder files are not acceptable for a public release.

| **Doc Page** | **Required Content** |
| --- | --- |
| docs/reference/farm.md | Farm, Field, Crop class signatures, all constructor arguments, return types, exceptions raised |
| docs/reference/state.md | All four dataclass schemas (PlantState, SoilVoxelState, FieldState, EnvironmentState) with field descriptions and units |
| docs/reference/loaders.md | Weather.from_csv and Soil.from_csv with all keyword arguments, expected column names, unit conversion table |
| docs/reference/events.md | Event.irrigation, Event.fertiliser, Event.custom — arguments, timing behaviour, interaction with @step phase system |
| docs/reference/runtime.md | @farm.step decorator — phase argument, execution order, error contract, CropForgeStepError |
| docs/reference/logger.md | Parquet schema (all three tables), partitioning structure, custom_json column format, how to read logs externally |
| docs/reference/viz.md | farm.visualize() — launch behaviour, port configuration, memory ceiling, Panel descriptions, postMessage bridge |

## **2.3 Memory Ceiling Documentation**

The audit identified that the BufferStore holds all timestep buffers in browser memory. Before public release, the documentation must explicitly state:

* Approximate memory usage formula: (num_plants × 36 bytes × num_days) loaded into browser
* Practical ceiling for a standard laptop (8 GB RAM browser tab limit ~2 GB): approximately 15 million plant-days
* For a 40×60 grid over 90 days: 40 × 60 × 90 × 36 = ~7.8 MB — well within limits
* For a 100×100 grid over 365 days: 100 × 100 × 365 × 36 = ~131 MB — document as a known-large session
* Streaming fallback is a v0.3 feature; v0.1 is memory-bound by design and this must be stated

---

# **3. v0.2.0 Feature Overview**

v0.2.0 introduces five new subsystems. Each is independent and can be implemented in parallel. Each is opt-in.

| **Subsystem** | **What It Provides** | **Who Benefits** | **Complexity** |
| --- | --- | --- | --- |
| Penman-Monteith ET₀ Engine | Built-in daily ET₀ calculation from weather data. Replaces researcher need to compute ET₀ in their step function. | Any researcher modelling water balance | Medium |
| Root Growth Engine | Built-in daily root depth extension with soil layer resistance. Hard pan constraint built in. | Agronomists studying soil-root interaction | Medium |
| Soil Water Balance | Built-in daily soil moisture update accounting for rainfall, irrigation, ET₀, and drainage. Per-layer. | Any researcher modelling irrigation or drought | High |
| Lateral Water Flow | D8-based downslope water redistribution on each timestep. Requires elevation grid. | Researchers with topographically complex fields | High |
| Multi-Field Side-by-Side View | 3D viewport split to show two fields simultaneously with synchronised scrubber. | Comparative experiment researchers | Medium |

---

# **4. Subsystem 1 — Penman-Monteith ET₀ Engine**

## **4.1 Scientific Basis**

The FAO-56 Penman-Monteith equation is the international standard for computing reference evapotranspiration. It is used by DSSAT, APSIM, and every major irrigation scheduling system globally. Implementing it as a built-in service means researchers no longer need to code it themselves, and results are directly comparable to published literature that uses the same equation.

The equation computes ET₀ (mm/day) from: maximum and minimum temperature, solar radiation, wind speed, humidity, and site latitude/elevation. All of these are already present in EnvironmentState and the Farm definition in v0.1.0.

## **4.2 Implementation Specification**

```python
# How the researcher enables it — one line added to farm definition
farm = Farm(
    name="Trial 2026-A",
    location=(23.4, 85.3),
    et0_engine="penman-monteith"   # NEW in v0.2.0 — default is None
)
 
# When enabled, EnvironmentState.et0_mm is automatically populated
# each day by the engine BEFORE @farm.step functions run.
# The researcher reads it like any other env variable:
 
@farm.step(interval="daily")
def water_balance(state, env):
    for plant in state.plants:
        soil = state.soil[plant.row][plant.col][0]
        # env.et0_mm is now a real computed value, not 0.0
        soil.moisture_pct -= env.et0_mm * crop_coefficient(plant.phenological_stage)
    return state
```

## **4.3 Internal Computation**

The engine computes ET₀ once per day, before any @farm.step functions run. It uses:

* Net radiation (Rn) — derived from solar radiation, albedo (fixed at 0.23 for reference surface), and long-wave back-radiation using temperature
* Soil heat flux (G) — approximated as 0 for daily timestep per FAO-56
* Psychrometric constant (γ) — computed from site elevation stored in Farm.location
* Slope of saturation vapour pressure curve (Δ) — computed from mean temperature
* Vapour pressure deficit — computed from Tmax, Tmin, and humidity
* Wind speed at 2m height — converted from input height if researcher specifies measurement height

## **4.4 Unit Handling**

> [!NOTE]
> ****⚠  NOTE****: *v0.1.0 normalised wind speed input to m/s internally. The ET₀ engine requires wind speed at 2 m height. If the researcher's weather station measures at a different height, they specify anemometer_height_m in Weather.from_csv() and the loader applies the logarithmic wind profile correction automatically.*

## **4.5 Tests Required**

* FAO-56 Appendix I example data (Etzion, Israel) — computed ET₀ must match published value of 3.95 mm/day ± 0.05
* ET₀ is zero when radiation is zero (night check)
* EnvironmentState.et0_mm remains 0.0 when et0_engine=None (backward compatibility)
* Wind speed height correction applies correctly for anemometer_height_m values of 2, 5, and 10 m

---

# **5. Subsystem 2 — Root Growth Engine**

## **5.1 Scientific Basis**

Root depth extension in most crops follows a thermal-time-based model — roots extend at a rate proportional to accumulated heat units above a base temperature, modulated by soil mechanical resistance. The hard pan scenario in the maize experiment is the canonical use case: when the root front reaches a layer with penetration resistance above a threshold (typically 2–3 MPa), vertical extension stops and horizontal proliferation begins.

## **5.2 Implementation Specification**

```python
# Enabled per-field
field_b.set_root_engine(
    base_temp_c=8.0,           # Below this, no root extension (maize default)
    max_extension_cm_day=2.5,  # Maximum daily root depth increase
    resistance_threshold_mpa=2.0  # Hard pan triggers above this value
)
 
# PlantState.root_depth_cm is updated automatically each day
# BEFORE @farm.step functions run (same priority as ET₀).
# Researcher can read it and override it in their step function.
 
@farm.step(interval="daily")
def check_root_stress(state, env):
    for plant in state.plants:
        # Engine has already updated root_depth_cm today
        # Researcher can inspect which layer roots are in:
        active_layer = get_layer_at_depth(
            state.soil[plant.row][plant.col],
            plant.root_depth_cm
        )
        if active_layer.penetration_resistance > 2.0:
            # Roots are blocked — can apply horizontal stress modifier
            plant.custom['root_blocked'] = True
    return state
```

## **5.3 Hard Pan Behaviour**

When a plant's root front encounters a SoilVoxelState where penetration_resistance exceeds the threshold:

* Vertical root extension rate is clamped to zero for that plant
* plant.custom['root_blocked'] is set to True by the engine (readable in step functions)
* An entry is written to the Event Log: 'Plant {id} root blocked at {depth} cm on day {d}'
* The Farm Inspector soil cross-section chart visually marks the blocked layer in amber

## **5.4 Soil Layer Helper Function**

A utility function is added to the public API to avoid boilerplate in researcher step functions:

```python
from cropforge.utils import get_layer_at_depth, get_layers_in_range
 
# Returns the SoilVoxelState at a given depth for a given cell
layer = get_layer_at_depth(state.soil[row][col], depth_cm=25.0)
 
# Returns all layers within a depth range (for root zone averaging)
root_zone_layers = get_layers_in_range(state.soil[row][col], 0, plant.root_depth_cm)
```

## **5.5 Tests Required**

* Root depth increases by thermal-time-proportional amount each day above base temp
* Root depth does not increase when daily temp_mean < base_temp_c
* Root depth clamps at hard pan layer; plant.custom['root_blocked'] set correctly
* Event Log entry written exactly once per plant when block first occurs
* get_layer_at_depth returns correct layer for boundary depths (exactly at layer boundary)
* Root engine disabled: PlantState.root_depth_cm stays 0.0 (backward compatibility)

---

# **6. Subsystem 3 — Soil Water Balance**

## **6.1 Scientific Basis**

The FAO-56 two-stage soil water depletion model is the standard for daily water balance in crop simulation. It tracks total available water in the root zone across soil layers, computes actual evapotranspiration (ETc) as the crop-coefficient-adjusted ET₀, subtracts ETc from root-zone layers (weighted by root density per layer), adds rainfall and irrigation, and drains excess water below the field capacity of each layer downward.

> [!NOTE]
> ****⚠  NOTE****: *The soil water balance subsystem depends on the ET₀ engine. If soil_water_balance=True is set, et0_engine='penman-monteith' is automatically enabled. The researcher does not need to enable them separately.*

## **6.2 Implementation Specification**

```python
field_a.set_soil_water_balance(
    field_capacity_pct=32.0,      # Volumetric water content at FC (%)
    wilting_point_pct=14.0,       # Permanent wilting point (%)
    drainage_coefficient=0.5,     # Fraction of excess water draining per day
    crop_coefficient_fn=None      # Optional: fn(phenological_stage) -> Kc
                                  # If None, Kc = 1.0 (reference surface)
)
 
# SoilVoxelState.moisture_pct is updated automatically each day:
#   + rainfall_mm / (layer_thickness_cm * 10) per layer (surface only)
#   + irrigation_mm on event days (surface layer)
#   - ETc (distributed across root zone layers by root fraction)
#   - drainage to next layer if moisture > field_capacity_pct
# Researcher reads the updated values in their @farm.step
```

## **6.3 Root-Zone Moisture Extraction**

Water extraction from soil is distributed across layers in proportion to root length density per layer. In v0.2.0 a simplified uniform distribution is used within the root zone (total root depth / number of layers occupied). A non-uniform distribution weighted by root age per layer is deferred to v0.3.0.

## **6.4 Interaction with Event System**

When Event.irrigation fires, the irrigation amount (mm) is added to the surface soil layer (layer 0) moisture on that day, before ET₀ extraction. This is already how v0.1.0 events work — the soil water balance engine simply reads and writes the same SoilVoxelState fields the researcher already accesses.

## **6.5 Tests Required**

* Moisture decreases daily by ETc when no rainfall or irrigation
* Moisture increases by correct mm amount on irrigation days
* Excess moisture above field capacity drains to next layer at drainage_coefficient rate
* Moisture does not drop below 0.0 (floor enforced)
* Moisture does not exceed soil porosity (ceiling enforced)
* When root_engine is also enabled, extraction is distributed correctly across active layers
* Soil water balance disabled: SoilVoxelState.moisture_pct unchanged by engine (backward compatibility)

---

# **7. Subsystem 4 — Lateral Water Flow (D8 Algorithm)**

## **7.1 Scientific Basis**

On any non-flat terrain, gravity drives water laterally from high elevation cells to lower ones after rainfall or irrigation events. The D8 (Deterministic Eight-neighbour) algorithm assigns each grid cell a single downslope flow direction to one of its eight neighbours based on the steepest descent in the elevation grid. Water in excess of field capacity flows in that direction at a rate proportional to slope.

This is the mechanism that creates the nutrient and moisture accumulation in valley cells observed in the maize Plot A scenario, and the corresponding depletion on slope crests. Without this subsystem, Plot A behaves identically to flat terrain in the simulation.

## **7.2 Prerequisite**

Lateral water flow requires an elevation grid. It is automatically disabled for fields where set_elevation() has not been called. A warning is written to the Event Log if lateral_flow=True is set but no elevation grid exists.

## **7.3 Implementation Specification**

```python
field_a.set_lateral_flow(
    enabled=True,
    flow_fraction=0.3,      # Fraction of excess moisture that flows laterally per day
                            # (remainder stays in cell or drains vertically)
    n_transport=True        # Whether mobile N (nitrate) also flows with water
)
 
# Execution order within a daily timestep:
# 1. ET₀ computed (if enabled)
# 2. Root depth updated (if enabled)
# 3. Soil water balance updated (rainfall + irrigation - ETc - vertical drainage)
# 4. Lateral flow: excess water redistributed across grid (this subsystem)
# 5. @farm.step functions run (researcher reads updated values)
```

## **7.4 D8 Flow Direction Computation**

The D8 flow direction grid is computed once at simulation initialisation from the elevation array and cached. It does not change during the simulation. The computation:

1. For each cell, compute slope to each of 8 neighbours: slope = (elevation[cell] - elevation[neighbour]) / cell_distance
1. Assign flow direction to the neighbour with steepest positive slope
1. Cells with no downslope neighbour (sinks or flat) retain water
1. Cell distance for cardinal neighbours = grid_resolution_m; diagonal neighbours = grid_resolution_m × √2

## **7.5 Nitrogen Transport**

When n_transport=True, mobile nitrogen (nitrate-N in SoilVoxelState.nitrogen_kg_ha) flows with water in proportion to the water flow fraction. Organic N (immobile) does not move. This is the mechanism responsible for N accumulation in valley cells and N depletion on slope crests in Plot A.

## **7.6 Tests Required**

* Water flows from high elevation cell to lowest neighbour, not to all neighbours
* Water accumulates in sink cells (no downslope neighbour)
* Flow is zero when moisture is below field capacity (no excess)
* N transport moves N in proportion to water flow fraction
* Total water + N conserved across the grid (no creation or destruction)
* Lateral flow disabled for flat fields (elevation array all zeros): moisture unchanged
* Warning written to Event Log when lateral_flow=True but no elevation grid set

---

# **8. Subsystem 5 — Multi-Field Side-by-Side 3D View**

## **8.1 Purpose**

When a farm has two or more fields, the researcher needs to compare them visually in real time. The current v0.1.0 frontend renders one field at a time. v0.2.0 introduces a split-viewport mode that places two Three.js canvases side by side with a synchronised timeline scrubber.

## **8.2 Activation**

```python
# No API change required. If farm has 2+ fields, farm.visualize() automatically
# opens in split-viewport mode.
 
# Researcher can override:
farm.visualize(layout="split")     # Force split (2 fields)
farm.visualize(layout="single")    # Force single (dropdown to select field)
farm.visualize(layout="auto")      # Default: split if 2 fields, single if 1
```

## **8.3 Frontend Specification**

| **Element** | **Specification** |
| --- | --- |
| Left viewport | First field added to farm (farm.fields[0]). Labelled with field name at top. |
| Right viewport | Second field (farm.fields[1]). Labelled. If farm has 3+ fields, a dropdown selects which field appears right. |
| Scrubber | Single timeline scrubber below both viewports. Both canvases advance together. One play/pause control. |
| Variable selector | Each viewport has its own colour-mapping variable dropdown. Researcher can compare biomass (left) vs stress_index (right) simultaneously. |
| Click-to-inspect | Clicking a plant in either viewport opens Farm Inspector for that plant. Inspector header shows which field it belongs to. |
| Metrics dashboard | Panel 2 shows both fields on the same Plotly chart with distinct line colours and field-name legend entries. |

## **8.4 Performance Requirement**

Both viewports must scrub at full speed simultaneously on a machine with a standard integrated GPU (Intel Iris Xe or equivalent). This is achievable because both viewports read from the same preloaded buffer store — there is no additional data loading cost for the second viewport.

## **8.5 Tests Required**

* farm.visualize() opens split layout when farm has 2 fields
* Single scrubber advances both canvases to the same day
* Each viewport maintains independent variable selection
* Farm Inspector correctly identifies field name of clicked plant
* Metrics dashboard renders distinct lines for both fields

---

# **9. Full Daily Timestep Execution Order**

With all v0.2.0 subsystems enabled, the daily timestep executes in the following strict order. This order is not configurable by the researcher in v0.2.0.

| **Step** | **Action** |
| --- | --- |
| 1. Environment resolve | Weather data for this day loaded into EnvironmentState. All existing v0.1.0 fields populated. |
| 2. ET₀ computation | If et0_engine enabled: Penman-Monteith computed. EnvironmentState.et0_mm updated. |
| 3. Root extension | If root_engine enabled: PlantState.root_depth_cm updated for all living plants. Hard pan check and event log entry written. |
| 4. Soil water balance | If soil_water_balance enabled: rainfall and irrigation added; ETc extracted per layer; vertical drainage applied. |
| 5. Lateral water flow | If lateral_flow enabled: excess water (and N if configured) redistributed across grid via D8. |
| 6. @farm.step functions | All researcher-registered step functions run in phase order. They read the fully updated state from steps 1–5. |
| 7. Event resolution | Events scheduled for this day fire (e.g. fertiliser application). Events that modify soil state run after researcher step functions. |
| 8. State logging | Full FieldState, all PlantState, all SoilVoxelState, EnvironmentState serialised to Parquet. |

> ****Why This Order****: *Steps 1–5 ensure that when the researcher's step function runs, all physical environment values are already updated and correct. The researcher's model receives computed, not raw, inputs. This mirrors how real crops experience their environment.*

---

# **10. Backward Compatibility Requirements**

v0.2.0 must not break any simulation written for v0.1.0. This is a hard requirement, not a goal.

| **Scenario** | **Required Behaviour** |
| --- | --- |
| v0.1.0 script run on v0.2.0 with no changes | Runs identically. All new subsystems default to disabled. EnvironmentState.et0_mm remains 0.0. PlantState.root_depth_cm remains 0.0 unless researcher sets it in their step function. |
| v0.1.0 script that manually computes ET₀ in @farm.step | Continues to work. If researcher also enables et0_engine, they will get double-counting — document this clearly and raise a warning if et0_mm is written in a step function while et0_engine is enabled. |
| wheat_basic.py from v0.1.0 | Must run with zero changes on v0.2.0 and produce identical output. |
| All 230 v0.1.0 tests | Must pass without modification on v0.2.0 codebase. |

---

# **11. Schema Changes to SimulationState**

v0.2.0 makes minimal additions to the SimulationState schema. No existing fields are removed or renamed.

## **11.1 Additions to FieldState**

```python
@dataclass
class FieldState:
    # ... all v0.1.0 fields unchanged ...
    
    # NEW in v0.2.0
    d8_flow_grid: Optional[np.ndarray] = None   # Cached at init, None if no elevation
    active_engines: List[str] = field(default_factory=list)
    # e.g. ["et0:penman-monteith", "root-growth", "soil-water-balance", "lateral-flow"]
```

## **11.2 Additions to SoilVoxelState**

```python
@dataclass
class SoilVoxelState:
    # ... all v0.1.0 fields unchanged ...
    
    # NEW in v0.2.0
    field_capacity_pct: float = 0.0    # Set from set_soil_water_balance()
    wilting_point_pct: float = 0.0     # Set from set_soil_water_balance()
    drainage_mm_today: float = 0.0     # Computed daily by soil water balance
    lateral_inflow_mm: float = 0.0     # Computed daily by lateral flow engine
```

## **11.3 Additions to PlantState**

```python
@dataclass  
class PlantState:
    # ... all v0.1.0 fields unchanged ...
    
    # NEW in v0.2.0 (populated by root engine if enabled, else stays 0.0)
    root_depth_cm: float = 0.0         # Already existed — now engine-managed
```

> [!NOTE]
> ****⚠  NOTE****: *root_depth_cm existed in v0.1.0 PRD but was researcher-managed. In v0.2.0 it becomes engine-managed when root_engine is enabled. If both the engine and the researcher write to root_depth_cm in the same timestep, the engine value applies first (step 3) and the researcher overwrites it (step 6). This is by design — the researcher always has final authority.*

---

# **12. Parquet Log Changes**

The Parquet log gains new columns in all three tables. The cropforge_version field in file metadata changes to '0.2.0'. Old v0.1.0 logs remain readable by v0.2.0's visualiser — missing columns are filled with None.

| **Table** | **New Columns** | **Type** |
| --- | --- | --- |
| plants | root_depth_cm | float32 |
| soil | field_capacity_pct, wilting_point_pct, drainage_mm_today, lateral_inflow_mm | float32 each |
| environment | et0_mm (already existed as 0.0 in v0.1), active_engines | float32, string |

---

# **13. Build Phases**

> [!NOTE]
> ****⚠  NOTE****: *Phase 0 must be completed before any v0.2.0 work begins. It is the v0.1.0 completion work identified in the audit.*

## **Phase 0 — Complete v0.1.0 (Pre-condition)**

1. Write maize_dual_plot.py example with inline comments — researcher-coded physics, meaningful plot divergence
1. Write all 7 API reference documentation pages with substantive content
1. Add memory ceiling documentation to docs/reference/viz.md
1. Run full test suite: all 230 tests pass
1. git tag v0.1.0 && git push --tags — triggers PyPI auto-publish
1. Enable GitHub Discussions

## **Phase 1 — ET₀ Engine (2 weeks)**

1. Implement cropforge/engines/et0.py — FAO-56 Penman-Monteith
1. Integrate into runtime.py daily timestep as step 2 (before @farm.step)
1. Add et0_engine parameter to Farm constructor
1. Add anemometer_height_m parameter to Weather.from_csv()
1. Write all tests specified in §4.5
1. Verify: wheat_basic.py output unchanged when et0_engine=None

## **Phase 2 — Root Growth Engine (2 weeks)**

1. Implement cropforge/engines/roots.py — thermal-time root extension with resistance check
1. Implement get_layer_at_depth() and get_layers_in_range() in cropforge/utils.py
1. Integrate into runtime.py as step 3
1. Add hard pan event log entry and Farm Inspector amber layer marker
1. Write all tests specified in §5.5

## **Phase 3 — Soil Water Balance (3 weeks)**

1. Implement cropforge/engines/soil_water.py — FAO-56 two-stage depletion
1. Add field_capacity_pct and wilting_point_pct to SoilVoxelState
1. Integrate into runtime.py as step 4
1. Add set_soil_water_balance() method to Field
1. Write all tests specified in §6.5
1. Verify: v0.1.0 soil moisture behaviour unchanged when soil_water_balance=False

## **Phase 4 — Lateral Water Flow (3 weeks)**

1. Implement cropforge/engines/lateral_flow.py — D8 flow direction grid + daily redistribution
1. Implement N transport with water
1. Integrate into runtime.py as step 5
1. Add set_lateral_flow() method to Field
1. Write all tests specified in §7.6
1. Write maize_dual_plot_v2.py — full physics version using all four engines

## **Phase 5 — Multi-Field Frontend (2 weeks)**

1. Implement split-viewport layout in viz/static/ Three.js
1. Implement shared scrubber controlling both canvases
1. Implement per-viewport variable selector
1. Update Plotly dashboard (Panel 2) to render both fields with legend
1. Write frontend tests specified in §8.5

## **Phase 6 — Integration and Release (1 week)**

1. Run full test suite: 230 v0.1.0 tests + all new v0.2.0 tests — all must pass
1. Run wheat_basic.py from v0.1.0 without changes — output must be identical to v0.1.0
1. Run maize_dual_plot.py from v0.1.0 without changes — output must be identical to v0.1.0
1. Run maize_dual_plot_v2.py with all engines enabled — verify correct physical behaviour
1. Update CHANGELOG.md with all new features
1. Update all documentation to cover new subsystem APIs
1. git tag v0.2.0 && git push --tags

---

# **14. Testing Strategy**

## **14.1 Unit Tests (per subsystem)**

Each engine has its own test file under tests/engines/. All physics functions are tested with known inputs and expected outputs from published literature or first principles.

## **14.2 Integration Tests**

Two integration test scenarios run the full farm.run() pipeline with all engines enabled and verify final state against known correct values:

* Scenario A: Flat field, uniform soil, uniform weather, 30 days — verify moisture drawdown matches manual FAO-56 calculation
* Scenario B: 2% slope, N-deficit top cells, 90 days — verify that bottom cells have higher moisture and N than top cells after 90 days of irrigation

## **14.3 Regression Tests**

wheat_basic.py is run as a regression test in CI on every push. Its final Parquet log is compared against a stored reference output. Any difference in any column value fails the build.

## **14.4 Backward Compatibility Test**

A dedicated test loads a v0.1.0 Parquet log (stored as a test fixture) into the v0.2.0 visualiser and verifies it renders without error and all columns display correctly.

---

# **15. Items Explicitly Deferred to v0.3.0**

| **Feature** | **Reason for Deferral** |
| --- | --- |
| Non-uniform root density distribution by layer | Requires root architecture model; uniform distribution in v0.2 covers 90% of use cases |
| Capillary rise (upward water movement) | Uncommon in field crops under irrigation; deferred pending researcher demand |
| Runoff (water leaving the field boundary) | D8 routes water to sinks within the grid; boundary runoff deferred to v0.3 |
| Multi-field lateral flow (flow between fields) | Different fields may have different soil schemas; cross-field flow requires schema unification |
| 3D voxel soil cross-section view | Frontend feature deferred; 2D chart in Farm Inspector covers scientific need |
| Plugin API | Stabilising the engine API in v0.2.0 before exposing it to third-party plugins |
| Streaming Parquet playback | Memory ceiling adequate for current use cases; streaming deferred to v0.3 |
| R bindings | Deferred pending researcher demand; priority queue managed via GitHub Discussions |
| cropforge.examples module | After two examples are stable, the module will package them for pip install |

---

# **16. v0.2.0 Success Criteria**

| **Criterion** | **Measurement** |
| --- | --- |
| All v0.1.0 tests pass | CI shows 230/230 green on v0.2.0 codebase |
| ET₀ accuracy | FAO-56 reference example matches within ±0.05 mm/day |
| Maize dual-plot physical accuracy | After 90 days, bottom slope cells have measurably higher moisture and N than crest cells in Plot A; Plot B plants breach PWP before Plot A crest plants |
| Hard pan constraint | Root depth correctly clamps at 19 cm for Plot B in maize_dual_plot_v2.py |
| Backward compatibility | wheat_basic.py (v0.1.0, unmodified) produces byte-identical Parquet on v0.2.0 |
| Multi-field frontend | Split viewport scrubs at ≥24 fps on Intel Iris Xe integrated GPU |
| Documentation completeness | All new APIs documented with examples before tag |

*CropForge PRD v0.2.0 — Prepared June 2026 — ICAR-IARI Jharkhand*

