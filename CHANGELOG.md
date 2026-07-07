# CHANGELOG

All notable changes to CropForge are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

## [0.9.0] - 2026-07-08

### Added

- **Model Registry** — `cropforge.models.ModelRegistry` maps `(species, stage)` pairs to GLTF file paths. Three.js viewport routes each plant to the correct `InstancedMesh` based on its current phenological stage. Cylinder fallback is always active — no model installed → no error.
- **Binary buffer expansion** — Float32Array payload grows from 9 → 11 floats/plant, adding `stage_index` (int, remapped to float) and `lai_scale` (0.0–1.0). The `/api/buffer` endpoint is unchanged; third-party consumers must update to 11 floats/plant.
- **Stage progress tracking** — `PlantState` gains `stage_progress` (0.0–1.0, fraction through current phenological stage). Updated live by `StandardWheat` and `StandardMaize` plugins from accumulated thermal time.
- **PBR rendering** — `farm.visualize(quality="enhanced")` activates `MeshStandardMaterial` on terrain (roughness 0.9) and plants (roughness 0.6), shadow casting from a sun-angle directional light, and SSAO. `quality="standard"` (default) is bit-identical to v0.8.0.
- **Per-stage InstancedMesh routing** — Three.js maintains a dict of `InstancedMesh`es keyed by `model_index`. Each day, plants are dispatched to the correct mesh; registered GLTF models replace the cylinder for that stage.
- **GLTF scene export (Python)** — `farm.export_scene(day, filepath, field)` writes a valid binary GLB containing the simulation terrain and alive plant geometry (coloured boxes). Requires `pip install cropforge[export]` (`pygltflib>=1.16`).
- **GLTF scene export (frontend)** — "⬇ Export 3D Scene (.glb)" button in the dashboard topbar. Uses Three.js `GLTFExporter` to capture the live scene and trigger a browser download.
- **Plotly terrain 4× upsampling** — The `go.Surface` terrain chart applies `scipy.ndimage.zoom(order=3)` to elevation and `zoom(order=1)` to the colour overlay before display. Physics data is never modified. `z` and `surfacecolor` are guaranteed to match shape.
- **PBR terrain lighting in Plotly** — `go.Surface` now uses `lighting=dict(roughness=0.9, specular=0.1, ambient=0.7)` for matte-soil aesthetics. Axis grid lines, tick labels, and backgrounds are hidden for a clean digital-twin look.
- **Collapsible sidebars** — "◀ L" and "R ▶" toggle buttons in the topbar collapse left/right sidebar panels to give the 3D viewport the full screen.
- **`Mulching` land prep** — Cover fraction, evaporation reduction, C-factor modification.
- **`BroadBedFurrow` land prep** — BBF geometry with P factor = 0.45.
- **`terrain_feedback` flag** — `@farm.use_physics(terrain_feedback=True/False)` controls whether the elevation grid updates daily from net sediment flux.
- **New docs** — `docs/features/visual_engine.md`, `docs/tutorials/3d_assets_and_export.md`.
- **Capstone example** — `examples/photorealistic_twin_trial.py` demonstrates the full v0.9.0 feature stack end-to-end.

### Changed

- Binary buffer: 9 → 11 floats/plant (`stage_index`, `lai_scale` appended). Backward-incompatible for third-party binary consumers.

### Tested

- **838 tests passing**, 1 skipped, 0 failures. Zero regressions.
- Shape-match guarantee verified: `test_upsample_shapes_match_10x10` asserts `z.shape == surfacecolor.shape == (40, 40)`.
- GLB export verified: `test_export_scene_glb_header` reads first 4 bytes and asserts `b"glTF"`.
- Cylinder fallback verified: all 798 pre-v0.9.0 tests still pass with no models installed.

---

## [0.8.0] - 2026-07-05


### Added

- **Sub-metre resolution**: `Terrain` now accepts `resolution_m` (default `1.0` m — fully backward compatible). All physics engines (runoff, LS factor, D8 routing, nutrient flux) scale correctly at any cell size. `Terrain.from_array(arr, resolution_m=…)` added for direct NumPy ingestion.
- **Sediment transport**: eroded soil is routed downslope via D8 flow and deposited in accumulation zones. Mass is conserved to floating-point precision — every gram eroded either deposits in a downslope cell or exits the field boundary. New `SoilState` fields: `sediment_flux_kg_m2`, `sediment_deposited_kg_m2`, `cumulative_sediment_loss_kg_m2`, `cumulative_deposition_kg_m2`.
- **Geomorphological feedback**: the terrain elevation grid updates daily from net sediment flux; slope and aspect recompute automatically for the next timestep. Layer 0 topsoil depth expands with deposition and contracts with erosion (1 mm bedrock floor).
- **`TiedRidges` land prep**: ridge-furrow geometry with periodic perpendicular tie-dams that block D8 lateral flow and form micro-catchments. Proved to reduce cumulative field erosion vs. plain ridge-furrow.
- **`VegetativeFilterStrip` land prep**: dense grass strip at the field boundary stamps per-cell `surface_roughness_index = 0.95`, reducing soil detachment by ~95% in strip rows. Upslope cells are unaffected.
- **Terrain LOD renderer**: the Three.js viewport now chunks terrain into 64×64-cell tiles with two pre-built geometry levels (hi-res and 16× downsampled lo-res). Distant tiles switch to lo-res automatically; off-screen tiles are frustum-culled. For a 500×500 sub-metre field this reduces the rendered vertex count from ~270 K to ~18 K (14× reduction) when zoomed out.

### Changed

- **Exception standardisation**: two bare `RuntimeError` calls replaced with typed exceptions — `CropForgeStateError` (save before run) and `CropForgeConfigError` (season change during run).

### Tested

- **798 tests passing**, 1 skipped, 0 failures. Zero regressions across all prior physics, terrain, and visualization modules.
- Mass conservation verified analytically: `Σ(erosion) = Σ(export) + Σ(deposition)` across all sediment routing tests.
- LOD crucible: 500×500 sub-metre field loads successfully; distant polygon count confirmed at 18,496 vs 270,400 at full resolution.

---

## [0.7.0] - 2026-06-28

### Added
- **Topographical Physics:** Advanced opt-in physics modules that leverage the 3D terrain system.
- **Solar Incidence Engine:** Modifies radiation absorption dynamically based on slope, aspect, and solar declination calculations.
- **Wind Shadow Engine:** Models localized wind fields based on prevailing direction, offering leeward shelter and ridgeline intensification.
- **Clod Dynamics:** Exponential decay of soil surface roughness during heavy rainfall events.
- **Topographical Erosion Engine:** Incorporates a grid-based RUSLE model evaluating slope gradient, daily surface runoff, vegetation cover, and roughness dampening.
- **3D Observable Updates:** "Cumulative Erosion Index" and "Surface Runoff" can now be mapped directly onto the WebGL terrain viewport via the Parquet data layer.

### Fixed
- **Root Clamping:** Implemented strict hook mapping to clamp downward root growth into `effective_soil_depth_m` preventing penetration beyond the bedrock boundaries of carved terraces or deep furrows.

### Tested
- Extensively audited backwards-compatibility across flat-field runs and legacy datasets.
- Test suite expanded to 742 passing tests with full coverage for Erosion and Clod mechanics.

## [0.6.0] - 2026-06-27

### Added
- **Terrain Engine:** Procedural, CSV, and GeoTIFF topographies supported via the new `Terrain` class.
- **Land Preparation Modifiers:** `RidgeFurrow`, `ContourBund`, `Terrace`, `DeepTillage`, and `ConservationTillage` allow dynamic modification of the elevation grid and soil properties prior to simulation.
- **D8 Hydrology Coupling:** The deterministic-8 (D8) steepest-descent routing algorithm is now fully coupled with the new terrain system. Water routes natively over land preparation geometries (e.g. into furrows and behind bunds).
- **3D Plotly Modal:** New Terrain View toggle in the visualization dashboard seamlessly transitions from 2D heatmap to 3D procedural WebGL viewport.
- **Variable Overlays:** Users can now map agronomic variables (Nitrogen, Moisture, LAI, Stress) directly onto 3D plant models via instance coloring.

### Fixed
- **WebGL Performance Optimizations:** Eliminated geometry duplication, resolved `vertexColors` pipeline collisions, and implemented Matrix4 caching in the animation loop. 3D view now runs smoothly at 60fps.
- **Animation Anchoring:** Fixed an issue where scaling plants floated above the terrain; raycast hitboxes and instanced matrices now strictly anchor to the procedural `elevY` grid.
- **Dash Z-Index Bugs:** Corrected sidebar overlap and dropdown transparency issues introduced by the new Terrain View modal.

### Tested
- Extended unit test suite to 667 passing integration tests covering D8 terrain routing, mass conservation, and flat-grid fallbacks.

## [0.1.0] to [0.5.0]
- Core simulation loop, basic soil physics, crop phenology, dashboard rendering, and spatial integrations completed across previous PRD phases.
