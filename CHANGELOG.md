# CHANGELOG

All notable changes to CropForge are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [1.0.1] - 2026-07-14

v1.0.1 is the first stable release of CropForge. The visual arc is complete,
all known rendering regressions are resolved, and the example suite runs
clean end-to-end.

### Fixed

- **3D plants now grow vertically** (was: sideways/horizontal in Three.js r152).
  Root causes eliminated:
  - `MeshPhysicalMaterial` on `InstancedMesh` corrupts per-instance normal/tangent
    matrices in r152 — replaced with `MeshStandardMaterial` (no visual regression,
    no `transmission` dependency).
  - GLTF Z-up to Y-up rotation was discarded when cloning geometry for instancing;
    now baked in via `geo.applyMatrix4(child.matrixWorld)`.
  - `OutputPass` (Three.js r158+ only) was imported against r152 CDN, causing a
    silent 404 that triggered the 2D fallback; replaced with
    `ShaderPass(GammaCorrectionShader)` (r152-compatible).
- **Terrain glow removed** — bloom pass strength reduced and threshold raised so
  only bright canopy tops fire, not the soil mesh.
- `examples/irrigation_trial.py` — fixed broken field-detection logic
  (`plant_id.startswith()` never matched `r00c00`-format IDs); replaced with a
  single step that tracks field call-order per day.
- `examples/disease_outbreak_trial.py` — fixed `UnicodeEncodeError` crash on
  Windows cp1252 consoles caused by emoji in `print_disease_snapshot`; replaced
  with ASCII delimiters.

### Added

- `cropforge/viz/static/fallback.js` — offline / CDN-blocked fallback renderer.
- `cropforge/viz/static/parent_sanitize.js` — cross-origin iframe message sanitizer.
- `examples/disease_outbreak_trial.py` now calls `farm.visualize(quality="enhanced")`
  at the end so the dashboard launches automatically on the disease run.

### Changed

- Project version bumped to `1.0.1`.
- Repo root cleaned: removed `emoji_hits.txt`, `cropforge_crash.log`,
  `mid_season_export.glb`, `wheat_trial_day15.glb`, and all `__pycache__` /
  `.pytest_cache` directories from tracked files.

### Tested

- `wheat_basic_v2.py` — 600 plants, 90 days, all alive, biomass 316.29 g/plant.
- `disease_outbreak_trial.py` — 900 plants, 90 days, wind anisotropy E/W ratio
  2.50x confirmed.
- `irrigation_trial.py` — dual-field stress divergence simulation runs clean.

---

## [0.9.5] - 2026-07-09

v0.9.5 completes the CropForge visual architecture arc: bundled crop assets,
growth morphing, stress and disease shaders, machinery animation, weather
particles, enhanced PBR rendering, and release-ready visual documentation now
work together from the same compute-first Parquet pipeline.

### Fixed

- Packaged first-party GLTF stage assets in wheels and declared `cropforge.viz` package data.
- Added `scipy` as a runtime dependency for Plotly terrain upsampling.
- `terrain_feedback=False` now freezes terrain elevation geometry while still logging sediment flux.
- Python GLTF scene export now uses logged terrain elevation instead of a flat mesh.
- `ModelRegistry.register()` now accepts documented `species=` and `gltf_path=` aliases.

### Added

- **First-party asset bundles** for StandardWheat and StandardMaize now auto-register GLTF stage assets while preserving cylinder fallback for missing models.
- **Morph target interpolation** transmits `morph_weight` to the frontend and blends growth-stage geometry where GLTF morph targets exist.
- **Stress and disease visualizations** combine drought wilt deformation with enhanced-mode necrosis fragment shading from `disease_severity`.
- **Machinery animation framework** logs tillage, spray, and harvest paths to a separate `machinery` Parquet dataset and animates frontend proxy machines from daily metadata.
- **Weather particle effects** expose daily precipitation metadata and render bounded enhanced-only rain particles.
- **Capstone lifecycle example** added at `examples/digital_twin_full_lifecycle.py`.
- `farm.export_animation()` API entry point writes a valid terrain-aware GLB for the first requested keyframe; full animation channels remain scoped to v0.9.5 exporter work.
- Enhanced WebGL mode now wires SSAO, bloom, ACES tone mapping, and `MeshPhysicalMaterial` for plants behind `quality="enhanced"`.
- Morph interpolation and disease shader data now travel in the binary viewport buffer as `morph_weight`, `stress_ks`, and `disease_severity`, expanding the internal payload to 14 floats per plant.
- Machinery path logging now writes a separate `machinery` Parquet dataset, exposes day metadata via `/api/buffer/day/{day}`, and animates a lightweight frontend proxy along logged waypoints.
- Daily precipitation metadata now reaches `/api/buffer/day/{day}` and drives an enhanced-only bounded rain particle system in the Three.js viewport.

### Changed

- Project version bumped to `0.9.5`.
- Visual engine documentation updated for first-party assets, morphing, machinery, disease/stress shaders, and weather particles.

### Tested

- **861 tests passing**, 1 skipped, 0 failures. Zero regressions across the visual architecture release arc.

## [0.9.0] - 2026-07-08

### Added

- **Model Registry** — `cropforge.models.ModelRegistry` maps `(species, stage)` pairs to GLTF file paths. Three.js viewport routes each plant to the correct `InstancedMesh` based on its current phenological stage. Cylinder fallback is always active — no model installed → no error.
- **Binary buffer expansion** — Float32Array payload grows from 9 → 11 floats/plant, adding `stage_index` (int, remapped to float) and `lai_scale` (0.0–1.0). The `/api/buffer` endpoint is unchanged; third-party consumers must update to 11 floats/plant.
- **Stage progress tracking** — `PlantState` gains `stage_progress` (0.0–1.0, fraction through current phenological stage). Updated live by `StandardWheat` and `StandardMaize` plugins from accumulated thermal time.
- **PBR rendering** — `farm.visualize(quality="enhanced")` activates `MeshStandardMaterial` on terrain (roughness 0.9) and plants (roughness 0.6), shadow casting from a sun-angle directional light, and SSAO. `quality="standard"` (default) is bit-identical to v0.8.0.
- **Per-stage InstancedMesh routing** — Three.js maintains a dict of `InstancedMesh`es keyed by `model_index`. Each day, plants are dispatched to the correct mesh; registered GLTF models replace the cylinder for that stage.
- **GLTF scene export (Python)** — `farm.export_scene(day, filepath, field)` writes a valid binary GLB containing the simulation terrain and alive plant geometry (coloured boxes). Requires `pip install cropforge[export]` (`pygltflib>=1.16`).
- **GLTF scene export (frontend)** — "Export 3D Scene (.glb)" button in the dashboard topbar. Uses Three.js `GLTFExporter` to capture the live scene and trigger a browser download.
- **Plotly terrain 4× upsampling** — The `go.Surface` terrain chart applies `scipy.ndimage.zoom(order=3)` to elevation and `zoom(order=1)` to the colour overlay before display. Physics data is never modified. `z` and `surfacecolor` are guaranteed to match shape.
- **PBR terrain lighting in Plotly** — `go.Surface` now uses `lighting=dict(roughness=0.9, specular=0.1, ambient=0.7)` for matte-soil aesthetics. Axis grid lines, tick labels, and backgrounds are hidden for a clean digital-twin look.
- **Collapsible sidebars** — "L" and "R" toggle buttons in the topbar collapse left/right sidebar panels to give the 3D viewport the full screen.
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
