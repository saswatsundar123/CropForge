# CHANGELOG

All notable changes to CropForge are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

## [0.7.0] - 2026-07-05

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

## [0.6.0] - 2026-07-04

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
