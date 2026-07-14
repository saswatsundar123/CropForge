/**
 * main.js - CropForge 3D Viewport
 *
 * PRD Section 7.3:
 *   - Three.js r152 instanced cylinder meshes
 *   - height = plant.height_cm (scaled), radius = LAI proxy
 *   - Full HSL gradient colour-mapped to selected variable
 *   - Dead plants: colour #8B6914, height collapses
 *   - 2D ground plane (soil surface)
 *   - Binary Float32Array from /api/buffer?day=N (no JSON)
 *   - All days preloaded into browser memory on page load
 *   - Timeline scrubber driven by postMessage from Dash parent
 *   - Raycasting: click a plant -> highlight + postMessage PLANT_CLICKED
 *
 * Binary frame layout (14 float32 per plant, 56 bytes):
 *   [0]  x              grid col position (world units)
 *   [1]  y              half-height (cylinder centre above ground)
 *   [2]  z              grid row position (world units)
 *   [3]  scaleY         full cylinder height (world units)
 *   [4]  radius         cylinder radius
 *   [5]  r              colour red   [0..1]
 *   [6]  g              colour green [0..1]
 *   [7]  b              colour blue  [0..1]
 *   [8]  alive          1.0 = alive, 0.0 = dead
 *   [9]  model_index    int key into meta.model_index_map (0 = cylinder fallback)
 *   [10] stage_progress fractional progress within current pheno stage [0.0, 1.0]
 *   [11] morph_weight   stage morph interpolation weight [0.0, 1.0]
 *   [12] stress_ks      water stress coefficient for wilt deformation [0.0, 1.0]
 *   [13] disease_severity disease necrosis shader severity [0.0, 1.0]
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { SSAOPass } from 'three/addons/postprocessing/SSAOPass.js';

window.__cfThreeViewportStarted = true;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

const API = window.location.origin;           // same origin as iframe parent
const FLOATS = 14;                                // float32 values per plant (v0.9.5 Phase 3)
const BYTES = FLOATS * 4;                        // bytes per plant

let meta = null;   // buffer metadata from /api/buffer/meta
let qualityEnhanced = false;  // true when quality="enhanced" (PBR shadows + HDR)
let terrainGrid = null;   // Float32Array [rows*cols] row-major elevation values (v0.6.0)
let terrainRows = 0;
let terrainCols = 0;
let frames = {};     // {day: Float32Array}  - all preloaded frames
let dayMeta = {};    // {day: {machinery: [...]}} lightweight JSON metadata
let currentDay = 1;
let currentField = null;   // active field name (null = server default)
let playing = false;
let speedMult = 1;
let playTimer = null;
// v0.9.0 Phase 3: meshes map, keyed by model_index (0 = cylinder fallback).
// ponytail: keep `mesh` as alias to meshes[0] - all selection/colour code unchanged.
let meshes = {};   // {model_index: THREE.InstancedMesh}
let mesh = null;   // alias -> meshes[0], cylinder fallback
let soilMesh = null;
let machineryMesh = null;
let machineryAnim = null;
let rainSystem = null;
let rainPositions = null;
let rainActiveCount = 0;
let sprinklerSystem = null;
let sprinklerPositions = null;
let weedTintMesh = null;
let weedMesh = null;
let renderer, scene, camera, controls, composer;

// ponytail: cache terrain elev constants once at init - not per frame, not per plant
let _terrainEMin = 0.0;
let _terrainElevScale = 0.0;
// ponytail: pre-allocated objects reused every frame - avoid GC churn in hot loop
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();  // identity, never rotated
const _scl = new THREE.Vector3();
const _mat = new THREE.Matrix4();

// Terrain LOD chunk tracking (v0.8.0 Phase 5)
// ponytail: two pre-built geometries per chunk, swap on distance threshold.
// Ceiling: >256 chunks with animated terrain would need GPU-side LOD; add when needed.
let _terrainChunks = [];   // [{mesh, cx, cz, geoHi, geoLo, isLo}]
const _CHUNK_CELLS = 64;   // cells per chunk edge
const _LOD_LO_SQ  = 80 * 80;  // world-unit² distance to switch to lo-res
const _LOD_HI_SQ  = 55 * 55;  // hysteresis - switch back to hi when closer than this

// Raycasting
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let selectedInstance = -1;    // instanceId of currently selected plant
let selectionBox = null;  // THREE.LineSegments wireframe around selection

const colourBuf = new THREE.Color();

// ---------------------------------------------------------------------------
// DOM handles
// ---------------------------------------------------------------------------

const loader = document.getElementById('loader');
const loaderFill = document.getElementById('loader-bar-fill');
const loaderStatus = document.getElementById('loader-status');
const hudDay = document.getElementById('hud-day');
const hudInfo = document.getElementById('hud-info');
const legendTitle = document.getElementById('legend-title');
const legendMin = document.getElementById('legend-min');
const legendMax = document.getElementById('legend-max');
const legendBar = document.getElementById('legend-bar');
const varSelect = document.getElementById('var-select');
const btnPlay = document.getElementById('btn-play');
const btn1x = document.getElementById('btn-1x');
const btn2x = document.getElementById('btn-2x');
const btn5x = document.getElementById('btn-5x');
const btnExportGlb = document.getElementById('btn-export-glb');

// ---------------------------------------------------------------------------
// 1. Bootstrap: fetch metadata, preload all frames
// ---------------------------------------------------------------------------

async function bootstrap(fieldName) {
  /* fieldName: string | null - pass null to load the server-default field */
  if (fieldName !== undefined && fieldName !== null) {
    currentField = fieldName;
  }

  const fieldParam = currentField ? `?field=${encodeURIComponent(currentField)}` : '';

  setStatus('Fetching session metadata...', 2);

  try {
    const r = await fetch(`${API}/api/buffer/meta${fieldParam}`);
    if (!r.ok) throw new Error(`meta HTTP ${r.status}`);
    meta = await r.json();
    qualityEnhanced = (meta.quality_mode === 'enhanced');
  } catch (e) {
    setStatus(`ERROR: ${e.message}`, 0);
    return;
  }

  // Update HUD with field name if available
  if (meta.field_name) {
    const hudField = document.getElementById('hud-field');
    if (hudField) hudField.textContent = meta.field_name;
  }

  setStatus(`Preloading ${meta.n_days} days x ${meta.n_plants} plants...`, 5);

  frames = {};   // clear old frames
  dayMeta = {};
  const days = meta.days;

  for (let i = 0; i < days.length; i++) {
    const day = days[i];
    try {
      const sep = fieldParam ? '&' : '?';
      const r = await fetch(`${API}/api/buffer?day=${day}${fieldParam ? fieldParam.replace('?', '&') : ''}`);

      if (!r.ok) throw new Error(`buffer HTTP ${r.status} day=${day}`);
      const ab = await r.arrayBuffer();
      frames[day] = new Float32Array(ab);

      const mr = await fetch(`${API}/api/buffer/day/${day}${fieldParam}`);
      if (mr.ok) dayMeta[day] = await mr.json();
    } catch (e) {
      console.error(e);
    }
    // Update loading bar
    const pct = Math.round(((i + 1) / days.length) * 100);
    loaderFill.style.width = pct + '%';
    setStatus(`Preloading day ${day} / ${days[days.length - 1]}  (${pct}%)`, pct);
  }

  setStatus('Building 3D scene...', 98);
  await new Promise(r => setTimeout(r, 50));  // yield to let browser paint

  // v0.6.0 - Fetch terrain elevation grid (modified by LandPrep if any)
  try {
    const tr = await fetch(`${API}/api/buffer/terrain${fieldParam}`);
    if (tr.ok) {
      const td = await tr.json();
      terrainRows = td.rows;
      terrainCols = td.cols;
      const flat = td.elevation_flat;
      terrainGrid = new Float32Array(flat.length);
      for (let i = 0; i < flat.length; i++) terrainGrid[i] = flat[i];
    } else {
      terrainGrid = null;  // flat fallback
    }
  } catch (e) {
    console.warn('Terrain fetch failed, using flat plane:', e);
    terrainGrid = null;
  }

  initScene();
  // Async GLTF loads: fire-and-forget; cylinders render immediately, GLTF meshes appear when ready
  loadGltfMeshes();
  showDay(days[0]);

  // Hide iframe loader
  window.__cfThreeViewportReady = true;
  loader.classList.add('hidden');
  setTimeout(() => { loader.style.display = 'none'; }, 650);

  // Notify Dash parent that viewport is fully loaded (PRD v0.5.0 Section4.5)
  // The Dash layout listens for this and hides the Dash-level loading overlay.
  try {
    window.parent.postMessage(
      { type: 'LOAD_COMPLETE', total_days: days.length },
      window.location.origin
    );
  } catch (e) {
    // cross-origin or no parent - safe to ignore
  }

  // Update legend
  updateLegendMeta();

  // Start animation loop (idempotent - animate() guards with RAF)
  animate();
}

function setStatus(msg, pct) {
  loaderStatus.textContent = msg;
  if (pct !== undefined) loaderFill.style.width = pct + '%';
}

function failToFallback(err) {
  console.error('CropForge Three.js viewport failed:', err);
  const message = err && err.message ? err.message : String(err || 'unknown error');
  setStatus(`Viewport fallback: ${message}`, 0);
  if (window.CropForgeFallback && !window.__cfFallbackViewportStarted) {
    window.CropForgeFallback.start('Three.js bootstrap error');
  }
}

// ---------------------------------------------------------------------------
// 2. Three.js scene setup
// ---------------------------------------------------------------------------

function initScene() {
  const wrapper = document.getElementById('canvas-wrapper');
  const W = wrapper.clientWidth;
  const H = wrapper.clientHeight;

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  if (qualityEnhanced) {
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
  }
  wrapper.appendChild(renderer.domElement);

  // Scene - bright light background
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9c9c9c);
  scene.fog = new THREE.FogExp2(0x9c9c9c, 0.016);

  // Camera - isometric-ish perspective above the field
  camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 500);
  const fieldW = meta.cols * meta.grid_spacing;
  const fieldD = meta.rows * meta.grid_spacing;
  camera.position.set(fieldW * 0.5, fieldW * 0.9, fieldD * 1.5);
  camera.lookAt(fieldW * 0.5, 0, fieldD * 0.5);

  // Orbit controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(fieldW * 0.5, 0, fieldD * 0.5);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 3;
  controls.maxDistance = 120;
  controls.maxPolarAngle = Math.PI / 2.1;
  controls.update();

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.45);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff0d0, 1.4);
  sun.position.set(fieldW * 0.6, 20, fieldD * 0.3);
  // ponytail: shadows are expensive - only enable in enhanced quality mode
  sun.castShadow = qualityEnhanced;
  if (qualityEnhanced) {
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 200;
    sun.shadow.camera.left = -fieldW * 0.7;
    sun.shadow.camera.right = fieldW * 1.3;
    sun.shadow.camera.top = fieldD * 1.3;
    sun.shadow.camera.bottom = -fieldD * 0.7;
  }
  scene.add(sun);

  const fill = new THREE.HemisphereLight(0x4060a0, 0x2c4a2c, 0.35);
  scene.add(fill);

  // Ground plane (soil surface) - chunked LOD terrain (v0.8.0 Phase 5)
  buildTerrainChunks(fieldW, fieldD);

  // Grid helper (subtle) - skip for huge grids to avoid overdraw
  if (meta.cols <= 200 && meta.rows <= 200) {
    const grid = new THREE.GridHelper(
      Math.max(fieldW, fieldD) + 4, Math.max(meta.cols, meta.rows) + 2,
      0xEAEAEA, 0xF9F9F8
    );
    grid.position.set(fieldW * 0.5 - 0.5, -0.01, fieldD * 0.5 - 0.5);
    scene.add(grid);
  }

  // Build instanced plant mesh
  buildInstancedMesh();

  if (qualityEnhanced) initRainSystem(fieldW, fieldD);
  if (qualityEnhanced) initEnhancedRendering(W, H);

  // ---- Raycasting click listener -------------------------------------------
  renderer.domElement.addEventListener('click', onCanvasClick, false);

  // Window resize
  window.addEventListener('resize', onResize);
}

// Build a single chunk's PlaneGeometry, optionally downsampled.
// r0/c0 = top-left cell index; rN/cN = exclusive end. step = 1 (hi) or 4 (lo).
function _buildChunkGeo(r0, c0, rN, cN, step, spacing, eMin, elevScale) {
  const sampR = Math.ceil((rN - r0) / step);
  const sampC = Math.ceil((cN - c0) / step);
  const w = (sampC - 1) * step * spacing;
  const d = (sampR - 1) * step * spacing;
  const geo = new THREE.PlaneGeometry(
    Math.max(step * spacing, w),
    Math.max(step * spacing, d),
    Math.max(1, sampC - 1),
    Math.max(1, sampR - 1),
  );
  geo.rotateX(-Math.PI / 2);

  if (terrainGrid && elevScale > 0) {
    const pos = geo.attributes.position;
    let vi = 0;
    for (let ri = 0; ri < sampR; ri++) {
      for (let ci = 0; ci < sampC; ci++) {
        const row = Math.min(r0 + ri * step, rN - 1);
        const col = Math.min(c0 + ci * step, cN - 1);
        const elev = (terrainGrid[row * terrainCols + col] - eMin) * elevScale;
        pos.setY(vi, elev - 0.02);
        vi++;
      }
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
  }
  return geo;
}

function buildTerrainChunks(fieldW, fieldD) {
  // Dispose any existing chunks (field-switch teardown)
  for (const ch of _terrainChunks) {
    scene.remove(ch.mesh);
    ch.geoHi.dispose();
    ch.geoLo.dispose();
  }
  _terrainChunks = [];

  const cols = meta.cols;
  const rows = meta.rows;
  const spacing = meta.grid_spacing;

  // Compute elevation scale once (same logic as old buildGroundPlane)
  let eMin = 0, elevScale = 0;
  if (terrainGrid && terrainGrid.length === rows * cols) {
    let eMax = -Infinity;
    eMin = Infinity;
    for (let i = 0; i < terrainGrid.length; i++) {
      if (terrainGrid[i] < eMin) eMin = terrainGrid[i];
      if (terrainGrid[i] > eMax) eMax = terrainGrid[i];
    }
    const eRange = eMax - eMin || 1.0;
    elevScale = Math.min(3.0, (cols * spacing) * 0.15) / eRange;
  }
  _terrainEMin = eMin;
  _terrainElevScale = elevScale;

  // PBR: MeshStandardMaterial for terrain - roughness=0.9 for soil, low metalness
  // ponytail: shadows conditionally enabled by qualityEnhanced flag
  const mat = new THREE.MeshStandardMaterial({
    color: 0xC8A97E,
    roughness: 0.9,
    metalness: 0.0,
    side: THREE.FrontSide,
  });

  for (let r0 = 0; r0 < rows; r0 += _CHUNK_CELLS) {
    const rN = Math.min(r0 + _CHUNK_CELLS, rows);
    for (let c0 = 0; c0 < cols; c0 += _CHUNK_CELLS) {
      const cN = Math.min(c0 + _CHUNK_CELLS, cols);

      const geoHi = _buildChunkGeo(r0, c0, rN, cN, 1, spacing, eMin, elevScale);
      // ponytail: lo-res step=4; clamp so tiny chunks don't go below 1 sample
      const loStep = Math.min(4, Math.max(1, Math.floor((cN - c0) / 2)));
      const geoLo = _buildChunkGeo(r0, c0, rN, cN, loStep, spacing, eMin, elevScale);

      const chunkMesh = new THREE.Mesh(geoHi, mat);

      // Position: chunk centre in world space
      const worldX = (c0 + (cN - c0) / 2) * spacing;
      const worldZ = (r0 + (rN - r0) / 2) * spacing;
      chunkMesh.position.set(worldX, 0.0, worldZ);
      chunkMesh.receiveShadow = qualityEnhanced;
      chunkMesh.frustumCulled = true;
      scene.add(chunkMesh);

      _terrainChunks.push({ mesh: chunkMesh, cx: worldX, cz: worldZ, geoHi, geoLo, isLo: false });
    }
  }
}

function updateTerrainLOD() {
  if (!_terrainChunks.length) return;
  const cx = camera.position.x;
  const cz = camera.position.z;
  for (const ch of _terrainChunks) {
    const dx = cx - ch.cx, dz = cz - ch.cz;
    const distSq = dx * dx + dz * dz;
    if (!ch.isLo && distSq > _LOD_LO_SQ) {
      ch.mesh.geometry = ch.geoLo;
      ch.isLo = true;
    } else if (ch.isLo && distSq < _LOD_HI_SQ) {
      ch.mesh.geometry = ch.geoHi;
      ch.isLo = false;
    }
  }
}

// ---------------------------------------------------------------------------
// Plant geometry factory - cylinder (index=0) or GLTF-sourced BufferGeometry
// ---------------------------------------------------------------------------

function _cylinderGeo() {
  return new THREE.CylinderGeometry(1, 1, 1, 6, 1);
}

function _makePlantMat() {
  // PBR plant material - instanced colour via instanceColor
  // ponytail: transmission removed - semi-transparency on instanced meshes
  // causes ghost/arrow artifacts; standard roughness is sufficient for crops.
  return new THREE.MeshStandardMaterial({
    roughness: qualityEnhanced ? 0.6 : 0.7,
    metalness: 0.0,
  });
}

function initEnhancedRendering(width, height) {
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  // SSAO gives soft ambient occlusion depth cues without the
  // bloom halo that caused terrain glow on bright diffuse surfaces.
  // ponytail: bloom removed - threshold 0.85 fired on sunlit cream soil;
  // bloom is correct only for emissive objects (LEDs, headlights, etc.)
  const ssaoPass = new SSAOPass(scene, camera, width, height);
  ssaoPass.kernelRadius = 8;
  ssaoPass.minDistance = 0.001;
  ssaoPass.maxDistance = 0.3;
  composer.addPass(ssaoPass);
}

function _installPlantMorphShader(material, geometry) {
  const morphPositions = geometry.morphAttributes?.position || [];
  const hasMorphTargets = morphPositions.length >= 2;
  const useDiseaseShader = qualityEnhanced;
  if (!hasMorphTargets && !useDiseaseShader) {
    material.userData.cfMorphTargetsEnabled = false;
    material.userData.cfWiltEnabled = false;
    material.userData.cfDiseaseShaderEnabled = false;
    return;
  }

  if (hasMorphTargets) {
    geometry.setAttribute('cfMorphStart', morphPositions[0]);
    geometry.setAttribute('cfMorphEnd', morphPositions[1]);
    if (morphPositions.length >= 3) geometry.setAttribute('cfMorphWilt', morphPositions[2]);
  }

  material.userData.cfMorphTargetsEnabled = hasMorphTargets;
  material.userData.cfWiltEnabled = hasMorphTargets;
  material.userData.cfWiltUsesTarget = hasMorphTargets && morphPositions.length >= 3;
  material.userData.cfDiseaseShaderEnabled = useDiseaseShader;
  material.defines = material.defines || {};
  material.defines.CF_USE_STAGE_MORPH = hasMorphTargets ? 1 : 0;
  material.defines.CF_HAS_WILT_TARGET = morphPositions.length >= 3 ? 1 : 0;
  material.defines.CF_USE_DISEASE_SHADER = useDiseaseShader ? 1 : 0;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.cfMorphTargetsRelative = { value: geometry.morphTargetsRelative !== false };
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        attribute float cfMorphWeight;
        attribute float cfWiltWeight;
        attribute float cfDiseaseSeverity;
        varying float vCfDiseaseSeverity;
        attribute vec3 cfMorphStart;
        attribute vec3 cfMorphEnd;
        #if CF_HAS_WILT_TARGET == 1
          attribute vec3 cfMorphWilt;
        #endif
        uniform bool cfMorphTargetsRelative;`
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        #if CF_USE_DISEASE_SHADER == 1
          vCfDiseaseSeverity = cfDiseaseSeverity;
        #endif
        #if CF_USE_STAGE_MORPH == 1
          vec3 cfStageStart = cfMorphTargetsRelative ? position + cfMorphStart : cfMorphStart;
          vec3 cfStageEnd = cfMorphTargetsRelative ? position + cfMorphEnd : cfMorphEnd;
          transformed = mix(cfStageStart, cfStageEnd, clamp(cfMorphWeight, 0.0, 1.0));
          #if CF_HAS_WILT_TARGET == 1
            vec3 cfWiltTarget = cfMorphTargetsRelative ? position + cfMorphWilt : cfMorphWilt;
            transformed = mix(transformed, cfWiltTarget, clamp(cfWiltWeight, 0.0, 1.0));
          #else
            float cfWilt = clamp(cfWiltWeight, 0.0, 1.0);
            float cfTipWeight = smoothstep(0.05, 1.0, position.y);
            transformed.y -= cfTipWeight * cfWilt * 0.25;
            transformed.xz *= 1.0 + cfTipWeight * cfWilt * 0.08;
          #endif
        #endif`
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying float vCfDiseaseSeverity;`
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        #if CF_USE_DISEASE_SHADER == 1
          vec3 cfNecroticColor = vec3(0.55, 0.35, 0.10);
          diffuseColor.rgb = mix(diffuseColor.rgb, cfNecroticColor, clamp(vCfDiseaseSeverity, 0.0, 1.0));
        #endif`
      );
  };
  material.needsUpdate = true;
}

function _configurePlantMesh(im, n) {
  const morphWeights = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
  const wiltWeights = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
  const diseaseSeverities = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
  morphWeights.setUsage(THREE.DynamicDrawUsage);
  wiltWeights.setUsage(THREE.DynamicDrawUsage);
  diseaseSeverities.setUsage(THREE.DynamicDrawUsage);
  im.geometry.setAttribute('cfMorphWeight', morphWeights);
  im.geometry.setAttribute('cfWiltWeight', wiltWeights);
  im.geometry.setAttribute('cfDiseaseSeverity', diseaseSeverities);
  _installPlantMorphShader(im.material, im.geometry);
}

function _wiltWeightFromStress(stressKs) {
  if (!Number.isFinite(stressKs) || stressKs >= 0.5) return 0.0;
  return Math.max(0.0, Math.min(1.0, (0.5 - stressKs) / 0.5));
}

function _terrainYAt(row, col) {
  if (!terrainGrid) return 0.0;
  const r = Math.max(0, Math.min(terrainRows - 1, Math.round(row)));
  const c = Math.max(0, Math.min(terrainCols - 1, Math.round(col)));
  return (_terrainElevScale > 0)
    ? (terrainGrid[r * terrainCols + c] - _terrainEMin) * _terrainElevScale
    : 0.0;
}

function initRainSystem(fieldW, fieldD) {
  if (!qualityEnhanced || rainSystem || !scene) return;

  const count = _RAIN_PARTICLE_COUNT;
  rainPositions = new Float32Array(count * 3);
  const topY = Math.max(6.0, Math.min(32.0, Math.max(fieldW, fieldD) * 0.35));
  for (let i = 0; i < count; i++) {
    const base = i * 3;
    rainPositions[base + 0] = Math.random() * fieldW;
    rainPositions[base + 1] = 0.5 + Math.random() * topY;
    rainPositions[base + 2] = Math.random() * fieldD;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(rainPositions, 3));
  geo.setDrawRange(0, 0);

  const mat = new THREE.PointsMaterial({
    color: 0x9ecfff,
    size: 0.035,
    transparent: true,
    opacity: 0.0,
    depthWrite: false,
  });

  rainSystem = new THREE.Points(geo, mat);
  rainSystem.visible = false;
  rainSystem.frustumCulled = false;
  rainSystem.userData = { fieldW, fieldD, topY, fallSpeed: 0.12 };
  scene.add(rainSystem);
}

function _updateRainForDay(day) {
  if (!qualityEnhanced || !rainSystem) return;
  const precipitation = Number(dayMeta[day]?.precipitation_mm || 0.0);
  if (!Number.isFinite(precipitation) || precipitation < _RAIN_HIDE_THRESHOLD_MM) {
    rainActiveCount = 0;
    rainSystem.visible = false;
    rainSystem.geometry.setDrawRange(0, 0);
    rainSystem.material.opacity = 0.0;
    return;
  }

  const intensity = Math.max(0.0, Math.min(1.0, precipitation / 40.0));
  rainActiveCount = Math.max(250, Math.floor(_RAIN_PARTICLE_COUNT * intensity));
  rainSystem.geometry.setDrawRange(0, rainActiveCount);
  rainSystem.material.opacity = 0.18 + intensity * 0.42;
  rainSystem.userData.fallSpeed = 0.08 + intensity * 0.24;
  rainSystem.visible = true;
}

function _animateRain() {
  if (!qualityEnhanced || !rainSystem || !rainSystem.visible || rainActiveCount <= 0) return;
  const positions = rainSystem.geometry.attributes.position.array;
  const { fieldW, fieldD, topY, fallSpeed } = rainSystem.userData;
  const speed = fallSpeed * (speedMult || 1);

  for (let i = 0; i < rainActiveCount; i++) {
    const base = i * 3;
    positions[base + 1] -= speed;
    if (positions[base + 1] < 0.0) {
      positions[base + 0] = Math.random() * fieldW;
      positions[base + 1] = topY;
      positions[base + 2] = Math.random() * fieldD;
    }
  }
  rainSystem.geometry.attributes.position.needsUpdate = true;
}

function _ensureWeedMeshes() {
  if (!scene || weedTintMesh) return;
  const n = meta.n_plants;

  const tintGeo = new THREE.PlaneGeometry(0.82, 0.82);
  tintGeo.rotateX(-Math.PI / 2);
  const tintMat = new THREE.MeshBasicMaterial({
    color: 0x8b6f3d,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  });
  weedTintMesh = new THREE.InstancedMesh(tintGeo, tintMat, n);
  weedTintMesh.count = 0;
  weedTintMesh.frustumCulled = true;
  scene.add(weedTintMesh);

  if (qualityEnhanced) {
    const geo = new THREE.ConeGeometry(0.12, 0.35, 5);
    const mat = new THREE.MeshStandardMaterial({ color: 0x3f7f3a, roughness: 0.8 });
    weedMesh = new THREE.InstancedMesh(geo, mat, n);
    weedMesh.count = 0;
    weedMesh.castShadow = qualityEnhanced;
    weedMesh.frustumCulled = true;
    scene.add(weedMesh);
  }
}

function _renderWeedsForDay(day) {
  const weeds = dayMeta[day]?.weeds || [];
  if (!weeds.length) {
    if (weedTintMesh) weedTintMesh.count = 0;
    if (weedMesh) weedMesh.count = 0;
    return;
  }

  _ensureWeedMeshes();
  let count = 0;
  for (const weed of weeds) {
    if (!weed.alive) continue;
    const row = Number(weed.row) || 0;
    const col = Number(weed.col) || 0;
    const x = col * meta.grid_spacing;
    const z = row * meta.grid_spacing;
    const y = _terrainYAt(row, col) + 0.015;
    const lai = Math.max(0.05, Math.min(2.5, Number(weed.lai) || 0.2));

    _pos.set(x, y, z);
    _scl.set(0.7 + lai * 0.08, 1.0, 0.7 + lai * 0.08);
    _mat.compose(_pos, _quat, _scl);
    weedTintMesh.setMatrixAt(count, _mat);

    if (weedMesh) {
      _pos.set(x + 0.16, y + 0.17, z - 0.12);
      _scl.set(0.7, Math.min(1.8, 0.7 + lai * 0.35), 0.7);
      _mat.compose(_pos, _quat, _scl);
      weedMesh.setMatrixAt(count, _mat);
    }
    count++;
  }
  weedTintMesh.count = count;
  weedTintMesh.instanceMatrix.needsUpdate = true;
  if (weedMesh) {
    weedMesh.count = count;
    weedMesh.instanceMatrix.needsUpdate = true;
  }
}

function _ensureMachineryMesh(machineType) {
  if (machineryMesh) {
    if (machineryMesh.material?.color) {
      machineryMesh.material.color.setHex(
        machineType === 'harvester' ? 0xc9a227 :
        machineType === 'sprinkler' || machineType === 'pivot' ? 0x4aa3df :
        0x335f35
      );
    }
    return machineryMesh;
  }
  const geo = new THREE.BoxGeometry(0.8, 0.35, 0.45);
  const mat = new THREE.MeshStandardMaterial({
    color: machineType === 'harvester' ? 0xc9a227 :
      machineType === 'sprinkler' || machineType === 'pivot' ? 0x4aa3df :
      0x335f35,
    roughness: 0.65,
    metalness: 0.05,
  });
  machineryMesh = new THREE.Mesh(geo, mat);
  machineryMesh.castShadow = qualityEnhanced;
  machineryMesh.receiveShadow = false;
  machineryMesh.visible = false;
  scene.add(machineryMesh);
  return machineryMesh;
}

function _ensureSprinklerParticles() {
  if (!qualityEnhanced || sprinklerSystem || !scene) return sprinklerSystem;
  const count = 360;
  sprinklerPositions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const base = i * 3;
    sprinklerPositions[base + 0] = 0;
    sprinklerPositions[base + 1] = -999;
    sprinklerPositions[base + 2] = 0;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(sprinklerPositions, 3));
  const mat = new THREE.PointsMaterial({
    color: 0x58b7ff,
    size: 0.055,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  sprinklerSystem = new THREE.Points(geo, mat);
  sprinklerSystem.visible = false;
  sprinklerSystem.frustumCulled = false;
  sprinklerSystem.userData = { count, spread: 0.55, fall: 0.12 };
  scene.add(sprinklerSystem);
  return sprinklerSystem;
}

function _stopSprinklerEmission() {
  if (sprinklerSystem) sprinklerSystem.visible = false;
}

function _emitSprinklerBurst(x, y, z) {
  if (!qualityEnhanced) return;
  const system = _ensureSprinklerParticles();
  if (!system || !sprinklerPositions) return;
  const { count, spread, fall } = system.userData;
  for (let i = 0; i < count; i++) {
    const base = i * 3;
    const age = (i % 24) / 24;
    const angle = (i * 2.399963) % (Math.PI * 2);
    const radius = spread * Math.sqrt(age);
    sprinklerPositions[base + 0] = x + Math.cos(angle) * radius;
    sprinklerPositions[base + 1] = Math.max(0.04, y - age * (1.8 + fall * speedMult));
    sprinklerPositions[base + 2] = z + Math.sin(angle) * radius;
  }
  system.geometry.attributes.position.needsUpdate = true;
  system.visible = true;
}

function _stopMachineryAnimation() {
  if (machineryAnim) {
    cancelAnimationFrame(machineryAnim);
    machineryAnim = null;
  }
  _stopSprinklerEmission();
}

function _animateMachineryForDay(day) {
  _stopMachineryAnimation();
  const machinery = dayMeta[day]?.machinery || [];
  const item = machinery[0];
  if (!item || !Array.isArray(item.path) || item.path.length < 2) {
    if (machineryMesh) machineryMesh.visible = false;
    _stopSprinklerEmission();
    return;
  }

  const machineType = item.machine_type || 'machine';
  const machine = _ensureMachineryMesh(machineType);
  machine.visible = true;
  machine.castShadow = qualityEnhanced;
  const isSprinkler = machineType === 'sprinkler' || machineType === 'pivot';

  const path = item.path;
  const durationMs = playing ? Math.max(250, Math.round(1000 / (6 * speedMult))) : 1200;
  const startedAt = performance.now();

  const placeAt = (waypoint) => {
    const x = Number(waypoint[0]) || 0.0;
    const z = Number(waypoint[1]) || 0.0;
    const y = _terrainYAt(z, x) + 0.22;
    machine.position.set(x, y, z);
    if (isSprinkler) _emitSprinklerBurst(x, y + 0.28, z);
  };

  const tick = (now) => {
    const t = Math.min(1.0, (now - startedAt) / durationMs);
    const scaled = t * (path.length - 1);
    const i = Math.min(path.length - 2, Math.floor(scaled));
    const localT = scaled - i;
    const a = path[i];
    const b = path[i + 1];
    const x = (Number(a[0]) || 0.0) + ((Number(b[0]) || 0.0) - (Number(a[0]) || 0.0)) * localT;
    const z = (Number(a[1]) || 0.0) + ((Number(b[1]) || 0.0) - (Number(a[1]) || 0.0)) * localT;
    const y = _terrainYAt(z, x) + 0.22;
    machine.position.set(x, y, z);
    machine.rotation.y = Math.atan2((Number(b[0]) || 0.0) - (Number(a[0]) || 0.0), (Number(b[1]) || 0.0) - (Number(a[1]) || 0.0));
    if (isSprinkler) _emitSprinklerBurst(x, y + 0.28, z);
    if (t < 1.0) machineryAnim = requestAnimationFrame(tick);
    else _stopSprinklerEmission();
  };

  placeAt(path[0]);
  machineryAnim = requestAnimationFrame(tick);
}

function buildInstancedMesh() {
  const n = meta.n_plants;
  const geo = _cylinderGeo();
  const mat = _makePlantMat();
  const im = new THREE.InstancedMesh(geo, mat, n);
  _configurePlantMesh(im, n);
  im.castShadow = qualityEnhanced;
  im.receiveShadow = false;
  im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  im.frustumCulled = true;
  scene.add(im);
  meshes[0] = im;
  mesh = im;   // backward-compat alias for selection/colour helpers
}

// Async GLTF loader - fires after initScene, adds meshes[model_index] when ready.
// ponytail: one loader instance, reused for all URIs.
const _gltfLoader = new GLTFLoader();

async function loadGltfMeshes() {
  const indexMap = meta.model_index_map || {};
  const n = meta.n_plants;
  for (const [uri, idx] of Object.entries(indexMap)) {
    try {
      const gltf = await new Promise((res, rej) => _gltfLoader.load(uri, res, undefined, rej));
      // Extract first BufferGeometry from the loaded scene
      let geo = null;
      gltf.scene.traverse(child => {
        if (!geo && child.isMesh) geo = child.geometry.clone();
      });
      if (!geo) { console.warn(`GLTF ${uri}: no mesh found, using cylinder`); continue; }

      const im = new THREE.InstancedMesh(geo, _makePlantMat(), n);
      _configurePlantMesh(im, n);
      im.castShadow = qualityEnhanced;
      im.receiveShadow = false;
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      im.count = 0;   // hidden until showDay populates it
      im.frustumCulled = true;
      scene.add(im);
      meshes[idx] = im;
      console.log(`GLTF loaded: model_index=${idx} uri=${uri}`);
    } catch (e) {
      console.warn(`GLTF load failed for ${uri}:`, e, '- cylinder fallback active');
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Apply a binary frame to the InstancedMesh
// ---------------------------------------------------------------------------

// ponytail: LOD distance threshold - beyond this, dead plants are skipped entirely.
// Upgrade path: separate billboard InstancedMesh when profiler shows >30% overdraw.
const _LOD_DEAD_SKIP_SQ = 60 * 60;  // 60 world units squared
const _RAIN_PARTICLE_COUNT = 6000;
const _RAIN_HIDE_THRESHOLD_MM = 2.0;

function showDay(day) {
  const frame = frames[day];
  if (!frame || !mesh) return;

  currentDay = day;
  const n = meta.n_plants;
  const cols = meta.cols;
  // ponytail: raised floors - at 0.05/0.08 early seedlings were sub-pixel specks
  const minScale = 0.20;
  const minRadius = 0.14;
  const camX = camera.position.x;
  const camZ = camera.position.z;

  // Reset per-mesh instance counter (each mesh tracks its own count this frame)
  // ponytail: use a plain object literal - cheap reset, no allocation.
  const perMeshCount = {};
  for (const idx of Object.keys(meshes)) perMeshCount[idx] = 0;

  for (let i = 0; i < n; i++) {
    const base = i * FLOATS;
    const x            = frame[base + 0];
    const z            = frame[base + 2];
    const scaleY       = frame[base + 3];
    const radius       = frame[base + 4];
    let r              = frame[base + 5];
    let g              = frame[base + 6];
    let b              = frame[base + 7];
    const alive        = frame[base + 8];
    const model_index  = Math.round(frame[base + 9]);   // 0 = cylinder
    const stage_progress = frame[base + 10];
    const morph_weight = frame[base + 11];
    const stress_ks = frame[base + 12];
    const disease_severity = frame[base + 13];

    // Route to the correct InstancedMesh; fall back to cylinder (index 0) if not loaded yet
    const targetMesh = meshes[model_index] || meshes[0];
    const slot = perMeshCount[model_index] ?? perMeshCount[0];
    const useIndex0 = !meshes[model_index];
    const slotMesh = useIndex0 ? meshes[0] : meshes[model_index];
    // ponytail: instanceId within each mesh = its own sequential slot index
    const slotIdx = useIndex0 ? (perMeshCount[0]++) : (perMeshCount[model_index]++);
    const morphAttr = slotMesh.geometry.attributes.cfMorphWeight;
    const wiltAttr = slotMesh.geometry.attributes.cfWiltWeight;
    const diseaseAttr = slotMesh.geometry.attributes.cfDiseaseSeverity;
    if (morphAttr) morphAttr.setX(slotIdx, Number.isFinite(morph_weight) ? morph_weight : stage_progress);
    if (wiltAttr) wiltAttr.setX(slotIdx, _wiltWeightFromStress(stress_ks));
    if (diseaseAttr) diseaseAttr.setX(slotIdx, Number.isFinite(disease_severity) ? disease_severity : 0.0);

    // LOD: skip distant dead plants
    if (!alive) {
      const dx = x - camX, dz = z - camZ;
      if (dx * dx + dz * dz > _LOD_DEAD_SKIP_SQ) {
        _scl.set(0, 0, 0);
        _pos.set(x, 0, z);
        _mat.compose(_pos, _quat, _scl);
        slotMesh.setMatrixAt(slotIdx, _mat);
        continue;
      }
    }

    let elevY = 0.0;
    if (terrainGrid) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      elevY = (_terrainElevScale > 0)
        ? (terrainGrid[row * terrainCols + col] - _terrainEMin) * _terrainElevScale
        : 0.0;
    }

    if (i === selectedInstance) { r = 1.0; g = 1.0; b = 0.0; }

    const clampedH = Math.max(scaleY, minScale) * (1.0 + stage_progress * 0.05);
    _pos.set(x, elevY + clampedH * 0.5, z);
    _scl.set(Math.max(radius, minRadius), clampedH, Math.max(radius, minRadius));
    _mat.compose(_pos, _quat, _scl);

    slotMesh.setMatrixAt(slotIdx, _mat);
    slotMesh.setColorAt(slotIdx, colourBuf.setRGB(r, g, b));
  }

  // Flush all active meshes; hide overflow slots by zeroing unused trailing instances
  for (const [idxStr, im] of Object.entries(meshes)) {
    const usedCount = perMeshCount[idxStr] ?? 0;
    // Zero any slots beyond what we wrote this frame (in case count shrank)
    if (usedCount < im.count) {
      _scl.set(0, 0, 0); _pos.set(0, 0, 0);
      _mat.compose(_pos, _quat, _scl);
      for (let j = usedCount; j < im.count; j++) im.setMatrixAt(j, _mat);
    }
    im.count = usedCount;
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    if (im.geometry.attributes.cfMorphWeight) im.geometry.attributes.cfMorphWeight.needsUpdate = true;
    if (im.geometry.attributes.cfWiltWeight) im.geometry.attributes.cfWiltWeight.needsUpdate = true;
    if (im.geometry.attributes.cfDiseaseSeverity) im.geometry.attributes.cfDiseaseSeverity.needsUpdate = true;
    if (day === meta.days[0]) im.geometry.computeBoundingSphere();
  }

  // Update selection box position if a plant is selected
  if (selectedInstance >= 0) {
    updateSelectionBox(selectedInstance, frame);
  }

  // HUD
  hudDay.textContent = `Day ${day} / ${meta.days[meta.days.length - 1]}`;
  hudInfo.textContent = `${n} plants  |  ${meta.rows}x${meta.cols} grid`;

  // Notify parent Dash app about day change (for scrubber sync)
  if (window.parent !== window) {
    window.parent.postMessage({ type: 'cf_day_changed', day: day }, '*');
  }

  _animateMachineryForDay(day);
  _updateRainForDay(day);
  _renderWeedsForDay(day);
}

// ---------------------------------------------------------------------------
// 4. Raycasting - click handler
// ---------------------------------------------------------------------------

function onCanvasClick(event) {
  if (!mesh || !meta) return;

  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(pointer, camera);
  // Check all loaded plant meshes (cylinder + any GLTF meshes)
  const allMeshes = Object.values(meshes);
  const hits = raycaster.intersectObjects(allMeshes);

  if (hits.length === 0) {
    clearSelection();
    return;
  }

  const instanceId = hits[0].instanceId;
  if (instanceId === undefined) return;

  selectPlant(instanceId);
}

function selectPlant(instanceId) {
  // Deselect old
  if (selectedInstance >= 0 && selectedInstance !== instanceId) {
    restorePlantColour(selectedInstance);
  }

  selectedInstance = instanceId;

  // Compute row/col from instanceId (row-major: instanceId = row * cols + col)
  const row = Math.floor(instanceId / meta.cols);
  const col = instanceId % meta.cols;
  const plantId = `r${String(row).padStart(2, '0')}c${String(col).padStart(2, '0')}`;

  // Apply highlight colour (bright yellow)
  mesh.setColorAt(instanceId, colourBuf.setRGB(1.0, 1.0, 0.0));
  mesh.instanceColor.needsUpdate = true;

  // Draw wireframe selection box around the plant
  const frame = frames[currentDay];
  if (frame) updateSelectionBox(instanceId, frame);

  // Update HUD
  hudInfo.textContent = `Selected: ${plantId}  (row ${row}, col ${col})  |  Day ${currentDay}`;

  // Notify parent Dash app
  if (window.parent !== window) {
    window.parent.postMessage({
      type: 'PLANT_CLICKED',
      plant_id: plantId,
      row: row,
      col: col,
      day: currentDay,
      instance: instanceId,
    }, '*');
  }
}

function restorePlantColour(instanceId) {
  if (!frames[currentDay] || selectedInstance < 0) return;
  const frame = frames[currentDay];
  const base = instanceId * FLOATS;
  const r = frame[base + 5];
  const g = frame[base + 6];
  const b = frame[base + 7];
  mesh.setColorAt(instanceId, colourBuf.setRGB(r, g, b));
  mesh.instanceColor.needsUpdate = true;
}

function clearSelection() {
  if (selectedInstance >= 0) {
    restorePlantColour(selectedInstance);
  }
  selectedInstance = -1;
  removeSelectionBox();
  hudInfo.textContent = `${meta.n_plants} plants  |  ${meta.rows}x${meta.cols} grid`;

  // Notify parent to close Panel 4
  if (window.parent !== window) {
    window.parent.postMessage({ type: 'PLANT_DESELECTED' }, '*');
  }
}

// ---------------------------------------------------------------------------
// 5. Wireframe selection box
// ---------------------------------------------------------------------------

function updateSelectionBox(instanceId, frame) {
  removeSelectionBox();

  const base = instanceId * FLOATS;
  const cx = frame[base + 0];          // x centre
  const halfH = frame[base + 1];
  const cz = frame[base + 2];          // z centre
  const height = frame[base + 3] || 0.05;
  const radius = frame[base + 4] || 0.15;

  // Build a box slightly larger than the cylinder bounding box
  const pad = 0.1;
  const w = radius * 2 + pad;
  const h = height + pad;
  const geo = new THREE.BoxGeometry(w, h, w);
  const edges = new THREE.EdgesGeometry(geo);
  const mat = new THREE.LineBasicMaterial({
    color: 0xffff00,
    linewidth: 2,
    depthTest: false,
  });

  const row = Math.floor(instanceId / meta.cols);
  const col = instanceId % meta.cols;
  let elevY = 0.0;
  if (terrainGrid) {
    elevY = (_terrainElevScale > 0)
      ? (terrainGrid[row * terrainCols + col] - _terrainEMin) * _terrainElevScale
      : 0.0;
  }

  selectionBox = new THREE.LineSegments(edges, mat);
  selectionBox.position.set(cx, elevY + halfH, cz);
  selectionBox.renderOrder = 999;
  scene.add(selectionBox);
}

function removeSelectionBox() {
  if (selectionBox) {
    scene.remove(selectionBox);
    selectionBox.geometry.dispose();
    selectionBox.material.dispose();
    selectionBox = null;
  }
}

// ---------------------------------------------------------------------------
// 6. Colour variable rebuild
// ---------------------------------------------------------------------------

async function rebuildColour(variable) {
  loaderStatus.textContent = `Rebuilding colours for ${variable}...`;
  loader.style.display = 'flex';
  loader.classList.remove('hidden');
  loaderFill.style.width = '30%';

  try {
    const r = await fetch(`${API}/api/buffer/rebuild?variable=${variable}`);
    if (!r.ok) throw new Error(`rebuild HTTP ${r.status}`);
    const result = await r.json();

    loaderFill.style.width = '60%';
    setStatus('Reloading frames...', 60);

    // Re-fetch all frames
    frames = {};
    const days = meta.days;
    for (let i = 0; i < days.length; i++) {
      const day = days[i];
      const fr = await fetch(`${API}/api/buffer?day=${day}`);
      if (fr.ok) frames[day] = new Float32Array(await fr.arrayBuffer());
      const pct = 60 + Math.round(((i + 1) / days.length) * 38);
      loaderFill.style.width = pct + '%';
    }

    // Re-fetch meta for updated vmin/vmax
    const mr = await fetch(`${API}/api/buffer/meta`);
    if (mr.ok) meta = await mr.json();

    updateLegendMeta();
    showDay(currentDay);

  } catch (e) {
    console.error(e);
  } finally {
    loader.classList.add('hidden');
    setTimeout(() => { loader.style.display = 'none'; }, 500);
  }
}

function updateLegendMeta() {
  if (!meta) return;
  const v = meta.variable || 'biomass_g';
  const labels = {
    biomass_g: 'Biomass (g/plant)',
    lai: 'LAI (m2/m2)',
    weed_lai: 'Weed LAI',
    height_cm: 'Height (cm)',
    stress_index: 'Stress Index',
  };
  const gradients = {
    biomass_g: 'linear-gradient(90deg, #d4a60a, #2e8b57)',
    lai: 'linear-gradient(90deg, #8fbc00, #228b22)',
    weed_lai: 'linear-gradient(90deg, #d9c98e, #3f7f3a)',
    height_cm: 'linear-gradient(90deg, #5599cc, #003399)',
    stress_index: 'linear-gradient(90deg, #2e8b57, #cc2200)',
  };
  legendTitle.textContent = labels[v] || v;
  legendBar.style.background = gradients[v] || gradients.biomass_g;
  legendMin.textContent = (meta.vmin || 0).toFixed(1);
  legendMax.textContent = (meta.vmax || 0).toFixed(1);
}

// ---------------------------------------------------------------------------
// 7. Playback
// ---------------------------------------------------------------------------

function startPlayback() {
  if (playTimer) clearInterval(playTimer);
  const fps = 6 * speedMult;
  const intervalMs = Math.round(1000 / fps);
  playTimer = setInterval(() => {
    const days = meta.days;
    const idx = days.indexOf(currentDay);
    const next = days[(idx + 1) % days.length];
    showDay(next);
  }, intervalMs);
  playing = true;
  btnPlay.textContent = 'Pause';
  btnPlay.classList.add('active');
}

function stopPlayback() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  playing = false;
  btnPlay.textContent = 'Play';
  btnPlay.classList.remove('active');
}

btnPlay.addEventListener('click', () => {
  if (playing) stopPlayback(); else startPlayback();
});

function setSpeed(m) {
  speedMult = m;
  [btn1x, btn2x, btn5x].forEach(b => b.classList.remove('active'));
  document.getElementById(`btn-${m}x`).classList.add('active');
  if (playing) { stopPlayback(); startPlayback(); }
}
btn1x.addEventListener('click', () => setSpeed(1));
btn2x.addEventListener('click', () => setSpeed(2));
btn5x.addEventListener('click', () => setSpeed(5));

// ---------------------------------------------------------------------------
// Export scene as .glb - captures terrain + plant meshes for current day
// ---------------------------------------------------------------------------

function exportSceneAsGlb() {
  if (!scene) return;
  const exporter = new GLTFExporter();
  // ponytail: export whole scene; binary=true -> single .glb blob
  exporter.parse(
    scene,
    (glb) => {
      const blob = new Blob([glb], { type: 'model/gltf-binary' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `cropforge_scene_day_${currentDay}.glb`;
      a.click();
      URL.revokeObjectURL(url);
    },
    (err) => console.error('GLTFExporter error:', err),
    { binary: true },
  );
}

if (btnExportGlb) btnExportGlb.addEventListener('click', exportSceneAsGlb);


// ---------------------------------------------------------------------------
// 8. postMessage listener (Dash -> iframe day sync + plant selection)
// ---------------------------------------------------------------------------

window.addEventListener('message', (event) => {
  if (!event.data || typeof event.data !== 'object') return;

  if (event.data.type === 'cf_set_day') {
    const day = parseInt(event.data.day, 10);
    if (!isNaN(day) && frames[day]) {
      stopPlayback();
      showDay(day);
    }
  }

  if (event.data.type === 'cf_set_field') {
    /* PRD v0.2.0 Section8 - field selector change triggers a full re-bootstrap.
       We tear down the existing scene/renderer, show the loader, and
       reload all frames for the new field from the server. */
    const newField = event.data.field;
    if (!newField || newField === currentField) return;

    stopPlayback();
    clearSelection();
    _stopMachineryAnimation();

    // Tear down Three.js objects
    if (scene) {
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
          else obj.material.dispose();
        }
      });
      scene.clear();
    }
    if (renderer) {
      if (composer) {
        composer.dispose();
        composer = null;
      }
      renderer.dispose();
      const wrapper = document.getElementById('canvas-wrapper');
      if (wrapper && renderer.domElement && wrapper.contains(renderer.domElement)) {
        wrapper.removeChild(renderer.domElement);
      }
      renderer = null;
    }
    scene = null; camera = null; controls = null; mesh = null; soilMesh = null; machineryMesh = null;
    rainSystem = null; rainPositions = null; rainActiveCount = 0;
    sprinklerSystem = null; sprinklerPositions = null;
    weedTintMesh = null; weedMesh = null;
    frames = {}; dayMeta = {}; meta = null; terrainGrid = null; terrainRows = 0; terrainCols = 0;
    _terrainChunks = [];

    // Show loader again before re-bootstrapping
    loader.style.display = 'flex';
    loader.classList.remove('hidden');
    loaderFill.style.width = '0%';

    bootstrap(newField);
  }

  if (event.data.type === 'cf_set_variable') {
    varSelect.value = event.data.variable;
    rebuildColour(event.data.variable);
  }
  if (event.data.type === 'cf_deselect') {
    clearSelection();
  }
});

varSelect.addEventListener('change', () => {
  rebuildColour(varSelect.value);
});

// ---------------------------------------------------------------------------
// 9. Animation loop
// ---------------------------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);
  // Guard: renderer can be null during field-switch teardown
  if (!renderer) return;
  if (controls) controls.update();
  updateTerrainLOD();
  _animateRain();
  if (composer) {
    composer.render();
  } else if (scene && camera) {
    renderer.render(scene, camera);
  }
}

function onResize() {
  const wrapper = document.getElementById('canvas-wrapper');
  const W = wrapper.clientWidth;
  const H = wrapper.clientHeight;
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
  renderer.setSize(W, H);
  if (composer) composer.setSize(W, H);
}

// ---------------------------------------------------------------------------
// 10. Start
// ---------------------------------------------------------------------------

if (!window.__cfFallbackViewportStarted) {
  bootstrap(null).catch(failToFallback);
}
