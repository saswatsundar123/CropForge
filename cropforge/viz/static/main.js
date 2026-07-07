/**
 * main.js — CropForge 3D Viewport
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
 *   - Raycasting: click a plant → highlight + postMessage PLANT_CLICKED
 *
 * Binary frame layout (11 × float32 per plant, 44 bytes):
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
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

const API = window.location.origin;           // same origin as iframe parent
const FLOATS = 11;                                // float32 values per plant (v0.9.0 Phase 2)
const BYTES = FLOATS * 4;                        // bytes per plant

let meta = null;   // buffer metadata from /api/buffer/meta
let qualityEnhanced = false;  // true when quality="enhanced" (PBR shadows + HDR)
let terrainGrid = null;   // Float32Array [rows*cols] row-major elevation values (v0.6.0)
let terrainRows = 0;
let terrainCols = 0;
let frames = {};     // {day: Float32Array}  — all preloaded frames
let currentDay = 1;
let currentField = null;   // active field name (null = server default)
let playing = false;
let speedMult = 1;
let playTimer = null;
// v0.9.0 Phase 3: meshes map, keyed by model_index (0 = cylinder fallback).
// ponytail: keep `mesh` as alias to meshes[0] — all selection/colour code unchanged.
let meshes = {};   // {model_index: THREE.InstancedMesh}
let mesh = null;   // alias → meshes[0], cylinder fallback
let soilMesh = null;
let renderer, scene, camera, controls;

// ponytail: cache terrain elev constants once at init — not per frame, not per plant
let _terrainEMin = 0.0;
let _terrainElevScale = 0.0;
// ponytail: pre-allocated objects reused every frame — avoid GC churn in hot loop
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
const _LOD_HI_SQ  = 55 * 55;  // hysteresis — switch back to hi when closer than this

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
  /* fieldName: string | null — pass null to load the server-default field */
  if (fieldName !== undefined && fieldName !== null) {
    currentField = fieldName;
  }

  const fieldParam = currentField ? `?field=${encodeURIComponent(currentField)}` : '';

  setStatus('Fetching session metadata…', 2);

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

  setStatus(`Preloading ${meta.n_days} days × ${meta.n_plants} plants…`, 5);

  frames = {};   // clear old frames
  const days = meta.days;

  for (let i = 0; i < days.length; i++) {
    const day = days[i];
    try {
      const sep = fieldParam ? '&' : '?';
      const r = await fetch(`${API}/api/buffer?day=${day}${fieldParam ? fieldParam.replace('?', '&') : ''}`);

      if (!r.ok) throw new Error(`buffer HTTP ${r.status} day=${day}`);
      const ab = await r.arrayBuffer();
      frames[day] = new Float32Array(ab);
    } catch (e) {
      console.error(e);
    }
    // Update loading bar
    const pct = Math.round(((i + 1) / days.length) * 100);
    loaderFill.style.width = pct + '%';
    setStatus(`Preloading day ${day} / ${days[days.length - 1]}  (${pct}%)`, pct);
  }

  setStatus('Building 3D scene…', 98);
  await new Promise(r => setTimeout(r, 50));  // yield to let browser paint

  // v0.6.0 — Fetch terrain elevation grid (modified by LandPrep if any)
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
  loader.classList.add('hidden');
  setTimeout(() => { loader.style.display = 'none'; }, 650);

  // Notify Dash parent that viewport is fully loaded (PRD v0.5.0 §4.5)
  // The Dash layout listens for this and hides the Dash-level loading overlay.
  try {
    window.parent.postMessage(
      { type: 'LOAD_COMPLETE', total_days: days.length },
      window.location.origin
    );
  } catch (e) {
    // cross-origin or no parent — safe to ignore
  }

  // Update legend
  updateLegendMeta();

  // Start animation loop (idempotent — animate() guards with RAF)
  animate();
}

function setStatus(msg, pct) {
  loaderStatus.textContent = msg;
  if (pct !== undefined) loaderFill.style.width = pct + '%';
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
  wrapper.appendChild(renderer.domElement);

  // Scene — bright light background
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9c9c9c);
  scene.fog = new THREE.FogExp2(0x9c9c9c, 0.016);

  // Camera — isometric-ish perspective above the field
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
  // ponytail: shadows are expensive — only enable in enhanced quality mode
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

  // Ground plane (soil surface) — chunked LOD terrain (v0.8.0 Phase 5)
  buildTerrainChunks(fieldW, fieldD);

  // Grid helper (subtle) — skip for huge grids to avoid overdraw
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

  // PBR: MeshStandardMaterial for terrain — roughness=0.9 for soil, low metalness
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
// Plant geometry factory — cylinder (index=0) or GLTF-sourced BufferGeometry
// ---------------------------------------------------------------------------

function _cylinderGeo() {
  return new THREE.CylinderGeometry(1, 1, 1, 6, 1);
}

function _makePlantMat() {
  // PBR plant material — instanced colour via instanceColor
  return new THREE.MeshStandardMaterial({
    roughness: 0.6,
    metalness: 0.0,
  });
}

function buildInstancedMesh() {
  const n = meta.n_plants;
  const geo = _cylinderGeo();
  const mat = _makePlantMat();
  const im = new THREE.InstancedMesh(geo, mat, n);
  im.castShadow = qualityEnhanced;
  im.receiveShadow = false;
  im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  im.frustumCulled = true;
  scene.add(im);
  meshes[0] = im;
  mesh = im;   // backward-compat alias for selection/colour helpers
}

// Async GLTF loader — fires after initScene, adds meshes[model_index] when ready.
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
      im.castShadow = qualityEnhanced;
      im.receiveShadow = false;
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      im.count = 0;   // hidden until showDay populates it
      im.frustumCulled = true;
      scene.add(im);
      meshes[idx] = im;
      console.log(`GLTF loaded: model_index=${idx} uri=${uri}`);
    } catch (e) {
      console.warn(`GLTF load failed for ${uri}:`, e, '— cylinder fallback active');
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Apply a binary frame to the InstancedMesh
// ---------------------------------------------------------------------------

// ponytail: LOD distance threshold — beyond this, dead plants are skipped entirely.
// Upgrade path: separate billboard InstancedMesh when profiler shows >30% overdraw.
const _LOD_DEAD_SKIP_SQ = 60 * 60;  // 60 world units squared

function showDay(day) {
  const frame = frames[day];
  if (!frame || !mesh) return;

  currentDay = day;
  const n = meta.n_plants;
  const cols = meta.cols;
  const minScale = 0.05;
  const minRadius = 0.08;
  const camX = camera.position.x;
  const camZ = camera.position.z;

  // Reset per-mesh instance counter (each mesh tracks its own count this frame)
  // ponytail: use a plain object literal — cheap reset, no allocation.
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

    // Route to the correct InstancedMesh; fall back to cylinder (index 0) if not loaded yet
    const targetMesh = meshes[model_index] || meshes[0];
    const slot = perMeshCount[model_index] ?? perMeshCount[0];
    const useIndex0 = !meshes[model_index];
    const slotMesh = useIndex0 ? meshes[0] : meshes[model_index];
    // ponytail: instanceId within each mesh = its own sequential slot index
    const slotIdx = useIndex0 ? (perMeshCount[0]++) : (perMeshCount[model_index]++);

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
    if (day === meta.days[0]) im.geometry.computeBoundingSphere();
  }

  // Update selection box position if a plant is selected
  if (selectedInstance >= 0) {
    updateSelectionBox(selectedInstance, frame);
  }

  // HUD
  hudDay.textContent = `Day ${day} / ${meta.days[meta.days.length - 1]}`;
  hudInfo.textContent = `${n} plants  |  ${meta.rows}×${meta.cols} grid`;

  // Notify parent Dash app about day change (for scrubber sync)
  if (window.parent !== window) {
    window.parent.postMessage({ type: 'cf_day_changed', day: day }, '*');
  }
}

// ---------------------------------------------------------------------------
// 4. Raycasting — click handler
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
  hudInfo.textContent = `${meta.n_plants} plants  |  ${meta.rows}×${meta.cols} grid`;

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
  loaderStatus.textContent = `Rebuilding colours for ${variable}…`;
  loader.style.display = 'flex';
  loader.classList.remove('hidden');
  loaderFill.style.width = '30%';

  try {
    const r = await fetch(`${API}/api/buffer/rebuild?variable=${variable}`);
    if (!r.ok) throw new Error(`rebuild HTTP ${r.status}`);
    const result = await r.json();

    loaderFill.style.width = '60%';
    setStatus('Reloading frames…', 60);

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
    lai: 'LAI (m²/m²)',
    height_cm: 'Height (cm)',
    stress_index: 'Stress Index',
  };
  const gradients = {
    biomass_g: 'linear-gradient(90deg, #d4a60a, #2e8b57)',
    lai: 'linear-gradient(90deg, #8fbc00, #228b22)',
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
  btnPlay.textContent = '⏸ Pause';
  btnPlay.classList.add('active');
}

function stopPlayback() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  playing = false;
  btnPlay.textContent = '▶ Play';
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
// Export scene as .glb — captures terrain + plant meshes for current day
// ---------------------------------------------------------------------------

function exportSceneAsGlb() {
  if (!scene) return;
  const exporter = new GLTFExporter();
  // ponytail: export whole scene; binary=true → single .glb blob
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
// 8. postMessage listener (Dash → iframe day sync + plant selection)
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
    /* PRD v0.2.0 §8 — field selector change triggers a full re-bootstrap.
       We tear down the existing scene/renderer, show the loader, and
       reload all frames for the new field from the server. */
    const newField = event.data.field;
    if (!newField || newField === currentField) return;

    stopPlayback();
    clearSelection();

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
      renderer.dispose();
      const wrapper = document.getElementById('canvas-wrapper');
      if (wrapper && renderer.domElement && wrapper.contains(renderer.domElement)) {
        wrapper.removeChild(renderer.domElement);
      }
      renderer = null;
    }
    scene = null; camera = null; controls = null; mesh = null; soilMesh = null;
    frames = {}; meta = null; terrainGrid = null; terrainRows = 0; terrainCols = 0;
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
  if (controls) controls.update();
  updateTerrainLOD();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

function onResize() {
  const wrapper = document.getElementById('canvas-wrapper');
  const W = wrapper.clientWidth;
  const H = wrapper.clientHeight;
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
  renderer.setSize(W, H);
}

// ---------------------------------------------------------------------------
// 10. Start
// ---------------------------------------------------------------------------

bootstrap(null);
