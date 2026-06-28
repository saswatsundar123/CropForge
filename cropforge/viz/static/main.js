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
 * Binary frame layout (9 × float32 per plant, 36 bytes):
 *   [0] x          grid col position (world units)
 *   [1] y          half-height (cylinder centre above ground)
 *   [2] z          grid row position (world units)
 *   [3] scaleY     full cylinder height (world units)
 *   [4] radius     cylinder radius
 *   [5] r          colour red   [0..1]
 *   [6] g          colour green [0..1]
 *   [7] b          colour blue  [0..1]
 *   [8] alive      1.0 = alive, 0.0 = dead
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

const API       = window.location.origin;           // same origin as iframe parent
const FLOATS    = 9;                                 // float32 values per plant
const BYTES     = FLOATS * 4;                        // bytes per plant

let meta         = null;   // buffer metadata from /api/buffer/meta
let frames       = {};     // {day: Float32Array}  — all preloaded frames
let currentDay   = 1;
let currentField = null;   // active field name (null = server default)
let playing      = false;
let speedMult    = 1;
let playTimer    = null;
let mesh         = null;   // THREE.InstancedMesh
let soilMesh     = null;
let renderer, scene, camera, controls;

// Raycasting
const raycaster        = new THREE.Raycaster();
const pointer          = new THREE.Vector2();
let   selectedInstance = -1;    // instanceId of currently selected plant
let   selectionBox     = null;  // THREE.LineSegments wireframe around selection
let   selectionOrigColors = {}; // {instanceId: THREE.Color}  — backup for deselect

const dummy     = new THREE.Object3D();
const colourBuf = new THREE.Color();

// ---------------------------------------------------------------------------
// DOM handles
// ---------------------------------------------------------------------------

const loader       = document.getElementById('loader');
const loaderFill   = document.getElementById('loader-bar-fill');
const loaderStatus = document.getElementById('loader-status');
const hudDay       = document.getElementById('hud-day');
const hudInfo      = document.getElementById('hud-info');
const legendTitle  = document.getElementById('legend-title');
const legendMin    = document.getElementById('legend-min');
const legendMax    = document.getElementById('legend-max');
const legendBar    = document.getElementById('legend-bar');
const varSelect    = document.getElementById('var-select');
const btnPlay      = document.getElementById('btn-play');
const btn1x        = document.getElementById('btn-1x');
const btn2x        = document.getElementById('btn-2x');
const btn5x        = document.getElementById('btn-5x');

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

  initScene();
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
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 200;
  sun.shadow.camera.left  = -fieldW * 0.7;
  sun.shadow.camera.right =  fieldW * 1.3;
  sun.shadow.camera.top   =  fieldD * 1.3;
  sun.shadow.camera.bottom = -fieldD * 0.7;
  scene.add(sun);

  const fill = new THREE.HemisphereLight(0x4060a0, 0x2c4a2c, 0.35);
  scene.add(fill);

  // Ground plane (soil surface texture)
  buildGroundPlane(fieldW, fieldD);

  // Grid helper (subtle)
  const grid = new THREE.GridHelper(
    Math.max(fieldW, fieldD) + 4, Math.max(meta.cols, meta.rows) + 2,
    0xEAEAEA, 0xF9F9F8
  );
  grid.position.set(fieldW * 0.5 - 0.5, -0.01, fieldD * 0.5 - 0.5);
  scene.add(grid);

  // Build instanced plant mesh
  buildInstancedMesh();

  // ---- Raycasting click listener -------------------------------------------
  renderer.domElement.addEventListener('click', onCanvasClick, false);

  // Window resize
  window.addEventListener('resize', onResize);
}

function buildGroundPlane(fieldW, fieldD) {
  const geo = new THREE.PlaneGeometry(fieldW + 4, fieldD + 4, 1, 1);
  const mat = new THREE.MeshLambertMaterial({
    color: 0xEAEAEA,
    side: THREE.FrontSide,
  });
  soilMesh = new THREE.Mesh(geo, mat);
  soilMesh.rotation.x = -Math.PI / 2;
  soilMesh.position.set(fieldW * 0.5 - 0.5, -0.02, fieldD * 0.5 - 0.5);
  soilMesh.receiveShadow = true;
  scene.add(soilMesh);
}

function buildInstancedMesh() {
  const n = meta.n_plants;

  // Cylinder: height=1 (scaled per instance), radius=1 (scaled per instance)
  // radialSegments=6 → hexagonal prism (fast, looks like a plant stem)
  const geo = new THREE.CylinderGeometry(1, 1, 1, 6, 1);
  const mat = new THREE.MeshLambertMaterial({ vertexColors: true });

  // Use instanced colour buffer attribute
  const colours = new Float32Array(n * 3);
  geo.setAttribute('color', new THREE.InstancedBufferAttribute(colours, 3));

  mesh = new THREE.InstancedMesh(geo, mat, n);
  mesh.castShadow = true;
  mesh.receiveShadow = false;
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(mesh);
}

// ---------------------------------------------------------------------------
// 3. Apply a binary frame to the InstancedMesh
// ---------------------------------------------------------------------------

function showDay(day) {
  const frame = frames[day];
  if (!frame || !mesh) return;

  currentDay = day;
  const n = meta.n_plants;

  for (let i = 0; i < n; i++) {
    const base = i * FLOATS;

    const x      = frame[base + 0];
    const halfH  = frame[base + 1];
    const z      = frame[base + 2];
    const scaleY = frame[base + 3];
    const radius = frame[base + 4];
    let   r      = frame[base + 5];
    let   g      = frame[base + 6];
    let   b      = frame[base + 7];
    // alive = frame[base + 8]  (informational)

    // Keep the highlight colour for the selected plant
    if (i === selectedInstance) {
      r = 1.0; g = 1.0; b = 0.0;  // bright yellow highlight
    }

    // Build transform: translate to (x, halfH, z), scale (radius, scaleY, radius)
    // Minimum scale enforced so plants are always visible even at Day 1 low-biomass
    const minScale = 0.05;
    const minRadius = 0.08;
    dummy.position.set(x, Math.max(halfH, minScale * 0.5), z);
    dummy.scale.set(Math.max(radius, minRadius), Math.max(scaleY, minScale), Math.max(radius, minRadius));
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    mesh.setColorAt(i, colourBuf.setRGB(r, g, b));
  }

  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate  = true;

  // Update selection box position if a plant is selected
  if (selectedInstance >= 0) {
    updateSelectionBox(selectedInstance, frame);
  }

  // HUD
  hudDay.textContent  = `Day ${day} / ${meta.days[meta.days.length - 1]}`;
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

  // Prevent firing when OrbitControls has just dragged (use a tiny drag threshold)
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width)  * 2 - 1;
  pointer.y = -((event.clientY - rect.top)  / rect.height) * 2 + 1;

  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(mesh);

  if (hits.length === 0) {
    // Clicked empty space → deselect
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
  const row     = Math.floor(instanceId / meta.cols);
  const col     = instanceId % meta.cols;
  const plantId = `r${String(row).padStart(2,'0')}c${String(col).padStart(2,'0')}`;

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
      type:      'PLANT_CLICKED',
      plant_id:  plantId,
      row:       row,
      col:       col,
      day:       currentDay,
      instance:  instanceId,
    }, '*');
  }
}

function restorePlantColour(instanceId) {
  if (!frames[currentDay] || selectedInstance < 0) return;
  const frame = frames[currentDay];
  const base  = instanceId * FLOATS;
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

  const base   = instanceId * FLOATS;
  const cx     = frame[base + 0];          // x centre
  const halfH  = frame[base + 1];
  const cz     = frame[base + 2];          // z centre
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
  selectionBox = new THREE.LineSegments(edges, mat);
  selectionBox.position.set(cx, halfH, cz);
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
    biomass_g:   'Biomass (g/plant)',
    lai:         'LAI (m²/m²)',
    height_cm:   'Height (cm)',
    stress_index:'Stress Index',
  };
  const gradients = {
    biomass_g:    'linear-gradient(90deg, #d4a60a, #2e8b57)',
    lai:          'linear-gradient(90deg, #8fbc00, #228b22)',
    height_cm:    'linear-gradient(90deg, #5599cc, #003399)',
    stress_index: 'linear-gradient(90deg, #2e8b57, #cc2200)',
  };
  legendTitle.textContent     = labels[v] || v;
  legendBar.style.background  = gradients[v] || gradients.biomass_g;
  legendMin.textContent       = (meta.vmin || 0).toFixed(1);
  legendMax.textContent       = (meta.vmax || 0).toFixed(1);
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
    const idx  = days.indexOf(currentDay);
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
    frames = {}; meta = null;

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
