/*
 * Offline-safe CropForge viewport fallback.
 *
 * The full renderer in main.js depends on Three.js ES modules. If the browser
 * cannot fetch those modules, this lightweight canvas view still renders the
 * field from the same binary buffer API and lets the Dash scrubber drive days.
 */
(function () {
  const FLOATS = 14;
  const API = window.location.origin;
  let started = false;
  let meta = null;
  let currentDay = 1;
  let frames = {};

  const loader = document.getElementById('loader');
  const loaderFill = document.getElementById('loader-bar-fill');
  const loaderStatus = document.getElementById('loader-status');
  const wrapper = document.getElementById('canvas-wrapper');
  const hudDay = document.getElementById('hud-day');
  const hudInfo = document.getElementById('hud-info');
  const legendTitle = document.getElementById('legend-title');
  const legendMin = document.getElementById('legend-min');
  const legendMax = document.getElementById('legend-max');
  const btnPlay = document.getElementById('btn-play');
  const btn1x = document.getElementById('btn-1x');
  const btn2x = document.getElementById('btn-2x');
  const btn5x = document.getElementById('btn-5x');
  const varSelect = document.getElementById('var-select');

  let canvas = null;
  let ctx = null;
  let playing = false;
  let speed = 1;
  let timer = null;

  function setStatus(message, pct) {
    if (loaderStatus) loaderStatus.textContent = message;
    if (loaderFill && pct !== undefined) loaderFill.style.width = pct + '%';
  }

  function hideLoader() {
    if (!loader) return;
    loader.classList.add('hidden');
    setTimeout(() => { loader.style.display = 'none'; }, 450);
  }

  function postReady() {
    try {
      window.parent.postMessage({ type: 'LOAD_COMPLETE', fallback: true }, window.location.origin);
    } catch (_err) {
      // Parent messaging is optional for standalone viewport use.
    }
  }

  function resizeCanvas() {
    if (!canvas || !wrapper) return;
    const w = Math.max(1, wrapper.clientWidth);
    const h = Math.max(1, wrapper.clientHeight);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(currentDay);
  }

  function ensureCanvas() {
    if (canvas) return;
    canvas = document.createElement('canvas');
    canvas.setAttribute('aria-label', 'CropForge field viewport fallback');
    canvas.style.display = 'block';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    wrapper.appendChild(canvas);
    ctx = canvas.getContext('2d');
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  }

  async function fetchFrame(day) {
    if (frames[day]) return frames[day];
    const r = await fetch(`${API}/api/buffer?day=${day}`);
    if (!r.ok) throw new Error(`buffer HTTP ${r.status}`);
    frames[day] = new Float32Array(await r.arrayBuffer());
    return frames[day];
  }

  function colour(r, g, b, alpha) {
    return `rgba(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)}, ${alpha})`;
  }

  function draw(day) {
    if (!ctx || !meta || !frames[day]) return;

    const frame = frames[day];
    const w = wrapper.clientWidth;
    const h = wrapper.clientHeight;
    const cols = Math.max(1, meta.cols || 1);
    const rows = Math.max(1, meta.rows || 1);
    const pad = 42;
    const scale = Math.min((w - pad * 2) / cols, (h - pad * 2) / rows);
    const originX = (w - cols * scale) * 0.5;
    const originY = (h - rows * scale) * 0.5 + rows * scale * 0.08;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#9c9c9c';
    ctx.fillRect(0, 0, w, h);

    ctx.save();
    ctx.translate(originX, originY);
    ctx.fillStyle = '#c8a97e';
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.lineWidth = 1;
    ctx.fillRect(0, 0, cols * scale, rows * scale);

    const n = Math.min(meta.n_plants || 0, Math.floor(frame.length / FLOATS));
    for (let i = 0; i < n; i++) {
      const base = i * FLOATS;
      const x = frame[base + 0] || 0;
      const z = frame[base + 2] || 0;
      const height = Math.max(2, (frame[base + 3] || 0.1) * scale * 0.55);
      const radius = Math.max(2, (frame[base + 4] || 0.08) * scale * 1.2);
      const alive = frame[base + 8] > 0.5;
      const px = x * scale + scale * 0.5;
      const py = z * scale + scale * 0.5;
      const r = alive ? frame[base + 5] : 0.45;
      const g = alive ? frame[base + 6] : 0.36;
      const b = alive ? frame[base + 7] : 0.12;

      ctx.fillStyle = 'rgba(0,0,0,0.16)';
      ctx.beginPath();
      ctx.ellipse(px + 2, py + 2, radius * 1.4, radius * 0.7, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = colour(r * 0.65, g * 0.65, b * 0.65, 1);
      ctx.lineWidth = Math.max(1, radius);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px, py - height);
      ctx.stroke();

      ctx.fillStyle = colour(r, g, b, 0.95);
      ctx.beginPath();
      ctx.arc(px, py - height, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.24)';
    ctx.strokeRect(0.5, 0.5, cols * scale - 1, rows * scale - 1);
    ctx.restore();

    if (hudDay) hudDay.textContent = `Day ${day} / ${meta.days[meta.days.length - 1]}`;
    if (hudInfo) hudInfo.textContent = `${meta.n_plants} plants | ${rows}x${cols} grid | fallback renderer`;
    if (legendTitle) legendTitle.textContent = meta.variable || 'Biomass (g/plant)';
    if (legendMin) legendMin.textContent = Number(meta.vmin || 0).toFixed(1);
    if (legendMax) legendMax.textContent = Number(meta.vmax || 0).toFixed(1);
  }

  async function showDay(day) {
    currentDay = day;
    await fetchFrame(day);
    draw(day);
  }

  function setSpeed(nextSpeed) {
    speed = nextSpeed;
    [btn1x, btn2x, btn5x].forEach((btn) => btn && btn.classList.remove('active'));
    const active = document.getElementById(`btn-${nextSpeed}x`);
    if (active) active.classList.add('active');
    if (playing) {
      stop();
      play();
    }
  }

  function play() {
    if (!meta || timer) return;
    const fps = 4 * speed;
    timer = setInterval(() => {
      const days = meta.days || [currentDay];
      const idx = days.indexOf(currentDay);
      showDay(days[(idx + 1) % days.length]);
    }, Math.round(1000 / fps));
    playing = true;
    if (btnPlay) {
      btnPlay.textContent = 'Pause';
      btnPlay.classList.add('active');
    }
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    playing = false;
    if (btnPlay) {
      btnPlay.textContent = 'Play';
      btnPlay.classList.remove('active');
    }
  }

  async function start(reason) {
    if (started || window.__cfThreeViewportReady) return;
    started = true;
    window.__cfFallbackViewportStarted = true;
    setStatus(reason ? `Starting fallback renderer (${reason})` : 'Starting fallback renderer', 5);

    try {
      const mr = await fetch(`${API}/api/buffer/meta`);
      if (!mr.ok) throw new Error(`meta HTTP ${mr.status}`);
      meta = await mr.json();
      ensureCanvas();
      const firstDay = (meta.days && meta.days[0]) || 1;
      await showDay(firstDay);
      hideLoader();
      postReady();
    } catch (err) {
      setStatus(`Viewport error: ${err.message}`, 0);
      console.error(err);
    }
  }

  if (btnPlay) btnPlay.addEventListener('click', () => (playing ? stop() : play()));
  if (btn1x) btn1x.addEventListener('click', () => setSpeed(1));
  if (btn2x) btn2x.addEventListener('click', () => setSpeed(2));
  if (btn5x) btn5x.addEventListener('click', () => setSpeed(5));
  if (varSelect) varSelect.addEventListener('change', () => draw(currentDay));

  window.addEventListener('message', (event) => {
    if (!event.data || typeof event.data !== 'object' || !started) return;
    if (event.data.type === 'cf_set_day') {
      const day = parseInt(event.data.day, 10);
      if (!Number.isNaN(day)) {
        stop();
        showDay(day);
      }
    }
  });

  window.CropForgeFallback = { start };
}());
