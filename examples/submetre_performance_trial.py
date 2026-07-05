"""
examples/submetre_performance_trial.py
========================================
CropForge v0.8.0 Phase 5 — Terrain LOD Rendering Performance Trial.

Creates a 500×500 cell field at resolution_m=0.2 (0.2m per cell = 100m×100m).
Total cells: 250,000. At 1:1 vertex resolution this would be ~250K vertices in
a single PlaneGeometry — enough to stress older WebGL contexts.

With the Phase 5 chunked LOD renderer:
  - Chunks: ceil(500/64)^2 = 8x8 = 64 chunks
  - Hi-res per chunk: 65x65 = 4,225 vertices
  - Lo-res per chunk (step=4): 17x17 = 289 vertices
  - Distant chunks (>80 world units) drop to lo-res automatically
  - Camera starts ~90 units above → most chunks immediately switch to lo-res
  - Result: initial draw is ~64 × 289 = ~18,000 vertices (18× reduction)

Usage:
    python examples/submetre_performance_trial.py

Then open the printed URL in a browser and verify:
  1. Field loads without WebGL context loss.
  2. Zooming out → polygon count drops (chrome://gpu shows draw calls).
  3. Zooming in → central chunks switch back to full resolution.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""
import numpy as np
from cropforge import Farm, Field, Terrain, TiedRidges
from cropforge.loaders import Weather
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Build a 500×500 sub-metre field with procedural undulating terrain
# ---------------------------------------------------------------------------

ROWS, COLS = 500, 500
RESOLUTION_M = 0.2   # 0.2 m per cell → 100m × 100m field

print(f"Building {ROWS}×{COLS} grid at {RESOLUTION_M}m resolution "
      f"({ROWS * RESOLUTION_M:.0f}m × {COLS * RESOLUTION_M:.0f}m field)...")

# Undulating terrain: sine waves in both axes + linear slope
xs = np.linspace(0, 4 * np.pi, COLS)
zs = np.linspace(0, 2 * np.pi, ROWS)
xg, zg = np.meshgrid(xs, zs)
elev = (
    2.0 * np.sin(xg) * np.cos(zg * 0.5)     # primary undulation
    + 0.5 * np.sin(xg * 3) * np.sin(zg * 3)  # fine detail
    + zg * 0.3                                 # macro slope N→S
)
elev = elev.astype(np.float32)

terrain = Terrain.from_array(elev, resolution_m=RESOLUTION_M)

# ---------------------------------------------------------------------------
# 2. Field setup with TiedRidges (most complex land prep modifier)
# ---------------------------------------------------------------------------

field = Field(name="LOD_StressTest", rows=ROWS, cols=COLS)
field.set_terrain(terrain)
field.set_land_prep(TiedRidges(
    ridge_height_m=0.20,
    ridge_spacing_m=2.0,
    tie_spacing_m=6.0,
    tie_height_m=0.12,
))

# Minimal weather — 1 day only, we just need the viz to load
weather_df = pd.DataFrame([{
    "day": 1, "doy": 1,
    "temp_max_c": 35.0, "temp_min_c": 22.0, "temp_mean_c": 28.0,
    "radiation_mj_m2": 22.0, "rainfall_mm": 20.0,
    "et0_mm": 7.0, "wind_speed_ms": 3.0, "humidity_pct": 60.0,
    "co2_ppm": 415.0,
}]).set_index("day")
field.set_weather(Weather(weather_df))

# ---------------------------------------------------------------------------
# 3. Run 1 day (minimum to populate the buffer server)
# ---------------------------------------------------------------------------

farm = Farm(name="LOD_Farm", location=(23.0, 82.0))
farm.add_field(field)
farm.use_physics(erosion=True, sediment_transport=True)

print("Running 1-day simulation...")
farm.run(days=1)
print(f"Simulation complete. Plants: {ROWS * COLS:,}")

# ---------------------------------------------------------------------------
# 4. Launch visualisation dashboard
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("LOD PERFORMANCE STATS")
print("="*60)
chunk_cells = 64
n_chunks = (ROWS // chunk_cells + int(ROWS % chunk_cells > 0)) * \
           (COLS // chunk_cells + int(COLS % chunk_cells > 0))
hi_verts_per_chunk = (chunk_cells + 1) ** 2
lo_verts_per_chunk = (chunk_cells // 4 + 1) ** 2
print(f"  Total cells   : {ROWS * COLS:,}")
print(f"  Chunks (64×64): {n_chunks}")
print(f"  Hi-res verts  : {n_chunks * hi_verts_per_chunk:,}  (all close)")
print(f"  Lo-res verts  : {n_chunks * lo_verts_per_chunk:,}  (all distant)  "
      f"[{n_chunks * hi_verts_per_chunk // max(1, n_chunks * lo_verts_per_chunk)}× reduction]")
print(f"  Camera start  : ~90 world units above → most chunks → lo-res on load")
print("="*60)
print("\nStarting dashboard... open the printed URL in your browser.")
print("Verify: field loads, zooming changes polygon density, no WebGL crash.\n")

farm.visualize()
