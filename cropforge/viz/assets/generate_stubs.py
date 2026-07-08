"""
Generate placeholder GLTF stage files for StandardWheat and StandardMaize.

Each file is a minimal valid GLTF 2.0 with:
  - One mesh representing the plant at that stage
  - Two morph targets: "stage_start" and "stage_end"
  - PBR material matching PRD §4.4 / §5.2 specs

Triangle budgets (PRD §4.3):
  Stages 0-1: <50 tri  → use 1 quad = 2 tri (near-zero for stage 0)
  Stages 2-3: <150 tri → 6 quads = 12 tri  (leaf blades)
  Stages 4-5: <300 tri → 16 quads = 32 tri (spike complexity)
  Stage 6:    <100 tri → 4 quads = 8 tri   (collapsed)

These are stub/placeholder meshes. Real artist-authored models ship in a future update.
"""
import json
import base64
import struct
import os
import math

WHEAT_DIR = "cropforge/viz/assets/standard_wheat"
MAIZE_DIR = "cropforge/viz/assets/standard_maize"

# Stage name → (n_quads_for_plant, height_m, color_hex_rgb, description)
# n_quads drives triangle count (2 triangles per quad)
WHEAT_STAGES = [
    # idx, name, n_quads, height_m, color (r,g,b 0-1 floats), morph_end_height_factor
    (0, "stage_0_germination",   1,  0.001, (0.34, 0.20, 0.08), 1.0),  # near-invisible, soil colour
    (1, "stage_1_emergence",     4,  0.04,  (0.62, 0.78, 0.30), 1.2),  # pale green coleoptile
    (2, "stage_2_tillering",    12,  0.20,  (0.30, 0.61, 0.23), 1.3),  # 3-5 leaf blades
    (3, "stage_3_stem_ext",     12,  0.45,  (0.18, 0.45, 0.11), 1.3),  # internode elongation
    (4, "stage_4_anthesis",     20,  0.90,  (0.22, 0.55, 0.13), 1.1),  # flag leaf + spike
    (5, "stage_5_grain_fill",   20,  0.85,  (0.68, 0.72, 0.15), 1.0),  # nodding spike, yellowing
    (6, "stage_6_senescence",    8,  0.75,  (0.55, 0.41, 0.08), 0.9),  # golden-brown, drooping
]

MAIZE_STAGES = [
    (0, "stage_0_germination",   1,  0.001, (0.34, 0.20, 0.08), 1.0),
    (1, "stage_1_emergence",     4,  0.05,  (0.65, 0.80, 0.32), 1.2),  # mesocotyl shoot
    (2, "stage_2_veg_early",    16,  0.40,  (0.30, 0.67, 0.25), 1.3),  # broad maize leaves
    (3, "stage_3_veg_late",     16,  1.20,  (0.18, 0.50, 0.12), 1.2),  # 8-12 leaves
    (4, "stage_4_anthesis",     28,  2.50,  (0.20, 0.52, 0.12), 1.05), # tassel + silks
    (5, "stage_5_grain_fill",   28,  2.40,  (0.52, 0.58, 0.12), 1.0),  # ear husk visible
    (6, "stage_6_senescence",   12,  2.20,  (0.45, 0.33, 0.08), 0.85), # dried stalk
]


def _make_gltf(stage_idx: int, height_m: float, color: tuple, n_quads: int, morph_factor: float) -> dict:
    """Build a minimal GLTF dict for a single-mesh plant with 2 morph targets."""
    # Build a simple vertical quad-strip representing the plant.
    # n_quads stacked vertically from y=0 to y=height_m.
    # Each quad = 4 verts (2 shared), contributing 2 triangles.
    # Morph target 0 (stage_start): y coords compressed to 0
    # Morph target 1 (stage_end):   y coords scaled by morph_factor

    verts = []       # [x, y, z] * n_verts
    normals = []     # [nx, ny, nz] * n_verts
    indices = []     # triangle indices

    half_w = min(0.05 + stage_idx * 0.015, 0.25)   # plant gets wider per stage

    for q in range(n_quads):
        y0 = height_m * q / n_quads
        y1 = height_m * (q + 1) / n_quads
        # Two front-facing verts per row
        base = len(verts)
        verts += [
            [-half_w, y0, 0.0],
            [ half_w, y0, 0.0],
            [-half_w, y1, 0.0],
            [ half_w, y1, 0.0],
        ]
        normals += [[0, 0, 1]] * 4
        indices += [base, base+1, base+2, base+1, base+3, base+2]

    n_verts = len(verts)

    # --- pack positions ---
    pos_bytes = b"".join(struct.pack("<fff", *v) for v in verts)
    # morph target 0 (stage_start): displace y to near 0 (compressed)
    mt0_bytes = b"".join(struct.pack("<fff", 0.0, -v[1], 0.0) for v in verts)
    # morph target 1 (stage_end): grow by factor
    mt1_bytes = b"".join(struct.pack("<fff", 0.0, v[1] * (morph_factor - 1.0), 0.0) for v in verts)

    idx_bytes = b"".join(struct.pack("<H", i) for i in indices)

    def _pad4(b: bytes) -> bytes:
        return b + b"\x00" * ((4 - len(b) % 4) % 4)

    pos_bytes  = _pad4(pos_bytes)
    mt0_bytes  = _pad4(mt0_bytes)
    mt1_bytes  = _pad4(mt1_bytes)
    idx_bytes  = _pad4(idx_bytes)

    # encode to base64 data URIs
    def _uri(b): return "data:application/octet-stream;base64," + base64.b64encode(b).decode()

    buffers = [
        {"uri": _uri(pos_bytes),  "byteLength": len(pos_bytes)},
        {"uri": _uri(mt0_bytes),  "byteLength": len(mt0_bytes)},
        {"uri": _uri(mt1_bytes),  "byteLength": len(mt1_bytes)},
        {"uri": _uri(idx_bytes),  "byteLength": len(idx_bytes)},
    ]

    float3_count = n_verts
    idx_count    = len(indices)

    def _bv(buf_idx, stride, count, target):
        return {"buffer": buf_idx, "byteOffset": 0, "byteLength": stride * count, "target": target}

    buffer_views = [
        _bv(0, 12, float3_count, 34962),  # positions
        _bv(1, 12, float3_count, 34962),  # morph 0
        _bv(2, 12, float3_count, 34962),  # morph 1
        _bv(3,  2, idx_count,    34963),  # indices
    ]

    min_pos = [min(v[i] for v in verts) for i in range(3)]
    max_pos = [max(v[i] for v in verts) for i in range(3)]

    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": float3_count, "type": "VEC3",
         "min": min_pos, "max": max_pos},          # positions
        {"bufferView": 1, "componentType": 5126, "count": float3_count, "type": "VEC3"},  # mt0
        {"bufferView": 2, "componentType": 5126, "count": float3_count, "type": "VEC3"},  # mt1
        {"bufferView": 3, "componentType": 5123,  "count": idx_count,  "type": "SCALAR"}, # indices
    ]

    r, g, b = color
    gltf = {
        "asset": {"version": "2.0", "generator": "CropForge v0.9.5 asset generator"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": f"plant_stage_{stage_idx}"}],
        "meshes": [{
            "name": f"plant_stage_{stage_idx}",
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 3,
                "material": 0,
                "targets": [
                    {"POSITION": 1},   # stage_start morph
                    {"POSITION": 2},   # stage_end morph
                ],
            }],
            "extras": {"morphTargetNames": ["stage_start", "stage_end"]},
        }],
        "materials": [{
            "name": f"plant_material_stage_{stage_idx}",
            "pbrMetallicRoughness": {
                "baseColorFactor": [r, g, b, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.65 if stage_idx < 5 else 0.90,
            },
            "doubleSided": True,
        }],
        "buffers": buffers,
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    return gltf


def generate_all():
    for crop_dir, stages in [(WHEAT_DIR, WHEAT_STAGES), (MAIZE_DIR, MAIZE_STAGES)]:
        os.makedirs(crop_dir, exist_ok=True)
        for idx, name, n_quads, height_m, color, morph_factor in stages:
            gltf = _make_gltf(idx, height_m, color, n_quads, morph_factor)
            path = os.path.join(crop_dir, f"{name}.gltf")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(gltf, f, separators=(",", ":"))
            n_tri = n_quads * 2
            print(f"  wrote {path}  ({n_tri} triangles, height={height_m}m)")


if __name__ == "__main__":
    print("Generating StandardWheat GLTF stubs...")
    generate_all()
    print("Done.")
