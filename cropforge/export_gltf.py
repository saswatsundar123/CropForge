"""
cropforge/export_gltf.py
========================
Headless Python GLTF scene exporter (PRD v0.9.0 §5).

Writes a .glb file containing:
  - Terrain mesh — triangulated flat grid (accurate plan positions)
  - Plant geometry — alive plants as coloured boxes (PRD simplification rule)

Requires: pygltflib >= 1.16
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def export_scene(
    log_path: str,
    day: int,
    filepath: str = "scene.glb",
    field: Optional[str] = None,
) -> Path:
    """Export the farm scene for *day* to a .glb file.

    Parameters
    ----------
    log_path:
        Path to the Parquet session directory (same as used by farm.visualize).
    day:
        Simulation day to export (must exist in the log).
    filepath:
        Destination .glb file path.
    field:
        Field name to export. Defaults to first field found.

    Returns
    -------
    Path
        Resolved path of the written .glb file.

    Raises
    ------
    ImportError  — pygltflib not installed.
    FileNotFoundError — Parquet log missing.
    ValueError — requested day not in log.

    Example
    -------
    >>> farm.run(days=30)
    >>> farm.export_scene(day=15, filepath="out/day15.glb")
    """
    try:
        import pygltflib
    except ImportError as exc:
        raise ImportError(
            "pygltflib is required for GLTF export. "
            "Install with: pip install pygltflib  or  pip install cropforge[export]"
        ) from exc

    import pyarrow as pa
    import pyarrow.dataset as ds

    log_dir = Path(log_path)
    plants_dir = log_dir / "plants"
    if not plants_dir.exists():
        raise FileNotFoundError(f"plants/ subdirectory not found in: {log_path}")

    # Read Hive-partitioned log (field_name=X/day=Y/part-*.parquet)
    part_schema = pa.schema([
        pa.field("field_name", pa.string()),
        pa.field("day", pa.int32()),
    ])
    dataset = ds.dataset(
        str(plants_dir),
        format="parquet",
        partitioning=ds.partitioning(part_schema, flavor="hive"),
    )
    df = dataset.to_table().to_pandas()

    if "day" in df.columns:
        df["day"] = df["day"].astype(int)

    # Resolve field
    if field is None:
        field = df["field_name"].iloc[0]
    df = df[df["field_name"] == field]

    available = sorted(df["day"].unique().tolist())
    if day not in available:
        raise ValueError(f"Day {day} not in log. Available: {available}")

    day_df = df[df["day"] == day].copy()

    rows = int(day_df["row"].max()) + 1
    cols = int(day_df["col"].max()) + 1
    spacing = 1.0

    # ---- Build combined geometry (terrain + plants as boxes) ----
    all_positions = []
    all_indices   = []
    vertex_offset  = 0

    tp, ti = _build_flat_terrain(rows, cols, spacing)
    all_positions.append(tp)
    all_indices.append(ti + vertex_offset)
    vertex_offset += len(tp)

    for _, row_data in day_df.iterrows():
        alive = bool(row_data.get("alive", True))
        if not alive:
            continue
        x = float(row_data["col"]) * spacing
        z = float(row_data["row"]) * spacing
        h = max(float(row_data.get("height_cm", 5.0)) / 100.0, 0.05)
        pp, pi = _build_box(x, 0.0, z, 0.04, h, 0.04)
        all_positions.append(pp)
        all_indices.append(pi + vertex_offset)
        vertex_offset += len(pp)

    positions = np.concatenate(all_positions, axis=0).astype(np.float32)
    indices   = np.concatenate(all_indices,   axis=0).astype(np.uint32)

    pos_bytes = positions.tobytes()
    idx_bytes = indices.tobytes()

    pos_min = positions.min(axis=0).tolist()
    pos_max = positions.max(axis=0).tolist()

    gltf = pygltflib.GLTF2()
    gltf.scene = 0
    gltf.scenes = [pygltflib.Scene(nodes=[0])]
    gltf.nodes  = [pygltflib.Node(mesh=0)]
    gltf.meshes = [pygltflib.Mesh(name=f"CropForge_day{day}", primitives=[
        pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=0),
            indices=1,
        )
    ])]
    gltf.accessors = [
        pygltflib.Accessor(
            bufferView=0, componentType=pygltflib.FLOAT, count=len(positions),
            type=pygltflib.VEC3, min=pos_min, max=pos_max,
        ),
        pygltflib.Accessor(
            bufferView=1, componentType=pygltflib.UNSIGNED_INT, count=len(indices),
            type=pygltflib.SCALAR,
        ),
    ]
    gltf.bufferViews = [
        pygltflib.BufferView(buffer=0, byteOffset=0,              byteLength=len(pos_bytes), target=pygltflib.ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=len(pos_bytes), byteLength=len(idx_bytes), target=pygltflib.ELEMENT_ARRAY_BUFFER),
    ]
    gltf.buffers = [pygltflib.Buffer(byteLength=len(pos_bytes) + len(idx_bytes))]
    gltf.asset   = pygltflib.Asset(version="2.0", generator="CropForge")

    gltf.set_binary_blob(pos_bytes + idx_bytes)

    out = Path(filepath).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(out))
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_flat_terrain(rows: int, cols: int, spacing: float):
    positions = np.array(
        [[c * spacing, 0.0, r * spacing] for r in range(rows) for c in range(cols)],
        dtype=np.float32,
    )
    indices = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            tl = r * cols + c
            indices += [tl, tl + cols, tl + 1, tl + 1, tl + cols, tl + cols + 1]
    return positions, np.array(indices, dtype=np.uint32)


def _build_box(cx: float, y_base: float, cz: float, rx: float, h: float, rz: float):
    x0, x1 = cx - rx, cx + rx
    y0, y1 = y_base, y_base + h
    z0, z1 = cz - rz, cz + rz
    positions = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=np.float32)
    indices = np.array([
        0,1,2, 0,2,3, 5,4,7, 5,7,6,
        4,0,3, 4,3,7, 1,5,6, 1,6,2,
        3,2,6, 3,6,7, 4,5,1, 4,1,0,
    ], dtype=np.uint32)
    return positions, indices
