"""
cropforge/terrain.py
====================
Terrain geometry source of truth for CropForge v0.6.0.

Provides the Terrain dataclass and four factory constructors:
  Terrain.from_geotiff()    -- import a real DEM (requires rasterio)
  Terrain.from_csv()        -- import from a flat CSV file
  Terrain.from_array()      -- wrap a researcher-supplied numpy array
  Terrain.from_generator()  -- call a researcher-supplied callable
  Terrain.procedural()      -- built-in deterministic generators

Slope and aspect grids are computed once and cached on construction.
Fields without terrain default to a flat plane (elevation = 0.0)
via FieldState.terrain = None -- no behaviour change from v0.5.0.

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Terrain dataclass
# ---------------------------------------------------------------------------

@dataclass
class Terrain:
    """Source of truth for field elevation geometry (PRD v0.6.0 §5).

    Attributes
    ----------
    elevation_grid:  metres, shape (rows, cols)
    resolution_m:    physical size of one grid cell in metres
    slope_grid:      degrees, per cell -- computed on construction
    aspect_grid:     degrees from north, per cell -- computed on construction
    source:          one of "geotiff", "csv", "array", "generator", "procedural"
    """

    elevation_grid: np.ndarray
    resolution_m: float
    slope_grid: np.ndarray = field(init=False)
    aspect_grid: np.ndarray = field(init=False)
    source: str = "array"

    def __post_init__(self):
        self.slope_grid, self.aspect_grid = _compute_slope_aspect(
            self.elevation_grid, self.resolution_m
        )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_array(cls, array: np.ndarray, resolution_m: float = 1.0) -> "Terrain":
        """Wrap a researcher-supplied numpy array.

        Formalises the existing ``field.set_elevation(array)`` behaviour.
        """
        arr = np.asarray(array, dtype=float)
        return cls(elevation_grid=arr, resolution_m=resolution_m, source="array")

    @classmethod
    def from_generator(
        cls,
        fn: Callable[[int, int], np.ndarray],
        rows: int,
        cols: int,
        resolution_m: float = 1.0,
    ) -> "Terrain":
        """Call a researcher-supplied function to produce the elevation grid.

        Parameters
        ----------
        fn:           callable(rows, cols) -> np.ndarray of shape (rows, cols)
        rows, cols:   field grid dimensions
        resolution_m: physical cell size in metres
        """
        arr = np.asarray(fn(rows, cols), dtype=float)
        if arr.shape != (rows, cols):
            raise ValueError(
                f"Generator returned shape {arr.shape}, expected ({rows}, {cols})"
            )
        return cls(elevation_grid=arr, resolution_m=resolution_m, source="generator")

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        rows: int,
        cols: int,
        resolution_m: float = 1.0,
    ) -> "Terrain":
        """Load elevation from a plain CSV file (one float per cell, row-major).

        The file may have a header row (auto-detected). Values are resampled
        to (rows, cols) if the loaded shape differs.
        """
        raw = np.genfromtxt(filepath, delimiter=",", filling_values=0.0)
        # Drop header row if it contains NaN (genfromtxt produces NaN for text)
        if raw.ndim == 2 and np.isnan(raw[0]).any():
            raw = raw[1:]
        arr = raw.reshape(-1) if raw.ndim != 2 else raw
        if arr.shape != (rows, cols):
            arr = _bilinear_resample(arr.reshape(-1, 1) if arr.ndim == 1 else arr, rows, cols)
        return cls(elevation_grid=arr.astype(float), resolution_m=resolution_m, source="csv")

    @classmethod
    def from_geotiff(cls, filepath: str, resolution_m: Optional[float] = None) -> "Terrain":
        """Import a GeoTIFF DEM.

        Requires ``rasterio``. Install with: pip install rasterio

        The elevation matrix is extracted from band 1. If the source resolution
        differs from ``resolution_m``, the grid is resampled via bilinear
        interpolation. If ``resolution_m`` is None, the native GeoTIFF
        resolution is used.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling
        except ImportError as exc:
            raise ImportError(
                "rasterio is required for Terrain.from_geotiff(). "
                "Install it with: pip install rasterio"
            ) from exc

        with rasterio.open(filepath) as src:
            native_res = src.res[0]  # metres per pixel (assumes square pixels)
            arr = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = 0.0

        res = resolution_m if resolution_m is not None else native_res
        return cls(elevation_grid=arr, resolution_m=res, source="geotiff")

    @classmethod
    def procedural(
        cls,
        rows: int,
        cols: int,
        generator: str = "slope",
        resolution_m: float = 1.0,
        seed: int = 42,
        # slope params
        grade_pct: float = 2.0,
        direction_deg: float = 0.0,
        # undulating / ridge params
        amplitude_m: float = 2.5,
        frequency: float = 0.08,
        # bowl params
        depth_m: float = 2.0,
        radius_m: Optional[float] = None,
        # ridge params
        ridge_height_m: Optional[float] = None,
        width_m: float = 10.0,
        orientation_deg: float = 90.0,
        # shared
        base_elevation_m: float = 0.0,
    ) -> "Terrain":
        """Generate a deterministic terrain grid.

        Generators
        ----------
        "slope"      : uniform linear gradient. ``grade_pct`` and ``direction_deg``.
        "undulating" : smooth rolling hills via seeded Perlin-like sum-of-sines.
        "bowl"       : radial depression centred at the field midpoint.
        "ridge"      : single linear ridge crossing the field.

        Same ``seed`` always produces the same output (fully deterministic).
        """
        generators = {
            "slope": _gen_slope,
            "undulating": _gen_undulating,
            "bowl": _gen_bowl,
            "ridge": _gen_ridge,
        }
        if generator not in generators:
            raise ValueError(
                f"Unknown generator '{generator}'. "
                f"Choose from: {list(generators)}"
            )

        kwargs = dict(
            rows=rows, cols=cols, seed=seed,
            amplitude_m=amplitude_m, frequency=frequency,
            grade_pct=grade_pct, direction_deg=direction_deg,
            depth_m=depth_m, radius_m=radius_m,
            ridge_height_m=ridge_height_m if ridge_height_m is not None else amplitude_m,
            width_m=width_m, orientation_deg=orientation_deg,
            base_elevation_m=base_elevation_m,
        )
        arr = generators[generator](**kwargs)
        return cls(elevation_grid=arr, resolution_m=resolution_m, source="procedural")


# ---------------------------------------------------------------------------
# Built-in procedural generators (pure numpy, fully deterministic)
# ---------------------------------------------------------------------------

def _gen_slope(
    rows: int, cols: int, grade_pct: float, direction_deg: float,
    base_elevation_m: float, **_
) -> np.ndarray:
    """Uniform linear slope. grade_pct is rise/run × 100. direction_deg is
    the downhill bearing (0 = south, 90 = west)."""
    grade = grade_pct / 100.0
    rad = math.radians(direction_deg)
    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    # Physical distance along slope direction
    dist = r_idx * math.cos(rad) + c_idx * math.sin(rad)
    return base_elevation_m + dist * grade


def _gen_undulating(
    rows: int, cols: int, seed: int, amplitude_m: float, frequency: float,
    base_elevation_m: float, **_
) -> np.ndarray:
    """Smooth rolling terrain via a seeded sum-of-sines (no scipy needed).

    ponytail: sum-of-sines instead of Perlin noise -- same visual quality
    for agricultural fields, zero extra dependency.
    """
    rng = np.random.default_rng(seed)
    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    result = np.zeros((rows, cols), dtype=float)
    # Four independent waves at orthogonal/diagonal directions
    phases = rng.uniform(0, 2 * math.pi, 4)
    freqs  = rng.uniform(frequency * 0.7, frequency * 1.3, 4)
    amps   = rng.uniform(0.5, 1.0, 4)
    amps  /= amps.sum()  # normalise so total amplitude == amplitude_m
    dirs   = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for i, (dr, dc) in enumerate(dirs):
        result += amps[i] * np.sin(
            2 * math.pi * freqs[i] * (dr * r_idx + dc * c_idx) + phases[i]
        )
    return base_elevation_m + result * amplitude_m


def _gen_bowl(
    rows: int, cols: int, depth_m: float, radius_m: Optional[float],
    base_elevation_m: float, **_
) -> np.ndarray:
    """Radial depression centred at the field midpoint."""
    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    cr, cc = rows / 2.0, cols / 2.0
    r = radius_m if radius_m is not None else min(rows, cols) / 2.0
    dist = np.sqrt((r_idx - cr) ** 2 + (c_idx - cc) ** 2)
    # Cosine bowl: deepest at centre, flat at radius
    depth = np.where(dist < r, depth_m * (1 - np.cos(math.pi * dist / r)) / 2, 0.0)
    return base_elevation_m - depth


def _gen_ridge(
    rows: int, cols: int, ridge_height_m: float, width_m: float,
    orientation_deg: float, base_elevation_m: float, **_
) -> np.ndarray:
    """Single linear ridge crossing the field centre."""
    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    cr, cc = rows / 2.0, cols / 2.0
    rad = math.radians(orientation_deg)
    # Perpendicular distance from each cell to the ridge centreline
    perp = abs((r_idx - cr) * math.sin(rad) - (c_idx - cc) * math.cos(rad))
    half_w = width_m / 2.0
    height = np.where(
        perp < half_w,
        ridge_height_m * (1 - np.cos(math.pi * perp / half_w)) / 2,
        0.0,
    )
    return base_elevation_m + height


# ---------------------------------------------------------------------------
# Slope / aspect computation (central differences, in degrees)
# ---------------------------------------------------------------------------

def _compute_slope_aspect(
    elev: np.ndarray, resolution_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (degrees from North, clockwise)
    using central finite differences. Edge cells use forward/backward diffs.

    Returns (slope_grid, aspect_grid), both shape == elev.shape.
    """
    rows, cols = elev.shape
    # np.gradient requires ≥ 2 elements per axis; return zeros for degenerate grids
    # ponytail: zero slope/aspect for 1-row or 1-col fields; fine for D8 (no meaningful gradient)
    if rows < 2 or cols < 2:
        return np.zeros_like(elev), np.zeros_like(elev)

    dz_dc = np.gradient(elev, resolution_m, axis=1)   # dz/dx  (col direction)
    dz_dr = np.gradient(elev, resolution_m, axis=0)   # dz/dy  (row direction)

    slope = np.degrees(np.arctan(np.sqrt(dz_dc ** 2 + dz_dr ** 2)))
    # Aspect: 0 = North (neg-row direction), clockwise positive
    aspect = np.degrees(np.arctan2(dz_dc, -dz_dr)) % 360.0

    return slope, aspect


# ---------------------------------------------------------------------------
# Bilinear resample (used by from_csv if shape mismatch)
# ---------------------------------------------------------------------------

def _bilinear_resample(src: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """Bilinear interpolation resize. Pure numpy, no scipy needed.

    ponytail: hand-rolled only because scipy is not a declared dependency.
    Replace with scipy.ndimage.zoom if scipy is added in future.
    """
    in_rows, in_cols = src.shape
    r_ratio = (in_rows - 1) / max(out_rows - 1, 1)
    c_ratio = (in_cols - 1) / max(out_cols - 1, 1)

    r_idx = np.arange(out_rows) * r_ratio
    c_idx = np.arange(out_cols) * c_ratio

    r0 = np.floor(r_idx).astype(int).clip(0, in_rows - 2)
    c0 = np.floor(c_idx).astype(int).clip(0, in_cols - 2)
    r1 = (r0 + 1).clip(0, in_rows - 1)
    c1 = (c0 + 1).clip(0, in_cols - 1)

    dr = (r_idx - r0)[:, None]
    dc = (c_idx - c0)[None, :]

    return (
        src[r0[:, None], c0[None, :]] * (1 - dr) * (1 - dc)
        + src[r1[:, None], c0[None, :]] * dr * (1 - dc)
        + src[r0[:, None], c1[None, :]] * (1 - dr) * dc
        + src[r1[:, None], c1[None, :]] * dr * dc
    )
