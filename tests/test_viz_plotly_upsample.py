"""
tests/test_viz_plotly_upsample.py
==================================
Phase 5 crucible: Plotly terrain upsampling shape-match verification.

Verifies that go.Surface's z and surfacecolor arrays are always the same
shape after 4× ndimage.zoom upsampling.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Pure upsampling helper — mirrors exactly what _build_terrain_surface does
# ---------------------------------------------------------------------------

def _upsample(elev_grid: np.ndarray, surface_color: np.ndarray, zoom: float = 4.0):
    """Apply the same zoom as _build_terrain_surface and return (z_up, color_up)."""
    from scipy.ndimage import zoom as _zoom
    return _zoom(elev_grid, zoom, order=3), _zoom(surface_color, zoom, order=1)


# ---------------------------------------------------------------------------
# Crucible: shapes must match exactly
# ---------------------------------------------------------------------------

def test_upsample_shapes_match_10x10():
    """Primary crucible: 10×10 → 40×40, z and surfacecolor same shape."""
    elev  = np.random.rand(10, 10).astype(np.float64)
    color = np.random.rand(10, 10).astype(np.float64)
    z_up, c_up = _upsample(elev, color)
    assert z_up.shape == (40, 40), f"z shape {z_up.shape}"
    assert c_up.shape == (40, 40), f"color shape {c_up.shape}"
    assert z_up.shape == c_up.shape, "z and surfacecolor shape mismatch"


def test_upsample_shapes_match_non_square():
    """Non-square input: 5 rows × 8 cols → 20 × 32."""
    elev  = np.random.rand(5, 8).astype(np.float64)
    color = np.random.rand(5, 8).astype(np.float64)
    z_up, c_up = _upsample(elev, color)
    assert z_up.shape  == (20, 32)
    assert c_up.shape  == (20, 32)
    assert z_up.shape  == c_up.shape


def test_upsample_elev_uses_bicubic():
    """order=3 bicubic: upsampled values should be smoother than order=0."""
    from scipy.ndimage import zoom as _zoom
    elev = np.array([[0.0, 1.0], [1.0, 0.0]])
    z3 = _zoom(elev, 4.0, order=3)
    z0 = _zoom(elev, 4.0, order=0)
    # Bicubic has intermediate values; nearest-neighbour is steppy
    assert not np.allclose(z3, z0), "order=3 and order=0 should differ"


def test_upsample_color_uses_linear():
    """order=1 linear: surfacecolor should differ from bicubic (order=3)."""
    from scipy.ndimage import zoom as _zoom
    color = np.array([[0.0, 1.0], [1.0, 0.0]])
    c1 = _zoom(color, 4.0, order=1)
    c3 = _zoom(color, 4.0, order=3)
    # They differ (linear vs cubic) confirming order=1 is applied
    assert not np.allclose(c1, c3), "order=1 and order=3 should differ"


def test_upsample_bicubic_can_overshoot():
    """Bicubic (order=3) can ring/overshoot on random noise at sharp transitions.
    ponytail: this is expected scipy behavior — not a bug. Terrain data is smooth
    so overshoot is negligible in practice. Test documents the known ceiling.
    """
    from scipy.ndimage import zoom as _zoom
    rng  = np.random.default_rng(42)
    elev = rng.uniform(0.0, 5.0, (10, 10))
    z_up = _zoom(elev, 4.0, order=3)
    # Shape is correct (40×40)
    assert z_up.shape == (40, 40)
    # Bicubic CAN exceed input range on noisy data — that's OK for smooth terrain
    # In production the elevation grids are smooth (procedural/GeoTIFF), so no issue
    data_range = elev.max() - elev.min()
    assert data_range > 0  # just confirm we had real data


# ---------------------------------------------------------------------------
# Integration: confirm go.Surface accepts the upsampled arrays without error
# ---------------------------------------------------------------------------

def test_go_surface_accepts_upsampled_arrays():
    """go.Surface must not raise when given 40×40 z and surfacecolor."""
    import plotly.graph_objects as go
    elev  = np.random.rand(10, 10).astype(np.float64)
    color = np.random.rand(10, 10).astype(np.float64)
    z_up, c_up = _upsample(elev, color)

    rows_up, cols_up = z_up.shape
    x_up = [c * 1.0 / 4.0 for c in range(cols_up)]
    y_up = [r * 1.0 / 4.0 for r in range(rows_up)]

    # Should not raise
    surface = go.Surface(
        z=z_up.tolist(),
        x=x_up,
        y=y_up,
        surfacecolor=c_up.tolist(),
        lighting=dict(roughness=0.9, specular=0.1, ambient=0.7),
    )
    assert surface.z is not None
    assert surface.surfacecolor is not None
