"""
cropforge/physics/wind.py
===========================
Topographical wind speed multiplier engine for CropForge v0.7.0.

Computes a per-cell wind exposure index from terrain elevation.
Windward / ridgeline cells receive speed-up (multiplier > 1.0).
Leeward / sheltered cells receive reduced wind (multiplier < 1.0).
Flat terrain returns exactly 1.0 for every cell.

Algorithm: simplified shelter-index based on upwind elevation profile.
For each cell, the mean elevation of the nearest `fetch_cells` cells in
the upwind direction is compared to the cell's own elevation.
    shelter = mean_upwind_elevation - cell_elevation
    raw = 1.0 - sensitivity × shelter / elevation_range
    multiplier = clip(raw, min_mult, max_mult)

This naturally gives:
  ridge cells     → upwind terrain lower → shelter < 0 → multiplier > 1.0
  leeward cells   → upwind terrain higher → shelter > 0 → multiplier < 1.0
  flat terrain    → shelter ≈ 0 everywhere → multiplier ≈ 1.0

PRD Reference: v0.7.0 §6.3 (Topographical Wind Field)

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import math

import numpy as np


# ponytail: clamping bounds from PRD §6.3; upgrade path = more sophisticated shelter model
_MIN_MULT = 0.3
_MAX_MULT = 2.5


def calculate_wind_multiplier(
    elevation_grid: np.ndarray,
    wind_direction_deg: float,
    fetch_cells: int = 3,
    sensitivity: float = 0.3,
) -> np.ndarray:
    """Per-cell wind speed multiplier from terrain shelter index.

    Parameters
    ----------
    elevation_grid:
        2D array of cell elevations in metres, shape (rows, cols).
    wind_direction_deg:
        Meteorological bearing the wind blows FROM (0=N, 90=E, 180=S, 270=W).
    fetch_cells:
        Number of cells to sample in the upwind direction. Default 3.
    sensitivity:
        How strongly one full elevation-range of shelter changes the
        multiplier.  Default 0.3 (one range → ±30 % change in wind speed).

    Returns
    -------
    np.ndarray
        Multiplier grid, shape == elevation_grid.shape, values in
        [_MIN_MULT, _MAX_MULT].  Flat terrain returns all-ones.

    Examples
    --------
    >>> import numpy as np
    >>> flat = np.zeros((5, 5))
    >>> (calculate_wind_multiplier(flat, 270.0) == 1.0).all()
    True
    """
    elev = np.asarray(elevation_grid, dtype=float)
    rows, cols = elev.shape

    elev_range = float(elev.max() - elev.min())
    if elev_range < 1e-6:
        return np.ones((rows, cols), dtype=float)  # flat → all 1.0

    # Upwind direction unit vector in (row, col) space.
    # Bearing β: step upwind = (-cos β, sin β) with row↑ = north, col↑ = east.
    # sin(270°)=-1 correctly gives udc=-1 (look west) for wind FROM west.
    wd_rad = math.radians(wind_direction_deg)
    udr = -math.cos(wd_rad)
    udc =  math.sin(wd_rad)

    # ponytail: explicit loop over fetch_cells avoids scipy dependency;
    # ceiling: replace with vectorised np.roll approach if grids > 500×500.
    multiplier = np.empty((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            upwind_sum = 0.0
            upwind_n = 0
            for k in range(1, fetch_cells + 1):
                nr = int(round(r + udr * k))
                nc = int(round(c + udc * k))
                if 0 <= nr < rows and 0 <= nc < cols:
                    upwind_sum += elev[nr, nc]
                    upwind_n += 1

            if upwind_n == 0:
                # Edge cell fully upwind — no upwind sample → fully exposed
                raw = 1.0 + sensitivity  # slight speed-up at leading edge
            else:
                mean_upwind = upwind_sum / upwind_n
                shelter = mean_upwind - elev[r, c]
                raw = 1.0 - sensitivity * shelter / elev_range

            multiplier[r, c] = max(_MIN_MULT, min(_MAX_MULT, raw))

    return multiplier
