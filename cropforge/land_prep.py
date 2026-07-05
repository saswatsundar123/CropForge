"""
cropforge/land_prep.py
======================
Land preparation modifiers for CropForge v0.6.0 (PRD §6).

A LandPrep subclass transforms a base elevation grid by overlaying a
fine-scale geometric pattern (ridges, furrows, terraces) on top of the
base terrain macro-topology. The researcher attaches one instance to a
field via ``field.set_land_prep(modifier)``.

Extension pattern (§6.4):
    class TiedRidges(LandPrep):
        def apply(self, elevation_grid, resolution_m):
            modified = elevation_grid.copy()
            # ... researcher logic ...
            soil_modifiers = {"porosity_delta": 0.05, "bulk_density_delta": -0.1}
            return modified, soil_modifiers

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LandPrep(ABC):
    """Abstract base for land preparation geometric modifiers (PRD v0.6.0 §6).

    Subclass this and override ``apply()`` to define any hypothetical or
    custom land preparation practice.

    The base terrain's macro-topology is preserved — ``apply()`` receives
    the full elevation matrix and overlays a fine-scale pattern on top.
    """

    @abstractmethod
    def apply(
        self,
        elevation_grid: np.ndarray,
        resolution_m: float,
    ) -> tuple[np.ndarray, dict]:
        """Apply this modifier to a base elevation grid.

        Parameters
        ----------
        elevation_grid:
            Base terrain elevation matrix (metres), shape (rows, cols).
            **Do not modify in-place** — call ``.copy()`` first.
        resolution_m:
            Physical size of one grid cell in metres.

        Returns
        -------
        modified_elevation_grid:
            New elevation matrix after geometric transformation, same shape.
        soil_modifiers:
            Dict of soil property deltas to apply uniformly to every voxel.
            Supported keys: ``"porosity_delta"``, ``"bulk_density_delta"``,
            ``"surface_roughness_index"``.
            Return an empty dict if no soil property changes are needed.
        """


# ---------------------------------------------------------------------------
# Built-in modifiers
# ---------------------------------------------------------------------------

class RidgeFurrow(LandPrep):
    """Alternating raised ridges and depressed furrows across field columns
    (PRD v0.6.0 §6.3 — ridge-furrow).

    The pattern runs along the row axis (ridges are parallel to field rows).
    Each ridge-furrow cycle covers ``ridge_spacing_m`` of column width.
    The ridge peak is raised by ``ridge_height_m / 2`` above the base, the
    furrow trough is depressed by ``furrow_depth_m / 2`` below the base.

    Parameters
    ----------
    ridge_spacing_m:
        Centre-to-centre distance between adjacent ridge peaks (metres).
    ridge_height_m:
        Total height from furrow floor to ridge peak (metres).
    furrow_depth_m:
        Depth of the furrow below the unmodified ground level (metres).
        Defaults to ``ridge_height_m / 2`` if not specified.
    """

    def __init__(
        self,
        ridge_spacing_m: float = 0.75,
        ridge_height_m: float = 0.20,
        furrow_depth_m: float | None = None,
    ) -> None:
        if ridge_spacing_m <= 0:
            raise ValueError("ridge_spacing_m must be positive.")
        if ridge_height_m <= 0:
            raise ValueError("ridge_height_m must be positive.")
        self.ridge_spacing_m = ridge_spacing_m
        self.ridge_height_m = ridge_height_m
        self.furrow_depth_m = furrow_depth_m if furrow_depth_m is not None else ridge_height_m / 2.0

    def apply(self, elevation_grid: np.ndarray, resolution_m: float) -> tuple[np.ndarray, dict]:
        modified = elevation_grid.copy()
        _, cols = modified.shape

        # Physical column positions
        col_pos = np.arange(cols) * resolution_m  # metres from left edge

        # Cosine wave: peak = +ridge_height_m/2, trough = -(furrow_depth_m/2)
        # Peak amplitude (ridge above base)
        peak = self.ridge_height_m / 2.0
        # Trough amplitude (furrow below base)
        trough = self.furrow_depth_m / 2.0
        # Mean offset: cosine ranges [-1,1]; map to [trough, peak]
        midpoint = (peak - trough) / 2.0
        half_range = (peak + trough) / 2.0
        freq = 2 * math.pi / self.ridge_spacing_m
        wave = midpoint + half_range * np.cos(freq * col_pos)

        # Add wave to every row
        modified += wave[np.newaxis, :]  # broadcast over rows

        soil_mods = {
            "porosity_delta": 0.05,         # §6.5 — ridge position average
            "bulk_density_delta": -0.08,    # loosened soil
            "surface_roughness_index": 0.5, # medium roughness
        }
        return modified, soil_mods


class ContourBund(LandPrep):
    """Periodic raised barriers (bunds) running perpendicular to the
    primary slope direction, interrupting downslope water flow
    (PRD v0.6.0 §6.3 — contour bunds).

    Bunds are placed at equal elevation intervals along the row axis.
    Each bund occupies a single row band and raises those cells by
    ``bund_height_m`` above their base elevation.

    Parameters
    ----------
    bund_height_m:
        Height of each bund above the unmodified ground surface (metres).
    interval_m:
        Elevation interval between bund centre-lines (metres).
        The first bund is placed at the row whose base elevation first
        exceeds ``interval_m``; subsequent bunds at each additional
        ``interval_m`` of elevation gain.
    bund_width_cells:
        Half-width of each bund in cells (total bund width = 2 × this + 1).
        Default 1.
    """

    def __init__(
        self,
        bund_height_m: float = 0.25,
        interval_m: float = 1.5,
        bund_width_cells: int = 1,
    ) -> None:
        if bund_height_m <= 0:
            raise ValueError("bund_height_m must be positive.")
        if interval_m <= 0:
            raise ValueError("interval_m must be positive.")
        self.bund_height_m = bund_height_m
        self.interval_m = interval_m
        self.bund_width_cells = max(1, int(bund_width_cells))

    def apply(self, elevation_grid: np.ndarray, resolution_m: float) -> tuple[np.ndarray, dict]:
        modified = elevation_grid.copy()
        rows, _ = modified.shape

        # Row-mean elevations to determine where bunds go
        row_elev = modified.mean(axis=1)
        min_elev = row_elev.min()

        # Place bunds at rows where elevation crosses multiples of interval_m
        next_threshold = min_elev + self.interval_m
        for r in range(rows):
            if row_elev[r] >= next_threshold:
                # Place bund centred on row r, width = 2*bund_width_cells + 1
                for br in range(
                    max(0, r - self.bund_width_cells),
                    min(rows, r + self.bund_width_cells + 1),
                ):
                    modified[br, :] += self.bund_height_m
                next_threshold += self.interval_m

        soil_mods = {
            "porosity_delta": 0.03,
            "bulk_density_delta": -0.05,
            "surface_roughness_index": 0.3,
        }
        return modified, soil_mods


class Terrace(LandPrep):
    """Transform a continuous slope into a stepped staircase of level platforms
    (PRD v0.6.0 §6.3 — terraces).

    The field is divided into ``n_terraces`` equal-height bands along the
    primary slope (row axis). Within each band, all cells are levelled to the
    mean elevation of that band — forming flat terrace platforms. Between
    adjacent bands, a vertical riser drops to the next platform level.

    Parameters
    ----------
    n_terraces:
        Number of terrace steps to create.
    width_m:
        Horizontal width of each terrace platform (metres). If provided,
        overrides ``n_terraces`` by computing the number of terraces that
        fit given the field's row extent and ``resolution_m``.
        Defaults to None (use ``n_terraces``).
    drop_m:
        Maximum allowed step height between adjacent terraces (metres).
        Only used when ``width_m`` is provided to auto-compute ``n_terraces``.
        Ignored when ``n_terraces`` is set directly.
    """

    def __init__(
        self,
        n_terraces: int = 4,
        width_m: float | None = None,
        drop_m: float = 0.5,
    ) -> None:
        if n_terraces < 1:
            raise ValueError("n_terraces must be >= 1.")
        self.n_terraces = n_terraces
        self.width_m = width_m
        self.drop_m = drop_m

    def apply(self, elevation_grid: np.ndarray, resolution_m: float) -> tuple[np.ndarray, dict]:
        modified = elevation_grid.copy()
        rows, _ = modified.shape

        # If width_m given, compute n_terraces from field extent
        n = self.n_terraces
        if self.width_m is not None:
            field_height_m = rows * resolution_m
            n = max(1, round(field_height_m / self.width_m))

        # Divide rows into n equal bands; level each band to its mean elevation
        bands = np.array_split(np.arange(rows), n)
        for band in bands:
            if len(band) == 0:
                continue
            band_mean = modified[band, :].mean()
            modified[band, :] = band_mean

        soil_mods = {
            "porosity_delta": 0.08,          # §6.5 — platform
            "bulk_density_delta": -0.15,
            "surface_roughness_index": 0.2,  # low roughness on level platform
        }
        return modified, soil_mods


class ZeroTillage(LandPrep):
    """No geometric change — surface property change only (PRD v0.6.0 §6.3)."""

    def __init__(self, residue_cover_fraction: float = 0.6) -> None:
        self.residue_cover_fraction = residue_cover_fraction

    def apply(self, elevation_grid: np.ndarray, resolution_m: float) -> tuple[np.ndarray, dict]:
        return elevation_grid.copy(), {
            "porosity_delta": 0.0,
            "bulk_density_delta": 0.0,
            "surface_roughness_index": 0.1,  # low, residue-covered
        }


class ConventionalTill(LandPrep):
    """Flat ploughing — roughness and soil property change only (PRD v0.6.0 §6.3)."""

    def __init__(self, tillage_depth_cm: float = 15.0) -> None:
        self.tillage_depth_cm = tillage_depth_cm

    def apply(self, elevation_grid: np.ndarray, resolution_m: float) -> tuple[np.ndarray, dict]:
        return elevation_grid.copy(), {
            "porosity_delta": 0.08,
            "bulk_density_delta": -0.15,
            "surface_roughness_index": 0.8,  # high — freshly broken clods
        }


class TiedRidges(LandPrep):
    """Tied ridge-furrow: alternating ridges with periodic perpendicular tie dams
    that block D8 lateral flow and create micro-catchments (PRD v0.8.0 §7.1).

    Builds on the RidgeFurrow cosine wave, then overlays discrete tie dams
    across the furrows at ``tie_spacing_m`` intervals along the row axis.
    Each tie dam raises furrow-floor cells by ``tie_height_m``.

    Parameters
    ----------
    ridge_height_m:
        Height of each ridge above base (metres).  Default 0.20.
    ridge_spacing_m:
        Centre-to-centre distance between ridge peaks (metres). Default 0.75.
    tie_spacing_m:
        Row distance between successive tie dams (metres). Default 3.0.
    tie_height_m:
        Height of each tie dam above the furrow floor (metres). Default 0.10.
    """

    def __init__(
        self,
        ridge_height_m: float = 0.20,
        ridge_spacing_m: float = 0.75,
        tie_spacing_m: float = 3.0,
        tie_height_m: float = 0.10,
    ) -> None:
        if tie_spacing_m <= 0:
            raise ValueError("tie_spacing_m must be positive.")
        if tie_height_m < 0:
            raise ValueError("tie_height_m must be non-negative.")
        self._base = RidgeFurrow(
            ridge_spacing_m=ridge_spacing_m,
            ridge_height_m=ridge_height_m,
        )
        self.tie_spacing_m = tie_spacing_m
        self.tie_height_m = tie_height_m

    def apply(self, elevation_grid: np.ndarray, resolution_m: float) -> tuple[np.ndarray, dict]:
        modified, soil_mods = self._base.apply(elevation_grid, resolution_m)
        rows, _ = modified.shape

        # Place tie dams at every tie_spacing_m along the row axis
        tie_interval_rows = max(1, round(self.tie_spacing_m / resolution_m))
        for r in range(0, rows, tie_interval_rows):
            modified[r, :] += self.tie_height_m  # raise entire tie row

        # Higher roughness than plain RidgeFurrow (ties add micro-dams)
        soil_mods = dict(soil_mods)   # don't mutate _base's dict
        soil_mods["surface_roughness_index"] = 0.7
        return modified, soil_mods


class VegetativeFilterStrip(LandPrep):
    """Dense perennial grass strip at the downslope edge of a field that traps
    incoming sediment and runoff (PRD v0.8.0 §7.2).

    Unlike geometric modifiers, this class does not carve the elevation grid.
    Instead it returns per-cell soil overrides (``surface_roughness_index=0.95``)
    for the strip rows, exploiting the existing erosion / sediment damping equations.

    The strip occupies the last *N* rows of the field grid (``edge='bottom'``,
    which is the downslope boundary in a typical slope-to-valley setup).

    Parameters
    ----------
    width_m:
        Physical width of the strip (metres). Number of rows = ceil(width_m / resolution_m).
    edge:
        Which field edge the strip occupies: ``'bottom'`` (default) or ``'top'``.
    roughness:
        Surface roughness index assigned to strip cells.  Default 0.95.
    """

    def __init__(
        self,
        width_m: float = 3.0,
        edge: str = "bottom",
        roughness: float = 0.95,
    ) -> None:
        if width_m <= 0:
            raise ValueError("width_m must be positive.")
        if edge not in ("bottom", "top"):
            raise ValueError("edge must be 'bottom' or 'top'.")
        self.width_m = width_m
        self.edge = edge
        self.roughness = roughness

    def apply(
        self,
        elevation_grid: np.ndarray,
        resolution_m: float,
    ) -> tuple[np.ndarray, dict, dict]:
        """Return (unchanged elevation, empty global soil_mods, per_cell_mods).

        ``per_cell_mods`` maps ``(row, col) → {"surface_roughness_index": value}``.
        farm._init_field_state applies these per-cell (not broadcast to all cells).
        """
        rows, cols = elevation_grid.shape
        n_strip_rows = max(1, math.ceil(self.width_m / resolution_m))

        if self.edge == "bottom":
            strip_rows = range(rows - n_strip_rows, rows)
        else:
            strip_rows = range(0, n_strip_rows)

        per_cell: dict = {
            (r, c): {"surface_roughness_index": self.roughness}
            for r in strip_rows
            for c in range(cols)
        }
        # No global soil changes; no elevation change (vegetation doesn't reshape ground)
        return elevation_grid.copy(), {}, per_cell

