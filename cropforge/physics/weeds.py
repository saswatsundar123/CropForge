"""
cropforge/physics/weeds.py
==========================
Opt-in weed competition model for CropForge v1.0.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class WeedState:
    row: int
    col: int
    alive: bool = True
    biomass_g: float = 0.0
    lai: float = 0.2
    species: str = ""


@dataclass
class WeedParams:
    species: str = "generic_grass"
    initial_density_m2: float = 0.0
    emergence_doy: int = 1
    spread_rate: float = 0.0
    competitive_index: float = 1.0
    daily_lai_growth: float = 0.03
    max_lai: float = 2.5


def initialise_weed_grid(
    rows: int,
    cols: int,
    params: WeedParams,
    resolution_m: float,
    rng: np.random.Generator,
) -> List[List[Optional[WeedState]]]:
    """Seed weed plants at init based on initial_density_m2."""
    grid: List[List[Optional[WeedState]]] = [[None] * cols for _ in range(rows)]
    plants_per_cell = max(0.0, params.initial_density_m2) * (resolution_m ** 2)
    probability = min(1.0, plants_per_cell)
    for r in range(rows):
        for c in range(cols):
            if rng.random() < probability:
                grid[r][c] = WeedState(row=r, col=c, species=params.species)
    return grid


def step_weeds(weed_grid, soil_grid, plant_grid, env, params, doy, rng):
    """Run one daily weed timestep. Modifies weed_grid and soil_grid in place."""
    if not weed_grid or doy < params.emergence_doy:
        return

    rows = len(weed_grid)
    cols = len(weed_grid[0]) if rows else 0

    for r in range(rows):
        for c in range(cols):
            weed = weed_grid[r][c]
            if weed is None or not weed.alive:
                continue

            weed.lai = min(weed.lai + params.daily_lai_growth, params.max_lai)
            weed.biomass_g = max(weed.biomass_g, weed.lai * 2.0)

            crop_plant = _plant_at(plant_grid, r, c, cols)
            crop_lai = crop_plant.lai if crop_plant and crop_plant.alive else 0.0
            total_lai = crop_lai + weed.lai
            if total_lai <= 0:
                continue

            weed_fraction = min(1.0, (weed.lai / total_lai) * params.competitive_index)
            soil_layer = soil_grid[r][c][0]
            daily_depletion = max(0.0, env.et0_mm) * weed_fraction * 0.5
            soil_layer.moisture_pct = max(
                soil_layer.custom.get("wilting_point_pct", 0.0),
                soil_layer.moisture_pct - daily_depletion,
            )

    if params.spread_rate > 0:
        _spread_weeds(weed_grid, params, rng, rows, cols)


def _spread_weeds(weed_grid, params, rng, rows, cols):
    new_infestations = []
    for r in range(rows):
        for c in range(cols):
            weed = weed_grid[r][c]
            if weed is None or not weed.alive:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and weed_grid[nr][nc] is None:
                    if rng.random() < params.spread_rate:
                        new_infestations.append((nr, nc))

    for r, c in new_infestations:
        if weed_grid[r][c] is None:
            weed_grid[r][c] = WeedState(row=r, col=c, species=params.species)


def compute_weed_radiation_suppression(
    weed_grid,
    plant_grid,
    competitive_index: float,
) -> np.ndarray:
    """Return a 2D crop PAR suppression factor grid."""
    rows = len(weed_grid)
    cols = len(weed_grid[0]) if rows else 0
    suppression = np.ones((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            weed = weed_grid[r][c]
            if weed is None or not weed.alive:
                continue
            crop = _plant_at(plant_grid, r, c, cols)
            crop_lai = crop.lai if crop and crop.alive else 0.0
            total_lai = crop_lai + weed.lai
            if total_lai > 0:
                weed_shade = min(1.0, (weed.lai / total_lai) * competitive_index)
                suppression[r][c] = max(0.1, 1.0 - weed_shade)
    return suppression


def _plant_at(plant_grid, row: int, col: int, cols: int):
    if not plant_grid:
        return None
    first = plant_grid[0]
    if isinstance(first, list):
        return plant_grid[row][col]
    idx = row * cols + col
    return plant_grid[idx] if 0 <= idx < len(plant_grid) else None
