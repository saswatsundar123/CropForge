"""
cropforge/physics/pathology.py
================================
Spatial Disease/Pest Pressure Engine for CropForge v0.5.0.

Implements a spatially explicit SIR (Susceptible-Infected-Resistant) spread
model on the plant grid. Each plant cell has one of three disease states:

    S  — Susceptible (default; healthy plant)
    I  — Infected (spreading; accumulates disease_stress daily)
    R  — Resistant/Recovered (removed from spread; cannot be re-infected)

Spread is anisotropic: infection probability to each of the 4-connected
neighbours (N, S, E, W) is modulated by wind direction. Downwind cells
receive probability = base_infection_rate × wind_weight (high); upwind cells
receive base_infection_rate × (1 - wind_weight) (near zero).

PRD Reference: §8.2, §8.4 (v0.5.0)

Schema keys written to plant.custom (PRD §10.2):
    'disease_state'  : str   — 'S', 'I', or 'R'
    'disease_stress' : float — cumulative stress contribution (0–1 range)
    'days_infected'  : int   — days since first infection
    '_latent_days'   : int   — internal: days in latent (pre-symptomatic) phase

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cropforge.state import FieldState

# ---------------------------------------------------------------------------
# Direction encoding helpers
# ---------------------------------------------------------------------------

# Cardinal offsets: (Δrow, Δcol) — grid convention: row increases South
# Wind 0° = North, 90° = East, 180° = South, 270° = West
# (standard meteorological bearing from North, clockwise)

_NEIGHBOURS: dict[str, tuple[int, int]] = {
    "N": (-1,  0),   # decreasing row = North
    "E": ( 0,  1),   # increasing col = East
    "S": ( 1,  0),   # increasing row = South
    "W": ( 0, -1),   # decreasing col = West
}

_BEARING_TO_DIRECTION: dict[str, str] = {
    "N": "N",
    "E": "E",
    "S": "S",
    "W": "W",
}


def _bearing_to_unit_vector(wind_direction_deg: float) -> tuple[float, float]:
    """Convert met bearing (0°=N, 90°=E, CW) to unit (dx, dy) Cartesian.

    Grid convention: dx = East (+col), dy = North (-row).

    Returns
    -------
    (dx, dy) unit vector in Cartesian grid space.
    """
    # Convert met bearing to math angle: math angle = 90 - bearing
    rad = math.radians(90.0 - wind_direction_deg)
    dx = math.cos(rad)   # East component
    dy = math.sin(rad)   # North component (positive = North = decreasing row)
    return dx, dy


def _neighbour_wind_weights(
    wind_direction_deg: float,
    anisotropy: float = 0.80,
) -> dict[str, float]:
    """Compute infection probability weight for each cardinal neighbour.

    The downwind neighbour receives a high weight, the upwind neighbour
    near zero. Cross-wind neighbours receive intermediate weights.

    The weights are normalised so their mean equals 1.0 (preserving the
    expected number of transmission events relative to isotropic spread).

    Parameters
    ----------
    wind_direction_deg:
        Prevailing wind direction — the direction FROM which wind blows
        (met convention: 270° = wind from West → blows East).
        This means the HIGH-weight direction is the DOWNWIND cell:
        opposite to where the wind is blowing from.
    anisotropy:
        How strongly the wind biases spread. 0 = isotropic. 1 = fully
        directional (only downwind). Default 0.80 (strong anisotropy).

    Returns
    -------
    dict mapping 'N'/'E'/'S'/'W' to a probability weight ≥ 0.
    """
    # Wind blows FROM wind_direction_deg, so the downwind direction
    # is the OPPOSITE direction: add 180° for downwind
    downwind_deg = (wind_direction_deg + 180.0) % 360.0
    dx_wind, dy_wind = _bearing_to_unit_vector(downwind_deg)

    # Cardinal neighbour unit vectors: (col_delta, row_delta_flipped)
    # E=(+1,0), W=(-1,0), N=(0,+1), S=(0,-1) in Cartesian
    neighbour_vecs: dict[str, tuple[float, float]] = {
        "N": (0.0,  1.0),
        "S": (0.0, -1.0),
        "E": (1.0,  0.0),
        "W": (-1.0, 0.0),
    }

    weights: dict[str, float] = {}
    for dirn, (nx, ny) in neighbour_vecs.items():
        # dot product: 1 if perfectly downwind, -1 if upwind
        dot = dx_wind * nx + dy_wind * ny
        # Map [-1, 1] dot product to [1-anisotropy, 1+anisotropy] weight
        # Ensures weights are always positive
        weights[dirn] = 1.0 + anisotropy * dot

    # Clamp to [0, inf) and normalise so mean = 1.0
    for k in weights:
        weights[k] = max(0.0, weights[k])

    total = sum(weights.values())
    if total > 0:
        mean_raw = total / len(weights)
        weights = {k: v / mean_raw for k, v in weights.items()}

    return weights


# ---------------------------------------------------------------------------
# Core SIR spread function
# ---------------------------------------------------------------------------

def calculate_disease_spread(
    plant_grid: list[list],
    wind_speed_ms: float,
    wind_direction_deg: float,
    base_infection_rate: float,
    latency_days: int = 0,
    stress_increment: float = 0.04,
    anisotropy: float = 0.80,
    rng: random.Random | None = None,
) -> list[list]:
    """Advance the SIR disease spread model by one day on the plant grid.

    For every infected plant (disease_state == 'I'), calculates wind-weighted
    infection probabilities for each 4-connected neighbour that is still
    susceptible ('S'), and stochastically infects them.

    Also updates disease_stress and days_infected counters for all infected
    plants.

    Parameters
    ----------
    plant_grid:
        2D list of PlantState objects (rows × cols).
    wind_speed_ms:
        Wind speed (m/s). Higher wind speed amplifies the anisotropy effect
        and marginally increases spread probability. Clamped to [0, 20] m/s.
    wind_direction_deg:
        Prevailing wind direction in meteorological bearing (0°=N, 90°=E,
        180°=S, 270°=W). Direction FROM which the wind blows.
    base_infection_rate:
        Base probability per infected plant per day that each neighbour
        becomes infected (isotropic, before wind weighting). Range [0, 1].
    latency_days:
        Days from infection to when the plant becomes contagious (can spread).
        Default 0 (immediate spread).
    stress_increment:
        Daily increase in disease_stress for each infected plant. Default 0.04.
    anisotropy:
        How strongly wind direction biases the spread. Range [0, 1].
        Default 0.80 (strongly anisotropic). See ``_neighbour_wind_weights``.
    rng:
        Optional seeded ``random.Random`` instance for reproducibility in
        tests. If None, uses the module-level ``random`` functions.

    Returns
    -------
    list[list]
        The same ``plant_grid`` list, mutated in-place and returned for
        convenience (consistent with other hook return types).

    Notes
    -----
    This function mutates ``plant_grid`` in-place via ``plant.custom`` dict
    updates. The grid itself (list-of-lists structure) is not modified.
    """
    if rng is None:
        _rand = random.random
    else:
        _rand = rng.random

    n_rows = len(plant_grid)
    n_cols = len(plant_grid[0]) if n_rows > 0 else 0

    # Compute wind-weighted directional spread probabilities
    weights = _neighbour_wind_weights(wind_direction_deg, anisotropy=anisotropy)

    # Wind speed scaling: higher wind → slightly amplified spread
    # Capped at 2.0× to avoid unrealistic runaway
    wind_speed_factor = min(2.0, 1.0 + 0.05 * min(20.0, wind_speed_ms))

    # Build set of new infections to apply AFTER iterating
    # (avoids cascade within one time step)
    new_infections: set[tuple[int, int]] = set()

    for row in range(n_rows):
        for col in range(n_cols):
            plant = plant_grid[row][col]
            if not plant.alive:
                continue

            state = plant.custom.get("disease_state", "S")

            # Update stress for currently infected plants
            if state == "I":
                days_infected = plant.custom.get("days_infected", 0) + 1
                plant.custom["days_infected"] = days_infected
                plant.custom["disease_stress"] = min(
                    1.0,
                    plant.custom.get("disease_stress", 0.0) + stress_increment,
                )
                # Accumulate stress_index
                plant.stress_index = min(
                    1.0,
                    plant.stress_index + stress_increment * 0.5,
                )

                # Only spread if past the latency period
                if days_infected <= latency_days:
                    continue

                # Try to infect each 4-connected neighbour
                for dirn, (dr, dc) in _NEIGHBOURS.items():
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                        continue

                    neighbour = plant_grid[nr][nc]
                    if not neighbour.alive:
                        continue
                    if neighbour.custom.get("disease_state", "S") != "S":
                        continue

                    spread_prob = (
                        base_infection_rate
                        * weights[dirn]
                        * wind_speed_factor
                    )
                    if _rand() < spread_prob:
                        new_infections.add((nr, nc))

    # Apply new infections
    for (nr, nc) in new_infections:
        neighbour = plant_grid[nr][nc]
        if neighbour.custom.get("disease_state", "S") == "S":
            neighbour.custom["disease_state"] = "I"
            neighbour.custom["days_infected"] = 0
            neighbour.custom["disease_stress"] = 0.0

    return plant_grid


def seed_initial_foci(
    plant_grid: list[list],
    foci: list[tuple[int, int]],
) -> None:
    """Seed initial infection foci on a plant grid (PRD §8.2).

    Parameters
    ----------
    plant_grid:
        2D list of PlantState objects.
    foci:
        List of (row, col) tuples identifying plants to infect on day 1.

    Notes
    -----
    Called by the disease hook on the first simulation day only.
    """
    n_rows = len(plant_grid)
    n_cols = len(plant_grid[0]) if n_rows > 0 else 0
    for row, col in foci:
        if 0 <= row < n_rows and 0 <= col < n_cols:
            plant = plant_grid[row][col]
            plant.custom["disease_state"] = "I"
            plant.custom["days_infected"] = 0
            plant.custom["disease_stress"] = 0.0
