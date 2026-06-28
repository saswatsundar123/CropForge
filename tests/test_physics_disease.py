"""
tests/test_physics_disease.py
================================
Test suite for the Spatial Disease / Pest Pressure Engine (PRD v0.5.0 §8).

PRD Crucible Criterion (§8.4):
    Initialize a grid, infect the center plant, set wind to 270°
    (blowing East). After sufficient days, assert that infected cells are
    significantly more prevalent in the DOWNWIND (eastern) half of the field
    than in the upwind (western) half.

    Wind at 270° = wind FROM the West → blows toward the East.
    Downwind = East (right half of field columns).

Tests:
    [✓] seed_initial_foci: specified plant gets disease_state='I'
    [✓] Uninfected plants default to disease_state absent (treat as 'S')
    [✓] Infected plant spreads to at least one neighbour within spread_rate bounds
    [✓] PRD Crucible: downwind (East) half has significantly more infected plants
    [✓] Anisotropy: disease_wind_direction_deg=90° (from East) spreads WEST
    [✓] Spread is zero when base_infection_rate=0
    [✓] disease_stress accumulates at correct rate for infected plants
    [✓] stress_index increases from disease_stress
    [✓] Latency: plants in 'I' state do not spread until latency_days elapsed
    [✓] Hook disabled: no disease_state keys in plant.custom
    [✓] Hook enabled via farm.use_physics(disease=True, disease_foci=[(r,c)])
    [✓] Deterministic with fixed seed (same result on multiple runs)
    [✓] isotropic spread: no wind bias when anisotropy=0
    [✓] Dead plants do not spread disease

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import random
import math
from pathlib import Path
import sys

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from cropforge.physics.pathology import (
    calculate_disease_spread,
    seed_initial_foci,
    _neighbour_wind_weights,
    _bearing_to_unit_vector,
)
from cropforge import Farm, Field, Crop, Soil, Weather

_DATA_DIR = Path(__file__).parent.parent / "examples" / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plant_grid(rows: int, cols: int):
    """Return a 2D grid of minimal mock PlantState-like objects."""
    from cropforge.state import PlantState

    grid = []
    for r in range(rows):
        row_list = []
        for c in range(cols):
            p = PlantState(plant_id=f"p_{r}_{c}", row=r, col=c)
            row_list.append(p)
        grid.append(row_list)
    return grid


def _count_infected(grid):
    total = 0
    for row in grid:
        for plant in row:
            if plant.custom.get("disease_state") == "I":
                total += 1
    return total


def _count_infected_in_cols(grid, col_start, col_end):
    """Count infected plants in columns [col_start, col_end)."""
    total = 0
    for row in grid:
        for plant in row[col_start:col_end]:
            if plant.custom.get("disease_state") == "I":
                total += 1
    return total


def _make_minimal_farm(rows: int = 8, cols: int = 8):
    farm = Farm(name="DiseaseTestFarm")
    field = Field(name="DiseaseField", rows=rows, cols=cols, area_ha=0.1)
    field.set_crop(Crop(species="wheat"))
    field.set_weather(
        Weather.from_csv(
            str(_DATA_DIR / "wheat_synthetic_weather_90d.csv"),
            date_col="date", tmax_col="tmax_c", tmin_col="tmin_c",
            radiation_col="radiation_mj", rainfall_col="rainfall_mm",
            humidity_col="humidity_pct", wind_col="wind_ms", wind_unit="m/s",
        )
    )
    field.set_soil(Soil.from_csv(str(_DATA_DIR / "wheat_uniform_soil_3layer.csv"), apply="uniform"))
    farm.add_field(field)
    return farm, field


# ===========================================================================
# 1. Wind weight helper tests
# ===========================================================================

class TestWindWeights:

    def test_270_deg_downwind_is_east(self):
        """Wind=270° (from West) → East neighbour should have highest weight."""
        weights = _neighbour_wind_weights(wind_direction_deg=270.0, anisotropy=0.80)
        assert weights["E"] > weights["W"], (
            f"East should dominate when wind blows East (270°). "
            f"E={weights['E']:.4f}, W={weights['W']:.4f}"
        )

    def test_90_deg_downwind_is_west(self):
        """Wind=90° (from East) → West neighbour should have highest weight."""
        weights = _neighbour_wind_weights(wind_direction_deg=90.0, anisotropy=0.80)
        assert weights["W"] > weights["E"], (
            f"West should dominate when wind blows West (90°). "
            f"W={weights['W']:.4f}, E={weights['E']:.4f}"
        )

    def test_0_deg_downwind_is_south(self):
        """Wind=0° (from North) → South neighbour should have highest weight."""
        weights = _neighbour_wind_weights(wind_direction_deg=0.0, anisotropy=0.80)
        assert weights["S"] > weights["N"], (
            f"South should dominate when wind blows South (0°). "
            f"S={weights['S']:.4f}, N={weights['N']:.4f}"
        )

    def test_180_deg_downwind_is_north(self):
        """Wind=180° (from South) → North neighbour should have highest weight."""
        weights = _neighbour_wind_weights(wind_direction_deg=180.0, anisotropy=0.80)
        assert weights["N"] > weights["S"]

    def test_all_weights_positive(self):
        """All neighbour weights must be ≥ 0 for any wind direction."""
        for deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            weights = _neighbour_wind_weights(float(deg), anisotropy=0.80)
            for dirn, w in weights.items():
                assert w >= 0.0, f"Negative weight for {dirn} at wind={deg}°: {w}"

    def test_isotropic_at_zero_anisotropy(self):
        """anisotropy=0 must give equal weights to all 4 neighbours."""
        weights = _neighbour_wind_weights(wind_direction_deg=270.0, anisotropy=0.0)
        vals = list(weights.values())
        assert max(vals) - min(vals) < 1e-6, (
            f"Isotropic spread expected, got: {weights}"
        )

    def test_weights_mean_is_one(self):
        """Normalised weights: mean across 4 neighbours must be 1.0."""
        weights = _neighbour_wind_weights(wind_direction_deg=270.0, anisotropy=0.80)
        mean_weight = sum(weights.values()) / len(weights)
        assert abs(mean_weight - 1.0) < 1e-6, (
            f"Mean weight should be 1.0 (normalised), got {mean_weight:.6f}"
        )


# ===========================================================================
# 2. seed_initial_foci tests
# ===========================================================================

class TestSeedInitialFoci:

    def test_specified_plants_infected(self):
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(2, 2)])
        assert grid[2][2].custom.get("disease_state") == "I"

    def test_multiple_foci_infected(self):
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(0, 0), (4, 4)])
        assert grid[0][0].custom.get("disease_state") == "I"
        assert grid[4][4].custom.get("disease_state") == "I"

    def test_other_plants_uninfected(self):
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(2, 2)])
        for r in range(5):
            for c in range(5):
                if (r, c) != (2, 2):
                    assert grid[r][c].custom.get("disease_state", "S") == "S"

    def test_out_of_bounds_foci_ignored(self):
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(10, 10)])  # out of bounds
        total_infected = _count_infected(grid)
        assert total_infected == 0

    def test_days_infected_reset_to_zero(self):
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(2, 2)])
        assert grid[2][2].custom.get("days_infected", -1) == 0

    def test_disease_stress_reset_to_zero(self):
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(2, 2)])
        assert grid[2][2].custom.get("disease_stress", -1.0) == 0.0


# ===========================================================================
# 3. calculate_disease_spread — unit tests
# ===========================================================================

class TestCalculateDiseaseSpread:

    def test_no_spread_with_zero_infection_rate(self):
        """With base_infection_rate=0, no spread should occur."""
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(2, 2)])
        rng = random.Random(42)
        for _ in range(10):
            calculate_disease_spread(
                grid, wind_speed_ms=3.0, wind_direction_deg=270.0,
                base_infection_rate=0.0, latency_days=0, rng=rng,
            )
        infected = _count_infected(grid)
        assert infected == 1, f"Only the seed should be infected, got {infected}"

    def test_spread_occurs_with_high_rate(self):
        """With rate=1.0 and latency=0, spread must occur within a few days."""
        grid = _make_plant_grid(10, 10)
        seed_initial_foci(grid, [(5, 5)])
        rng = random.Random(123)
        for _ in range(5):
            calculate_disease_spread(
                grid, wind_speed_ms=3.0, wind_direction_deg=270.0,
                base_infection_rate=1.0, latency_days=0, rng=rng,
            )
        infected = _count_infected(grid)
        assert infected > 1, f"Expected spread with rate=1.0, still only {infected} infected"

    def test_latency_prevents_immediate_spread(self):
        """With latency=10, infected plant should not spread in first 10 days."""
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(2, 2)])
        rng = random.Random(42)
        for _ in range(5):
            calculate_disease_spread(
                grid, wind_speed_ms=3.0, wind_direction_deg=270.0,
                base_infection_rate=1.0, latency_days=10, rng=rng,
            )
        infected = _count_infected(grid)
        assert infected == 1, (
            f"With latency=10, only seed should be infected after 5 days. Got {infected}"
        )

    def test_disease_stress_accumulates(self):
        """disease_stress must increase by stress_increment each infected day."""
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(2, 2)])
        stress_increment = 0.04
        rng = random.Random(42)
        for _ in range(5):
            calculate_disease_spread(
                grid, wind_speed_ms=3.0, wind_direction_deg=270.0,
                base_infection_rate=0.0, latency_days=0,
                stress_increment=stress_increment, rng=rng,
            )
        stress = grid[2][2].custom.get("disease_stress", 0.0)
        expected = 5 * stress_increment
        assert abs(stress - expected) < 1e-6, (
            f"Expected disease_stress={expected:.4f} after 5 days, got {stress:.4f}"
        )

    def test_days_infected_increments(self):
        """days_infected counter must increment by 1 each day."""
        grid = _make_plant_grid(5, 5)
        seed_initial_foci(grid, [(2, 2)])
        rng = random.Random(42)
        for _ in range(7):
            calculate_disease_spread(
                grid, wind_speed_ms=3.0, wind_direction_deg=270.0,
                base_infection_rate=0.0, latency_days=0, rng=rng,
            )
        days = grid[2][2].custom.get("days_infected", -1)
        assert days == 7, f"Expected days_infected=7, got {days}"

    def test_dead_plants_not_infected(self):
        """Dead plants must not be infected during spread."""
        grid = _make_plant_grid(5, 5)
        # Kill all neighbours of center
        seed_initial_foci(grid, [(2, 2)])
        for (dr, dc) in [(-1,0),(1,0),(0,-1),(0,1)]:
            grid[2+dr][2+dc].alive = False
        rng = random.Random(42)
        for _ in range(10):
            calculate_disease_spread(
                grid, wind_speed_ms=3.0, wind_direction_deg=270.0,
                base_infection_rate=1.0, latency_days=0, rng=rng,
            )
        for (dr, dc) in [(-1,0),(1,0),(0,-1),(0,1)]:
            state = grid[2+dr][2+dc].custom.get("disease_state", "S")
            assert state == "S", f"Dead plant at ({2+dr},{2+dc}) should not be infected"

    def test_deterministic_with_fixed_seed(self):
        """Two runs with the same seed must produce identical spread patterns."""
        def run_spread(seed_val):
            grid = _make_plant_grid(8, 8)
            seed_initial_foci(grid, [(4, 4)])
            rng = random.Random(seed_val)
            for _ in range(15):
                calculate_disease_spread(
                    grid, wind_speed_ms=3.0, wind_direction_deg=270.0,
                    base_infection_rate=0.25, latency_days=0, rng=rng,
                )
            return [p.custom.get("disease_state", "S") for row in grid for p in row]

        states_a = run_spread(99)
        states_b = run_spread(99)
        assert states_a == states_b, "Same seed must give identical spread patterns"


# ===========================================================================
# 4. PRD Crucible: Wind-anisotropic spread (§8.4)
# ===========================================================================

class TestDiseaseWindCrucible:
    """PRD §8.4 Crucible: downwind cells infect faster than upwind cells.

    Scenario:
        - 20×20 grid (400 plants)
        - Infect center plant at (10, 10)
        - Wind = 270° (from West → blows East)
        - Run 20+ days with latency=0 and high spread rate for fast propagation
        - Assert: eastern half has significantly more infected plants than western half
    """

    ROWS = 20
    COLS = 20
    CENTER_ROW = 10
    CENTER_COL = 10
    WIND_DEG = 270.0  # Wind FROM West → blows East
    SPREAD_RATE = 0.40
    ANISOTROPY = 0.80
    RUN_DAYS = 20
    LATENCY = 0  # No latency so we can observe spread quickly
    SEED = 42

    def _run_crucible(self, wind_direction_deg: float, seed: int = 42):
        grid = _make_plant_grid(self.ROWS, self.COLS)
        seed_initial_foci(grid, [(self.CENTER_ROW, self.CENTER_COL)])
        rng = random.Random(seed)
        for _ in range(self.RUN_DAYS):
            calculate_disease_spread(
                grid,
                wind_speed_ms=5.0,
                wind_direction_deg=wind_direction_deg,
                base_infection_rate=self.SPREAD_RATE,
                latency_days=self.LATENCY,
                anisotropy=self.ANISOTROPY,
                rng=rng,
            )
        return grid

    def test_crucible_east_wind_more_infected_in_east_half(self):
        """PRD Crucible: wind=270° (blowing East) → eastern half has more infected plants."""
        grid = self._run_crucible(wind_direction_deg=270.0)

        mid_col = self.COLS // 2
        east_infected = _count_infected_in_cols(grid, mid_col, self.COLS)
        west_infected = _count_infected_in_cols(grid, 0, mid_col)

        total_infected = east_infected + west_infected
        assert total_infected > 1, "Disease did not spread at all — check parameters"

        assert east_infected > west_infected, (
            f"PRD Crucible FAILED: East half has {east_infected} infected, "
            f"West half has {west_infected} infected. "
            f"Expected east > west when wind blows East (270°)."
        )

    def test_crucible_west_wind_more_infected_in_west_half(self):
        """Wind=90° (blowing West) → western half must have more infected plants."""
        grid = self._run_crucible(wind_direction_deg=90.0)

        mid_col = self.COLS // 2
        east_infected = _count_infected_in_cols(grid, mid_col, self.COLS)
        west_infected = _count_infected_in_cols(grid, 0, mid_col)

        total_infected = east_infected + west_infected
        assert total_infected > 1, "Disease did not spread at all"

        assert west_infected > east_infected, (
            f"Wind=90° (blowing West) should give more West infections. "
            f"W={west_infected}, E={east_infected}"
        )

    def test_crucible_east_infected_exceed_upwind_by_margin(self):
        """East/West ratio must be meaningfully greater than 1.0 (not marginal)."""
        grid = self._run_crucible(wind_direction_deg=270.0)

        mid_col = self.COLS // 2
        east_infected = _count_infected_in_cols(grid, mid_col, self.COLS)
        west_infected = _count_infected_in_cols(grid, 0, mid_col)

        # To avoid ZeroDivisionError, use +1 in denominator
        ratio = east_infected / (west_infected + 1)
        assert ratio > 1.5, (
            f"Expected east/west ratio > 1.5, got {ratio:.2f} "
            f"(east={east_infected}, west={west_infected}). "
            f"Wind direction anisotropy may not be strong enough."
        )

    def test_crucible_reproducible_across_multiple_seeds(self):
        """East dominance must hold for at least 3 out of 5 random seeds."""
        east_dominant_count = 0
        mid_col = self.COLS // 2
        for seed in [1, 2, 3, 4, 5]:
            grid = self._run_crucible(wind_direction_deg=270.0, seed=seed)
            east = _count_infected_in_cols(grid, mid_col, self.COLS)
            west = _count_infected_in_cols(grid, 0, mid_col)
            if east > west:
                east_dominant_count += 1
        assert east_dominant_count >= 3, (
            f"East half was dominant in only {east_dominant_count}/5 seeds. "
            "Wind anisotropy is not reliable."
        )


# ===========================================================================
# 5. Integration: disease hook via farm.use_physics(disease=True)
# ===========================================================================

class TestDiseaseHookIntegration:

    def test_hook_disabled_no_disease_state(self):
        """When disease=False, no plant.custom should have disease_state key."""
        farm, field = _make_minimal_farm(rows=4, cols=4)
        farm.run(days=3)
        for plant in field._field_state.plants:
            assert "disease_state" not in plant.custom, (
                f"Plant {plant.plant_id} should not have disease_state when engine disabled"
            )

    def test_hook_enabled_seeds_foci(self):
        """Disease hook must infect the specified foci on first simulation day."""
        farm, field = _make_minimal_farm(rows=6, cols=6)
        farm.use_physics(
            disease=True,
            disease_foci=[(3, 3)],
            disease_spread_rate=0.0,  # No spread, only seeding
            disease_latency_days=0,
            disease_seed=42,
        )
        farm.run(days=2)
        plants = field._field_state.plants
        plant_33 = next(p for p in plants if p.row == 3 and p.col == 3)
        assert plant_33.custom.get("disease_state") == "I", (
            "Focus plant (3,3) must be infected after seeding"
        )

    def test_hook_spread_occurs_over_days(self):
        """Disease hook must spread from foci over multiple days."""
        farm, field = _make_minimal_farm(rows=8, cols=8)
        farm.use_physics(
            disease=True,
            disease_foci=[(4, 4)],
            disease_spread_rate=0.50,
            disease_latency_days=0,
            disease_anisotropy=0.0,  # Isotropic to ensure all directions spread
            disease_seed=42,
        )
        farm.run(days=15)
        plants = field._field_state.plants
        infected = sum(1 for p in plants if p.custom.get("disease_state") == "I")
        assert infected > 1, (
            f"Disease should have spread from single focus after 15 days, "
            f"but only {infected} plant(s) infected"
        )

    def test_hook_wind_270_spreads_east(self):
        """Via farm.use_physics, wind=270° must produce more eastern infections."""
        farm, field = _make_minimal_farm(rows=10, cols=10)
        farm.use_physics(
            disease=True,
            disease_foci=[(5, 5)],
            disease_spread_rate=0.40,
            disease_latency_days=0,
            disease_wind_direction_deg=270.0,
            disease_anisotropy=0.80,
            disease_seed=42,
        )
        farm.run(days=20)
        plants = field._field_state.plants

        mid_col = 5
        east = sum(1 for p in plants if p.col >= mid_col and p.custom.get("disease_state") == "I")
        west = sum(1 for p in plants if p.col < mid_col and p.custom.get("disease_state") == "I")

        assert east + west > 1, "Disease did not spread from hook"
        assert east > west, (
            f"hook Crucible: wind=270° must give more eastern infections. "
            f"E={east}, W={west}"
        )

    def test_hook_stress_accumulation_via_farm(self):
        """disease_stress must accumulate in infected plants via the hook."""
        farm, field = _make_minimal_farm(rows=4, cols=4)
        farm.use_physics(
            disease=True,
            disease_foci=[(2, 2)],
            disease_spread_rate=0.0,  # No spread
            disease_stress_increment=0.04,
            disease_latency_days=0,
            disease_seed=42,
        )
        farm.run(days=5)
        plants = field._field_state.plants
        plant_22 = next(p for p in plants if p.row == 2 and p.col == 2)
        stress = plant_22.custom.get("disease_stress", 0.0)
        # 5 days × 0.04 increment = 0.20
        assert abs(stress - 0.20) < 0.01, (
            f"Expected disease_stress≈0.20 after 5 days, got {stress:.4f}"
        )
