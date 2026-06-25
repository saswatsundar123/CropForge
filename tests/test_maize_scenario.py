"""
tests/test_maize_scenario.py
==============================
Verification test for the CropForge v0.2.0 Maize Dual-Plot Crucible.

This test suite:
1. Runs examples/maize_dual_plot.py as a subprocess to produce a fresh Parquet log.
2. Reads the resulting plant table.
3. Asserts the exact physical outcomes required by PRD v0.2.0 Section 16:

   HARD PAN PHYSICS (Plot B):
     "Root depth correctly clamps at 19 cm for Plot B in maize_dual_plot_v2.py"
     → By day 90, ALL plants in Plot_B_Hardpan have root_depth_cm <= 19.0 cm.

   FREE ROOT GROWTH (Plot A):
     → By day 90, ALL living plants in Plot_A_Slope have root_depth_cm > 19.0 cm.
     (Plot A soil has no impedance layer, roots grow ~0.35 cm/day × 90 days
     starting from 2 cm = ~33 cm by end of season.)

   DIVERGENCE:
     → Mean root depth of Plot A > Mean root depth of Plot B (significant gap).

PRD v0.2.0 References:
    Section 5.5 — Root depth clamps at hard pan layer
    Section 10  — Backward compatibility (Plot A unaffected)
    Section 16  — Hard pan constraint: root depth clamps at 19 cm for Plot B

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SCRIPT       = PROJECT_ROOT / "examples" / "maize_dual_plot.py"
OUTPUT_ROOT  = PROJECT_ROOT / "cropforge_output"

HARD_PAN_DEPTH_CM = 19.0   # boundary defined in maize_soil_plotB_hardpan.csv
SIMULATION_DAYS   = 90


# ---------------------------------------------------------------------------
# Session fixture: run the example once, load the resulting Parquet dataset
# ---------------------------------------------------------------------------

def _discover_latest_log(prefix: str = "Maize_DualPlot_2026") -> Path:
    """Return the most-recently written Maize_DualPlot session directory."""
    candidates = sorted(
        OUTPUT_ROOT.glob(f"{prefix}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No session directories matching '{prefix}_*' found under "
            f"{OUTPUT_ROOT}. Run examples/maize_dual_plot.py first."
        )
    return candidates[0]


def _load_plant_table(log_dir: Path) -> pd.DataFrame:
    """Read all plant Parquet files into a single DataFrame.

    The Parquet dataset is partitioned by hive (field_name=.../day=...) under
    a 'plants/' subdirectory.  We reconstruct the partition values from the
    directory names and return the concatenated DataFrame.
    """
    # Logger writes plant data into log_dir/plants/**/*.parquet
    plants_dir = log_dir / "plants"
    if not plants_dir.exists():
        raise FileNotFoundError(
            f"'plants' subdirectory not found under {log_dir}. "
            f"Available: {[p.name for p in log_dir.iterdir() if p.is_dir()]}"
        )

    parquet_files = list(plants_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {plants_dir}")

    frames = []
    for pf in parquet_files:
        tbl = pq.read_table(pf)
        frame = tbl.to_pandas()

        # Inject hive partition values if missing from the table columns
        for part in Path(pf).parts:
            if "=" in part:
                k, v = part.split("=", 1)
                if k not in frame.columns:
                    frame[k] = v

        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int32")
    return df


@pytest.fixture(scope="module")
def plant_df():
    """Run the example script and return the day-90 plant DataFrame."""
    # Run the example (produces fresh Parquet each time)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        # Print stdout/stderr for debugging, then fail
        print("STDOUT:", result.stdout[-3000:] if result.stdout else "(empty)")
        print("STDERR:", result.stderr[-3000:] if result.stderr else "(empty)")
        pytest.fail(
            f"maize_dual_plot.py exited with code {result.returncode}."
        )

    # Discover the log directory produced by this run
    log_dir = _discover_latest_log()
    df = _load_plant_table(log_dir)

    # Filter to simulation day 90 (plant table)
    # and exclude soil / environment partitions (they don't have root_depth_cm)
    if "root_depth_cm" not in df.columns:
        pytest.fail(
            "root_depth_cm column not found in Parquet plant table. "
            "Check that the plant table files were loaded (not soil/env tables)."
        )

    day90 = df[df["day"] == SIMULATION_DAYS].copy()
    if day90.empty:
        pytest.fail(f"No records found for day={SIMULATION_DAYS} in Parquet log.")

    return day90


# ---------------------------------------------------------------------------
# Test 1: Hard-pan physics — Plot B roots clamped at 19 cm
# ---------------------------------------------------------------------------

class TestHardPanConstraint:
    """PRD v0.2.0 Section 5.5 & 16: Root depth clamps at 19 cm for Plot B."""

    def test_plot_b_exists_in_log(self, plant_df):
        """Plot_B_Hardpan must have records in the day-90 plant table."""
        plot_b = plant_df[plant_df["field_name"] == "Plot_B_Hardpan"]
        assert not plot_b.empty, (
            "Plot_B_Hardpan has no records on day 90 — check that the field "
            "was added to the farm and the simulation ran correctly."
        )

    def test_plot_b_max_root_depth_at_or_below_hard_pan(self, plant_df):
        """ALL Plot B plants must have root_depth_cm <= 19.0 cm.

        This is the central PRD v0.2.0 assertion: the root impedance engine
        (phase=-1) sets root_growth_multiplier=0.0 when penetration_resistance
        >= 2.5 MPa. The researcher's root_growth step (phase=0) multiplies the
        base rate by 0.0, so root_depth_cm stops advancing at the hard-pan
        boundary (19 cm).
        """
        plot_b = plant_df[plant_df["field_name"] == "Plot_B_Hardpan"]
        max_root = plot_b["root_depth_cm"].max()

        assert max_root <= HARD_PAN_DEPTH_CM, (
            f"Plot B hard-pan FAILED: max root_depth_cm={max_root:.3f} cm "
            f"exceeds the hard-pan boundary of {HARD_PAN_DEPTH_CM} cm.\n"
            f"The root impedance engine should have produced multiplier=0.0 "
            f"for pen_resistance=3.0 MPa, blocking root growth at 19 cm."
        )

    def test_plot_b_mean_root_below_hard_pan(self, plant_df):
        """Mean root depth for Plot B must be at or below the 19 cm hard pan."""
        plot_b = plant_df[plant_df["field_name"] == "Plot_B_Hardpan"]
        mean_root = float(plot_b["root_depth_cm"].mean())

        assert mean_root <= HARD_PAN_DEPTH_CM, (
            f"Plot B mean root depth {mean_root:.2f} cm should be at or below "
            f"{HARD_PAN_DEPTH_CM} cm (hard pan boundary). "
            f"All plants clamped at exactly 19.0 cm is the correct outcome."
        )

    def test_plot_b_no_plant_exceeds_hard_pan(self, plant_df):
        """No individual plant in Plot B may have root_depth_cm > 19.0 cm.

        This test checks every single plant row — not just the max — to
        confirm the impedance engine applied uniformly to all cells.
        """
        plot_b = plant_df[plant_df["field_name"] == "Plot_B_Hardpan"]
        violators = plot_b[plot_b["root_depth_cm"] > HARD_PAN_DEPTH_CM]

        assert violators.empty, (
            f"{len(violators)} plant(s) in Plot_B_Hardpan exceeded the "
            f"19 cm hard-pan boundary:\n"
            f"{violators[['plant_id', 'root_depth_cm']].to_string()}"
        )


# ---------------------------------------------------------------------------
# Test 2: Free root growth — Plot A roots past 19 cm
# ---------------------------------------------------------------------------

class TestFreeRootGrowth:
    """Plot A (no impedance) must have roots deeper than 19 cm by day 90."""

    def test_plot_a_exists_in_log(self, plant_df):
        """Plot_A_Slope must have records in the day-90 plant table."""
        plot_a = plant_df[plant_df["field_name"] == "Plot_A_Slope"]
        assert not plot_a.empty, "Plot_A_Slope has no records on day 90."

    def test_plot_a_max_root_exceeds_hard_pan_depth(self, plant_df):
        """Living Plot A plants must reach beyond 19 cm.

        With base_rate=0.35 cm/day and no impedance:
        day-90 root depth ≈ 2.0 (initial) + 0.35 × 90 ≈ 33.5 cm.
        We assert > 19.0 cm as the strict floor.
        """
        plot_a = plant_df[plant_df["field_name"] == "Plot_A_Slope"]
        # Consider only living plants (dead plants stop growing)
        alive_a = plot_a[plot_a["alive"] == True]

        if alive_a.empty:
            pytest.skip(
                "All Plot A plants are dead on day 90 — increase soil moisture "
                "or reduce stress parameters to maintain living population."
            )

        max_root = alive_a["root_depth_cm"].max()
        assert max_root > HARD_PAN_DEPTH_CM, (
            f"Plot A max root depth {max_root:.2f} cm should exceed "
            f"{HARD_PAN_DEPTH_CM} cm by day 90 (no impedance in Plot A soil). "
            f"With base_rate=0.35 cm/day the expected depth is ~33.5 cm."
        )

    def test_plot_a_mean_root_depth_above_hard_pan(self, plant_df):
        """Mean root depth of living Plot A plants must exceed 19 cm."""
        plot_a = plant_df[plant_df["field_name"] == "Plot_A_Slope"]
        alive_a = plot_a[plot_a["alive"] == True]

        if alive_a.empty:
            pytest.skip("All Plot A plants are dead; cannot verify mean root depth.")

        mean_root = alive_a["root_depth_cm"].mean()
        assert mean_root > HARD_PAN_DEPTH_CM, (
            f"Plot A mean root depth {mean_root:.2f} cm should exceed "
            f"{HARD_PAN_DEPTH_CM} cm (hard pan boundary)."
        )


# ---------------------------------------------------------------------------
# Test 3: Divergence — Plot A roots significantly deeper than Plot B
# ---------------------------------------------------------------------------

class TestPlotDivergence:
    """The two plots must show clear, measurable root-depth divergence."""

    def test_root_depth_divergence(self, plant_df):
        """Plot A mean root depth must be substantially greater than Plot B.

        The hard pan physically prevents Plot B roots from growing past 19 cm.
        Plot A roots (expected ~33 cm) should be at least 10 cm deeper
        than Plot B (expected ≤ 19 cm).
        """
        plot_a = plant_df[plant_df["field_name"] == "Plot_A_Slope"]
        plot_b = plant_df[plant_df["field_name"] == "Plot_B_Hardpan"]

        mean_a = plot_a["root_depth_cm"].mean()
        mean_b = plot_b["root_depth_cm"].mean()
        gap    = mean_a - mean_b

        print(f"\nRoot depth divergence: Plot A={mean_a:.2f} cm, Plot B={mean_b:.2f} cm, gap={gap:.2f} cm")

        assert gap > 10.0, (
            f"Root depth divergence gap={gap:.2f} cm is less than 10 cm. "
            f"Expected Plot A ~33 cm, Plot B ~18 cm (hard pan blocked). "
            f"Check that root impedance engine is correctly wired."
        )

    def test_plot_a_roots_deeper_than_plot_b_max(self, plant_df):
        """Plot A maximum root depth must exceed Plot B maximum root depth."""
        plot_a = plant_df[plant_df["field_name"] == "Plot_A_Slope"]
        plot_b = plant_df[plant_df["field_name"] == "Plot_B_Hardpan"]

        max_a = plot_a["root_depth_cm"].max()
        max_b = plot_b["root_depth_cm"].max()

        assert max_a > max_b, (
            f"Plot A max root ({max_a:.2f} cm) should exceed "
            f"Plot B max root ({max_b:.2f} cm)."
        )

    def test_both_fields_have_plants_logged(self, plant_df):
        """Both fields must appear in the day-90 log — simulation ran for both."""
        field_names = set(plant_df["field_name"].unique())
        assert "Plot_A_Slope"   in field_names, "Plot_A_Slope missing from Parquet log"
        assert "Plot_B_Hardpan" in field_names, "Plot_B_Hardpan missing from Parquet log"


# ---------------------------------------------------------------------------
# Test 4: v0.1.0 backward compatibility sanity
# ---------------------------------------------------------------------------

class TestBackwardCompatSanity:
    """Verify no v0.2.0 additions broke the plant schema in the Parquet log."""

    def test_plant_schema_has_root_depth_cm(self, plant_df):
        """root_depth_cm must be present and numeric in the plant Parquet table."""
        assert "root_depth_cm" in plant_df.columns, (
            "root_depth_cm column missing from Parquet plant table."
        )
        assert pd.api.types.is_numeric_dtype(plant_df["root_depth_cm"]), (
            "root_depth_cm must be a numeric column."
        )

    def test_plant_schema_has_alive_column(self, plant_df):
        """alive column must be present (v0.1.0 schema field)."""
        assert "alive" in plant_df.columns

    def test_plant_count_per_field_is_correct(self, plant_df):
        """Each 10x10 field must have exactly 100 plant records on day 90."""
        for field_name in ["Plot_A_Slope", "Plot_B_Hardpan"]:
            sub = plant_df[plant_df["field_name"] == field_name]
            assert len(sub) == 100, (
                f"{field_name} has {len(sub)} plant records on day 90, "
                f"expected 100 (10x10 grid)."
            )
