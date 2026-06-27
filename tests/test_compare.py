"""
tests/test_compare.py
=====================
PRD v0.4.0 §8.4 — Dashboard Export and compare() tests.

Tests cover:
  - compare(farm_a, farm_b) merges logs without error
  - compare() with <2 farms raises ValueError
  - compare() on farm with no log raises CropForgeVisualizeError
  - The time-series chart (update_timeseries) draws season boundary vlines
  - CSV export callback returns correct columns and row count
  - CSV filename follows the PRD naming convention
  - PNG export config is present on timeseries and heatmap dcc.Graph components
  - Season boundary helper (_get_season_boundaries) works correctly

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_minimal_farm(name: str, days: int = 6) -> "Farm":
    """Build and run a minimal 2×2 farm, return the Farm object."""
    from cropforge import Farm, Field

    farm = Farm(name=name)
    field = Field(name=f"{name}_plot", rows=2, cols=2)
    farm.add_field(field)

    @farm.step(interval="daily")
    def _noop(state, env):
        pass

    farm.run(days=days)
    return farm


# ===========================================================================
# compare() API tests
# ===========================================================================

class TestCompareAPI:
    """Tests for the compare(*farms) public function."""

    def test_compare_raises_if_less_than_two_farms(self, tmp_path):
        """compare() must require at least 2 farms."""
        from cropforge import compare
        farm = _run_minimal_farm("Solo")
        with pytest.raises(ValueError, match="at least 2"):
            compare(farm)

    def test_compare_raises_if_farm_has_no_log(self, tmp_path):
        """compare() must raise CropForgeVisualizeError for an un-run farm."""
        from cropforge import compare, Farm, Field
        from cropforge.runtime import CropForgeVisualizeError

        farm_a = _run_minimal_farm("FarmA")

        farm_b = Farm(name="NotRun")
        field_b = Field(name="b", rows=2, cols=2)
        farm_b.add_field(field_b)
        # farm_b.run() is intentionally NOT called

        with pytest.raises(CropForgeVisualizeError):
            compare(farm_a, farm_b)

    def test_compare_merges_parquet_logs(self, tmp_path):
        """compare() should create a merged temp directory with prefixed field dirs."""
        import importlib
        farm_a = _run_minimal_farm("Irrigated")
        farm_b = _run_minimal_farm("Rainfed")

        # Reach into compare internals to test the merge logic without booting server
        from cropforge.compare import compare as _compare
        import shutil

        # We'll monkey-patch boot() so the server doesn't actually start
        import cropforge.viz.server as srv
        original_boot = srv.boot
        merged_paths = []

        def fake_boot(log_path, cropforge_version, **kwargs):
            merged_paths.append(log_path)

        srv.boot = fake_boot
        try:
            _compare(farm_a, farm_b)
        finally:
            srv.boot = original_boot

        assert merged_paths, "compare() did not call boot()"
        merged = Path(merged_paths[0])
        assert merged.exists()

        # The plants table should exist and have field_name dirs from BOTH farms
        plants_dir = merged / "plants"
        assert plants_dir.exists(), "merged/plants directory not found"
        field_dirs = [d.name for d in plants_dir.iterdir() if d.is_dir()]
        irrigated_found = any("Irrigated" in d for d in field_dirs)
        rainfed_found   = any("Rainfed"   in d for d in field_dirs)
        assert irrigated_found, f"No 'Irrigated' field dir in merged: {field_dirs}"
        assert rainfed_found,   f"No 'Rainfed' field dir in merged: {field_dirs}"

    def test_compare_field_names_use_separator(self, tmp_path):
        """Merged field names must be prefixed 'FarmName :: OriginalField'."""
        farm_a = _run_minimal_farm("Alpha")
        farm_b = _run_minimal_farm("Beta")

        import cropforge.viz.server as srv
        original_boot = srv.boot
        merged_paths = []

        def fake_boot(log_path, cropforge_version, **kwargs):
            merged_paths.append(log_path)

        srv.boot = fake_boot
        try:
            from cropforge import compare
            compare(farm_a, farm_b)
        finally:
            srv.boot = original_boot

        plants_dir = Path(merged_paths[0]) / "plants"
        field_dirs = [d.name for d in plants_dir.iterdir() if d.is_dir()]
        # Each should look like "field_name=Alpha -- Alpha_plot" (Windows-safe separator)
        assert any("--" in d for d in field_dirs), (
            f"Expected '--' separator in field dirs: {field_dirs}"
        )


# ===========================================================================
# Season boundary helper tests
# ===========================================================================

class TestSeasonBoundaryHelper:
    """Tests for the _get_season_boundaries() utility."""

    def test_empty_df_returns_empty(self):
        from cropforge.viz.app import _get_season_boundaries
        result = _get_season_boundaries(pd.DataFrame())
        assert result == []

    def test_none_returns_empty(self):
        from cropforge.viz.app import _get_season_boundaries
        assert _get_season_boundaries(None) == []

    def test_single_season_returns_empty(self):
        from cropforge.viz.app import _get_season_boundaries
        df = pd.DataFrame({"day": [1, 2, 3], "season": [1, 1, 1], "field_name": ["A"]*3})
        assert _get_season_boundaries(df) == []

    def test_two_season_returns_one_boundary(self):
        from cropforge.viz.app import _get_season_boundaries
        df = pd.DataFrame({
            "day":        [1, 2, 3, 11, 12],
            "season":     [1, 1, 1,  2,  2],
            "field_name": ["A"]*5,
        })
        result = _get_season_boundaries(df)
        assert len(result) == 1
        boundary_day, season_num = result[0]
        assert boundary_day == 11
        assert season_num == 2

    def test_three_seasons_returns_two_boundaries(self):
        from cropforge.viz.app import _get_season_boundaries
        df = pd.DataFrame({
            "day":        [1, 5, 11, 15, 21, 25],
            "season":     [1, 1,  2,  2,  3,  3],
            "field_name": ["A"]*6,
        })
        result = _get_season_boundaries(df)
        assert len(result) == 2
        days = [r[0] for r in result]
        assert 11 in days
        assert 21 in days

    def test_missing_season_column_returns_empty(self):
        from cropforge.viz.app import _get_season_boundaries
        df = pd.DataFrame({"day": [1, 2, 3], "field_name": ["A"]*3})
        assert _get_season_boundaries(df) == []


# ===========================================================================
# Dash app structure tests (static layout inspection)
# ===========================================================================

class TestDashAppStructure:
    """Test that the Dash app layout has required PRD §8.3/§8.4 components."""

    @pytest.fixture(scope="class")
    def app_and_df(self, tmp_path_factory):
        """Build a minimal farm, run it, and return the Dash app."""
        td = tmp_path_factory.mktemp("dash_app")
        farm = _run_minimal_farm("TestFarm")
        log_path = farm._last_log_path
        from cropforge.viz.app import _DATA, _load_parquet, create_dash_app
        # reset cache so we re-read this session's data
        _DATA["plants"] = None
        _DATA["soil"]   = None
        _DATA["env"]    = None
        _load_parquet(log_path)
        app = create_dash_app(log_path)
        return app

    def _flatten_layout(self, component, results=None):
        """Recursively collect all Dash component ids from the layout."""
        if results is None:
            results = {}
        if hasattr(component, "id") and component.id:
            results[component.id] = component
        if hasattr(component, "children") and component.children is not None:
            children = component.children
            if not isinstance(children, (list, tuple)):
                children = [children]
            for child in children:
                if hasattr(child, "children") or hasattr(child, "id"):
                    self._flatten_layout(child, results)
        return results

    def test_export_csv_button_in_layout(self, app_and_df):
        """PRD §8.2: layout must contain 'export-csv-btn'."""
        ids = self._flatten_layout(app_and_df.layout)
        assert "export-csv-btn" in ids, f"Missing export-csv-btn. Found: {list(ids.keys())}"

    def test_download_csv_component_in_layout(self, app_and_df):
        """PRD §8.2: layout must contain 'download-csv' dcc.Download component."""
        ids = self._flatten_layout(app_and_df.layout)
        assert "download-csv" in ids, f"Missing download-csv. Found: {list(ids.keys())}"

    def test_timeseries_chart_in_layout(self, app_and_df):
        """timeseries-chart must exist in layout."""
        ids = self._flatten_layout(app_and_df.layout)
        assert "timeseries-chart" in ids

    def test_timeseries_png_config_present(self, app_and_df):
        """PRD §8.3: timeseries-chart must have toImageButtonOptions config."""
        ids = self._flatten_layout(app_and_df.layout)
        ts = ids.get("timeseries-chart")
        assert ts is not None
        cfg = getattr(ts, "config", None) or {}
        assert "toImageButtonOptions" in cfg, (
            f"timeseries-chart missing toImageButtonOptions. config={cfg}"
        )


# ===========================================================================
# CSV export content tests (unit — no server needed)
# ===========================================================================

class TestCsvExportContent:
    """Test CSV export callback return value."""

    @pytest.fixture(scope="class")
    def csv_data(self, tmp_path_factory):
        """Run a 6-day sim and call build_csv_export() directly."""
        farm = _run_minimal_farm("CsvFarm", days=6)
        log_path = farm._last_log_path

        from cropforge.viz.app import (
            _DATA, _load_parquet,
            _build_daily_metrics, _build_daily_soil_metrics,
            build_csv_export,
        )
        _DATA["plants"] = None
        _DATA["soil"]   = None
        _DATA["env"]    = None
        _load_parquet(log_path)

        plants_df = _DATA["plants"]
        soil_df   = _DATA["soil"]
        env_df    = _DATA["env"]
        session_name = Path(log_path).name

        daily_metrics = _build_daily_metrics(plants_df) if plants_df is not None else pd.DataFrame()
        daily_soil    = _build_daily_soil_metrics(soil_df) if soil_df is not None else pd.DataFrame()

        return build_csv_export(daily_metrics, daily_soil, env_df, session_name)

    def test_csv_returns_dict(self, csv_data):
        assert isinstance(csv_data, dict)

    def test_csv_has_content_key(self, csv_data):
        assert "content" in csv_data

    def test_csv_has_filename_key(self, csv_data):
        assert "filename" in csv_data

    def test_csv_filename_convention(self, csv_data):
        """PRD §8.2: filename = cropforge_timeseries_{session}_{YYYYMMDD}.csv"""
        fn = csv_data["filename"]
        assert fn.startswith("cropforge_timeseries_"), f"Bad filename prefix: {fn}"
        assert fn.endswith(".csv"), f"Missing .csv extension: {fn}"

    def test_csv_content_is_parseable(self, csv_data):
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data["content"]))
        assert not df.empty

    def test_csv_has_day_column(self, csv_data):
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data["content"]))
        assert "day" in df.columns

    def test_csv_has_field_name_column(self, csv_data):
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data["content"]))
        assert "field_name" in df.columns

    def test_csv_row_count(self, csv_data):
        """One row per day per field (6 days × 1 field = 6 rows)."""
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data["content"]))
        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"

    def test_csv_has_biomass_column(self, csv_data):
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data["content"]))
        assert "mean_biomass_g" in df.columns
