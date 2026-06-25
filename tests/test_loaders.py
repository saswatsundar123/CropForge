"""
tests/test_loaders.py
=====================
Tests for Weather.from_csv and Soil.from_csv (PRD Section 8).

Covers:
  - Happy-path CSV parsing for both loaders
  - Flexible column mapping (non-default column names)
  - Wind unit conversion to m/s (m/s, km/h, knots)
  - Mean temperature computed from tmax/tmin when tmean column absent
  - tmean_col override
  - CO2 default value (415.0 ppm)
  - DOY from date column
  - DOY fallback (no date column)
  - Cyclic day wrapping for simulations longer than the weather file
  - EnvironmentState returned by get_day() has correct types and values
  - Missing required weather columns → ValueError
  - Missing file → FileNotFoundError
  - Empty file → ValueError
  - Soil apply=uniform: one layer per row in CSV
  - Soil apply=uniform: build_grid broadcasts correctly
  - Soil apply=spatial: requires row/col columns
  - Soil apply=spatial: build_grid respects cell coordinates
  - Soil missing required columns → ValueError
  - Soil invalid apply value → ValueError
  - Soil missing file → FileNotFoundError
  - n_layers property
  - repr

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
"""

import textwrap
from io import StringIO
from pathlib import Path

import pytest

from cropforge.loaders import Soil, Weather, _wind_to_ms
from cropforge.state import EnvironmentState, SoilVoxelState


# ===========================================================================
# Helper: write a temp CSV
# ===========================================================================

def _write_csv(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# Minimal valid weather CSV (2 rows)
_WEATHER_CSV = """\
    date,tmax_c,tmin_c,radiation_mj,rainfall_mm,humidity_pct,wind_ms
    2026-03-01,20.0,10.0,15.0,2.5,65.0,2.0
    2026-03-02,22.0,12.0,17.0,0.0,60.0,3.0
"""

# Minimal valid soil CSV (2 layers)
_SOIL_CSV = """\
    layer,depth_top_cm,depth_bottom_cm,moisture_pct,n_kg_ha,bulk_density,pen_resistance_mpa
    0,0.0,20.0,28.0,40.0,1.25,0.8
    1,20.0,40.0,22.0,15.0,1.35,1.2
"""


# ===========================================================================
# Wind unit conversion
# ===========================================================================

class TestWindConversion:
    def test_ms_passthrough(self):
        assert _wind_to_ms(3.0, "m/s") == pytest.approx(3.0)

    def test_ms_shorthand(self):
        assert _wind_to_ms(3.0, "ms") == pytest.approx(3.0)

    def test_kmh_to_ms(self):
        assert _wind_to_ms(36.0, "km/h") == pytest.approx(10.0, rel=1e-4)

    def test_kmh_shorthand(self):
        assert _wind_to_ms(36.0, "kmh") == pytest.approx(10.0, rel=1e-4)

    def test_knots_to_ms(self):
        # 10 knots → 5.14444 m/s
        assert _wind_to_ms(10.0, "knots") == pytest.approx(5.14444, rel=1e-4)

    def test_knots_shorthand(self):
        assert _wind_to_ms(10.0, "kt") == pytest.approx(5.14444, rel=1e-4)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown wind unit"):
            _wind_to_ms(5.0, "mph")

    def test_case_insensitive(self):
        assert _wind_to_ms(36.0, "KM/H") == pytest.approx(10.0, rel=1e-4)


# ===========================================================================
# Weather.from_csv — happy path
# ===========================================================================

class TestWeatherFromCsv:
    def test_loads_successfully(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        assert isinstance(w, Weather)

    def test_n_days(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        assert w.n_days == 2

    def test_repr(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        assert "n_days=2" in repr(w)

    def test_get_day_returns_environment_state(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env = w.get_day(1)
        assert isinstance(env, EnvironmentState)

    def test_get_day_values_day1(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env = w.get_day(1)
        assert env.temp_max_c == pytest.approx(20.0)
        assert env.temp_min_c == pytest.approx(10.0)
        assert env.radiation_mj_m2 == pytest.approx(15.0)
        assert env.rainfall_mm == pytest.approx(2.5)
        assert env.humidity_pct == pytest.approx(65.0)
        assert env.wind_speed_ms == pytest.approx(2.0)

    def test_get_day_values_day2(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env = w.get_day(2)
        assert env.temp_max_c == pytest.approx(22.0)
        assert env.rainfall_mm == pytest.approx(0.0)

    def test_tmean_computed_from_tmax_tmin(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env = w.get_day(1)
        assert env.temp_mean_c == pytest.approx((20.0 + 10.0) / 2)

    def test_tmean_override_column(self, tmp_path):
        content = """\
            date,tmax_c,tmin_c,tmean_c,radiation_mj,rainfall_mm,humidity_pct,wind_ms
            2026-01-01,20.0,10.0,17.0,15.0,0.0,60.0,2.0
        """
        csv = _write_csv(tmp_path, "w.csv", content)
        w = Weather.from_csv(str(csv), tmean_col="tmean_c")
        env = w.get_day(1)
        assert env.temp_mean_c == pytest.approx(17.0)

    def test_co2_default_415(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env = w.get_day(1)
        assert env.co2_ppm == pytest.approx(415.0)

    def test_co2_from_column(self, tmp_path):
        content = """\
            date,tmax_c,tmin_c,radiation_mj,rainfall_mm,humidity_pct,wind_ms,co2
            2026-01-01,20.0,10.0,15.0,0.0,60.0,2.0,450.0
        """
        csv = _write_csv(tmp_path, "w.csv", content)
        w = Weather.from_csv(str(csv), co2_col="co2")
        env = w.get_day(1)
        assert env.co2_ppm == pytest.approx(450.0)

    def test_et0_default_zero(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env = w.get_day(1)
        assert env.et0_mm == pytest.approx(0.0)

    def test_et0_from_column(self, tmp_path):
        content = """\
            date,tmax_c,tmin_c,radiation_mj,rainfall_mm,humidity_pct,wind_ms,et0
            2026-01-01,20.0,10.0,15.0,0.0,60.0,2.0,4.5
        """
        csv = _write_csv(tmp_path, "w.csv", content)
        w = Weather.from_csv(str(csv), et0_col="et0")
        env = w.get_day(1)
        assert env.et0_mm == pytest.approx(4.5)

    def test_doy_from_date_column(self, tmp_path):
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env = w.get_day(1)
        # 2026-03-01 → DOY 60
        assert env.doy == 60

    def test_doy_fallback_no_date_column(self, tmp_path):
        content = """\
            tmax_c,tmin_c,radiation_mj,rainfall_mm,humidity_pct,wind_ms
            20.0,10.0,15.0,0.0,60.0,2.0
            22.0,12.0,17.0,0.0,58.0,3.0
        """
        csv = _write_csv(tmp_path, "w.csv", content)
        w = Weather.from_csv(str(csv))
        # No date column → DOY 1, 2
        env1 = w.get_day(1)
        env2 = w.get_day(2)
        assert env1.doy == 1
        assert env2.doy == 2

    def test_start_doy_override(self, tmp_path):
        content = """\
            tmax_c,tmin_c,radiation_mj,rainfall_mm,humidity_pct,wind_ms
            20.0,10.0,15.0,0.0,60.0,2.0
        """
        csv = _write_csv(tmp_path, "w.csv", content)
        w = Weather.from_csv(str(csv), start_doy=200)
        assert w.get_day(1).doy == 200

    def test_cyclic_wrap_beyond_file_length(self, tmp_path):
        """Day 3 wraps to row 1 for a 2-day file."""
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env1 = w.get_day(1)
        env3 = w.get_day(3)   # should wrap → same as day 1
        assert env3.temp_max_c == pytest.approx(env1.temp_max_c)
        assert env3.temp_min_c == pytest.approx(env1.temp_min_c)

    def test_get_day_day_field_set_correctly(self, tmp_path):
        """env.day must equal the requested simulation day, not wrapped row."""
        csv = _write_csv(tmp_path, "w.csv", _WEATHER_CSV)
        w = Weather.from_csv(str(csv))
        env = w.get_day(99)
        assert env.day == 99

    def test_wind_conversion_applied(self, tmp_path):
        """Wind column in km/h must be converted to m/s internally."""
        content = """\
            date,tmax_c,tmin_c,radiation_mj,rainfall_mm,humidity_pct,wind_kmh
            2026-01-01,20.0,10.0,15.0,0.0,60.0,36.0
        """
        csv = _write_csv(tmp_path, "w.csv", content)
        w = Weather.from_csv(str(csv), wind_col="wind_kmh", wind_unit="km/h")
        env = w.get_day(1)
        assert env.wind_speed_ms == pytest.approx(10.0, rel=1e-4)

    def test_flexible_column_names(self, tmp_path):
        """All column names can be overridden via kwargs."""
        content = """\
            DATE,TMAX,TMIN,RAD,RAIN,HUM,WIND
            2026-01-01,25.0,15.0,18.0,1.0,55.0,2.5
        """
        csv = _write_csv(tmp_path, "w.csv", content)
        w = Weather.from_csv(
            str(csv),
            date_col="DATE",
            tmax_col="TMAX",
            tmin_col="TMIN",
            radiation_col="RAD",
            rainfall_col="RAIN",
            humidity_col="HUM",
            wind_col="WIND",
        )
        env = w.get_day(1)
        assert env.temp_max_c == pytest.approx(25.0)

    def test_comment_lines_skipped(self, tmp_path):
        """Lines beginning with # must be ignored."""
        content = """\
            # SYNTHETIC DATA — NOT FROM A WEATHER STATION
            date,tmax_c,tmin_c,radiation_mj,rainfall_mm,humidity_pct,wind_ms
            2026-01-01,20.0,10.0,15.0,0.0,60.0,2.0
        """
        csv = _write_csv(tmp_path, "w.csv", content)
        w = Weather.from_csv(str(csv))
        assert w.n_days == 1

    def test_missing_required_column_raises(self, tmp_path):
        content = """\
            date,tmax_c,tmin_c,rainfall_mm,humidity_pct,wind_ms
            2026-01-01,20.0,10.0,0.0,60.0,2.0
        """
        # Missing radiation_mj column
        csv = _write_csv(tmp_path, "w.csv", content)
        with pytest.raises(ValueError, match="missing required columns"):
            Weather.from_csv(str(csv))

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Weather.from_csv(str(tmp_path / "nonexistent.csv"))

    def test_empty_file_raises(self, tmp_path):
        content = "date,tmax_c,tmin_c,radiation_mj,rainfall_mm,humidity_pct,wind_ms\n"
        csv = _write_csv(tmp_path, "w.csv", content)
        with pytest.raises(ValueError, match="no data rows"):
            Weather.from_csv(str(csv))


# ===========================================================================
# Soil.from_csv — uniform
# ===========================================================================

class TestSoilFromCsvUniform:
    def test_loads_successfully(self, tmp_path):
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        soil = Soil.from_csv(str(csv), apply="uniform")
        assert isinstance(soil, Soil)

    def test_n_layers(self, tmp_path):
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        soil = Soil.from_csv(str(csv), apply="uniform")
        assert soil.n_layers == 2

    def test_repr(self, tmp_path):
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        soil = Soil.from_csv(str(csv), apply="uniform")
        assert "uniform" in repr(soil)

    def test_build_grid_shape(self, tmp_path):
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        soil = Soil.from_csv(str(csv), apply="uniform")
        grid = soil.build_grid(rows=3, cols=4)
        assert len(grid) == 3
        assert len(grid[0]) == 4
        assert len(grid[0][0]) == 2    # 2 layers from CSV

    def test_build_grid_values_topsoil(self, tmp_path):
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        soil = Soil.from_csv(str(csv), apply="uniform")
        grid = soil.build_grid(rows=2, cols=2)
        topsoil = grid[0][0][0]
        assert isinstance(topsoil, SoilVoxelState)
        assert topsoil.moisture_pct == pytest.approx(28.0)
        assert topsoil.nitrogen_kg_ha == pytest.approx(40.0)
        assert topsoil.depth_top_cm == pytest.approx(0.0)
        assert topsoil.depth_bottom_cm == pytest.approx(20.0)
        assert topsoil.bulk_density == pytest.approx(1.25)
        assert topsoil.penetration_resistance == pytest.approx(0.8)

    def test_build_grid_values_second_layer(self, tmp_path):
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        soil = Soil.from_csv(str(csv), apply="uniform")
        grid = soil.build_grid(rows=2, cols=2)
        layer1 = grid[0][0][1]
        assert layer1.depth_top_cm == pytest.approx(20.0)
        assert layer1.moisture_pct == pytest.approx(22.0)

    def test_build_grid_row_col_indices_set(self, tmp_path):
        """SoilVoxelState.row and .col must match grid position."""
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        soil = Soil.from_csv(str(csv), apply="uniform")
        grid = soil.build_grid(rows=3, cols=4)
        assert grid[2][3][0].row == 2
        assert grid[2][3][0].col == 3

    def test_uniform_broadcasts_same_values_all_cells(self, tmp_path):
        """Every cell must have identical layer values."""
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        soil = Soil.from_csv(str(csv), apply="uniform")
        grid = soil.build_grid(rows=5, cols=5)
        ref = grid[0][0][0].moisture_pct
        for r in range(5):
            for c in range(5):
                assert grid[r][c][0].moisture_pct == pytest.approx(ref)

    def test_comment_lines_skipped_soil(self, tmp_path):
        content = """\
            # Field capacity = 35%, PWP = 12%
            layer,depth_top_cm,depth_bottom_cm,moisture_pct,n_kg_ha,bulk_density,pen_resistance_mpa
            0,0.0,20.0,28.0,40.0,1.25,0.8
        """
        csv = _write_csv(tmp_path, "s.csv", content)
        soil = Soil.from_csv(str(csv), apply="uniform")
        assert soil.n_layers == 1

    def test_missing_required_column_raises(self, tmp_path):
        content = """\
            layer,depth_top_cm,depth_bottom_cm,moisture_pct,bulk_density,pen_resistance_mpa
            0,0.0,20.0,28.0,1.25,0.8
        """
        # Missing n_kg_ha
        csv = _write_csv(tmp_path, "s.csv", content)
        with pytest.raises(ValueError, match="missing columns"):
            Soil.from_csv(str(csv), apply="uniform")

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Soil.from_csv(str(tmp_path / "nope.csv"))

    def test_invalid_apply_value_raises(self, tmp_path):
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)
        with pytest.raises(ValueError, match="apply must be"):
            Soil.from_csv(str(csv), apply="random")


# ===========================================================================
# Soil.from_csv — spatial
# ===========================================================================

class TestSoilFromCsvSpatial:
    def _spatial_csv(self, tmp_path: Path) -> Path:
        content = """\
            row,col,layer,depth_top_cm,depth_bottom_cm,moisture_pct,n_kg_ha,bulk_density,pen_resistance_mpa
            0,0,0,0.0,20.0,28.0,40.0,1.25,0.8
            0,0,1,20.0,40.0,22.0,15.0,1.35,1.2
            0,1,0,0.0,20.0,30.0,38.0,1.20,0.7
            0,1,1,20.0,40.0,24.0,14.0,1.30,1.1
            1,0,0,0.0,20.0,26.0,42.0,1.28,0.9
            1,0,1,20.0,40.0,20.0,16.0,1.38,1.3
            1,1,0,0.0,20.0,29.0,39.0,1.22,0.75
            1,1,1,20.0,40.0,23.0,13.0,1.32,1.15
        """
        return _write_csv(tmp_path, "spatial.csv", content)

    def test_loads_spatial(self, tmp_path):
        csv = self._spatial_csv(tmp_path)
        soil = Soil.from_csv(str(csv), apply="spatial", rows=2, cols=2)
        assert isinstance(soil, Soil)

    def test_spatial_build_grid_shape(self, tmp_path):
        csv = self._spatial_csv(tmp_path)
        soil = Soil.from_csv(str(csv), apply="spatial", rows=2, cols=2)
        grid = soil.build_grid(rows=2, cols=2)
        assert len(grid) == 2
        assert len(grid[0]) == 2
        assert len(grid[0][0]) == 2   # 2 layers per cell

    def test_spatial_cell_values(self, tmp_path):
        csv = self._spatial_csv(tmp_path)
        soil = Soil.from_csv(str(csv), apply="spatial", rows=2, cols=2)
        grid = soil.build_grid(rows=2, cols=2)
        # cell (0,1), layer 0 has moisture_pct=30.0
        assert grid[0][1][0].moisture_pct == pytest.approx(30.0)

    def test_spatial_different_cells_different_values(self, tmp_path):
        csv = self._spatial_csv(tmp_path)
        soil = Soil.from_csv(str(csv), apply="spatial", rows=2, cols=2)
        grid = soil.build_grid(rows=2, cols=2)
        assert grid[0][0][0].moisture_pct != grid[0][1][0].moisture_pct

    def test_spatial_missing_row_col_columns_raises(self, tmp_path):
        csv = _write_csv(tmp_path, "s.csv", _SOIL_CSV)  # no row/col columns
        with pytest.raises(ValueError, match="'row'"):
            Soil.from_csv(str(csv), apply="spatial", rows=5, cols=5)

    def test_spatial_missing_dimensions_raises(self, tmp_path):
        content = """\
            row,col,layer,depth_top_cm,depth_bottom_cm,moisture_pct,n_kg_ha,bulk_density,pen_resistance_mpa
            0,0,0,0.0,20.0,28.0,40.0,1.25,0.8
        """
        csv = _write_csv(tmp_path, "s.csv", content)
        with pytest.raises(ValueError, match="rows > 0"):
            Soil.from_csv(str(csv), apply="spatial")  # rows/cols default to 0

    def test_spatial_missing_cell_gets_default_layer(self, tmp_path):
        """Cells not present in the CSV should get a fallback default layer."""
        content = """\
            row,col,layer,depth_top_cm,depth_bottom_cm,moisture_pct,n_kg_ha,bulk_density,pen_resistance_mpa
            0,0,0,0.0,20.0,28.0,40.0,1.25,0.8
        """
        csv = _write_csv(tmp_path, "s.csv", content)
        # 2×2 grid but only (0,0) is in CSV; other cells should get defaults
        soil = Soil.from_csv(str(csv), apply="spatial", rows=2, cols=2)
        grid = soil.build_grid(rows=2, cols=2)
        assert len(grid[1][1]) == 1   # default: 1 layer
