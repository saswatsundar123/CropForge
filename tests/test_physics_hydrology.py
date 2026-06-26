"""
tests/test_physics_hydrology.py
================================
Pure-math unit tests for cropforge/physics/hydrology.py.

All tests operate on plain Python dicts (layer descriptors), not on
CropForge simulation state objects. This guarantees complete isolation
of the physics from the engine.

PRD v0.3.0 Section 5.6 required tests:
  ✓ Moisture decreases daily by correct ETc amount (ET0 × Kc / active_layers)
  ✓ Moisture increases by rainfall_mm on correct days
  ✓ Excess above field_capacity drains to next layer at drainage_coefficient rate
  ✓ Moisture does not drop below 0.0 (floor) or exceed saturation_pct (ceiling)
  ✓ Ks = 1.0 when moisture = field_capacity (no stress)
  ✓ Ks = 0.0 when moisture = wilting_point (full stress)
  ✓ Water cascades correctly through multiple layers
  ✓ Deep percolation discards water past the bottom layer
  ✓ Root zone extraction bounded by root_depth_cm
  ✓ No extraction from layers below root_depth_cm

Author : Saswat Sundar Rath, ICAR-IARI Jharkhand
Licence: MIT
"""

from __future__ import annotations

import pytest

from cropforge.physics.hydrology import (
    calculate_tipping_bucket,
    calculate_water_extraction,
    _mm_to_pct,
    DEFAULT_DRAINAGE_COEFFICIENT,
    DEFAULT_FIELD_CAPACITY_PCT,
    DEFAULT_WILTING_POINT_PCT,
    DEFAULT_SATURATION_PCT,
)


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

def _make_layer(
    moisture_pct: float = 20.0,
    field_capacity_pct: float = 30.0,
    wilting_point_pct: float = 10.0,
    saturation_pct: float = 45.0,
    depth_top_cm: float = 0.0,
    depth_bottom_cm: float = 20.0,
    drainage_coefficient: float = 0.5,
) -> dict:
    """Return a single layer descriptor dict."""
    return {
        "moisture_pct":       moisture_pct,
        "field_capacity_pct": field_capacity_pct,
        "wilting_point_pct":  wilting_point_pct,
        "saturation_pct":     saturation_pct,
        "depth_top_cm":       depth_top_cm,
        "depth_bottom_cm":    depth_bottom_cm,
        "drainage_coefficient": drainage_coefficient,
    }


def _two_layer_profile(
    top_moisture: float = 20.0,
    bot_moisture: float = 15.0,
    fc: float = 30.0,
    wp: float = 10.0,
    sat: float = 45.0,
    coef: float = 0.5,
) -> list:
    """Return a 2-layer profile (0–20 cm, 20–40 cm)."""
    return [
        _make_layer(moisture_pct=top_moisture, field_capacity_pct=fc,
                    wilting_point_pct=wp, saturation_pct=sat,
                    depth_top_cm=0.0, depth_bottom_cm=20.0,
                    drainage_coefficient=coef),
        _make_layer(moisture_pct=bot_moisture, field_capacity_pct=fc,
                    wilting_point_pct=wp, saturation_pct=sat,
                    depth_top_cm=20.0, depth_bottom_cm=40.0,
                    drainage_coefficient=coef),
    ]


def _three_layer_profile(
    moistures: tuple = (20.0, 15.0, 10.0),
    fc: float = 30.0,
    wp: float = 10.0,
    sat: float = 45.0,
    coef: float = 0.5,
) -> list:
    return [
        _make_layer(moisture_pct=moistures[0], field_capacity_pct=fc,
                    wilting_point_pct=wp, saturation_pct=sat,
                    depth_top_cm=0.0,  depth_bottom_cm=20.0,
                    drainage_coefficient=coef),
        _make_layer(moisture_pct=moistures[1], field_capacity_pct=fc,
                    wilting_point_pct=wp, saturation_pct=sat,
                    depth_top_cm=20.0, depth_bottom_cm=40.0,
                    drainage_coefficient=coef),
        _make_layer(moisture_pct=moistures[2], field_capacity_pct=fc,
                    wilting_point_pct=wp, saturation_pct=sat,
                    depth_top_cm=40.0, depth_bottom_cm=60.0,
                    drainage_coefficient=coef),
    ]


# ---------------------------------------------------------------------------
# Unit tests: _mm_to_pct helper
# ---------------------------------------------------------------------------

class TestMmToPct:
    def test_standard_conversion(self):
        # 10 mm over 10 cm depth = 10 mm / (10 * 10) * 100 = 10 %
        assert _mm_to_pct(10.0, 10.0) == pytest.approx(10.0)

    def test_20cm_layer_10mm_rain(self):
        # 10 mm over 20 cm = 10 / (20 * 10) * 100 = 5.0 %
        assert _mm_to_pct(10.0, 20.0) == pytest.approx(5.0)

    def test_zero_mm(self):
        assert _mm_to_pct(0.0, 20.0) == pytest.approx(0.0)

    def test_zero_depth_returns_zero(self):
        assert _mm_to_pct(10.0, 0.0) == pytest.approx(0.0)

    def test_30mm_rain_20cm_layer(self):
        # 30 / (20 * 10) * 100 = 15 %
        assert _mm_to_pct(30.0, 20.0) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Unit tests: calculate_tipping_bucket
# ---------------------------------------------------------------------------

class TestTippingBucketInflow:
    """Rainfall and irrigation add water to layer 0 correctly."""

    def test_rainfall_adds_correct_pct_to_layer0(self):
        """10 mm rain on a 20 cm layer should add 5.0 % moisture."""
        layers = [_make_layer(moisture_pct=20.0, depth_top_cm=0.0, depth_bottom_cm=20.0)]
        result = calculate_tipping_bucket(layers, precipitation_mm=10.0, irrigation_mm=0.0)
        assert result[0]["moisture_pct"] == pytest.approx(25.0)

    def test_irrigation_adds_correct_pct_to_layer0(self):
        """
        30 mm irrigation on a 20 cm layer = +15% volumetric → 35%.
        But FC=30% (default), excess=5%, drainage_coefficient=0.5 → drains 2.5%.
        Final moisture = 35 - 2.5 = 32.5%.
        This is physically correct: tipping-bucket drainage fires in the same call.
        """
        layers = [_make_layer(moisture_pct=20.0)]
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=30.0)
        # 20 + 15(inflow) = 35%; excess above FC(30%) = 5%; drained = 5*0.5 = 2.5% → 32.5%
        assert result[0]["moisture_pct"] == pytest.approx(32.5)

    def test_combined_rain_and_irrigation(self):
        """10 mm rain + 20 mm irrigation = 30 mm total → 15 % increase on 20 cm layer."""
        layers = [_make_layer(moisture_pct=15.0)]
        result = calculate_tipping_bucket(layers, precipitation_mm=10.0, irrigation_mm=20.0)
        assert result[0]["moisture_pct"] == pytest.approx(30.0)

    def test_inflow_only_to_layer0_not_layer1(self):
        """Rainfall is added to layer 0 only; layer 1 is not directly affected."""
        layers = _two_layer_profile(top_moisture=20.0, bot_moisture=15.0)
        # 5 mm rain → 5 / (20*10) * 100 = 2.5 % increase in layer 0
        result = calculate_tipping_bucket(layers, precipitation_mm=5.0, irrigation_mm=0.0)
        # Layer 0 gets rain
        assert result[0]["moisture_pct"] > 20.0
        # Layer 1 only changes if layer 0 drains, which it doesn't here (below FC=30%)
        # After inflow, layer 0 = 22.5 % < FC=30 % → no drainage
        assert result[1]["moisture_pct"] == pytest.approx(15.0)

    def test_no_inflow_no_change_below_fc(self):
        """With 0 mm rain and moisture < FC, nothing changes."""
        layers = [_make_layer(moisture_pct=20.0, field_capacity_pct=30.0)]
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)
        assert result[0]["moisture_pct"] == pytest.approx(20.0)

    def test_empty_layers_returns_empty(self):
        result = calculate_tipping_bucket([], precipitation_mm=10.0, irrigation_mm=0.0)
        assert result == []


class TestTippingBucketDrainage:
    """Excess above FC drains at drainage_coefficient rate."""

    def test_excess_drains_at_correct_rate(self):
        """Layer at saturation (45%) above FC (30%) → excess=15%, drains 50% → 7.5%."""
        layers = [_make_layer(
            moisture_pct=45.0,         # at saturation
            field_capacity_pct=30.0,
            saturation_pct=45.0,
            drainage_coefficient=0.5,
            depth_top_cm=0.0,
            depth_bottom_cm=20.0,
        )]
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)
        # Excess = 45 - 30 = 15%; drained = 15 * 0.5 = 7.5%; remaining = 37.5%
        assert result[0]["moisture_pct"] == pytest.approx(37.5)

    def test_drainage_cascades_to_layer1(self):
        """
        Layer 0 at 45 % (sat). FC=30%. coef=0.5.
        Drainage from layer 0 = 7.5 % of 20 cm layer = 15 mm.
        15 mm added to layer 1 (depth 20 cm) = 15/200*100 = 7.5 % increase.
        Layer 1 starts at 15 %, becomes 22.5 % (still below FC=30%).
        """
        layers = _two_layer_profile(
            top_moisture=45.0, bot_moisture=15.0,
            fc=30.0, sat=45.0, coef=0.5,
        )
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)

        # Layer 0: 45 - 7.5 = 37.5 %
        assert result[0]["moisture_pct"] == pytest.approx(37.5)
        # Layer 1: 15 + 7.5 = 22.5 %
        assert result[1]["moisture_pct"] == pytest.approx(22.5)

    def test_drainage_cascade_three_layers(self):
        """Water cascades through all three layers correctly."""
        layers = _three_layer_profile(moistures=(45.0, 15.0, 5.0), fc=30.0, sat=45.0, coef=1.0)
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)

        # Layer 0: excess = 15 %, coef=1.0 → all drains. moisture = 30.0 (FC)
        assert result[0]["moisture_pct"] == pytest.approx(30.0)

        # Layer 1: receives 15 % × 20 cm = 30 mm → 30/200*100=15% increase.
        # layer 1 was 15% → becomes 30% (at FC). Still no excess (30 - 30 = 0).
        assert result[1]["moisture_pct"] == pytest.approx(30.0)

        # Layer 2: layer 1 had no excess (30 = FC) → layer 2 unchanged = 5.0 %
        assert result[2]["moisture_pct"] == pytest.approx(5.0)

    def test_deep_percolation_discards_water_from_bottom_layer(self):
        """Excess draining from the bottom layer is lost — not returned or stored."""
        layers = [_make_layer(
            moisture_pct=45.0, field_capacity_pct=30.0,
            saturation_pct=45.0, drainage_coefficient=0.5,
        )]
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)
        # Moisture reduced by drainage; nothing returned
        assert result[0]["moisture_pct"] == pytest.approx(37.5)
        assert result[0]["drainage_mm_today"] == pytest.approx(15.0)

    def test_drainage_mm_today_recorded(self):
        """drainage_mm_today key must be present in every returned layer."""
        layers = _two_layer_profile(top_moisture=45.0, bot_moisture=15.0, fc=30.0, sat=45.0)
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)
        for layer in result:
            assert "drainage_mm_today" in layer

    def test_drainage_zero_when_below_fc(self):
        """If moisture < FC, drainage_mm_today must be 0.0."""
        layers = [_make_layer(moisture_pct=20.0, field_capacity_pct=30.0)]
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)
        assert result[0]["drainage_mm_today"] == pytest.approx(0.0)

    def test_drainage_coefficient_zero_means_no_drainage(self):
        """drainage_coefficient=0.0: no water drains regardless of excess."""
        layers = [_make_layer(moisture_pct=45.0, field_capacity_pct=30.0,
                              saturation_pct=45.0, drainage_coefficient=0.0)]
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)
        assert result[0]["moisture_pct"] == pytest.approx(45.0)

    def test_moisture_does_not_exceed_saturation(self):
        """Even with very heavy rain, moisture cannot exceed saturation_pct."""
        layers = [_make_layer(moisture_pct=40.0, saturation_pct=45.0)]
        # 200 mm rain on a 20 cm layer = 100 % increase → would go to 140 %
        result = calculate_tipping_bucket(layers, precipitation_mm=200.0, irrigation_mm=0.0)
        assert result[0]["moisture_pct"] <= 45.0

    def test_moisture_floor_zero(self):
        """Moisture cannot go below 0.0 (prevents negative water content)."""
        layers = [_make_layer(moisture_pct=0.0)]
        result = calculate_tipping_bucket(layers, precipitation_mm=0.0, irrigation_mm=0.0)
        assert result[0]["moisture_pct"] >= 0.0

    def test_inputs_not_mutated(self):
        """calculate_tipping_bucket must not modify the input layer dicts."""
        layers = [_make_layer(moisture_pct=20.0)]
        original_moisture = layers[0]["moisture_pct"]
        _ = calculate_tipping_bucket(layers, precipitation_mm=10.0, irrigation_mm=0.0)
        assert layers[0]["moisture_pct"] == original_moisture


# ---------------------------------------------------------------------------
# Unit tests: calculate_water_extraction
# ---------------------------------------------------------------------------

class TestWaterExtractionKs:
    """Stress coefficient (Ks) is computed correctly from moisture."""

    def test_ks_1_at_field_capacity(self):
        """At FC, Ks must be exactly 1.0 (no stress)."""
        layers = [_make_layer(moisture_pct=30.0, field_capacity_pct=30.0,
                              wilting_point_pct=10.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=0.0  # no demand to avoid extraction
        )
        assert result["ks"] == pytest.approx(1.0)

    def test_ks_0_at_wilting_point(self):
        """At WP, Ks must be exactly 0.0 (full stress)."""
        layers = [_make_layer(moisture_pct=10.0, field_capacity_pct=30.0,
                              wilting_point_pct=10.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=0.0
        )
        assert result["ks"] == pytest.approx(0.0)

    def test_ks_midpoint(self):
        """At midpoint between WP and FC, Ks = 0.5."""
        # WP=10, FC=30 → midpoint = 20 → Ks = (20-10)/(30-10) = 0.5
        layers = [_make_layer(moisture_pct=20.0, field_capacity_pct=30.0,
                              wilting_point_pct=10.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=0.0
        )
        assert result["ks"] == pytest.approx(0.5)

    def test_ks_clamped_above_1(self):
        """Ks cannot exceed 1.0 even if moisture > FC."""
        layers = [_make_layer(moisture_pct=45.0, field_capacity_pct=30.0,
                              wilting_point_pct=10.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=0.0
        )
        assert result["ks"] <= 1.0

    def test_ks_clamped_below_0(self):
        """Ks cannot be negative even if moisture < WP (shouldn't happen normally)."""
        layers = [_make_layer(moisture_pct=0.0, field_capacity_pct=30.0,
                              wilting_point_pct=10.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=0.0
        )
        assert result["ks"] >= 0.0


class TestWaterExtractionMoisture:
    """Water is removed from root-zone layers correctly."""

    def test_etc_deducted_from_single_layer(self):
        """
        ET0=5 mm, Kc=1.0 → ETc=5 mm.
        Layer depth = 20 cm → 5 mm / (20*10) * 100 = 2.5 % reduction.
        Initial 25 % → 22.5 %.
        """
        layers = [_make_layer(moisture_pct=25.0, field_capacity_pct=30.0,
                              wilting_point_pct=10.0, saturation_pct=45.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=5.0, crop_coefficient=1.0
        )
        assert result["layers"][0]["moisture_pct"] == pytest.approx(22.5)

    def test_etc_distributed_across_two_root_layers(self):
        """
        Two layers in root zone: ETc distributed equally.
        ET0=5 mm, Kc=1.0 → ETc=5 mm → 2.5 mm per layer.
        Layer depth=20 cm: 2.5 / 200 * 100 = 1.25 % per layer.
        """
        layers = _two_layer_profile(top_moisture=25.0, bot_moisture=25.0,
                                    fc=30.0, wp=10.0)
        # Root reaches 30 cm: both layers (0-20, 20-40) are in root zone
        result = calculate_water_extraction(
            layers, root_depth_cm=30.0, et0_demand=5.0, crop_coefficient=1.0
        )
        assert result["layers"][0]["moisture_pct"] == pytest.approx(23.75)
        assert result["layers"][1]["moisture_pct"] == pytest.approx(23.75)

    def test_moisture_floors_at_zero(self):
        """Extraction cannot drive moisture below 0.0."""
        layers = [_make_layer(moisture_pct=0.5, field_capacity_pct=30.0,
                              wilting_point_pct=10.0)]
        # Very high ET demand
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=50.0
        )
        assert result["layers"][0]["moisture_pct"] >= 0.0

    def test_kc_scales_etc(self):
        """crop_coefficient=0.5 halves the extracted water."""
        layers = [_make_layer(moisture_pct=25.0, field_capacity_pct=30.0,
                              wilting_point_pct=10.0)]
        # Kc=0.5: ETc = 5 * 0.5 = 2.5 mm → 2.5 / 200 * 100 = 1.25 %
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=5.0, crop_coefficient=0.5
        )
        assert result["layers"][0]["moisture_pct"] == pytest.approx(23.75)

    def test_no_extraction_when_et0_zero(self):
        """With et0_demand=0.0, moisture must remain unchanged."""
        layers = [_make_layer(moisture_pct=25.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=0.0
        )
        assert result["layers"][0]["moisture_pct"] == pytest.approx(25.0)

    def test_inputs_not_mutated(self):
        """calculate_water_extraction must not modify the input layer dicts."""
        layers = [_make_layer(moisture_pct=25.0)]
        original = layers[0]["moisture_pct"]
        _ = calculate_water_extraction(layers, root_depth_cm=20.0, et0_demand=5.0)
        assert layers[0]["moisture_pct"] == original

    def test_empty_layers_returns_ks_1(self):
        """Empty profile: no extraction, Ks defaults to 1.0 (no stress signal)."""
        result = calculate_water_extraction([], root_depth_cm=20.0, et0_demand=5.0)
        assert result["ks"] == pytest.approx(1.0)
        assert result["layers"] == []


class TestRootZoneBoundary:
    """Water extraction is bounded by root_depth_cm."""

    def test_extraction_only_from_root_zone_layers(self):
        """
        Root depth = 15 cm: only layer 0 (0-20 cm top edge = 0 cm < 15 cm) is active.
        Layer 1 starts at 20 cm and should NOT be extracted from.
        """
        layers = _two_layer_profile(top_moisture=25.0, bot_moisture=25.0, fc=30.0, wp=10.0)
        # root_depth_cm=15: layer 0 top=0 < 15 → active. Layer 1 top=20 > 15 → inactive.
        result = calculate_water_extraction(
            layers, root_depth_cm=15.0, et0_demand=5.0
        )
        # Layer 0 should decrease
        assert result["layers"][0]["moisture_pct"] < 25.0
        # Layer 1 should be unchanged (5mm ET extracted from layer 0 only)
        assert result["layers"][1]["moisture_pct"] == pytest.approx(25.0)

    def test_extraction_from_both_layers_when_root_deep(self):
        """Root at 30 cm covers both layers; both should lose moisture."""
        layers = _two_layer_profile(top_moisture=25.0, bot_moisture=25.0, fc=30.0, wp=10.0)
        result = calculate_water_extraction(
            layers, root_depth_cm=30.0, et0_demand=5.0
        )
        assert result["layers"][0]["moisture_pct"] < 25.0
        assert result["layers"][1]["moisture_pct"] < 25.0

    def test_root_depth_zero_uses_layer0(self):
        """root_depth_cm=0.0 falls back to layer 0 as the minimum root zone."""
        layers = _two_layer_profile(top_moisture=25.0, bot_moisture=25.0, fc=30.0, wp=10.0)
        result = calculate_water_extraction(
            layers, root_depth_cm=0.0, et0_demand=5.0
        )
        # Layer 0 must have been extracted from (fallback to layer 0)
        assert result["layers"][0]["moisture_pct"] < 25.0
        # Layer 1 must be unchanged
        assert result["layers"][1]["moisture_pct"] == pytest.approx(25.0)

    def test_three_layers_root_bounded(self):
        """Root at 25 cm: layers 0 (0-20) and 1 (20-40) are active. Layer 2 is not."""
        layers = _three_layer_profile(moistures=(25.0, 25.0, 25.0), fc=30.0, wp=10.0)
        result = calculate_water_extraction(
            layers, root_depth_cm=25.0, et0_demand=5.0
        )
        assert result["layers"][0]["moisture_pct"] < 25.0
        assert result["layers"][1]["moisture_pct"] < 25.0
        # Layer 2 starts at 40 cm → above root depth 25 cm → not extracted
        assert result["layers"][2]["moisture_pct"] == pytest.approx(25.0)


class TestExtractionReturnDict:
    """Result dict keys and types are correct."""

    def test_result_keys(self):
        layers = [_make_layer()]
        result = calculate_water_extraction(layers, root_depth_cm=20.0, et0_demand=5.0)
        assert "layers" in result
        assert "ks" in result
        assert "etc_mm" in result
        assert "extracted_mm" in result

    def test_etc_mm_is_et0_times_kc(self):
        layers = [_make_layer(moisture_pct=25.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=6.0, crop_coefficient=0.8
        )
        assert result["etc_mm"] == pytest.approx(4.8)

    def test_extracted_mm_matches_moisture_change(self):
        """extracted_mm should equal the actual moisture removed in mm units."""
        layers = [_make_layer(moisture_pct=25.0, depth_top_cm=0.0, depth_bottom_cm=20.0)]
        result = calculate_water_extraction(
            layers, root_depth_cm=20.0, et0_demand=5.0, crop_coefficient=1.0
        )
        # Expected: 2.5% removed from 20 cm layer = 2.5 * 200 / 100 = 5.0 mm
        assert result["extracted_mm"] == pytest.approx(5.0, abs=0.01)


class TestTippingBucketConservation:
    """Water conservation: what goes in must go somewhere (drainage or storage)."""

    def test_water_mass_balance_single_layer(self):
        """
        For a single layer:
          moisture_in = initial + rain + irrigation
          moisture_out = final_moisture + drainage
        Mass balance: moisture_in == moisture_out (within floating point)
        """
        initial = 20.0  # %
        rain = 10.0     # mm
        irrig = 5.0     # mm
        depth_cm = 20.0

        layers = [_make_layer(moisture_pct=initial, field_capacity_pct=30.0,
                              saturation_pct=45.0, depth_top_cm=0.0,
                              depth_bottom_cm=depth_cm, drainage_coefficient=0.5)]

        result = calculate_tipping_bucket(layers, precipitation_mm=rain, irrigation_mm=irrig)

        # Convert initial + inflow to %
        inflow_pct = _mm_to_pct(rain + irrig, depth_cm)
        moisture_in = initial + inflow_pct

        final_pct = result[0]["moisture_pct"]
        drainage_pct = result[0]["drainage_mm_today"] / (depth_cm * 10.0) * 100.0

        assert (final_pct + drainage_pct) == pytest.approx(
            min(moisture_in, 45.0), abs=1e-6
        )
