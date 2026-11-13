"""Faz 9 Unit & Physical Invariant Tests: Carbon Fractionation, Stock & Spatial Kriging."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from st_ds.carbon import (
    PROVISIONAL_TAXONOMY_VERSION,
    LayerStockInput,
    classify_soil_provisional,
    compute_layer_stock,
    compute_profile_stock,
    partition_carbon,
    propagate_stock_uncertainty,
)
from st_ds.domain.errors import SchemaError, UnitError
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.spatial import (
    fit_variogram,
    kriging_to_geojson,
    ordinary_kriging,
)


class TestCarbonFractions:
    """Carbon fractionation, unit normalization, closure checks, and no zero-clipping."""

    def test_unit_normalization_g_per_kg_vs_pct(self) -> None:
        res_pct = partition_carbon(soc=1.2, sic=0.8, tc=2.0, unit="pct")
        res_g_kg = partition_carbon(soc=12.0, sic=8.0, tc=20.0, unit="g/kg")

        assert res_pct.soc_pct == pytest.approx(1.2, rel=1e-9)
        assert res_g_kg.soc_pct == pytest.approx(1.2, rel=1e-9)
        assert res_g_kg.sic_pct == pytest.approx(0.8, rel=1e-9)
        assert res_g_kg.tc_pct == pytest.approx(2.0, rel=1e-9)
        assert res_g_kg.closure_residual_pct == pytest.approx(0.0, abs=1e-12)
        assert res_g_kg.quality_status == QualityStatus.OK
        assert res_g_kg.quality_flags == []

    def test_closure_residual_flags_exceeding_tolerance(self) -> None:
        res = partition_carbon(
            soc=1.5,
            sic=0.5,
            tc=2.45,
            unit="pct",
            closure_tolerance_pct=0.2,
        )
        assert res.closure_residual_pct == pytest.approx(0.45, abs=1e-12)
        assert QualityFlag.CLOSURE_RESIDUAL_HIGH in res.quality_flags
        assert res.quality_status == QualityStatus.WARNING

    def test_derive_soc_from_difference_forbids_negative_zero_clipping(self) -> None:
        valid = partition_carbon(
            tc=2.3,
            sic=0.8,
            unit="pct",
            derive_soc_from_difference=True,
        )
        assert valid.soc_is_derived is True
        assert valid.soc_pct == pytest.approx(1.5, abs=1e-12)
        assert valid.closure_residual_pct == pytest.approx(0.0, abs=1e-12)

        with pytest.raises(SchemaError, match="Negative derived SOC"):
            partition_carbon(
                tc=1.0,
                sic=1.2,
                unit="pct",
                derive_soc_from_difference=True,
            )

    def test_rejects_non_dry_matter_basis_and_missing_unit(self) -> None:
        with pytest.raises(SchemaError, match="dry-matter"):
            partition_carbon(soc=1.5, dry_matter_basis=False)

        with pytest.raises(UnitError):
            partition_carbon(soc=1.5, unit="")

    def test_provisional_taxonomy_labeling(self) -> None:
        c1 = classify_soil_provisional(soc=15.0, sic=0.2)
        c2 = classify_soil_provisional(soc=1.0, sic=4.0)
        c3 = classify_soil_provisional(soc=3.5, sic=0.5)
        c4 = classify_soil_provisional(soc=8.0, sic=5.0, unit="g/kg")

        assert c1.taxonomy_version == PROVISIONAL_TAXONOMY_VERSION
        assert c1.is_provisional is True
        assert c1.label == "organic_rich"
        assert c2.label == "calcareous_mineral"
        assert c3.label == "humic_mineral"
        assert c4.label == "low_carbon_mineral"


class TestCarbonStockAndPropagation:
    """Soil carbon stock calculation, DEF-005 invariance, and covariance propagation."""

    def test_def005_exact_stock_equivalence_pct_and_g_per_kg(self) -> None:
        res_pct = compute_layer_stock(
            carbon_value=1.2,
            carbon_unit="pct",
            bulk_density=1.30,
            bulk_density_unit="g/cm3",
            depth_top=0.0,
            depth_bottom=30.0,
            depth_unit="cm",
            coarse_fraction=0.10,
            coarse_unit="fraction",
        )
        res_g_kg = compute_layer_stock(
            carbon_value=12.0,
            carbon_unit="g/kg",
            bulk_density=1300.0,
            bulk_density_unit="kg/m3",
            depth_top=0.0,
            depth_bottom=0.30,
            depth_unit="m",
            coarse_fraction=10.0,
            coarse_unit="%",
        )

        assert res_pct.stock_mg_ha == pytest.approx(42.12, rel=1e-9, abs=1e-12)
        assert res_g_kg.stock_mg_ha == pytest.approx(42.12, rel=1e-9, abs=1e-12)
        assert res_pct.quality_status == QualityStatus.OK
        assert res_g_kg.quality_status == QualityStatus.OK

    def test_missing_bulk_density_returns_none_never_defaults_to_135(self) -> None:
        res = compute_layer_stock(
            carbon_value=1.5,
            carbon_unit="pct",
            bulk_density=None,
            depth_top=0.0,
            depth_bottom=30.0,
            coarse_fraction=0.0,
        )
        assert res.stock_mg_ha is None
        assert res.stock_std_mg_ha is None
        assert res.quality_status == QualityStatus.INSUFFICIENT_DATA
        assert QualityFlag.MISSING_BULK_DENSITY in res.quality_flags

    def test_profile_stock_aggregation_and_depth_overlap_rejection(self) -> None:
        profile = compute_profile_stock(
            [
                LayerStockInput(
                    carbon_value=2.0,
                    carbon_unit="pct",
                    bulk_density=1.2,
                    depth_top=0.0,
                    depth_bottom=10.0,
                    coarse_fraction=0.0,
                    carbon_std=0.2,
                    label="0-10cm",
                ),
                LayerStockInput(
                    carbon_value=10.0,
                    carbon_unit="g/kg",
                    bulk_density=1.4,
                    depth_top=10.0,
                    depth_bottom=30.0,
                    coarse_fraction=0.1,
                    bulk_density_std=0.1,
                    label="10-30cm",
                ),
            ],
            require_contiguous=True,
        )
        # Layer 1: 2.0 * 1.2 * 10 * 1.0 = 24.0; std = 0.2 * 12 = 2.4
        # Layer 2: 1.0 * 1.4 * 20 * 0.9 = 25.2; std = 0.1 * (1.0 * 20 * 0.9) = 1.8
        assert profile.total_stock_mg_ha == pytest.approx(49.2, rel=1e-9)
        assert profile.total_depth_cm == pytest.approx(30.0, rel=1e-9)
        assert profile.total_stock_std_mg_ha == pytest.approx(math.hypot(2.4, 1.8), rel=1e-9)
        assert profile.dominant_uncertainty_source == "carbon_pct"

        # Missing bulk density in one layer invalidates total profile stock
        incomplete = compute_profile_stock(
            [
                LayerStockInput(carbon_value=2.0, bulk_density=1.2, depth_top=0.0, depth_bottom=10.0),
                LayerStockInput(carbon_value=1.0, bulk_density=None, depth_top=10.0, depth_bottom=30.0),
            ]
        )
        assert incomplete.total_stock_mg_ha is None
        assert incomplete.quality_status == QualityStatus.INSUFFICIENT_DATA
        assert QualityFlag.MISSING_BULK_DENSITY in incomplete.quality_flags

        # Overlapping depths must fail closed
        with pytest.raises(SchemaError, match="Overlapping depth"):
            compute_profile_stock(
                [
                    LayerStockInput(carbon_value=2.0, bulk_density=1.2, depth_top=0.0, depth_bottom=15.0),
                    LayerStockInput(carbon_value=1.0, bulk_density=1.3, depth_top=10.0, depth_bottom=30.0),
                ]
            )

        # Non-contiguous gap when require_contiguous=True
        with pytest.raises(SchemaError, match="Non-contiguous"):
            compute_profile_stock(
                [
                    LayerStockInput(carbon_value=2.0, bulk_density=1.2, depth_top=0.0, depth_bottom=10.0),
                    LayerStockInput(carbon_value=1.0, bulk_density=1.3, depth_top=15.0, depth_bottom=30.0),
                ],
                require_contiguous=True,
            )

    def test_covariance_uncertainty_propagation_correlation_and_dominant_source(self) -> None:
        c_pct, rho_b, d_cm, fc = 1.2, 1.30, 30.0, 0.10
        sig_c, sig_rho = 0.05, 0.15

        diag_res = propagate_stock_uncertainty(
            carbon_pct=c_pct,
            bulk_density_g_cm3=rho_b,
            depth_cm=d_cm,
            coarse_fraction=fc,
            carbon_std_pct=sig_c,
            bulk_density_std_g_cm3=sig_rho,
        )
        g_c = rho_b * d_cm * (1.0 - fc)       # 35.1
        g_rho = c_pct * d_cm * (1.0 - fc)     # 32.4
        expected_diag_var = (g_c * sig_c) ** 2 + (g_rho * sig_rho) ** 2
        assert diag_res.stock_variance_mg_ha2 == pytest.approx(expected_diag_var, rel=1e-9)
        assert diag_res.dominant_uncertainty_source == "bulk_density"

        cov_pos = np.diag([sig_c**2, sig_rho**2, 0.0, 0.0])
        cov_pos[0, 1] = cov_pos[1, 0] = 0.6 * sig_c * sig_rho
        pos_res = propagate_stock_uncertainty(
            carbon_pct=c_pct,
            bulk_density_g_cm3=rho_b,
            depth_cm=d_cm,
            coarse_fraction=fc,
            cov_matrix=cov_pos,
        )

        cov_neg = np.diag([sig_c**2, sig_rho**2, 0.0, 0.0])
        cov_neg[0, 1] = cov_neg[1, 0] = -0.6 * sig_c * sig_rho
        neg_res = propagate_stock_uncertainty(
            carbon_pct=c_pct,
            bulk_density_g_cm3=rho_b,
            depth_cm=d_cm,
            coarse_fraction=fc,
            cov_matrix=cov_neg,
        )

        cross = 2.0 * g_c * g_rho * (0.6 * sig_c * sig_rho)
        assert pos_res.stock_variance_mg_ha2 == pytest.approx(expected_diag_var + cross, rel=1e-9)
        assert neg_res.stock_variance_mg_ha2 == pytest.approx(expected_diag_var - cross, rel=1e-9)
        assert neg_res.stock_std_mg_ha < diag_res.stock_std_mg_ha < pos_res.stock_std_mg_ha


class TestSpatialVariogramAndKriging:
    """CRS checks, variogram fitting, Ordinary Kriging guarantees, and GeoJSON output."""

    @staticmethod
    def _synthetic_planar_field() -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(0.0, 100.0, 4)
        ys = np.linspace(0.0, 100.0, 4)
        gx, gy = np.meshgrid(xs, ys)
        coords = np.column_stack((gx.ravel(), gy.ravel()))
        vals = (
            2.0
            + 0.01 * coords[:, 0]
            - 0.005 * coords[:, 1]
            + 0.5 * np.exp(-(((coords[:, 0] - 50.0) ** 2 + (coords[:, 1] - 50.0) ** 2) / 900.0))
        )
        return coords, vals

    def test_ordinary_kriging_unbiasedness_exact_interpolation_and_extrapolation(self) -> None:
        sample_coords, sample_vals = self._synthetic_planar_field()
        v_model = fit_variogram(
            sample_coords,
            sample_vals,
            crs_epsg="LOCAL:METERS",
            coord_order="xy_m",
            model="spherical",
            n_lags=6,
            min_points=8,
        )
        assert v_model.partial_sill > 0.0
        assert v_model.range_m > 0.0
        assert v_model.nugget >= 0.0

        # Query: 3 exact sample points, 1 interior midpoint, 1 distant extrapolation point
        query_coords = np.array(
            [
                sample_coords[0],
                sample_coords[5],
                sample_coords[10],
                [50.0, 50.0],
                [500.0, 500.0],
            ],
            dtype=np.float64,
        )
        pred = ordinary_kriging(
            sample_coords,
            sample_vals,
            query_coords,
            variogram=v_model,
            crs_epsg="LOCAL:METERS",
            coord_order="xy_m",
        )

        # Unbiasedness constraint: sum(lambda) == 1
        assert pred.weights_sum_max_error < 1e-8

        # Exact interpolation at observed sites
        assert pred.predicted_values[0] == pytest.approx(sample_vals[0], abs=1e-4)
        assert pred.predicted_values[1] == pytest.approx(sample_vals[5], abs=1e-4)
        assert pred.predicted_values[2] == pytest.approx(sample_vals[10], abs=1e-4)
        assert np.all(pred.kriging_variance[:3] < 1e-4 * v_model.sill + 1e-8)

        # Extrapolation detection & sill-level variance far outside support
        assert list(pred.is_extrapolated_mask) == [False, False, False, False, True]
        assert QualityFlag.OUT_OF_CALIBRATION_DOMAIN in pred.quality_flags
        assert pred.quality_status == QualityStatus.WARNING
        assert pred.kriging_variance[4] > pred.kriging_variance[3]
        assert pred.kriging_variance[4] >= v_model.sill * 0.95

        # Valid JSON-serializable GeoJSON FeatureCollection
        geojson = kriging_to_geojson(pred)
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 5
        assert geojson["features"][4]["properties"]["is_extrapolated"] is True
        json.dumps(geojson)

    def test_wgs84_lon_lat_projection_support(self) -> None:
        # 12 points over ~1 km x 1 km in Ankara region (lon_lat order)
        rng = np.random.default_rng(42)
        lons = 32.85 + rng.uniform(-0.005, 0.005, size=12)
        lats = 39.92 + rng.uniform(-0.004, 0.004, size=12)
        coords = np.column_stack((lons, lats))
        vals = 1.8 + 40.0 * (lons - 32.85) - 25.0 * (lats - 39.92)

        pred = ordinary_kriging(
            coords,
            vals,
            [[32.85, 39.92], [32.95, 40.00]],
            crs_epsg="EPSG:4326",
            coord_order="lon_lat",
            model="exponential",
        )
        assert pred.weights_sum_max_error < 1e-8
        assert pred.is_extrapolated_mask[0] == False  # noqa: E712
        assert pred.is_extrapolated_mask[1] == True  # noqa: E712
        assert pred.variogram is not None
        # Local tangent-plane distances should be in hundreds of meters, not tiny degrees
        assert pred.variogram.range_m > 10.0

    def test_rejects_small_collapsed_duplicate_or_invalid_crs(self) -> None:
        # Too few points (< 8)
        with pytest.raises(SchemaError, match="Insufficient spatial sample count"):
            fit_variogram(
                np.column_stack((np.arange(5, dtype=float), np.zeros(5))),
                np.ones(5),
                crs_epsg="LOCAL:METERS",
                coord_order="xy_m",
                min_points=8,
            )

        # All coordinates collapsed to one location
        collapsed = np.tile([32.85, 39.92], (10, 1))
        with pytest.raises(SchemaError, match="collapse"):
            fit_variogram(collapsed, np.linspace(1.0, 2.0, 10))

        # Invalid latitude on EPSG:4326
        bad_lat = np.column_stack((np.linspace(32.8, 32.9, 10), np.linspace(85.0, 120.0, 10)))
        with pytest.raises(SchemaError, match="Latitude"):
            fit_variogram(bad_lat, np.ones(10), crs_epsg="EPSG:4326", coord_order="lon_lat")

        # Incompatible EPSG:4326 + xy_m coord order
        with pytest.raises(SchemaError, match="coord_order='lon_lat'"):
            fit_variogram(
                np.column_stack((np.arange(10, dtype=float), np.arange(10, dtype=float))),
                np.arange(10, dtype=float),
                crs_epsg="EPSG:4326",
                coord_order="xy_m",
            )

        # Duplicate sample locations rejected cleanly in kriging
        coords, vals = self._synthetic_planar_field()
        dup_coords = np.vstack((coords[:9], coords[0]))
        dup_vals = np.append(vals[:9], vals[0])
        with pytest.raises(SchemaError, match="Duplicate"):
            ordinary_kriging(
                dup_coords,
                dup_vals,
                [[20.0, 20.0]],
                crs_epsg="LOCAL:METERS",
                coord_order="xy_m",
            )
