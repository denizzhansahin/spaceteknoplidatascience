"""Faz 1 Unit & Contract Tests: Domain Schemas, UnitRegistry, InstrumentProfiles, and Fail-Closed Rules."""

from __future__ import annotations

import json
import numpy as np
import pytest

from st_ds.domain import (
    BUILTIN_PROFILES,
    AnalyzerProtocol,
    ArtifactMetadata,
    CalibratorProtocol,
    CovariateSet,
    DepthInterval,
    EISMeasurement,
    InstrumentProfile,
    Measurement,
    PredictionReport,
    QualityFlag,
    QualityMetadata,
    QualityStatus,
    RegressorProtocol,
    SchemaError,
    SignalQCProtocol,
    SoilReference,
    SpatialCoordinates,
    SpectralModality,
    SpectralSignalMode,
    Spectrum,
    TargetLabelSchema,
    UnitError,
    UnitRegistry,
    get_builtin_profile,
)


class TestUnitRegistry:
    """Verify canonical unit conversions and strict fail-closed validation (ADR-002)."""

    def test_carbon_concentration_conversions(self) -> None:
        assert pytest.approx(UnitRegistry.to_canonical("carbon_concentration", 1.85, "pct")) == 1.85
        assert pytest.approx(UnitRegistry.to_canonical("carbon_concentration", 18.5, "g/kg")) == 1.85
        assert pytest.approx(UnitRegistry.to_canonical("carbon_concentration", 18500.0, "mg/kg")) == 1.85
        assert pytest.approx(UnitRegistry.to_canonical("carbon_concentration", 18500.0, "ppm")) == 1.85
        assert pytest.approx(UnitRegistry.to_canonical("carbon_concentration", 0.0185, "g/g")) == 1.85

    def test_moisture_and_bulk_density_conversions(self) -> None:
        assert pytest.approx(UnitRegistry.to_canonical("moisture_volumetric", 0.24, "m3/m3")) == 24.0
        assert pytest.approx(UnitRegistry.to_canonical("moisture_volumetric", 24.0, "%")) == 24.0
        assert pytest.approx(UnitRegistry.to_canonical("bulk_density", 1320.0, "kg/m3")) == 1.32
        assert pytest.approx(UnitRegistry.to_canonical("bulk_density", 1.32, "g/cm3")) == 1.32
        assert pytest.approx(UnitRegistry.to_canonical("bulk_density", 1.32, "Mg/m3")) == 1.32

    def test_temperature_ec_depth_and_coarse_fragments_conversions(self) -> None:
        assert pytest.approx(UnitRegistry.to_canonical("temperature", 293.15, "K")) == 20.0
        assert pytest.approx(UnitRegistry.to_canonical("temperature", 68.0, "degF")) == 20.0
        assert pytest.approx(UnitRegistry.to_canonical("ec", 0.25, "S/m")) == 2.5
        assert pytest.approx(UnitRegistry.to_canonical("depth", 0.30, "m")) == 30.0
        assert pytest.approx(UnitRegistry.to_canonical("coarse_fragments", 15.0, "%")) == 0.15
        assert pytest.approx(UnitRegistry.to_canonical("wavelength", 2.335, "um")) == 2335.0

    def test_missing_or_unknown_unit_raises_unit_error(self) -> None:
        with pytest.raises(UnitError):
            UnitRegistry.to_canonical("carbon_concentration", 1.5, None)
        with pytest.raises(UnitError):
            UnitRegistry.to_canonical("carbon_concentration", 1.5, "")
        with pytest.raises(UnitError):
            UnitRegistry.to_canonical("bulk_density", 1.3, "ounces_per_bushel")

    def test_out_of_physical_bounds_raises_schema_error(self) -> None:
        with pytest.raises(SchemaError):
            UnitRegistry.to_canonical("ph", 14.5, "dimensionless")
        with pytest.raises(SchemaError):
            UnitRegistry.to_canonical("carbon_concentration", -0.5, "pct")
        with pytest.raises(SchemaError):
            UnitRegistry.to_canonical("bulk_density", 0.0, "g/cm3")
        with pytest.raises(SchemaError):
            UnitRegistry.to_canonical("coarse_fragments", 1.05, "fraction")
        with pytest.raises(SchemaError):
            UnitRegistry.to_canonical("temperature", float("nan"), "degC")


class TestInstrumentProfiles:
    """Verify multi-device spectral profiles replacing hardcoded 24-band assumptions (DEF-003)."""

    def test_builtin_profiles_cover_all_hardware_tiers(self) -> None:
        assert set(BUILTIN_PROFILES.keys()) == {
            "led_5band_v1",
            "multispectral_18ch_v1",
            "visnir_24ch_v1",
            "ftnir_1300_2550nm_v1",
            "libs_ci_hi_v1",
        }
        led = get_builtin_profile("led_5band_v1")
        assert led.n_channels == 5
        assert led.has_water_bands is True
        assert led.has_carbonate_bands is False

        ms18 = get_builtin_profile("multispectral_18ch_v1")
        assert ms18.n_channels == 18
        assert ms18.has_carbonate_bands is False

        ftnir = get_builtin_profile("ftnir_1300_2550nm_v1")
        assert ftnir.n_channels == 126
        assert ftnir.is_uniform_grid is True
        assert ftnir.has_water_bands is True
        assert ftnir.has_organic_band is True
        assert ftnir.has_clay_band is True
        assert ftnir.has_carbonate_bands is True

        libs = get_builtin_profile("libs_ci_hi_v1")
        assert libs.modality == SpectralModality.LIBS
        assert libs.has_carbonate_bands is True

    def test_non_increasing_wavelength_grid_rejected(self) -> None:
        with pytest.raises(SchemaError):
            InstrumentProfile(
                profile_id="bad_grid",
                sensor_model="Test",
                modality=SpectralModality.VIS_NIR_SWIR,
                wavelength_grid_nm=(500.0, 700.0, 650.0),
                channel_ids=("B1", "B2", "B3"),
            )


class TestDomainSchemasAndFailClosedRules:
    """Verify CovariateSet, SoilReference, Measurement, and PredictionReport contracts (DEF-005)."""

    def test_missing_bulk_density_never_defaults_to_1_35(self) -> None:
        cov = CovariateSet.from_raw(
            {
                "ph": {"value": 7.1, "unit": "dimensionless"},
                "moisture_volumetric": {"value": 0.18, "unit": "m3/m3"},
                "temperature": {"value": 21.5, "unit": "degC"},
            }
        )
        assert cov.bulk_density_g_cm3 is None
        assert cov.has_bulk_density is False
        assert cov.missing_reasons["bulk_density"] == "not_measured"
        assert pytest.approx(cov.moisture_volumetric_pct) == 18.0

        prof = get_builtin_profile("led_5band_v1")
        spec = Spectrum(
            values=np.array([0.12, 0.18, 0.25, 0.28, 0.22]),
            wavelength_grid_nm=prof.wavelength_grid_nm,
            channel_ids=prof.channel_ids,
        )
        m = Measurement(
            sample_id="SMP_001",
            timestamp_utc="2026-11-12T10:00:00Z",
            instrument_id="DEV_LED_1",
            instrument_profile_id="led_5band_v1",
            calibration_id="CAL_01",
            spectrum=spec,
            covariates=cov,
        )
        assert QualityFlag.MISSING_BULK_DENSITY in m.quality.flags
        assert QualityFlag.CARBONATE_BAND_UNAVAILABLE in m.quality.flags

    def test_bare_covariate_without_unit_rejected(self) -> None:
        with pytest.raises(UnitError):
            CovariateSet.from_raw({"ph": 7.2})  # type: ignore[arg-type]

    def test_soil_reference_closure_and_negative_derived_soc_rejection(self) -> None:
        ref = SoilReference.from_units(
            tc=25.0,
            sic=10.0,
            unit="g/kg",
            derive_soc_from_difference=True,
        )
        assert pytest.approx(ref.tc_pct) == 2.5
        assert pytest.approx(ref.sic_pct) == 1.0
        assert pytest.approx(ref.soc_pct) == 1.5
        assert ref.is_soc_derived_from_difference is True
        assert pytest.approx(ref.closure_residual_pct) == 0.0

        # Negative TC - SIC difference must NOT be silently clipped to zero (Section 22, Page 24)
        with pytest.raises(SchemaError):
            SoilReference.from_units(
                tc=0.8,
                sic=1.2,
                unit="pct",
                derive_soc_from_difference=True,
            )

    def test_eis_measurement_and_prediction_report_serialization(self) -> None:
        eis = EISMeasurement.from_units(
            frequencies=[0.1, 1.0, 10.0],
            z_real=[1.2, 0.9, 0.7],
            z_imag=[-0.3, -0.2, -0.1],
            freq_unit="kHz",
            impedance_unit="kohm",
        )
        assert eis.frequencies_hz == (100.0, 1000.0, 10000.0)
        assert eis.z_real_ohm == (1200.0, 900.0, 700.0)

        report = PredictionReport(
            identity={"sample_id": "SMP_001", "instrument_profile_id": "visnir_24ch_v1"},
            status=QualityStatus.OK,
            carbon_metrics={"soc_pct": 1.8, "sic_pct": 0.7, "tc_pct": 2.5, "unit": "pct", "dry_matter_basis": True},
            carbon_stock={"stock_mg_ha": None, "reason": "missing_bulk_density"},
            classification=None,
            quality=QualityMetadata(status=QualityStatus.OK, flags=[QualityFlag.MISSING_BULK_DENSITY]),
            xai={"local_sensitivity": {}, "hsic_dependence": {"status": "insufficient_sample_count"}},
            provenance={"model_version": "0.1.0", "artifact_hash": "sha256:abc"},
            diagnostics={"closure_residual_pct": 0.0, "ood_score": 0.12},
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["status"] == "ok"
        assert parsed["carbon_metrics"]["tc_pct"] == 2.5
        assert "missing_bulk_density" in parsed["quality"]["flags"]

    def test_target_label_schema_artifact_metadata_and_protocols(self) -> None:
        tls = TargetLabelSchema()
        assert tls.targets == ("soc_pct", "sic_pct", "tc_pct")
        assert tls.dry_matter_basis is True
        assert tls.to_dict()["carbon_unit"] == "pct"

        with pytest.raises(SchemaError):
            TargetLabelSchema(dry_matter_basis=False)

        art_meta = ArtifactMetadata(
            artifact_id="art_001",
            semantic_version="1.0.0",
            schema_version="1.0.0",
            instrument_profile_id="visnir_24ch_v1",
            calibration_id="cal_01",
            pipeline_version="pipe_v1",
            dataset_hash="deadbeef",
            dependency_versions={"numpy": "2.2.6"},
        )
        assert art_meta.to_dict()["semantic_version"] == "1.0.0"

        # Protocols are runtime-checkable contracts
        assert hasattr(RegressorProtocol, "__protocol_attrs__") or callable(RegressorProtocol)
        assert hasattr(CalibratorProtocol, "__protocol_attrs__") or callable(CalibratorProtocol)
        assert hasattr(SignalQCProtocol, "__protocol_attrs__") or callable(SignalQCProtocol)
        assert hasattr(AnalyzerProtocol, "__protocol_attrs__") or callable(AnalyzerProtocol)
