"""Faz 5 Unit & Contract Tests:
FeatureRegistry, Multi-Modality Extractors, Fail-Closed Covariate Validation,
Time-Tolerant Sensor Alignment, and Spectral-Covariate FeatureFusionPipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from st_ds.chemometrics import ChemometricPipeline, PreprocessConfig
from st_ds.domain import (
    CalibrationError,
    CovariateSet,
    EISMeasurement,
    Measurement,
    QualityFlag,
    SchemaError,
    SpectralSignalMode,
    Spectrum,
    get_builtin_profile,
)
from st_ds.features import (
    COVARIATE_CANONICAL_UNITS,
    FeatureDescriptor,
    FeatureFusionConfig,
    FeatureFusionPipeline,
    FeatureProfileSpec,
    FeatureRegistry,
    align_spectral_and_covariates,
    available_visnir_band_names,
    extract_derived_covariates,
    extract_eis_features,
    extract_visnir_band_features,
    validate_covariates,
)
from st_ds.io import SyntheticConfig, SyntheticGeneratorV2


def _make_measurement(
    *,
    sample_id: str,
    profile_id: str = "visnir_24ch_v1",
    timestamp_utc: str = "2026-11-12T10:00:00Z",
    ph: float | None = 6.8,
    moisture_volumetric_pct: float | None = 18.0,
    temperature_c: float | None = 22.0,
    ec_ds_m: float | None = 0.8,
    bulk_density_g_cm3: float | None = None,
    missing_reasons: dict[str, str] | None = None,
    eis: EISMeasurement | None = None,
    seed: int = 0,
) -> Measurement:
    rng = np.random.default_rng(seed)
    profile = get_builtin_profile(profile_id)
    wl = np.asarray(profile.wavelength_grid_nm, dtype=float)
    reflectance = np.clip(
        0.35 + 0.12 * np.sin((wl - 500.0) / 300.0) + rng.normal(0.0, 0.01, wl.size),
        0.05,
        0.9,
    )
    spec = Spectrum(
        values=reflectance,
        wavelength_grid_nm=profile.wavelength_grid_nm,
        channel_ids=profile.channel_ids,
        mode=SpectralSignalMode.REFLECTANCE,
    )
    cov = CovariateSet(
        ph=ph,
        moisture_volumetric_pct=moisture_volumetric_pct,
        temperature_c=temperature_c,
        ec_ds_m=ec_ds_m,
        bulk_density_g_cm3=bulk_density_g_cm3,
        eis=eis,
        missing_reasons=dict(missing_reasons or {}),
    )
    return Measurement(
        sample_id=sample_id,
        timestamp_utc=timestamp_utc,
        instrument_id="inst_alpha",
        instrument_profile_id=profile_id,
        calibration_id="cal_v1",
        spectrum=spec,
        covariates=cov,
        site_id="site_a",
    )


class TestFeatureRegistry:
    """Verify dynamic named feature registry across heterogeneous sensor profiles."""

    def test_multi_device_profile_registration_and_lookup(self) -> None:
        registry = FeatureRegistry()

        prof_5 = get_builtin_profile("led_5band_v1")
        prof_24 = get_builtin_profile("visnir_24ch_v1")
        prof_ft = get_builtin_profile("ftnir_1300_2550nm_v1")

        bands_5 = available_visnir_band_names(prof_5)
        bands_24 = available_visnir_band_names(prof_24)

        spec_5 = registry.build_profile_from_instrument(
            prof_5,
            visnir_band_names=bands_5,
            derived_names=("ec_25c_ds_m",),
        )
        spec_24 = registry.build_profile_from_instrument(
            prof_24,
            visnir_band_names=bands_24,
            eis_frequencies_hz=(100.0, 1000.0),
            eis_protocol_id="eis_2freq_v1",
        )
        spec_ft = registry.build_profile_from_instrument(prof_ft)

        # spectral_feature_names covers raw spectral channels + visnir_band summaries + libs_peak
        assert len(spec_5.spectral_feature_names) == prof_5.n_channels + len(bands_5)
        assert len(spec_24.spectral_feature_names) == prof_24.n_channels + len(bands_24)
        assert len(spec_ft.spectral_feature_names) == prof_ft.n_channels

        retrieved = registry.get_profile(spec_24.profile_id)
        assert retrieved.schema_hash == spec_24.schema_hash
        assert retrieved.index_of("ph") > retrieved.index_of(spec_24.spectral_feature_names[-1])
        assert retrieved.index_of("eis_mag_1000hz") > retrieved.index_of("ph")

    def test_duplicate_feature_names_and_missing_profile_raise_schema_error(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(SchemaError, match="Unknown feature profile"):
            registry.get_profile("nonexistent_profile")

        dup_desc = FeatureDescriptor(name="ph", group="covariate", unit="dimensionless")
        with pytest.raises(SchemaError, match="feature names must be unique"):
            FeatureProfileSpec(
                profile_id="bad_dup",
                instrument_profile_id="led_5band_v1",
                descriptors=(dup_desc, dup_desc),
            )


class TestModalityFeatureExtractors:
    """Verify Vis-NIR chemical bands, EIS impedance features, and derived covariates."""

    def test_visnir_band_availability_and_extraction(self) -> None:
        led5 = get_builtin_profile("led_5band_v1")
        bands_5 = available_visnir_band_names(led5)
        assert "band_carbonate_2330nm" not in bands_5
        assert "band_carbonate_2520nm" not in bands_5

        prof24 = get_builtin_profile("visnir_24ch_v1")
        bands_24 = available_visnir_band_names(prof24)
        assert "band_carbonate_2330nm" in bands_24
        assert "band_water_1430nm" in bands_24

        meas24 = _make_measurement(sample_id="m24", profile_id="visnir_24ch_v1", seed=1)
        extracted = extract_visnir_band_features(meas24.spectrum, prof24)
        assert set(extracted.keys()) == set(bands_24)
        assert all(np.isfinite(v) for v in extracted.values())

    def test_eis_feature_extraction_and_mismatch_rejection(self) -> None:
        eis = EISMeasurement.from_units(
            frequencies=[100.0, 1000.0],
            z_real=[120.0, 80.0],
            z_imag=[-40.0, -20.0],
            protocol_id="eis_2freq_v1",
        )
        feats = extract_eis_features(
            eis,
            expected_frequencies_hz=[100.0, 1000.0],
            expected_protocol_id="eis_2freq_v1",
        )
        assert feats["eis_zreal_100hz"] == pytest.approx(120.0)
        assert feats["eis_zimag_100hz"] == pytest.approx(-40.0)
        assert feats["eis_mag_100hz"] == pytest.approx(np.hypot(120.0, -40.0))
        assert feats["eis_phase_rad_1000hz"] == pytest.approx(np.arctan2(-20.0, 80.0))

        with pytest.raises(SchemaError, match="protocol mismatch"):
            extract_eis_features(eis, expected_protocol_id="other_proto")

        with pytest.raises(SchemaError, match="EIS frequency vector does not match"):
            extract_eis_features(eis, expected_frequencies_hz=[100.0, 5000.0])

    def test_derived_ec_25c_and_none_propagation(self) -> None:
        cov_ok = CovariateSet(temperature_c=30.0, ec_ds_m=1.1)
        derived_ok = extract_derived_covariates(cov_ok)
        assert derived_ok["ec_25c_ds_m"] == pytest.approx(1.1 / (1.0 + 0.02 * (30.0 - 25.0)))

        cov_missing = CovariateSet(temperature_c=None, ec_ds_m=1.1)
        derived_missing = extract_derived_covariates(cov_missing)
        assert derived_missing["ec_25c_ds_m"] is None


class TestFailClosedCovariatesAndAlignment:
    """Verify explicit missingness taxonomy, ADR-002 bulk_density, and time alignment."""

    def test_validate_covariates_distinguishes_missing_vs_qc_failed_and_fail_closed(self) -> None:
        cov = CovariateSet(
            ph=6.5,
            moisture_volumetric_pct=None,
            temperature_c=20.0,
            ec_ds_m=None,
            bulk_density_g_cm3=None,
            missing_reasons={
                "moisture_volumetric": "not_measured",
                "ec": "qc_failed",
            },
        )
        assert cov.bulk_density_g_cm3 is None

        res = validate_covariates(cov, allow_missing=True)
        assert res.is_valid is True
        assert res.missing_flags == {
            "moisture_volumetric_pct": "not_measured",
            "ec_ds_m": "qc_failed",
        }
        assert QualityFlag.MISSING_COVARIATE in res.quality_flags
        assert QualityFlag.COVARIATE_QC_FAILED in res.quality_flags
        assert COVARIATE_CANONICAL_UNITS["bulk_density_g_cm3"] == "g/cm3"

        with pytest.raises(SchemaError, match="fail-closed policy"):
            validate_covariates(cov, allow_missing=False)

    def test_align_spectral_and_covariates_within_and_beyond_tolerance(self) -> None:
        m1 = _make_measurement(sample_id="s1", timestamp_utc="2026-11-12T10:00:00Z", ph=None, seed=1)
        m2 = _make_measurement(sample_id="s2", timestamp_utc="2026-11-12T10:10:00Z", ph=None, seed=2)
        c1 = CovariateSet(ph=7.1, moisture_volumetric_pct=19.0, temperature_c=21.0, ec_ds_m=0.5)
        c2 = CovariateSet(ph=6.4, moisture_volumetric_pct=22.0, temperature_c=20.0, ec_ds_m=0.7)

        aligned = align_spectral_and_covariates(
            [m1, m2],
            [("s1", "2026-11-12T10:02:00Z", c1), ("s2", "2026-11-12T10:08:00Z", c2)],
            max_time_delta_s=300.0,
        )
        assert aligned[0].covariates.ph == pytest.approx(7.1)
        assert aligned[1].covariates.ph == pytest.approx(6.4)
        # Missing bulk density must stay None and retain MISSING_BULK_DENSITY flag
        assert aligned[0].covariates.bulk_density_g_cm3 is None
        assert QualityFlag.MISSING_BULK_DENSITY.value in aligned[0].quality.flags

        with pytest.raises(SchemaError, match="exceeds tolerance"):
            align_spectral_and_covariates(
                [m1],
                [("s1", "2026-11-12T10:20:00Z", c1)],
                max_time_delta_s=300.0,
            )

        with pytest.raises(SchemaError, match="No matching covariate record"):
            align_spectral_and_covariates(
                [m1, m2],
                [("s1", "2026-11-12T10:00:00Z", c1)],
                max_time_delta_s=300.0,
            )


class TestFeatureFusionPipelineEndToEnd:
    """Verify spectral + covariate fusion, separate branch scaling, and synthetic v2 integration."""

    def test_fusion_pipeline_with_chemometrics_and_missing_indicators(self) -> None:
        gen = SyntheticGeneratorV2(
            SyntheticConfig(
                n_samples=16,
                n_sites=2,
                profile_id="visnir_24ch_v1",
                seed=42,
            )
        )
        ds = gen.generate_dataset()
        train_meas = list(ds.records[:12])
        test_meas = list(ds.records[12:])

        # Inject controlled missing covariates into one test sample (qc_failed & not_measured)
        test_meas[0].covariates = CovariateSet(
            ph=None,
            moisture_volumetric_pct=test_meas[0].covariates.moisture_volumetric_pct,
            temperature_c=test_meas[0].covariates.temperature_c,
            ec_ds_m=None,
            bulk_density_g_cm3=None,
            eis=test_meas[0].covariates.eis,
            missing_reasons={"ph": "qc_failed", "ec": "not_measured"},
        )

        # Unfitted fusion must fail closed
        fusion_cfg = FeatureFusionConfig(
            covariate_names=("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m"),
            missing_strategy="median_with_indicator",
            spectral_scaler="standard",
            covariate_scaler="robust",
            include_eis=True,
            include_derived=True,
            include_visnir_bands=True,
        )
        fusion = FeatureFusionPipeline(config=fusion_cfg)
        with pytest.raises(CalibrationError, match="fitted"):
            fusion.transform_covariates(test_meas)

        chemo = ChemometricPipeline(
            PreprocessConfig(
                use_absorbance=True,
                use_snv=True,
                use_savgol=True,
                sg_window=7,
                sg_polyorder=2,
                sg_deriv=1,
                suppression_mode="none",
            )
        ).fit(train_meas[0].spectrum.wavelength_grid_nm)
        fusion.fit(train_meas, chemometric_pipeline=chemo)

        train_fused = fusion.build_feature_vector(train_meas, chemometric_pipeline=chemo)
        test_fused = fusion.build_feature_vector(test_meas, chemometric_pipeline=chemo)

        assert train_fused.X_fused.shape[0] == 12
        assert test_fused.X_fused.shape[0] == 4
        assert train_fused.X_fused.shape[1] == len(train_fused.fused_feature_names)
        assert test_fused.X_fused.shape[1] == train_fused.X_fused.shape[1]
        assert np.all(np.isfinite(test_fused.X_fused))

        # Check explicit missingness mask and indicator columns for test_meas[0]
        ph_idx = fusion_cfg.covariate_names.index("ph")
        ec_idx = fusion_cfg.covariate_names.index("ec_ds_m")
        assert bool(test_fused.missingness_mask[0, ph_idx]) is True
        assert bool(test_fused.missingness_mask[0, ec_idx]) is True
        assert test_fused.missing_reasons[0]["ph"] == "qc_failed"
        assert test_fused.missing_reasons[0]["ec_ds_m"] == "not_measured"

        ph_ind_idx = test_fused.covariate_feature_names.index("is_missing_ph")
        assert test_fused.X_covariates[0, ph_ind_idx] == pytest.approx(1.0)
        temp_ind_idx = test_fused.covariate_feature_names.index("is_missing_temperature_c")
        assert test_fused.X_covariates[0, temp_ind_idx] == pytest.approx(0.0)

        # Spectral branch scaler should standardize training spectral columns near zero mean
        assert np.allclose(np.mean(train_fused.X_spectral, axis=0), 0.0, atol=1e-10)

    def test_all_nan_training_covariate_and_fail_closed_strategy_raise_schema_error(self) -> None:
        m1 = _make_measurement(sample_id="a1", ph=None, seed=1)
        m2 = _make_measurement(sample_id="a2", ph=None, seed=2)
        m3 = _make_measurement(sample_id="a3", ph=6.5, seed=3)

        fusion_all_nan = FeatureFusionPipeline(FeatureFusionConfig(covariate_names=("ph", "temperature_c")))
        with pytest.raises(SchemaError, match="100% missing across training samples"):
            fusion_all_nan.fit([m1, m2], spectral_matrix=np.ones((2, 24)))

        fusion_strict = FeatureFusionPipeline(
            FeatureFusionConfig(
                covariate_names=("ph", "temperature_c"),
                missing_strategy="fail_closed",
            )
        )
        with pytest.raises(SchemaError, match="fail-closed policy"):
            fusion_strict.fit([m1, m3], spectral_matrix=np.ones((2, 24)))
