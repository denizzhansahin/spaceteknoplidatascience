"""Faz 7 Unit, Coverage & Contract Tests:
Conformal Prediction, Aleatoric/Epistemic Ensemble Decomposition,
Subgroup Calibration Curves, and Multi-Signal Out-of-Domain Detection.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from st_ds.domain import (
    CalibrationError,
    CovariateSet,
    ModelInferenceError,
    OutOfDomainError,
    QualityFlag,
    QualityStatus,
)
from st_ds.io import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models import RandomForestSoilRegressor
from st_ds.training import GroupedSplitBuilder, STSoilTrainer, TrainerConfig
from st_ds.uncertainty import (
    ConformalConfig,
    ConformalUncertaintyCalibrator,
    GroupedBootstrapEnsemble,
    OODConfig,
    OutOfDomainDetector,
    compute_calibration_curve,
    detect_out_of_domain,
    estimate_ensemble_spread,
    evaluate_subgroup_coverage,
    fit_uncertainty_calibrator,
    predict_interval,
)


class TestConformalPrediction:
    """Verify finite-sample conformal calibration, adaptive widths, and coverage curves."""

    def test_normalized_vs_absolute_conformal_and_empirical_coverage(self) -> None:
        rng = np.random.default_rng(123)
        n_cal, n_test = 200, 300

        # Heteroscedastic residuals: half low noise (sigma=0.1), half high noise (sigma=0.8)
        pred_std_cal = np.where(np.arange(n_cal) % 2 == 0, 0.1, 0.8)
        y_pred_cal = rng.uniform(1.0, 5.0, size=n_cal)
        y_cal = y_pred_cal + rng.normal(0.0, pred_std_cal)

        pred_std_test = np.where(np.arange(n_test) % 2 == 0, 0.1, 0.8)
        y_pred_test = rng.uniform(1.0, 5.0, size=n_test)
        y_test = y_pred_test + rng.normal(0.0, pred_std_test)

        cal_norm = fit_uncertainty_calibrator(
            y_cal,
            y_pred_cal,
            pred_std_cal=pred_std_cal,
            config=ConformalConfig(method="normalized", default_alpha=0.10, non_negative=True),
        )
        interval_norm = predict_interval(cal_norm, y_pred_test, pred_std=pred_std_test)

        # Normalized intervals must adapt strictly to heteroscedastic scale
        low_widths = interval_norm.half_width[0::2]
        high_widths = interval_norm.half_width[1::2]
        assert np.mean(high_widths) > 5.0 * np.mean(low_widths)

        emp_cov = float(np.mean((y_test >= interval_norm.lower_bound) & (y_test <= interval_norm.upper_bound)))
        assert 0.84 <= emp_cov <= 0.96
        assert np.all(interval_norm.lower_bound >= 0.0)

        # Absolute conformal mode check + calibration curve monotonicity
        cal_abs = fit_uncertainty_calibrator(
            y_cal,
            y_pred_cal,
            config=ConformalConfig(method="absolute", default_alpha=0.10, non_negative=True),
        )
        curve = compute_calibration_curve(
            cal_abs,
            y_test,
            y_pred_test,
            coverage_levels=(0.50, 0.80, 0.90, 0.95),
        )
        assert [p.nominal_coverage for p in curve] == [0.50, 0.80, 0.90, 0.95]
        # Empirical coverage must increase monotonically across increasing nominal levels
        emp_levels = [p.empirical_coverage for p in curve]
        assert emp_levels[0] < emp_levels[1] <= emp_levels[2] <= emp_levels[3]

    def test_unfitted_and_insufficient_calibration_samples_fail_closed(self) -> None:
        unfitted = ConformalUncertaintyCalibrator()
        with pytest.raises(CalibrationError, match="must be fitted"):
            predict_interval(unfitted, [1.0, 2.0], pred_std=[0.2, 0.2])

        with pytest.raises(ModelInferenceError, match="at least 5"):
            fit_uncertainty_calibrator([1.0, 2.0, 3.0], [1.1, 1.9, 3.1], pred_std_cal=[0.2, 0.2, 0.2])


class TestEnsembleSpread:
    """Verify exact aleatoric vs epistemic decomposition and grouped cluster bootstrap."""

    def test_exact_aleatoric_epistemic_decomposition(self) -> None:
        # 3 members, 2 samples
        member_means = np.array(
            [
                [1.0, 4.0],
                [2.0, 4.0],
                [3.0, 4.0],
            ]
        )
        member_stds = np.array(
            [
                [0.3, 0.5],
                [0.4, 0.5],
                [0.5, 0.5],
            ]
        )
        res = estimate_ensemble_spread(member_means, member_stds)
        # Sample 0: epistemic_var = ((1-2)^2 + 0 + (3-2)^2)/3 = 2/3, aleatoric_var = (0.09+0.16+0.25)/3 = 0.5/3
        # Sample 1: epistemic_std = 0, aleatoric_std = 0.5
        np.testing.assert_allclose(res.mean_prediction, [2.0, 4.0], atol=1e-12)
        np.testing.assert_allclose(res.epistemic_std, [np.sqrt(2.0 / 3.0), 0.0], atol=1e-12)
        np.testing.assert_allclose(res.aleatoric_std, [np.sqrt(0.5 / 3.0), 0.5], atol=1e-12)
        np.testing.assert_allclose(
            res.total_std,
            np.sqrt(res.aleatoric_std**2 + res.epistemic_std**2),
            atol=1e-12,
        )

    def test_grouped_bootstrap_ensemble_predict_and_unfitted_guard(self) -> None:
        rng = np.random.default_rng(77)
        X = rng.normal(0.0, 1.0, size=(36, 6))
        y = np.maximum(0.2, 2.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0.0, 0.1, size=36))
        groups = np.repeat(["site_1", "site_2", "site_3", "site_4", "site_5", "site_6"], 6)

        ens = GroupedBootstrapEnsemble(n_bootstraps=6, seed=42)
        with pytest.raises(CalibrationError, match="must be fitted"):
            ens.predict_distribution(X[:3])

        ens.fit(
            lambda seed: RandomForestSoilRegressor(n_estimators=20, random_state=seed),
            X,
            y,
            groups=groups,
        )
        spread = ens.predict_distribution(X[:4])
        assert spread.n_members == 6
        assert spread.has_aleatoric_component is True
        assert spread.total_std.shape == (4,)
        assert np.all(spread.total_std >= spread.epistemic_std - 1e-12)


class TestOutOfDomainDetection:
    """Verify multi-signal OOD detection, confidence penalty, and fail-closed flags."""

    def test_in_domain_vs_feature_shift_and_envelope_ood(self) -> None:
        gen = SyntheticGeneratorV2(
            SyntheticConfig(n_samples=24, n_sites=3, profile_id="visnir_24ch_v1", seed=42)
        )
        ds = gen.generate_dataset()
        X_train = np.vstack([r.spectrum.values for r in ds.records])

        detector = OutOfDomainDetector(OODConfig(knn_k=4, fail_on_ood=False))
        with pytest.raises(CalibrationError, match="must be fitted"):
            detect_out_of_domain(detector, X_train[:2])

        detector.fit(X_train, measurements=ds.records)

        # In-distribution check on subset of nominal samples
        in_res = detect_out_of_domain(detector, X_train[:5], measurements=ds.records[:5])
        assert in_res.n_ood == 0
        for a in in_res.assessments:
            assert a.quality_status == QualityStatus.OK
            assert a.confidence_score > 0.55

        # 1) Feature shift OOD
        X_ood = X_train[:3].copy()
        X_ood[0] += 5.0  # Huge spectral shift

        # 2) Extreme out-of-envelope covariate OOD (moisture/pH out of training bounds)
        m_env_ood = dataclasses.replace(
            ds.records[1],
            covariates=CovariateSet(
                ph=9.4,
                moisture_volumetric_pct=62.0,
                temperature_c=22.0,
                ec_ds_m=0.8,
            ),
        )

        # 3) Unseen hardware profile ID OOD
        m_hw_ood = dataclasses.replace(
            ds.records[2],
            instrument_profile_id="unknown_experimental_device_v9",
        )

        batch_ood = detect_out_of_domain(
            detector,
            X_ood,
            measurements=[ds.records[0], m_env_ood, m_hw_ood],
        )
        assert batch_ood.n_ood == 3
        for assessment in batch_ood.assessments:
            assert assessment.is_ood is True
            assert assessment.quality_status == QualityStatus.OUT_OF_DOMAIN
            assert QualityFlag.OUT_OF_CALIBRATION_DOMAIN in assessment.quality_flags
            # Critical Risk 5 requirement: OOD sample must never receive high confidence score
            assert assessment.confidence_score < 0.5

        assert "ph" in batch_ood.assessments[1].out_of_range_covariates
        assert "moisture_volumetric_pct" in batch_ood.assessments[1].out_of_range_covariates
        assert "instrument_profile_id:unknown_experimental_device_v9" in batch_ood.assessments[2].unseen_categories

        with pytest.raises(OutOfDomainError, match="out_of_domain"):
            detect_out_of_domain(detector, X_ood, fail_on_ood=True)


class TestFaz6ToFaz7Integration:
    """Verify STSoilTrainer OOF predictions integrate directly with Conformal & Subgroup Coverage."""

    def test_trainer_oof_feeds_conformal_and_subgroup_coverage(self) -> None:
        gen = SyntheticGeneratorV2(
            SyntheticConfig(n_samples=32, n_sites=4, profile_id="visnir_24ch_v1", seed=42)
        )
        ds = gen.generate_dataset()
        folds = GroupedSplitBuilder.group_kfold_split(ds, n_splits=4, group_by="site_id")

        trainer = STSoilTrainer(TrainerConfig(target_name="soc_pct", seed=42))
        report = trainer.run_benchmark(ds, split_folds=folds, include_optional_models=False)

        best_name = report.evidence_ranking[0]
        entry = report.entries[best_name]
        y_true = np.asarray([float(r.reference.soc_pct) for r in ds.records if r.reference is not None])

        cal = fit_uncertainty_calibrator(
            y_true,
            entry.oof_predictions,
            pred_std_cal=entry.oof_uncertainties,
            config=ConformalConfig(method="normalized", default_alpha=0.10),
        )
        cov_report = evaluate_subgroup_coverage(
            cal,
            ds.records,
            y_true,
            entry.oof_predictions,
            pred_std=entry.oof_uncertainties,
        )
        assert len(cov_report.curve) == 4
        assert "site_id" in cov_report.subgroup_coverage
        assert len(cov_report.subgroup_coverage["site_id"]) == 4
        # Overall empirical coverage at 90% nominal should be at least 0.85 on calibration OOF
        point_90 = [p for p in cov_report.curve if abs(p.nominal_coverage - 0.90) < 1e-6][0]
        assert point_90.empirical_coverage >= 0.85
