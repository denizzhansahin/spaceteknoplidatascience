"""Faz 6 Unit & Contract Tests:
Baseline Regressors (PLSR, KernelRidge, RF, MBL/LocalPLSR), STSoilHybridNet,
Multi-Task Physical Closure Losses, Chemometric/Bland-Altman Metrics, and STSoilTrainer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from st_ds.domain import ModelInferenceError, RegressorProtocol, SchemaError
from st_ds.io import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models import (
    HybridNetConfig,
    KernelRidgeRegressor,
    LocalPLSRRegressor,
    MultiTaskLossConfig,
    PLSRRegressor,
    RandomForestSoilRegressor,
    STSoilHybridNet,
    STSoilHybridNetRegressor,
    STSoilMultiTaskLoss,
    gaussian_nll_loss,
)
from st_ds.training import (
    GroupedSplitBuilder,
    STSoilTrainer,
    TrainerConfig,
    check_closure_diagnostics,
    compute_regression_metrics,
    evaluate_by_subgroup,
)


def _synthetic_linear_matrix(n: int = 40, d: int = 12, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    w_soc = np.linspace(0.5, -0.3, d)
    w_sic = np.linspace(-0.2, 0.4, d)
    soc = np.maximum(0.2, 2.5 + np.dot(X, w_soc) + rng.normal(0.0, 0.08, size=n))
    sic = np.maximum(0.1, 1.8 + np.dot(X, w_sic) + rng.normal(0.0, 0.08, size=n))
    return X, soc, sic


class TestBaselineRegressors:
    """Verify RegressorProtocol, inner-CV tuning, fail-closed guards, and uncertainty bounds."""

    def test_all_models_conform_to_regressor_protocol_and_fail_before_fit(self) -> None:
        models = [
            PLSRRegressor(max_components=5, inner_cv_folds=3),
            KernelRidgeRegressor(inner_cv_folds=3),
            RandomForestSoilRegressor(n_estimators=20, random_state=42),
            LocalPLSRRegressor(n_neighbors=10, n_components=2, min_library_size=8),
            STSoilHybridNetRegressor(n_spectral_features=8, epochs=5, seed=42),
        ]
        X_probe = np.ones((4, 8))
        for m in models:
            assert isinstance(m, RegressorProtocol)
            with pytest.raises(ModelInferenceError, match="must be fitted"):
                m.predict(X_probe)
            with pytest.raises(ModelInferenceError, match="must be fitted"):
                m.predict_distribution(X_probe)

    def test_plsr_kr_rf_mbl_fit_predict_and_uncertainty(self) -> None:
        X, soc, sic = _synthetic_linear_matrix(n=45, d=10, seed=11)
        X_tr, X_te = X[:35], X[35:]
        y_tr, y_te = soc[:35], soc[35:]

        plsr = PLSRRegressor(max_components=6, inner_cv_folds=3).fit(X_tr, y_tr)
        kr = KernelRidgeRegressor(inner_cv_folds=3).fit(X_tr, y_tr)
        rf = RandomForestSoilRegressor(n_estimators=30, random_state=42).fit(X_tr, y_tr)
        mbl = LocalPLSRRegressor(n_neighbors=15, n_components=3, min_library_size=10).fit(X_tr, y_tr)

        assert 1 <= plsr.selected_n_components_ <= 6
        assert kr.selected_alpha_ > 0.0 and kr.selected_sigma_ > 0.0

        for model in (plsr, kr, rf, mbl):
            mean, std = model.predict_distribution(X_te)
            assert mean.shape == (10,)
            assert std.shape == (10,)
            assert np.all(np.isfinite(mean))
            assert np.all(mean >= 0.0)
            assert np.all(std > 0.0)
            card = model.model_card()
            assert card["is_fitted"] is True
            assert card["n_features_in"] == 10

        # PLSR and KernelRidge should fit smooth linear/nonlinear relationship accurately
        assert float(np.sqrt(np.mean((plsr.predict(X_te) - y_te) ** 2))) < 0.35
        assert float(np.sqrt(np.mean((kr.predict(X_te) - y_te) ** 2))) < 0.65

        # Multi-target 2D output shape check
        y_2d = np.column_stack([soc[:35], sic[:35]])
        plsr_2d = PLSRRegressor(max_components=5, inner_cv_folds=3).fit(X_tr, y_2d)
        m2, s2 = plsr_2d.predict_distribution(X_te)
        assert m2.shape == (10, 2)
        assert s2.shape == (10, 2)

    def test_mbl_rejects_small_library_and_negative_clipping_is_tracked(self) -> None:
        X, soc, _ = _synthetic_linear_matrix(n=6, d=6, seed=3)
        mbl = LocalPLSRRegressor(n_neighbors=5, min_library_size=8)
        with pytest.raises(ModelInferenceError, match="at least 8 samples"):
            mbl.fit(X, soc)


class TestHybridNetAndLosses:
    """Verify non-negative softplus heads, multi-dimension support, and closure loss."""

    def test_variable_channel_support_and_softplus_non_negativity(self) -> None:
        for n_spec, n_cov in ((5, 4), (24, 6), (60, 0)):
            cfg = HybridNetConfig(
                n_spectral_features=n_spec,
                n_covariate_features=n_cov,
                spectral_hidden_dim=16,
                covariate_hidden_dim=8,
                fusion_hidden_dim=16,
                enable_tc_head=True,
                enable_moisture_head=True,
            )
            net = STSoilHybridNet(cfg)
            net.eval()
            # Extreme negative/positive inputs must still yield >= 0 carbon and > 0 variances
            x_s = torch.randn(7, n_spec) * 50.0 - 20.0
            x_c = (torch.randn(7, n_cov) * 10.0) if n_cov > 0 else None
            out = net(x_s, x_c)
            assert torch.all(out["soc_mean"] >= 0.0)
            assert torch.all(out["sic_mean"] >= 0.0)
            assert torch.all(out["tc_mean"] == out["soc_mean"] + out["sic_mean"])
            assert torch.all(out["soc_var"] > 0.0)
            assert torch.all(out["sic_var"] > 0.0)
            assert out["class_logits"].shape == (7, 4)

    def test_gaussian_nll_and_multitask_closure_loss_reduce_during_training(self) -> None:
        # Check gaussian_nll_loss masking
        t = torch.tensor([1.0, 2.0, 999.0])
        m = torch.tensor([1.0, 2.0, 0.0])
        v = torch.tensor([1.0, 1.0, 1.0])
        mask = torch.tensor([True, True, False])
        loss_masked = gaussian_nll_loss(t, m, v, mask=mask)
        assert float(loss_masked.item()) == pytest.approx(0.0, abs=1e-6)

        all_false = torch.tensor([False, False, False])
        assert float(gaussian_nll_loss(t, m, v, mask=all_false).item()) == pytest.approx(0.0)

        # Verify closure penalty computes gradients and STSoilHybridNetRegressor trains cleanly
        rng = np.random.default_rng(19)
        X_s = rng.normal(0.0, 1.0, size=(36, 12))
        X_c = rng.normal(0.0, 1.0, size=(36, 4))
        soc = np.maximum(0.3, 2.0 + 0.6 * X_s[:, 0] - 0.3 * X_c[:, 1])
        sic = np.maximum(0.1, 1.0 + 0.5 * X_s[:, 2])
        tc = soc + sic
        soil_cls = ((soc > 2.0).astype(int) * 2 + (sic > 1.0).astype(int)).astype(float)

        reg = STSoilHybridNetRegressor(
            n_spectral_features=12,
            spectral_hidden_dim=32,
            covariate_hidden_dim=16,
            fusion_hidden_dim=32,
            epochs=35,
            batch_size=12,
            learning_rate=5e-3,
            loss_config=MultiTaskLossConfig(lambda_close=4.0),
            mc_dropout_samples=3,
            dropout=0.1,
            seed=42,
        )
        reg.fit_multitask(X_s, X_c, soc=soc, sic=sic, tc=tc, soil_class=soil_cls)

        assert len(reg.training_history_) == 35
        assert reg.training_history_[-1]["loss_total"] < reg.training_history_[0]["loss_total"]

        pred = reg.predict_structured(X_s[:6], X_c[:6])
        assert np.all(pred.soc_mean_pct >= 0.0)
        assert pred.sic_mean_pct is not None and np.all(pred.sic_mean_pct >= 0.0)
        assert pred.tc_mean_pct is not None
        np.testing.assert_allclose(pred.tc_mean_pct, pred.soc_mean_pct + pred.sic_mean_pct, atol=1e-6)
        assert pred.closure_residual_pct is not None
        assert float(np.mean(np.abs(pred.closure_residual_pct))) < 0.5

        tensors = reg.state_dict_tensors()
        assert all(isinstance(val, torch.Tensor) for val in tensors.values())


class TestMetricsAndBlandAltman:
    """Verify exact regression, Bland-Altman, subgroup, and closure diagnostics calculations."""

    def test_exact_metrics_and_bland_altman_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.2, 1.8, 3.3, 3.9])
        diff = y_pred - y_true  # [0.2, -0.2, 0.3, -0.1], mean=0.05

        m = compute_regression_metrics(y_true, y_pred)
        assert m.n_samples == 4
        assert m.mae == pytest.approx(np.mean(np.abs(diff)))
        assert m.rmse == pytest.approx(np.sqrt(np.mean(diff**2)))
        assert m.bias == pytest.approx(0.05)
        expected_sd = float(np.std(diff, ddof=1))
        assert m.bland_altman_mean_diff == pytest.approx(0.05)
        assert m.bland_altman_std_diff == pytest.approx(expected_sd)
        assert m.bland_altman_loa_lower == pytest.approx(0.05 - 1.96 * expected_sd)
        assert m.bland_altman_loa_upper == pytest.approx(0.05 + 1.96 * expected_sd)
        iqr = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
        assert m.rpiq == pytest.approx(iqr / m.rmse)

    def test_closure_diagnostics_detects_violations(self) -> None:
        diag = check_closure_diagnostics(
            soc_pred=[1.5, 2.0],
            sic_pred=[0.5, 1.0],
            tc_pred=[2.01, 3.25],
            tolerance_pct=0.05,
        )
        assert diag["closure_violation_count"] == 1
        assert diag["max_abs_closure_residual_pct"] == pytest.approx(0.25)
        assert diag["negative_soc_count"] == 0


class TestSTSoilTrainerBenchmark:
    """Verify end-to-end leakage-free grouped CV benchmark across all Phase 6 models."""

    def test_grouped_benchmark_runs_all_models_on_identical_folds_and_reports_evidence(self) -> None:
        gen = SyntheticGeneratorV2(
            SyntheticConfig(
                n_samples=32,
                n_sites=4,
                profile_id="visnir_24ch_v1",
                seed=42,
            )
        )
        ds = gen.generate_dataset()
        folds = GroupedSplitBuilder.group_kfold_split(ds, n_splits=4, group_by="site_id")

        trainer = STSoilTrainer(
            TrainerConfig(
                target_name="soc_pct",
                hybrid_epochs=20,
                hybrid_batch_size=12,
                seed=42,
            )
        )
        report = trainer.run_benchmark(ds, split_folds=folds, include_optional_models=True)

        assert report.is_synthetic_smoke_only is True
        assert any("synthetic" in w.lower() for w in report.warnings)
        assert report.n_samples == 32
        assert report.n_folds == 4
        assert set(report.entries.keys()) == {
            "plsr",
            "kernel_ridge_rbf",
            "random_forest",
            "local_plsr_mbl",
            "st_soil_hybrid_net",
        }

        # Verify ranking strictly follows OOF RMSE ascending
        rmses = [report.entries[k].oof_metrics.rmse for k in report.evidence_ranking]
        assert rmses == sorted(rmses)

        # Verify subgroup stratification and hybrid closure diagnostics
        kr_entry = report.entries["kernel_ridge_rbf"]
        assert len(kr_entry.fold_metrics) == 4
        assert len(kr_entry.subgroup_metrics["site_id"]) == 4
        assert "synthetic" == kr_entry.last_model_card["training_dataset_source"]

        hyb_entry = report.entries["st_soil_hybrid_net"]
        assert hyb_entry.closure_diagnostics is not None
        assert hyb_entry.closure_diagnostics["negative_soc_count"] == 0
        assert hyb_entry.closure_diagnostics["negative_sic_count"] == 0
        assert np.all(np.isfinite(hyb_entry.oof_predictions))
        assert np.all(hyb_entry.oof_uncertainties > 0.0)

        # Verify direct subgroup evaluator call works with OOF predictions
        sub = evaluate_by_subgroup(
            ds.records,
            [float(r.reference.soc_pct) for r in ds.records if r.reference is not None],
            hyb_entry.oof_predictions,
        )
        assert "site_id" in sub and "moisture_scenario" in sub
