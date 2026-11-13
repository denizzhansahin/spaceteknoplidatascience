"""Faz 8 Unit & Invariant Tests:
Local Sensitivity, Integrated Gradients, Normalized HSIC (B < 4 Fail-Closed),
True Second-Order Interaction Score (DEF-004c), and SoilXAIEngine Report Separation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from st_ds.chemometrics import ChemometricPipeline, PreprocessConfig
from st_ds.domain import QualityFlag
from st_ds.features import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models import PLSRRegressor, STSoilHybridNetRegressor
from st_ds.xai import (
    SoilXAIEngine,
    compute_feature_hsic_ranking,
    compute_hsic,
    integrated_gradients,
    interaction_score,
    local_sensitivity,
)


class TestSensitivityAndIntegratedGradients:
    """Verify unit-normalized local gradients and Integrated Gradients completeness."""

    def test_scale_normalized_local_sensitivity_and_ig_completeness_on_plsr(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(0.0, 1.0, size=(30, 5))
        # Target driven primarily by feature 0 (slope 3.0) and weakly by feature 2 (slope 0.5)
        y = np.maximum(0.2, 5.0 + 3.0 * X[:, 0] + 0.5 * X[:, 2])

        plsr = PLSRRegressor(max_components=3, inner_cv_folds=3).fit(X, y)
        scales = np.array([1.0, 10.0, 1.0, 1.0, 1.0])
        x0 = np.array([0.4, -0.2, 0.7, 0.1, -0.5])

        sens = local_sensitivity(
            plsr,
            x0,
            feature_scales=scales,
            feature_names=("f0", "f1", "f2", "f3", "f4"),
            target_name="soc_pct",
        )
        assert sens.method == "central_difference"
        assert sens.raw_gradients[0] == pytest.approx(3.0, abs=0.25)
        assert sens.raw_gradients[2] == pytest.approx(0.5, abs=0.25)
        assert sens.normalized_sensitivity[0] > sens.normalized_sensitivity[2]

        ig = integrated_gradients(
            plsr,
            x0,
            baseline=np.zeros_like(x0),
            n_steps=24,
            feature_scales=scales,
            feature_names=("f0", "f1", "f2", "f3", "f4"),
        )
        assert ig.attributions is not None
        assert ig.completeness_residual is not None
        assert abs(ig.completeness_residual) < 1e-2

    def test_autograd_sensitivity_and_ig_completeness_on_hybrid_net(self) -> None:
        rng = np.random.default_rng(15)
        X_s = rng.normal(0.0, 1.0, size=(24, 6))
        X_c = rng.normal(0.0, 1.0, size=(24, 3))
        soc = np.maximum(0.3, 2.2 + 0.8 * X_s[:, 1] - 0.4 * X_c[:, 0])
        sic = np.maximum(0.2, 1.4 + 0.6 * X_s[:, 4])

        reg = STSoilHybridNetRegressor(
            n_spectral_features=6,
            spectral_hidden_dim=16,
            covariate_hidden_dim=8,
            fusion_hidden_dim=16,
            epochs=20,
            dropout=0.0,
            seed=42,
        )
        reg.fit_multitask(X_s, X_c, soc=soc, sic=sic)

        x_sample = np.concatenate([X_s[0], X_c[0]])
        x_base = np.concatenate([X_s[1], X_c[1]])
        sens = local_sensitivity(reg, x_sample, target_name="soc_pct")
        ig = integrated_gradients(reg, x_sample, baseline=x_base, n_steps=40, target_name="soc_pct")
        assert sens.method == "autograd"
        assert ig.method == "integrated_gradients_autograd"
        assert ig.completeness_residual is not None
        # Gauss-Legendre integration on smooth Softplus MLP recovers completeness tightly
        assert abs(ig.completeness_residual) < 1e-2


class TestHSICDependence:
    """Verify DEF-004b fail-closed B < 4 guard and normalized HSIC discrimination."""

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_hsic_returns_insufficient_sample_count_when_b_less_than_4(self, batch_size: int) -> None:
        rng = np.random.default_rng(99)
        X = rng.normal(size=(batch_size, 6))
        y = rng.normal(size=(batch_size,))

        res_1d = compute_hsic(X[:, 0], y)
        assert res_1d.status == "insufficient_sample_count"
        assert res_1d.raw_hsic is None
        assert res_1d.normalized_hsic is None
        assert QualityFlag.INSUFFICIENT_SAMPLE_COUNT in res_1d.quality_flags

        res_batch = compute_feature_hsic_ranking(X, y, feature_names=[f"b{i}" for i in range(6)])
        assert res_batch.status == "insufficient_sample_count"
        assert res_batch.feature_dependence is None
        assert QualityFlag.INSUFFICIENT_SAMPLE_COUNT in res_batch.quality_flags

    def test_hsic_discriminates_dependent_vs_independent_and_warns_below_8(self) -> None:
        rng = np.random.default_rng(5)
        X_small = rng.normal(size=(5, 2))
        y_small = X_small[:, 0] ** 2
        res_small = compute_hsic(X_small[:, 0], y_small)
        assert res_small.status == "low_sample_warning"
        assert res_small.normalized_hsic is not None

        n = 32
        X = rng.uniform(-2.0, 2.0, size=(n, 3))
        # y depends strongly on col 0 (nonlinear U-curve), moderately on col 1, independent of col 2
        y = 2.0 * (X[:, 0] ** 2) + 0.3 * X[:, 1]
        names = ("nonlinear_dep", "linear_dep", "pure_noise")
        ranked = compute_feature_hsic_ranking(X, y, feature_names=names)
        assert ranked.status == "ok"
        assert ranked.feature_dependence is not None
        assert ranked.feature_dependence["nonlinear_dep"] > ranked.feature_dependence["pure_noise"] + 0.2


class TestTrueInteractions:
    """Verify DEF-004c: additive models yield ~0 interaction, multiplicative yields true partial."""

    def test_additive_model_has_zero_interaction_while_bilinear_is_detected(self) -> None:
        def additive_fn(X: np.ndarray) -> np.ndarray:
            return 3.0 * X[:, 0] + 5.0 * X[:, 1] - 2.0 * X[:, 2]

        def bilinear_fn(X: np.ndarray) -> np.ndarray:
            return 3.0 * X[:, 0] + 5.0 * X[:, 1] + 4.0 * X[:, 0] * X[:, 1]

        x = np.array([1.2, -0.7, 0.5])
        scales = np.array([1.0, 2.0, 1.0])

        # Mandatory DEF-004c acceptance test: strictly <= 1e-7 on additive linear model
        inter_add = interaction_score(additive_fn, x, 0, 1, feature_scales=scales)
        assert inter_add.interaction_score <= 1e-7

        # Exact bilinear interaction: |d^2 f / (dx_0 dx_1)| * s_0 * s_1 = 4.0 * 1.0 * 2.0 = 8.0
        inter_bil = interaction_score(bilinear_fn, x, 0, 1, feature_scales=scales)
        assert inter_bil.interaction_score == pytest.approx(8.0, rel=1e-4)


class TestSoilXAIEngineReport:
    """Verify end-to-end XAI report structure, group/target separation, and serialization."""

    def test_xai_engine_separates_spectral_vs_covariate_and_single_vs_batch_hsic(self) -> None:
        gen = SyntheticGeneratorV2(
            SyntheticConfig(n_samples=20, n_sites=2, profile_id="visnir_24ch_v1", seed=42)
        )
        ds = gen.generate_dataset()
        chemo = ChemometricPipeline(
            PreprocessConfig(use_absorbance=True, use_snv=True, use_savgol=True, suppression_mode="none")
        ).fit(ds.records[0].spectrum.wavelength_grid_nm)

        fusion = FeatureFusionPipeline(
            FeatureFusionConfig(
                covariate_names=("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m"),
                include_derived=True,
                include_visnir_bands=True,
            )
        ).fit(ds.records, chemometric_pipeline=chemo)

        fused = fusion.build_feature_vector(ds.records, chemometric_pipeline=chemo)
        soc = np.asarray([float(r.reference.soc_pct) for r in ds.records if r.reference is not None])
        sic = np.asarray([float(r.reference.sic_pct) for r in ds.records if r.reference is not None])

        net_reg = STSoilHybridNetRegressor(
            n_spectral_features=fused.X_spectral.shape[1],
            spectral_hidden_dim=24,
            covariate_hidden_dim=12,
            fusion_hidden_dim=24,
            epochs=15,
            dropout=0.0,
            seed=42,
        )
        net_reg.fit_multitask(fused.X_spectral, fused.X_covariates, soc=soc, sic=sic)

        engine = SoilXAIEngine(ig_steps=12, max_interaction_pairs=4)

        # 1. Single-sample explanation without reference batch -> HSIC must fail closed (B=1 < 4)
        rep_single = engine.explain(
            net_reg,
            fused.X_fused[0],
            feature_profile=fused.feature_profile,
            target_name="soc_pct",
            artifact_id="art_demo_v1",
        )
        assert rep_single.dataset_dependence_hsic.status == "insufficient_sample_count"
        assert rep_single.dataset_dependence_hsic.feature_dependence is None
        assert QualityFlag.INSUFFICIENT_SAMPLE_COUNT in rep_single.dataset_dependence_hsic.quality_flags

        # 2. Explanation with reference batch (B=20) & separate SIC target ranking
        rep_batch_sic = engine.explain(
            net_reg,
            fused.X_fused[0],
            feature_profile=fused.feature_profile,
            reference_X=fused.X_fused,
            reference_y=sic,
            target_name="sic_pct",
            artifact_id="art_demo_v1",
        )
        assert rep_batch_sic.dataset_dependence_hsic.status == "ok"
        assert rep_batch_sic.dataset_dependence_hsic.feature_dependence is not None

        # DEF-004d verification: spectral bands and covariates live in strictly separate rankings
        spec_names = {r.name for r in rep_batch_sic.component_ranking.spectral_ranking}
        cov_names = {r.name for r in rep_batch_sic.component_ranking.covariate_ranking}
        assert "moisture_volumetric_pct" in cov_names
        assert "moisture_volumetric_pct" not in spec_names
        assert len(spec_names & cov_names) == 0
        assert rep_batch_sic.component_ranking.target_name == "sic_pct"

        # Provenance & JSON serialization verification
        payload = rep_batch_sic.to_dict()
        serialized = json.dumps(payload)
        assert "art_demo_v1" in serialized
        assert "not establish physical causality" in payload["provenance"]["disclaimer"]
