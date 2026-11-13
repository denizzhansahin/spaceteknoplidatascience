"""Faz 10 End-to-End Integration Tests: Training -> Signed Bundle -> Analyzer -> Appendix B JSON."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from st_ds import SCHEMA_VERSION
from st_ds.api import STSoilAnalyzer
from st_ds.artifact import inspect_artifact, save_artifact
from st_ds.chemometrics.pipeline import ChemometricPipeline, PreprocessConfig
from st_ds.domain.instrument import get_builtin_profile
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.domain.schemas import ArtifactMetadata, PredictionReport
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models.baselines import KernelRidgeRegressor, LocalPLSRRegressor, RandomForestSoilRegressor
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import STSoilTrainer, TrainerConfig, _extract_target_vector
from st_ds.uncertainty.conformal import ConformalConfig, ConformalUncertaintyCalibrator
from st_ds.uncertainty.ood import OODConfig, OutOfDomainDetector


def _train_and_export_bundle(
    tmp_path: Path,
    *,
    model_cls_name: str,
) -> tuple[Path, list, list]:
    gen = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id=f"e2e_{model_cls_name}",
            profile_id="visnir_24ch_v1",
            n_samples=36,
            n_sites=3,
            seed=77,
        )
    )
    ds = gen.generate_dataset()
    dry_pairs, wet_pairs = gen.generate_paired_wet_dry_dataset(n_pairs=12)
    profile = get_builtin_profile("visnir_24ch_v1")

    train_recs = [r for r in ds.records if r.site_id in ("site_00", "site_01")]
    test_recs = [r for r in ds.records if r.site_id == "site_02"]

    pre_cfg = PreprocessConfig(suppression_mode="epo", epo_q=2)
    chemo = ChemometricPipeline(pre_cfg).fit_from_datasets(wet_pairs, dry_pairs)
    fusion_cfg = FeatureFusionConfig(
        covariate_names=("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m"),
    )
    fusion = FeatureFusionPipeline(fusion_cfg).fit(train_recs, chemometric_pipeline=chemo)
    fused_tr = fusion.build_feature_vector(train_recs, chemometric_pipeline=chemo)

    y_tr = _extract_target_vector(train_recs, "soc_pct")
    if model_cls_name == "kernel_ridge_rbf":
        model = KernelRidgeRegressor(target_names=("soc_pct",))
        model.set_metadata(feature_names=fused_tr.fused_feature_names, target_names=("soc_pct",), dataset_source=ds.dataset_source)
        model.fit(fused_tr.X_fused, y_tr, groups=np.asarray([r.site_id for r in train_recs], dtype=str))
        p_cal, s_cal = model.predict_distribution(fused_tr.X_fused)
    elif model_cls_name == "random_forest":
        model = RandomForestSoilRegressor(n_estimators=25, random_state=42, target_names=("soc_pct",))
        model.set_metadata(feature_names=fused_tr.fused_feature_names, target_names=("soc_pct",), dataset_source=ds.dataset_source)
        model.fit(fused_tr.X_fused, y_tr)
        p_cal, s_cal = model.predict_distribution(fused_tr.X_fused)
    elif model_cls_name == "local_plsr_mbl":
        model = LocalPLSRRegressor(n_neighbors=10, n_components=2, min_library_size=8, target_names=("soc_pct",))
        model.set_metadata(feature_names=fused_tr.fused_feature_names, target_names=("soc_pct",), dataset_source=ds.dataset_source)
        model.fit(fused_tr.X_fused, y_tr)
        p_cal, s_cal = model.predict_distribution(fused_tr.X_fused)
    elif model_cls_name == "st_soil_hybrid_net":
        model = STSoilHybridNetRegressor(
            n_spectral_features=fused_tr.X_spectral.shape[1],
            epochs=20,
            batch_size=12,
            seed=42,
            target_names=("soc_pct",),
        )
        model.set_metadata(feature_names=fused_tr.fused_feature_names, target_names=("soc_pct",), dataset_source=ds.dataset_source)
        sic_tr = np.asarray([float(r.reference.sic_pct) for r in train_recs], dtype=float)  # type: ignore[union-attr]
        tc_tr = np.asarray([float(r.reference.tc_pct) for r in train_recs], dtype=float)  # type: ignore[union-attr]
        model.fit_multitask(fused_tr.X_spectral, fused_tr.X_covariates, soc=y_tr, sic=sic_tr, tc=tc_tr)
        struct = model.predict_structured(fused_tr.X_spectral, fused_tr.X_covariates)
        p_cal = struct.soc_mean_pct
        assert struct.soc_std_pct is not None
        s_cal = struct.soc_std_pct
    else:
        raise ValueError(model_cls_name)

    conformal = ConformalUncertaintyCalibrator(ConformalConfig(method="normalized")).fit(
        y_tr, p_cal, pred_std_cal=s_cal
    )
    ood = OutOfDomainDetector(OODConfig()).fit(fused_tr.X_fused, measurements=train_recs)

    meta = ArtifactMetadata(
        artifact_id=f"art_{model_cls_name}",
        semantic_version="1.0.0",
        schema_version=SCHEMA_VERSION,
        instrument_profile_id=profile.profile_id,
        calibration_id=train_recs[0].calibration_id,
        pipeline_version=pre_cfg.pipeline_version,
        dataset_hash=ds.compute_sha256(),
        dependency_versions={"python": "3.13"},
    )
    out_dir = tmp_path / f"bundle_{model_cls_name}"
    save_artifact(
        out_dir,
        metadata=meta,
        instrument_profile=profile,
        model=model,
        chemometric_pipeline=chemo,
        fusion_pipeline=fusion,
        conformal_calibrator=conformal,
        ood_detector=ood,
        validation_metrics={"rmse": 0.15},
        background_features=fused_tr.X_fused,
        background_targets=y_tr,
    )
    return out_dir, train_recs, test_recs


class TestEndToEndPipelineIntegration:
    """End-to-end integration tests across classical and neural artifact bundles."""

    @pytest.mark.parametrize(
        "model_cls_name",
        ["kernel_ridge_rbf", "random_forest", "local_plsr_mbl", "st_soil_hybrid_net"],
    )
    def test_full_artifact_round_trip_and_prediction_report(
        self,
        tmp_path: Path,
        model_cls_name: str,
    ) -> None:
        bundle_dir, _, test_recs = _train_and_export_bundle(tmp_path, model_cls_name=model_cls_name)

        inspected = inspect_artifact(bundle_dir)
        assert inspected["integrity_verified"] is True
        assert inspected["weight_format"].startswith("safetensors_")
        assert inspected["has_conformal_calibrator"] is True
        assert inspected["has_ood_detector"] is True
        assert inspected["model_card"]["is_synthetic_smoke_only"] is True

        analyzer = STSoilAnalyzer.from_artifact(bundle_dir)
        sample = test_recs[0]
        report = analyzer.predict(sample, include_xai=True, include_stock=True)
        assert isinstance(report, PredictionReport)

        rep_dict = report.to_dict()
        # Check all 9 Appendix B sections
        for section in (
            "identity",
            "status",
            "carbon_metrics",
            "carbon_stock",
            "classification",
            "quality",
            "xai",
            "provenance",
            "diagnostics",
        ):
            assert section in rep_dict

        assert rep_dict["carbon_metrics"]["unit"] == "pct"
        assert rep_dict["carbon_metrics"]["dry_matter_basis"] is True
        assert rep_dict["carbon_metrics"]["soc_pct"] >= 0.0
        assert rep_dict["carbon_stock"]["stock_mg_ha"] is not None
        assert rep_dict["classification"]["is_provisional"] is True
        assert rep_dict["provenance"]["artifact_sha256"] == inspected["bundle_sha256"]
        assert "stage_timings_ms" in rep_dict["diagnostics"]
        # Raw spectrum values must never leak into diagnostics/logs (Section 27)
        json_text = report.to_json()
        assert json.loads(json_text)["identity"]["sample_id"] == sample.sample_id

    def test_batch_partial_failure_missing_bd_and_explain_single_vs_batch(self, tmp_path: Path) -> None:
        bundle_dir, _, test_recs = _train_and_export_bundle(tmp_path, model_cls_name="kernel_ridge_rbf")
        analyzer = STSoilAnalyzer.from_artifact(bundle_dir)

        valid_with_bd = test_recs[0]
        missing_bd = replace(
            test_recs[1],
            covariates=replace(test_recs[1].covariates, bulk_density_g_cm3=None),
        )
        # Deliberately corrupted measurement (channel count mismatch -> Calibration/Schema error)
        bad_profile = get_builtin_profile("led_5band_v1")
        invalid_meas = replace(
            test_recs[2],
            instrument_profile_id="led_5band_v1",
            spectrum=replace(
                test_recs[2].spectrum,
                values=np.ones(5, dtype=float) * 0.25,
                wavelength_grid_nm=bad_profile.wavelength_grid_nm,
                channel_ids=bad_profile.channel_ids,
                dark_reference=None,
                white_reference=None,
                saturation_flags=None,
                valid_mask=None,
            ),
        )
        # Out-of-domain sample (extreme pH & moisture far outside synthetic domain)
        ood_meas = replace(
            test_recs[3],
            covariates=replace(test_recs[3].covariates, ph=3.5, moisture_volumetric_pct=62.0),
        )

        batch = analyzer.predict_batch(
            [valid_with_bd, missing_bd, invalid_meas, ood_meas],
            fail_fast=False,
            include_xai=True,
        )
        assert batch.n_total == 4
        assert batch.n_succeeded == 3
        assert batch.n_failed == 1
        assert batch.results[2] is None
        assert len(batch.errors) == 1
        assert batch.errors[0].index == 2
        assert batch.errors[0].sample_id == invalid_meas.sample_id

        # DEF-005 check on missing bulk_density sample: stock_mg_ha is None + MISSING_BULK_DENSITY
        miss_rep = batch.results[1]
        assert miss_rep is not None
        assert miss_rep.carbon_stock["stock_mg_ha"] is None
        assert QualityFlag.MISSING_BULK_DENSITY.value in miss_rep.carbon_stock["quality_flags"]

        # OOD sample must be flagged OUT_OF_DOMAIN with low confidence (< 0.5)
        ood_rep = batch.results[3]
        assert ood_rep is not None
        assert ood_rep.status == QualityStatus.OUT_OF_DOMAIN
        assert ood_rep.diagnostics["is_ood"] is True
        assert ood_rep.diagnostics["confidence_score"] < 0.5

        # Direct explain on 4+ samples computes HSIC (`status == "ok"`), while B=1 without background fails closed
        xai_batch = analyzer.explain(test_recs[:4])
        assert xai_batch.dataset_dependence_hsic.status in ("ok", "low_sample_warning")

        json.loads(batch.to_json())
