"""End-to-end Python quickstart using synthetic STDS soil datasets.

IMPORTANT: Synthetic datasets validate pipeline schemas, preprocessing, training,
uncertainty calibration, XAI, carbon stock propagation, and artifact integrity ONLY.
They do NOT represent real laboratory or field soil prediction accuracy (Risk 9).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from st_ds import (
    SCHEMA_VERSION,
    STSoilAnalyzer,
    VisualizationConfig,
    __version__,
    generate_full_visualization_suite,
    inspect_artifact,
    save_artifact,
)
from st_ds.artifact.bundle import collect_dependency_versions
from st_ds.carbon.stock import compute_layer_stock
from st_ds.domain.instrument import get_builtin_profile
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.domain.schemas import ArtifactMetadata
from st_ds.features.fusion import FeatureFusionPipeline
from st_ds.io.dataset import SoilDataset, save_dataset_json
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.io.validator import DataValidator
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.spatial.kriging import ordinary_kriging
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import (
    STSoilTrainer,
    TrainerConfig,
    _extract_target_vector,
    _fit_fold_chemometrics,
)
from st_ds.uncertainty.conformal import ConformalConfig, ConformalUncertaintyCalibrator
from st_ds.uncertainty.ood import OODConfig, OutOfDomainDetector


def main() -> None:
    print(f"STDS version: {__version__}")

    # 1. Generate synthetic dataset (development sites + unseen holdout records)
    gen = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id="quickstart_synthetic_v1",
            profile_id="visnir_24ch_v1",
            n_samples=72,
            n_sites=6,
            seed=42,
        )
    )
    dataset = gen.generate_dataset()
    validation = DataValidator.validate_dataset(dataset, for_training=True)
    assert validation.is_valid, validation.errors
    print(f"Generated {len(dataset.records)} records across {len(dataset.unique_site_ids)} sites.")

    train_recs = [r for r in dataset.records if r.site_id in {"site_01", "site_02", "site_03", "site_04"}]
    test_recs = [r for r in dataset.records if r.site_id in {"site_05", "site_06"}]
    GroupedSplitBuilder.verify_no_leakage(train_recs, test_recs, enforce_site_disjoint=True)

    with tempfile.TemporaryDirectory(prefix="stds_quickstart_") as tmp_dir:
        tmp = Path(tmp_dir)
        dataset_path = save_dataset_json(dataset, tmp / "synthetic_soil.json")
        print(f"Saved synthetic JSON dataset to: {dataset_path}")

        # 2. Grouped-CV benchmark and model training
        trainer_cfg = TrainerConfig(target_name="soc_pct", hybrid_epochs=8, seed=42)
        trainer = STSoilTrainer(trainer_cfg)
        train_ds = SoilDataset(
            dataset_id="quickstart_train",
            records=train_recs,
            dataset_source=dataset.dataset_source,
        )
        bench = trainer.run_benchmark(train_ds, n_splits=3, group_by="site_id")
        print("Grouped-CV Summary:", json.dumps(bench.summary_table(), indent=2))

        best_key = bench.evidence_ranking[0]
        best_entry = bench.entries[best_key]

        # 3. Fit full deployment bundle components and export verified artifact
        chemo = _fit_fold_chemometrics(trainer_cfg, train_ds.records)
        fusion = FeatureFusionPipeline(trainer_cfg.fusion_config).fit(
            train_ds.records, chemometric_pipeline=chemo
        )
        fused = fusion.build_feature_vector(train_ds.records, chemometric_pipeline=chemo)
        y_train = _extract_target_vector(train_ds.records, "soc_pct")

        factories = trainer.default_model_factories(include_optional=True)
        model = factories[best_key](fused.X_spectral.shape[1])
        model.set_metadata(
            feature_names=fused.fused_feature_names,
            target_names=("soc_pct",),
            dataset_source=train_ds.dataset_source,
        )
        if isinstance(model, STSoilHybridNetRegressor):
            model.fit_multitask(fused.X_spectral, fused.X_covariates, soc=y_train)
        else:
            model.fit(fused.X_fused, y_train)

        conformal = ConformalUncertaintyCalibrator(ConformalConfig()).fit(
            y_train,
            best_entry.oof_predictions,
            pred_std_cal=best_entry.oof_uncertainties,
        )
        ood = OutOfDomainDetector(OODConfig()).fit(fused.X_fused, measurements=train_ds.records)
        inst_prof = get_builtin_profile("visnir_24ch_v1")

        bundle_dir = tmp / "artifact_bundle"
        meta = ArtifactMetadata(
            artifact_id=f"quickstart_{best_key}",
            semantic_version=__version__,
            schema_version=SCHEMA_VERSION,
            instrument_profile_id=inst_prof.profile_id,
            calibration_id=train_ds.records[0].calibration_id,
            pipeline_version=trainer_cfg.preprocess_config.pipeline_version,
            dataset_hash=train_ds.compute_sha256(),
            dependency_versions=collect_dependency_versions(),
            seed=42,
        )
        save_artifact(
            bundle_dir,
            model=model,
            metadata=meta,
            instrument_profile=inst_prof,
            chemometric_pipeline=chemo,
            fusion_pipeline=fusion,
            conformal_calibrator=conformal,
            ood_detector=ood,
            validation_metrics=best_entry.oof_metrics.to_dict(),
            background_features=fused.X_fused[:16],
            background_targets=y_train[:16],
            target_schema=trainer_cfg.target_schema,
        )
        inspected = inspect_artifact(bundle_dir)
        print("Verified artifact ID:", inspected["metadata"]["artifact_id"])

        # 4. Offline Inference via STSoilAnalyzer (single + batch)
        analyzer = STSoilAnalyzer.from_artifact(bundle_dir)
        single_report = analyzer.predict(test_recs[0], include_xai=True, include_stock=True)
        print(
            f"Single Prediction ({test_recs[0].sample_id}): "
            f"SOC={single_report.carbon_metrics['soc_pct']:.3f}%, status={single_report.status.value}"
        )

        # Batch inference including one mismatched-profile item to show fail-isolated reporting
        led_gen = SyntheticGeneratorV2(
            SyntheticConfig(
                dataset_id="quickstart_led_mismatch",
                profile_id="led_5band_v1",
                n_samples=6,
                n_sites=2,
                seed=7,
            )
        )
        bad_item = led_gen.generate_dataset().records[0]
        batch_res = analyzer.predict_batch([test_recs[0], bad_item, test_recs[2]], fail_fast=False)
        print(f"Batch Inference: succeeded={batch_res.n_succeeded}, failed={batch_res.n_failed}")

        # 5. Fail-closed carbon stock check (DEF-005: missing bulk_density -> None stock)
        stock_ok = compute_layer_stock(
            carbon_value=float(single_report.carbon_metrics["soc_pct"]),
            bulk_density=1.28,
            depth_top=0.0,
            depth_bottom=30.0,
            carbon_std=0.15,
            bulk_density_std=0.05,
        )
        stock_missing_bd = compute_layer_stock(
            carbon_value=float(single_report.carbon_metrics["soc_pct"]),
            bulk_density=None,
            depth_top=0.0,
            depth_bottom=30.0,
        )
        assert stock_missing_bd.stock_mg_ha is None
        assert QualityFlag.MISSING_BULK_DENSITY in stock_missing_bd.quality_flags
        assert stock_missing_bd.quality_status == QualityStatus.INSUFFICIENT_DATA
        print(
            f"Carbon Stock OK={stock_ok.stock_mg_ha:.2f} Mg/ha | "
            f"Missing BD={stock_missing_bd.stock_mg_ha} ({stock_missing_bd.quality_status.value})"
        )

        # 6. Optional Ordinary Kriging on synthetic sample coordinates
        coords = np.asarray(
            [[32.85 + 0.002 * i, 39.92 + 0.001 * (i % 4)] for i in range(12)],
            dtype=float,
        )
        vals = np.linspace(1.1, 2.4, 12) + 0.05 * np.sin(np.arange(12))
        queries = np.asarray([[32.86, 39.921], [32.95, 39.98]], dtype=float)
        krig = ordinary_kriging(coords, vals, queries)
        print(
            f"Ordinary Kriging predictions={np.round(krig.predicted_values, 3).tolist()}, "
            f"extrapolated_mask={krig.is_extrapolated_mask.tolist()}"
        )

        # 7. Scientific Visualization Suite (26+ plots incl. 4x 3D, 5 CSV/MD tables, HTML dashboard)
        viz_report = generate_full_visualization_suite(
            train_ds,
            tmp / "viz_suite",
            analyzer=analyzer,
            benchmark=bench,
            config=VisualizationConfig(target_name="soc_pct", n_folds=3, dpi=110),
        )
        print(
            f"Visualization Suite generated: plots={viz_report.n_plots} "
            f"(3D={viz_report.n_3d_plots}), tables={viz_report.n_tables}, "
            f"dashboard={viz_report.dashboard_path}"
        )


if __name__ == "__main__":
    main()
