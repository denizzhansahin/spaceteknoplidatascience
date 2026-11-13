"""Release readiness harnesses: scientific holdout, deterministic replay, and profiling."""

from __future__ import annotations

import platform
import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from st_ds import SCHEMA_VERSION, __version__
from st_ds.api.analyzer import STSoilAnalyzer
from st_ds.artifact.bundle import collect_dependency_versions, load_artifact, save_artifact
from st_ds.domain.errors import ArtifactCompatibilityError, SchemaError
from st_ds.domain.instrument import get_builtin_profile
from st_ds.domain.schemas import ArtifactMetadata, Measurement
from st_ds.features.fusion import FeatureFusionPipeline
from st_ds.io.dataset import SoilDataset
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.training.metrics import compute_regression_metrics
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import (
    STSoilTrainer,
    TrainerConfig,
    _extract_target_vector,
    _fit_fold_chemometrics,
)
from st_ds.uncertainty.conformal import ConformalConfig, ConformalUncertaintyCalibrator
from st_ds.uncertainty.ood import OODConfig, OutOfDomainDetector

SYNTHETIC_DISCLAIMER = "SYNTHETIC_PIPELINE_VERIFICATION_ONLY — NOT_FIELD_PERFORMANCE"


def collect_environment_manifest() -> dict[str, Any]:
    """Capture host OS/CPU architecture, Python version, and pinned package versions."""
    import importlib.metadata as imp_meta

    deps = dict(collect_dependency_versions())
    for dist_name in ("pandas", "pydantic"):
        try:
            deps[dist_name] = imp_meta.version(dist_name)
        except imp_meta.PackageNotFoundError:
            pass
    return {
        "stds_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "dependencies": deps,
    }


def _subset_dataset(dataset: SoilDataset, records: list[Measurement], *, subset_id: str) -> SoilDataset:
    return SoilDataset(
        dataset_id=subset_id,
        records=records,
        dataset_source=dataset.dataset_source,
        description=dataset.description,
        allow_mixed_profiles=dataset.allow_mixed_profiles,
    )


def _train_and_export_bundle(
    train_ds: SoilDataset,
    output_dir: Path,
    *,
    seed: int = 42,
    hybrid_epochs: int = 8,
    model_key: str = "auto",
    n_folds: int = 3,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    """Train a leakage-verified model via STSoilTrainer and export a signed artifact bundle."""
    trainer_cfg = TrainerConfig(
        target_name="soc_pct",
        hybrid_epochs=hybrid_epochs,
        seed=seed,
    )
    trainer = STSoilTrainer(trainer_cfg)
    factories = trainer.default_model_factories(include_optional=True)

    if model_key != "auto":
        if model_key not in factories:
            raise SchemaError(f"Unknown model key '{model_key}'.", details={"supported": sorted(factories)})
        bench_factories = {model_key: factories[model_key]}
    else:
        bench_factories = factories

    n_sites = len(train_ds.unique_site_ids)
    folds = GroupedSplitBuilder.group_kfold_split(
        train_ds,
        n_splits=min(n_folds, n_sites),
        group_by="site_id",
    )
    bench = trainer.run_benchmark(
        train_ds,
        split_folds=folds,
        model_factories=bench_factories,
        include_optional_models=True,
    )
    chosen_key = bench.evidence_ranking[0] if model_key == "auto" else model_key
    chosen_entry = bench.entries[chosen_key]

    chemo = _fit_fold_chemometrics(trainer_cfg, train_ds.records)
    fusion = FeatureFusionPipeline(trainer_cfg.fusion_config).fit(
        train_ds.records, chemometric_pipeline=chemo
    )
    fused = fusion.build_feature_vector(train_ds.records, chemometric_pipeline=chemo)

    y_all = _extract_target_vector(train_ds.records, "soc_pct")
    final_model = factories[chosen_key](fused.X_spectral.shape[1])
    final_model.set_metadata(
        feature_names=fused.fused_feature_names,
        target_names=("soc_pct",),
        dataset_source=train_ds.dataset_source,
    )
    tr_groups = np.asarray([r.site_id for r in train_ds.records], dtype=str)
    if isinstance(final_model, STSoilHybridNetRegressor):
        final_model.fit_multitask(fused.X_spectral, fused.X_covariates, soc=y_all)
    else:
        try:
            final_model.fit(fused.X_fused, y_all, groups=tr_groups)  # type: ignore[call-arg]
        except TypeError:
            final_model.fit(fused.X_fused, y_all)

    conformal = ConformalUncertaintyCalibrator(ConformalConfig()).fit(
        y_all,
        chosen_entry.oof_predictions,
        pred_std_cal=chosen_entry.oof_uncertainties,
    )
    ood = OutOfDomainDetector(OODConfig()).fit(fused.X_fused, measurements=train_ds.records)
    inst_prof = fusion.instrument_profile or get_builtin_profile(train_ds.records[0].instrument_profile_id)

    meta = ArtifactMetadata(
        artifact_id=artifact_id or f"stds_{chosen_key}_{train_ds.dataset_id}",
        semantic_version=__version__,
        schema_version=SCHEMA_VERSION,
        instrument_profile_id=inst_prof.profile_id,
        calibration_id=train_ds.records[0].calibration_id,
        pipeline_version=trainer_cfg.preprocess_config.pipeline_version,
        dataset_hash=train_ds.compute_sha256(),
        dependency_versions=collect_dependency_versions(),
        seed=seed,
    )
    saved = save_artifact(
        output_dir,
        model=final_model,
        metadata=meta,
        instrument_profile=inst_prof,
        chemometric_pipeline=chemo,
        fusion_pipeline=fusion,
        conformal_calibrator=conformal,
        ood_detector=ood,
        validation_metrics=chosen_entry.oof_metrics.to_dict(),
        subgroup_metrics={
            k: {gk: gm.to_dict() for gk, gm in gv.items()}
            for k, gv in chosen_entry.subgroup_metrics.items()
        },
        background_features=fused.X_fused[: min(24, fused.X_fused.shape[0])],
        background_targets=y_all[: min(24, y_all.shape[0])],
        target_schema=trainer_cfg.target_schema,
    )
    return {
        "chosen_model": chosen_key,
        "benchmark": bench,
        "saved": saved,
        "bundle_dir": Path(output_dir),
    }


def run_scientific_holdout_check(
    *,
    seed: int = 42,
    n_samples: int = 96,
    n_sites: int = 6,
    holdout_site_count: int = 2,
    hybrid_epochs: int = 8,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify strict site-disjoint holdout generalization on synthetic data."""
    if holdout_site_count < 1 or n_sites - holdout_site_count < 3:
        raise SchemaError("Need at least 3 training sites and 1 holdout site.")

    gen = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id="stds_holdout_eval_v1",
            n_samples=n_samples,
            n_sites=n_sites,
            seed=seed,
        )
    )
    full_ds = gen.generate_dataset()
    sites = sorted(full_ds.unique_site_ids)
    train_sites = sites[:-holdout_site_count]
    holdout_sites = sites[-holdout_site_count:]

    train_recs = [r for r in full_ds.records if r.site_id in train_sites]
    holdout_recs = [r for r in full_ds.records if r.site_id in holdout_sites]
    GroupedSplitBuilder.verify_no_leakage(train_recs, holdout_recs, enforce_site_disjoint=True)

    train_ds = _subset_dataset(full_ds, train_recs, subset_id="stds_holdout_train")
    holdout_ds = _subset_dataset(full_ds, holdout_recs, subset_id="stds_holdout_test")

    def _execute(target_dir: Path) -> dict[str, Any]:
        trained = _train_and_export_bundle(
            train_ds,
            target_dir / "bundle",
            seed=seed,
            hybrid_epochs=hybrid_epochs,
            model_key="auto",
            n_folds=min(3, len(train_sites)),
        )
        analyzer = STSoilAnalyzer.from_artifact(trained["bundle_dir"])
        batch = analyzer.predict_batch(holdout_ds.records, include_stock=True)
        if batch.n_failed != 0:
            raise SchemaError(
                f"Holdout inference encountered {batch.n_failed} unexpected failures.",
                details={"errors": [e.to_dict() for e in batch.errors]},
            )

        y_true = _extract_target_vector(holdout_ds.records, "soc_pct")
        y_pred = np.asarray(
            [rep.carbon_metrics["soc_pct"] for rep in batch.results if rep is not None],
            dtype=float,
        )
        lowers = np.asarray(
            [
                rep.carbon_metrics["confidence_interval"]["lower_pct"]
                for rep in batch.results
                if rep is not None
            ],
            dtype=float,
        )
        uppers = np.asarray(
            [
                rep.carbon_metrics["confidence_interval"]["upper_pct"]
                for rep in batch.results
                if rep is not None
            ],
            dtype=float,
        )
        empirical_cov = float(np.mean((y_true >= lowers) & (y_true <= uppers)))
        ood_rate = float(
            np.mean(
                [
                    bool(rep.diagnostics.get("is_ood", False))
                    for rep in batch.results
                    if rep is not None
                ]
            )
        )
        holdout_metrics = compute_regression_metrics(y_true, y_pred)
        return {
            "disclaimer": SYNTHETIC_DISCLAIMER,
            "is_synthetic": True,
            "seed": seed,
            "n_train": len(train_recs),
            "n_holdout": len(holdout_recs),
            "train_sites": train_sites,
            "holdout_sites": holdout_sites,
            "selected_model": trained["chosen_model"],
            "bundle_sha256": analyzer.artifact.bundle_sha256,
            "oof_ranking": trained["benchmark"].summary_table(),
            "holdout_metrics": holdout_metrics.to_dict(),
            "empirical_ci90_coverage": empirical_cov,
            "holdout_ood_rate": ood_rate,
        }

    if work_dir is not None:
        return _execute(Path(work_dir))
    with tempfile.TemporaryDirectory(prefix="stds_holdout_") as tmp:
        return _execute(Path(tmp))


def verify_deterministic_replay(
    *,
    seed: int = 42,
    n_samples: int = 48,
    hybrid_epochs: int = 6,
    model_key: str = "st_soil_hybrid_net",
) -> dict[str, Any]:
    """Verify two independent runs with identical seeds yield identical weights and predictions."""
    ds = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id="stds_replay_check",
            n_samples=n_samples,
            n_sites=4,
            seed=seed,
        )
    ).generate_dataset()

    with tempfile.TemporaryDirectory(prefix="stds_replay_") as tmp:
        root = Path(tmp)
        run1 = _train_and_export_bundle(
            ds,
            root / "run1",
            seed=seed,
            hybrid_epochs=hybrid_epochs,
            model_key=model_key,
            n_folds=3,
            artifact_id="stds_replay_fixed_id",
        )
        run2 = _train_and_export_bundle(
            ds,
            root / "run2",
            seed=seed,
            hybrid_epochs=hybrid_epochs,
            model_key=model_key,
            n_folds=3,
            artifact_id="stds_replay_fixed_id",
        )

        # Verify weight bytes on disk & loaded predictions on identical held records
        w1 = (run1["bundle_dir"] / "model_weights.safetensors").read_bytes()
        w2 = (run2["bundle_dir"] / "model_weights.safetensors").read_bytes()
        weights_identical = w1 == w2

        an1 = STSoilAnalyzer.from_artifact(run1["bundle_dir"])
        an2 = STSoilAnalyzer.from_artifact(run2["bundle_dir"])
        probe = ds.records[:10]
        b1 = an1.predict_batch(probe, include_stock=True)
        b2 = an2.predict_batch(probe, include_stock=True)
        p1 = np.asarray([r.carbon_metrics["soc_pct"] for r in b1.results if r is not None], dtype=float)
        p2 = np.asarray([r.carbon_metrics["soc_pct"] for r in b2.results if r is not None], dtype=float)
        max_pred_diff = float(np.max(np.abs(p1 - p2)))

        if max_pred_diff > 1e-12:
            raise ArtifactCompatibilityError(
                f"Deterministic replay failed: prediction discrepancy {max_pred_diff} > 1e-12.",
                details={"max_prediction_diff": max_pred_diff, "weights_identical": weights_identical},
            )

        return {
            "seed": seed,
            "model_key": model_key,
            "weights_bytes_identical": weights_identical,
            "max_prediction_diff": max_pred_diff,
            "n_probe_samples": len(probe),
            "passed": max_pred_diff <= 1e-12,
        }


def profile_inference_and_load(
    *,
    seed: int = 42,
    n_samples: int = 48,
    n_warmup: int = 2,
    n_repeats: int = 12,
) -> dict[str, Any]:
    """Measure cold artifact load latency, single/batch inference ms, and peak RAM on this host."""
    ds = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id="stds_perf_profile",
            n_samples=n_samples,
            n_sites=4,
            seed=seed,
        )
    ).generate_dataset()

    with tempfile.TemporaryDirectory(prefix="stds_prof_") as tmp:
        bundle_dir = Path(tmp) / "bundle"
        _train_and_export_bundle(
            ds,
            bundle_dir,
            seed=seed,
            hybrid_epochs=5,
            model_key="st_soil_hybrid_net",
            n_folds=3,
        )

        tracemalloc.start()
        t_load0 = time.perf_counter()
        loaded = load_artifact(bundle_dir)
        analyzer = STSoilAnalyzer(loaded_artifact=loaded)
        cold_load_ms = (time.perf_counter() - t_load0) * 1000.0

        sample = ds.records[0]
        for _ in range(n_warmup):
            analyzer.predict(sample, include_stock=True)

        single_latencies_ms: list[float] = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            analyzer.predict(sample, include_stock=True)
            single_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        t_b0 = time.perf_counter()
        batch_res = analyzer.predict_batch(ds.records[:24], include_stock=True)
        batch_total_ms = (time.perf_counter() - t_b0) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports ru_maxrss in bytes, Linux in kilobytes
        rss_divisor = (1024.0 * 1024.0) if sys.platform == "darwin" else 1024.0
        max_rss_mb = float(usage.ru_maxrss) / rss_divisor

        lat_arr = np.asarray(single_latencies_ms, dtype=float)
        return {
            "host_machine": platform.machine(),
            "host_platform": platform.platform(),
            "cold_load_ms": float(cold_load_ms),
            "single_predict_p50_ms": float(np.percentile(lat_arr, 50.0)),
            "single_predict_p95_ms": float(np.percentile(lat_arr, 95.0)),
            "batch_size": int(batch_res.n_total),
            "batch_total_ms": float(batch_total_ms),
            "batch_ms_per_sample": float(batch_total_ms / max(batch_res.n_total, 1)),
            "peak_tracemalloc_mb": float(peak_bytes / (1024.0 * 1024.0)),
            "process_max_rss_mb": max_rss_mb,
            "note": "Measured strictly on current host hardware; Raspberry Pi 5 edge latency requires on-device execution.",
        }
