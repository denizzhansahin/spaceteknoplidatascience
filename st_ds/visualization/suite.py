"""Orchestrator generating the complete 26+ STDS scientific visualization suite, CSV/MD tables & dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from st_ds import SCHEMA_VERSION, __version__
from st_ds.api.analyzer import BatchPredictionResult, STSoilAnalyzer
from st_ds.artifact.bundle import LoadedArtifact, collect_dependency_versions
from st_ds.chemometrics.pipeline import ChemometricPipeline, PreprocessConfig
from st_ds.config import resolve_lang
from st_ds.domain.instrument import get_builtin_profile
from st_ds.domain.schemas import ArtifactMetadata, PredictionReport
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io.dataset import SoilDataset
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.reporting import generate_analysis_pdf_report
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import (
    BenchmarkReport,
    STSoilTrainer,
    TrainerConfig,
    _extract_target_vector,
    _fit_fold_chemometrics,
)
from st_ds.uncertainty.conformal import ConformalConfig, ConformalUncertaintyCalibrator
from st_ds.uncertainty.ood import OODBatchResult, OODConfig, OutOfDomainDetector
from st_ds.visualization.base import (
    DatasetArrays,
    PlotArtifactRecord,
    TableArtifactRecord,
    VisualizationConfig,
    extract_dataset_arrays,
)
from st_ds.visualization.model_plots import ModelPlotContext, generate_model_plots
from st_ds.visualization.science_plots import compute_spatial_kriging_bundle, generate_science_plots
from st_ds.visualization.spectral_plots import generate_spectral_plots
from st_ds.visualization.tables_and_dashboard import (
    build_and_export_tables,
    generate_optional_interactive_3d_html,
    plot_26_covariate_3d_response_surface,
    render_html_dashboard,
)
from st_ds.xai.engine import XAIExplanationReport


@dataclass(frozen=True)
class VisualizationReport:
    """Structured manifest returned by `generate_full_visualization_suite`."""

    output_dir: str
    manifest_path: str
    dashboard_path: str | None
    dataset_id: str
    dataset_source: str
    target_name: str
    n_plots: int
    n_3d_plots: int
    n_tables: int
    plots: tuple[PlotArtifactRecord, ...]
    tables: tuple[TableArtifactRecord, ...]
    interactive_3d_html: tuple[dict[str, str], ...] = field(default_factory=tuple)
    pdf_report_path: str | None = None
    lang: str = "tr"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stds_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "dashboard_path": self.dashboard_path,
            "pdf_report_path": self.pdf_report_path,
            "lang": self.lang,
            "dataset_id": self.dataset_id,
            "dataset_source": self.dataset_source,
            "target_name": self.target_name,
            "n_plots": self.n_plots,
            "n_3d_plots": self.n_3d_plots,
            "n_tables": self.n_tables,
            "plots": [p.to_dict() for p in self.plots],
            "tables": [t.to_dict() for t in self.tables],
            "interactive_3d_html": list(self.interactive_3d_html),
        }


def _run_benchmark_if_feasible(
    dataset: SoilDataset,
    *,
    config: VisualizationConfig,
) -> BenchmarkReport | None:
    """Run grouped cross-validation benchmark when sufficient sites and finite labels exist."""
    n_sites = len(dataset.unique_site_ids)
    if n_sites < 2 or len(dataset.records) < 8:
        return None
    try:
        y_vals = _extract_target_vector(dataset.records, config.target_name)
    except Exception:
        return None
    if not np.all(np.isfinite(y_vals)):
        return None

    t_cfg = TrainerConfig(
        target_name=config.target_name,
        preprocess_config=PreprocessConfig(suppression_mode="none"),
        hybrid_epochs=25,
        seed=config.seed,
    )
    trainer = STSoilTrainer(t_cfg)
    folds = GroupedSplitBuilder.group_kfold_split(
        dataset,
        n_splits=min(config.n_folds, n_sites),
        group_by="site_id",
    )
    return trainer.run_benchmark(
        dataset,
        split_folds=folds,
        include_optional_models=config.include_optional_models,
    )


def _build_in_memory_analyzer(
    dataset: SoilDataset,
    *,
    benchmark: BenchmarkReport | None,
    config: VisualizationConfig,
    wet_dataset: SoilDataset | None = None,
    dry_dataset: SoilDataset | None = None,
) -> STSoilAnalyzer:
    """Train a temporary in-memory LoadedArtifact when no saved artifact bundle was supplied."""
    pre_cfg = PreprocessConfig(
        suppression_mode="epo" if (wet_dataset is not None and dry_dataset is not None) else "none"
    )
    t_cfg = TrainerConfig(
        target_name=config.target_name,
        preprocess_config=pre_cfg,
        hybrid_epochs=25,
        seed=config.seed,
    )
    if wet_dataset is not None and dry_dataset is not None:
        chemo = ChemometricPipeline(pre_cfg).fit_from_datasets(wet_dataset, dry_dataset)
    else:
        chemo = _fit_fold_chemometrics(t_cfg, dataset.records)
    fusion = FeatureFusionPipeline(t_cfg.fusion_config).fit(dataset.records, chemometric_pipeline=chemo)
    fused = fusion.build_feature_vector(dataset.records, chemometric_pipeline=chemo)
    y_all = _extract_target_vector(dataset.records, config.target_name)

    trainer = STSoilTrainer(t_cfg)
    factories = trainer.default_model_factories(include_optional=True)
    chosen_key = benchmark.evidence_ranking[0] if (benchmark is not None and benchmark.evidence_ranking) else "plsr"
    if chosen_key not in factories:
        chosen_key = "plsr"

    model = factories[chosen_key](fused.X_spectral.shape[1])
    model.set_metadata(
        feature_names=fused.fused_feature_names,
        target_names=(config.target_name,),
        dataset_source=dataset.dataset_source,
    )
    groups = np.asarray([r.site_id for r in dataset.records], dtype=str)
    if isinstance(model, STSoilHybridNetRegressor):
        model.fit_multitask(fused.X_spectral, fused.X_covariates, soc=y_all)
    else:
        try:
            model.fit(fused.X_fused, y_all, groups=groups)  # type: ignore[call-arg]
        except TypeError:
            model.fit(fused.X_fused, y_all)

    conformal = None
    if benchmark is not None and chosen_key in benchmark.entries:
        entry = benchmark.entries[chosen_key]
        if len(dataset.records) >= 5 and entry.oof_predictions.size == len(dataset.records):
            conformal = ConformalUncertaintyCalibrator(ConformalConfig()).fit(
                y_all,
                entry.oof_predictions,
                pred_std_cal=entry.oof_uncertainties,
            )
    if conformal is None and len(dataset.records) >= 5:
        preds, stds = model.predict_distribution(fused.X_fused)
        conformal = ConformalUncertaintyCalibrator(ConformalConfig()).fit(
            y_all,
            np.asarray(preds, dtype=float).reshape(-1),
            pred_std_cal=np.asarray(stds, dtype=float).reshape(-1),
        )

    ood = OutOfDomainDetector(OODConfig()).fit(fused.X_fused, measurements=dataset.records)
    inst_prof = fusion.instrument_profile or get_builtin_profile(dataset.records[0].instrument_profile_id)
    meta = ArtifactMetadata(
        artifact_id=f"stds_viz_{chosen_key}_{dataset.dataset_id}",
        semantic_version=__version__,
        schema_version=SCHEMA_VERSION,
        instrument_profile_id=inst_prof.profile_id,
        calibration_id=dataset.records[0].calibration_id,
        pipeline_version=pre_cfg.pipeline_version,
        dataset_hash=dataset.compute_sha256(),
        dependency_versions=collect_dependency_versions(),
        seed=config.seed,
    )
    loaded = LoadedArtifact(
        bundle_dir=Path("."),
        manifest={"artifact_id": meta.artifact_id, "semantic_version": meta.semantic_version},
        model_card={},
        metadata=meta,
        instrument_profile=inst_prof,
        target_schema=t_cfg.target_schema,
        chemometric_pipeline=chemo,
        fusion_pipeline=fusion,
        model=model,
        conformal_calibrator=conformal,
        ood_detector=ood,
        feature_names=fused.fused_feature_names,
        background_features=fused.X_fused,
        background_targets=y_all,
        bundle_sha256="in_memory_viz_session",
    )
    return STSoilAnalyzer(loaded_artifact=loaded)


def _extract_predictions_from_batch(
    batch: BatchPredictionResult,
    arrays: DatasetArrays,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned y_true, y_pred, pred_std, ci_lower, ci_upper arrays from BatchPredictionResult."""
    n = len(arrays.records)
    if target_name == "sic_pct":
        y_true = arrays.sic.copy()
    elif target_name == "tc_pct":
        y_true = arrays.tc.copy()
    else:
        y_true = arrays.soc.copy()

    y_pred = np.full(n, np.nan, dtype=float)
    pred_std = np.full(n, np.nan, dtype=float)
    ci_lo = np.full(n, np.nan, dtype=float)
    ci_hi = np.full(n, np.nan, dtype=float)

    std_key = f"{target_name.replace('_pct', '')}_std_pct"
    for i, rep in enumerate(batch.results):
        if rep is None:
            continue
        cm = rep.carbon_metrics
        val = cm.get(target_name) if cm.get(target_name) is not None else cm.get("soc_pct")
        sd = cm.get(std_key) if cm.get(std_key) is not None else cm.get("soc_std_pct")
        ci = cm.get("confidence_interval") or {}
        if val is not None:
            y_pred[i] = float(val)
        if sd is not None:
            pred_std[i] = float(sd)
        if "lower_pct" in ci and "upper_pct" in ci:
            ci_lo[i] = float(ci["lower_pct"])
            ci_hi[i] = float(ci["upper_pct"])
        elif val is not None and sd is not None:
            ci_lo[i] = max(0.0, float(val) - 1.645 * float(sd))
            ci_hi[i] = float(val) + 1.645 * float(sd)

    return y_true, y_pred, pred_std, ci_lo, ci_hi


def generate_full_visualization_suite(
    dataset: SoilDataset,
    output_dir: str | Path,
    *,
    artifact_path: str | Path | None = None,
    analyzer: STSoilAnalyzer | None = None,
    benchmark: BenchmarkReport | None = None,
    wet_dataset: SoilDataset | None = None,
    dry_dataset: SoilDataset | None = None,
    config: VisualizationConfig | None = None,
) -> VisualizationReport:
    """Generate the complete 26+ scientific plots (including 4 3D plots), 5 tables, and HTML dashboard."""
    cfg = config or VisualizationConfig()
    out_root = Path(output_dir)
    plots_dir = out_root / "plots"
    tables_dir = out_root / "tables"
    interactive_dir = out_root / "interactive"
    out_root.mkdir(parents=True, exist_ok=True)

    arrays: DatasetArrays = extract_dataset_arrays(dataset)

    if benchmark is None:
        benchmark = _run_benchmark_if_feasible(dataset, config=cfg)

    if analyzer is None:
        if artifact_path is not None:
            analyzer = STSoilAnalyzer.from_artifact(artifact_path)
        else:
            analyzer = _build_in_memory_analyzer(
                dataset,
                benchmark=benchmark,
                config=cfg,
                wet_dataset=wet_dataset,
                dry_dataset=dry_dataset,
            )

    loaded = analyzer._artifact
    fused = loaded.fusion_pipeline.build_feature_vector(
        dataset.records,
        chemometric_pipeline=loaded.chemometric_pipeline,
    )
    batch_res = analyzer.predict_batch(
        dataset.records,
        fail_fast=False,
        include_xai=False,
        include_stock=True,
    )
    ood_batch: OODBatchResult | None = None
    if loaded.ood_detector is not None:
        ood_batch = loaded.ood_detector.detect(
            fused.X_fused,
            measurements=dataset.records,
            fail_on_ood=False,
        )

    y_true, y_pred, pred_std, ci_lo, ci_hi = _extract_predictions_from_batch(
        batch_res,
        arrays,
        cfg.target_name,
    )

    # Prefer OOF predictions for parity / conformal calibration curves if available and aligned
    best_entry = (
        benchmark.entries[benchmark.evidence_ranking[0]]
        if (benchmark is not None and benchmark.evidence_ranking)
        else None
    )
    if (
        best_entry is not None
        and best_entry.oof_predictions.size == len(dataset.records)
        and np.all(np.isfinite(y_true))
        and artifact_path is None
    ):
        y_eval_pred = best_entry.oof_predictions
        y_eval_std = best_entry.oof_uncertainties
        if loaded.conformal_calibrator is not None:
            iv = loaded.conformal_calibrator.predict_interval(y_eval_pred, pred_std=y_eval_std)
            ci_lo_eval, ci_hi_eval = iv.lower_bound, iv.upper_bound
        else:
            ci_lo_eval = np.maximum(0.0, y_eval_pred - 1.645 * y_eval_std)
            ci_hi_eval = y_eval_pred + 1.645 * y_eval_std
    else:
        y_eval_pred = y_pred
        y_eval_std = pred_std
        ci_lo_eval = ci_lo
        ci_hi_eval = ci_hi

    ctx = ModelPlotContext(
        target_name=cfg.target_name,
        y_true=y_true,
        y_pred=y_eval_pred,
        pred_std=y_eval_std,
        ci_lower=ci_lo_eval,
        ci_upper=ci_hi_eval,
        benchmark=benchmark,
        best_entry=best_entry,
        reports=tuple(batch_res.results),
        batch_result=batch_res,
        fused_features=fused.X_fused,
        ood_batch=ood_batch,
        conformal_calibrator=loaded.conformal_calibrator,
    )

    xai_report: XAIExplanationReport | None = None
    try:
        xai_report = analyzer.explain(dataset.records)
    except Exception:
        xai_report = None

    spatial_target = np.where(np.isfinite(y_true), y_true, y_pred)
    spatial_bundle = compute_spatial_kriging_bundle(arrays, spatial_target, grid_size=24)

    all_plots: list[PlotArtifactRecord] = []
    # Plots 01 - 07 (Spectral, Chemometrics, EPO, 3D Spectral, QC, EIS/Covariates)
    all_plots.extend(
        generate_spectral_plots(
            arrays,
            plots_dir=plots_dir,
            config=cfg,
            dataset_source=dataset.dataset_source,
            chem_pipeline=loaded.chemometric_pipeline,
            wet_dataset=wet_dataset,
            dry_dataset=dry_dataset,
        )
    )
    # Plots 08 - 16 (Benchmark, Parity, Bland-Altman, Residuals, Subgroups, Conformal, Uncertainty, OOD, 3D PCA)
    all_plots.extend(
        generate_model_plots(
            ctx,
            arrays,
            plots_dir=plots_dir,
            config=cfg,
            dataset_source=dataset.dataset_source,
        )
    )
    # Plots 17 - 23 (XAI IG/Sens, HSIC, Interactions, Fractions/Closure, Stock/Tornado, Variogram, 2D+3D Kriging)
    all_plots.extend(
        generate_science_plots(
            xai_report=xai_report,
            reports=tuple(batch_res.results),
            arrays=arrays,
            target_name=cfg.target_name,
            spatial=spatial_bundle,
            plots_dir=plots_dir,
            config=cfg,
            dataset_source=dataset.dataset_source,
        )
    )
    # Tables (5 CSV + MD) & Visual Table Figures #24 - #25
    tables, visual_table_plots = build_and_export_tables(
        dataset=dataset,
        arrays=arrays,
        ctx=ctx,
        xai_report=xai_report,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        config=cfg,
        dataset_source=dataset.dataset_source,
    )
    all_plots.extend(visual_table_plots)
    # Plot 26 (3D Covariate Response Surface)
    all_plots.append(
        plot_26_covariate_3d_response_surface(
            ctx,
            arrays,
            plots_dir=plots_dir,
            config=cfg,
            dataset_source=dataset.dataset_source,
        )
    )

    eff_lang = resolve_lang(cfg.lang)

    interactive_3d: list[dict[str, str]] = []
    if cfg.include_interactive_3d:
        interactive_3d = generate_optional_interactive_3d_html(
            interactive_dir=interactive_dir,
            arrays=arrays,
            ctx=ctx,
            spatial=spatial_bundle,
            lang=eff_lang,
        )

    artifact_info = {
        "artifact_id": loaded.metadata.artifact_id,
        "semantic_version": loaded.metadata.semantic_version,
        "bundle_sha256": loaded.bundle_sha256,
        "model_name": loaded.model.model_name,
    }

    pdf_report_path: str | None = None
    if cfg.include_pdf_report:
        pdf_out = out_root / "reports" / "stds_analysis_report.pdf"
        generate_analysis_pdf_report(
            pdf_out,
            dataset=dataset,
            arrays=arrays,
            ctx=ctx,
            plots=all_plots,
            tables=tables,
            xai_report=xai_report,
            artifact_info=artifact_info,
            chem_config=loaded.chemometric_pipeline.config.to_dict(),
            lang=eff_lang,
        )
        pdf_report_path = str(pdf_out)

    dash_path: str | None = None
    if cfg.include_html_dashboard:
        dash = render_html_dashboard(
            output_dir=out_root,
            dataset=dataset,
            plots=all_plots,
            tables=tables,
            interactive_3d=interactive_3d,
            ctx=ctx,
            artifact_info=artifact_info,
            pdf_report_path=pdf_report_path,
            lang=eff_lang,
        )
        dash_path = str(dash)

    manifest_path = out_root / "viz_manifest.json"
    n_3d = sum(1 for p in all_plots if p.is_3d)
    report = VisualizationReport(
        output_dir=str(out_root),
        manifest_path=str(manifest_path),
        dashboard_path=dash_path,
        dataset_id=dataset.dataset_id,
        dataset_source=dataset.dataset_source,
        target_name=cfg.target_name,
        n_plots=len(all_plots),
        n_3d_plots=n_3d,
        n_tables=len(tables),
        plots=tuple(all_plots),
        tables=tuple(tables),
        interactive_3d_html=tuple(interactive_3d),
        pdf_report_path=pdf_report_path,
        lang=eff_lang,
    )
    manifest_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return report
