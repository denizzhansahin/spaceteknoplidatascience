"""Unified STDS Command-Line Interface (`st-ds`, Section 27 & pyproject.toml).

Uses the exact same `st_ds.io`, `st_ds.training`, `st_ds.artifact`, and `st_ds.api`
components without duplicating analytical logic. Raw Python tracebacks are suppressed
unless `--debug` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from st_ds import SCHEMA_VERSION, __version__
from st_ds.api.analyzer import STSoilAnalyzer
from st_ds.artifact.bundle import collect_dependency_versions, inspect_artifact, save_artifact
from st_ds.chemometrics.pipeline import ChemometricPipeline, PreprocessConfig
from st_ds.config import get_runtime_config, resolve_lang
from st_ds.domain.errors import ReportSerializationError, STDSError, SchemaError
from st_ds.domain.instrument import get_builtin_profile, resolve_instrument_profile
from st_ds.domain.schemas import ArtifactMetadata, Measurement
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io.dataset import (
    SoilDataset,
    load_dataset_csv,
    load_dataset_json,
    measurement_from_dict,
)
from st_ds.io.validator import DataValidator
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.reporting import generate_prediction_pdf_report
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import (
    STSoilTrainer,
    TrainerConfig,
    _extract_target_vector,
    _fit_fold_chemometrics,
)
from st_ds.uncertainty.conformal import ConformalConfig, ConformalUncertaintyCalibrator
from st_ds.uncertainty.ood import OODConfig, OutOfDomainDetector
from st_ds.visualization import VisualizationConfig, generate_full_visualization_suite

MODEL_CLI_ALIASES: dict[str, str] = {
    "plsr": "plsr",
    "kernel_ridge": "kernel_ridge_rbf",
    "kernel_ridge_rbf": "kernel_ridge_rbf",
    "random_forest": "random_forest",
    "local_plsr": "local_plsr_mbl",
    "local_plsr_mbl": "local_plsr_mbl",
    "hybrid_net": "st_soil_hybrid_net",
    "st_soil_hybrid_net": "st_soil_hybrid_net",
}


def _load_dataset_auto(
    path: str | Path,
    *,
    profile_id: str | None = None,
    target_columns: Sequence[str] | None = None,
) -> SoilDataset:
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"Input dataset file not found: '{p}'.")
    prof = resolve_instrument_profile(profile_id) if profile_id else None
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return load_dataset_csv(p, profile=prof, target_columns=target_columns)
    ds = load_dataset_json(p)
    if prof is not None and ds.instrument_profile is None:
        ds.instrument_profile = prof
    return ds


def _load_measurements_for_predict(
    path: str | Path,
    *,
    profile_id: str | None = None,
    fallback_profile: Any = None,
) -> tuple[list[Measurement], bool]:
    """Load either a SoilDataset (JSON/CSV) or a single Measurement JSON dict.

    Returns ``(measurements, is_single_record)``.
    """
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"Input prediction file not found: '{p}'.")
    if p.suffix.lower() == ".csv":
        prof = resolve_instrument_profile(profile_id) if profile_id else fallback_profile
        ds = load_dataset_csv(p, profile=prof)
        return ds.records, False

    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "records" in raw and "manifest" in raw:
        ds = load_dataset_json(p)
        return ds.records, False
    if isinstance(raw, list):
        return [measurement_from_dict(item) for item in raw], False
    if isinstance(raw, dict):
        return [measurement_from_dict(raw)], True
    raise SchemaError("Prediction input JSON must be a Measurement object, list, or SoilDataset.")


def _emit_json(payload: dict[str, Any], output_path: str | None = None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


def _cmd_dataset_validate(args: argparse.Namespace) -> int:
    target_name = getattr(args, "target", "soc_pct")
    ds = _load_dataset_auto(
        args.input,
        profile_id=args.instrument_profile,
        target_columns=[target_name],
    )
    summary = DataValidator.validate_dataset(
        ds,
        for_training=bool(args.for_training),
        require_sic=not bool(args.no_require_sic),
        target_name=target_name,
    )
    payload = {
        "command": "dataset validate",
        "dataset_id": ds.dataset_id,
        "dataset_source": ds.dataset_source,
        "target": target_name,
        "is_valid": summary.is_valid,
        "n_records": summary.n_records,
        "n_errors": summary.n_errors,
        "errors": summary.errors,
        "warnings": summary.warnings,
    }
    _emit_json(payload, getattr(args, "output", None))
    return 0 if summary.is_valid else 1


def _cmd_calibration_fit_epo(args: argparse.Namespace) -> int:
    prof_arg = getattr(args, "instrument_profile", None)
    wet_ds = _load_dataset_auto(args.wet, profile_id=prof_arg)
    dry_ds = _load_dataset_auto(args.dry, profile_id=prof_arg)
    cfg = PreprocessConfig(
        use_absorbance=not bool(args.no_absorbance),
        use_snv=not bool(args.no_snv),
        use_savgol=not bool(args.no_savgol),
        suppression_mode="epo",
        epo_q=args.q,
    )
    chemo = ChemometricPipeline(cfg).fit_from_datasets(wet_ds, dry_ds)
    assert chemo.epo is not None

    out_path = Path(args.output)
    if out_path.suffix.lower() == ".json":
        json_path = out_path
        npz_path = out_path.with_suffix(".npz")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / "epo_state.json"
        npz_path = out_path / "epo_state.npz"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    invariants = chemo.epo.check_invariants()
    np.savez(
        npz_path,
        P=np.asarray(chemo.epo.P, dtype=np.float64),
        V_s=np.asarray(chemo.epo.V_s, dtype=np.float64),
        Q=np.asarray(chemo.epo.Q, dtype=np.float64),
        wavelength_grid_nm=np.asarray(chemo.wavelength_grid_nm, dtype=np.float64),
    )
    payload = {
        "command": "calibration fit-epo",
        "space_signature": chemo.space_signature,
        "n_bands": chemo.epo.n_bands,
        "selected_q": chemo.epo.q,
        "calibration_dataset_hash": chemo.epo.calibration_dataset_hash,
        "invariants": invariants,
        "state_npz": str(npz_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _emit_json(payload)
    return 0


def _parse_covariates_arg(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    if cleaned.lower() in ("none", "off", "-"):
        return ()
    return tuple(c.strip() for c in cleaned.split(",") if c.strip())


def _cmd_train(args: argparse.Namespace) -> int:
    prof_arg = getattr(args, "instrument_profile", None)
    ds = _load_dataset_auto(args.input, profile_id=prof_arg, target_columns=[args.target])
    wet_ds = _load_dataset_auto(args.wet, profile_id=prof_arg) if getattr(args, "wet", None) else None
    dry_ds = _load_dataset_auto(args.dry, profile_id=prof_arg) if getattr(args, "dry", None) else None
    pre_cfg = PreprocessConfig(suppression_mode=args.suppression_mode, epo_q=args.epo_q)
    cov_names = _parse_covariates_arg(getattr(args, "covariates", None))
    fusion_cfg = FeatureFusionConfig(covariate_names=cov_names) if cov_names is not None else FeatureFusionConfig()
    trainer_cfg = TrainerConfig(
        target_name=args.target,
        preprocess_config=pre_cfg,
        fusion_config=fusion_cfg,
        hybrid_epochs=args.hybrid_epochs,
        seed=args.seed,
    )
    trainer = STSoilTrainer(trainer_cfg)

    n_sites = len(ds.unique_site_ids)
    folds = GroupedSplitBuilder.group_kfold_split(
        ds,
        n_splits=min(int(args.folds), n_sites),
        group_by="site_id",
    )

    factories = trainer.default_model_factories(include_optional=True)
    if args.model != "auto":
        if args.model not in MODEL_CLI_ALIASES:
            raise SchemaError(
                f"Unsupported CLI model '{args.model}'.",
                details={"supported": sorted(set(MODEL_CLI_ALIASES.keys()))},
            )
        chosen_key = MODEL_CLI_ALIASES[args.model]
        bench_factories = {chosen_key: factories[chosen_key]}
    else:
        bench_factories = factories
        chosen_key = ""

    bench = trainer.run_benchmark(
        ds,
        split_folds=folds,
        model_factories=bench_factories,
        include_optional_models=True,
        wet_dataset=wet_ds,
        dry_dataset=dry_ds,
    )
    if args.model == "auto":
        chosen_key = bench.evidence_ranking[0]

    chosen_entry = bench.entries[chosen_key]

    # Fit final preprocessing + fusion + model + OOF conformal + OOD on full training set
    chemo = _fit_fold_chemometrics(
        trainer_cfg,
        ds.records,
        external_wet=wet_ds,
        external_dry=dry_ds,
    )
    fusion = FeatureFusionPipeline(trainer_cfg.fusion_config).fit(
        ds.records,
        chemometric_pipeline=chemo,
        profile=ds.instrument_profile,
    )
    fused = fusion.build_feature_vector(ds.records, chemometric_pipeline=chemo)

    y_all = _extract_target_vector(ds.records, args.target)
    final_target_names: tuple[str, ...] = (args.target,)
    y_fit: np.ndarray = y_all

    all_carbon_targets = ("soc_pct", "sic_pct", "tc_pct")
    if getattr(args, "multi_target", False):
        # Include all carbon fractions whose labels are completely finite in the training dataset
        candidate_cols: list[tuple[str, np.ndarray]] = []
        ordered_names = [args.target] + [t for t in all_carbon_targets if t != args.target]
        for tname in ordered_names:
            try:
                vec = _extract_target_vector(ds.records, tname)
                if np.all(np.isfinite(vec)):
                    candidate_cols.append((tname, vec))
            except Exception:
                continue
        if len(candidate_cols) >= 2:
            final_target_names = tuple(name for name, _ in candidate_cols)
            y_fit = np.column_stack([vec for _, vec in candidate_cols])

    final_model = factories[chosen_key](fused.X_spectral.shape[1])
    final_model.set_metadata(
        feature_names=fused.fused_feature_names,
        target_names=final_target_names,
        dataset_source=ds.dataset_source,
    )
    tr_groups = np.asarray([r.site_id for r in ds.records], dtype=str)
    if isinstance(final_model, STSoilHybridNetRegressor) and args.target in all_carbon_targets:
        soc_vals = (
            _extract_target_vector(ds.records, "soc_pct")
            if "soc_pct" in final_target_names
            else None
        )
        sic_vals = (
            _extract_target_vector(ds.records, "sic_pct")
            if "sic_pct" in final_target_names
            else None
        )
        tc_vals = (
            _extract_target_vector(ds.records, "tc_pct")
            if "tc_pct" in final_target_names
            else None
        )
        final_model.fit_multitask(
            fused.X_spectral,
            fused.X_covariates,
            soc=soc_vals,
            sic=sic_vals,
            tc=tc_vals,
        )
    elif hasattr(final_model, "fit"):
        try:
            final_model.fit(fused.X_fused, y_fit, groups=tr_groups)  # type: ignore[call-arg]
        except TypeError:
            final_model.fit(fused.X_fused, y_fit)

    conformal = None
    if len(ds.records) >= 5 and chosen_entry.oof_predictions.size == len(ds.records):
        conformal = ConformalUncertaintyCalibrator(ConformalConfig()).fit(
            y_all,
            chosen_entry.oof_predictions,
            pred_std_cal=chosen_entry.oof_uncertainties,
        )

    ood = OutOfDomainDetector(OODConfig()).fit(fused.X_fused, measurements=ds.records)
    profile = ds.records[0].spectrum
    inst_prof = fusion.instrument_profile or resolve_instrument_profile(prof_arg or ds.records[0].instrument_profile_id)
    del profile

    meta = ArtifactMetadata(
        artifact_id=args.artifact_id or f"stds_{chosen_key}_{ds.dataset_id}",
        semantic_version=args.semantic_version,
        schema_version=SCHEMA_VERSION,
        instrument_profile_id=inst_prof.profile_id,
        calibration_id=ds.records[0].calibration_id,
        pipeline_version=pre_cfg.pipeline_version,
        dataset_hash=ds.compute_sha256(),
        dependency_versions=collect_dependency_versions(),
        seed=args.seed,
    )

    out_dir = save_artifact(
        args.output,
        metadata=meta,
        instrument_profile=inst_prof,
        model=final_model,
        chemometric_pipeline=chemo,
        fusion_pipeline=fusion,
        conformal_calibrator=conformal,
        ood_detector=ood,
        validation_metrics=chosen_entry.oof_metrics.to_dict(),
        subgroup_metrics={
            k: {gk: gm.to_dict() for gk, gm in gv.items()}
            for k, gv in chosen_entry.subgroup_metrics.items()
        },
        background_features=fused.X_fused,
        background_targets=y_all,
        target_schema=trainer_cfg.target_schema,
    )

    inspected = inspect_artifact(out_dir)
    payload: dict[str, Any] = {
        "command": "train",
        "artifact_dir": str(out_dir),
        "selected_model": chosen_key,
        "target": args.target,
        "target_names": list(final_target_names),
        "suppression_mode": pre_cfg.suppression_mode,
        "multi_target": len(final_target_names) > 1,
        "bundle_sha256": inspected["bundle_sha256"],
        "is_synthetic_smoke_only": bench.is_synthetic_smoke_only,
        "oof_metrics": chosen_entry.oof_metrics.to_dict(),
    }
    pdf_out = getattr(args, "pdf", None)
    if pdf_out:
        eff_lang = resolve_lang(getattr(args, "lang", None))
        pdf_path = generate_prediction_pdf_report(payload, pdf_out, lang=eff_lang)
        payload["pdf_report_path"] = str(pdf_path)
        payload["lang"] = eff_lang
    _emit_json(payload)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    prof_arg = getattr(args, "instrument_profile", None)
    ds = _load_dataset_auto(args.input, profile_id=prof_arg, target_columns=[args.target])
    wet_ds = _load_dataset_auto(args.wet, profile_id=prof_arg) if getattr(args, "wet", None) else None
    dry_ds = _load_dataset_auto(args.dry, profile_id=prof_arg) if getattr(args, "dry", None) else None
    pre_cfg = PreprocessConfig(suppression_mode=getattr(args, "suppression_mode", "none"), epo_q=getattr(args, "epo_q", None))
    cov_names = _parse_covariates_arg(getattr(args, "covariates", None))
    fusion_cfg = FeatureFusionConfig(covariate_names=cov_names) if cov_names is not None else FeatureFusionConfig()
    cfg = TrainerConfig(target_name=args.target, preprocess_config=pre_cfg, fusion_config=fusion_cfg, seed=args.seed)
    trainer = STSoilTrainer(cfg)
    n_sites = len(ds.unique_site_ids)
    folds = GroupedSplitBuilder.group_kfold_split(
        ds,
        n_splits=min(int(args.folds), n_sites),
        group_by="site_id",
    )
    bench = trainer.run_benchmark(
        ds,
        split_folds=folds,
        include_optional_models=bool(args.include_optional),
        wet_dataset=wet_ds,
        dry_dataset=dry_ds,
    )
    payload = bench.to_dict()
    _emit_json(payload, args.output)
    return 0


def _cmd_artifact_inspect(args: argparse.Namespace) -> int:
    info = inspect_artifact(args.artifact)
    _emit_json(info, getattr(args, "output", None))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    analyzer = STSoilAnalyzer.from_artifact(args.artifact)
    measurements, is_single = _load_measurements_for_predict(
        args.input,
        profile_id=getattr(args, "instrument_profile", None),
        fallback_profile=analyzer.artifact.instrument_profile,
    )
    if is_single:
        report = analyzer.predict(
            measurements[0],
            include_xai=bool(args.include_xai),
            include_stock=not bool(args.no_stock),
        )
        payload = report.to_dict()
    else:
        batch = analyzer.predict_batch(
            measurements,
            fail_fast=bool(args.fail_fast),
            include_xai=bool(args.include_xai),
            include_stock=not bool(args.no_stock),
        )
        payload = batch.to_dict()

    _emit_json(payload, args.output)
    pdf_out = getattr(args, "pdf", None)
    if pdf_out:
        eff_lang = resolve_lang(getattr(args, "lang", None))
        generate_prediction_pdf_report(payload, pdf_out, lang=eff_lang)
    return 0


def _cmd_report_export(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.is_file():
        raise ReportSerializationError(f"Input report JSON not found: '{in_path}'.")
    raw = json.loads(in_path.read_text(encoding="utf-8"))

    req_single = {"identity", "status", "carbon_metrics", "carbon_stock", "quality", "provenance", "diagnostics"}
    if isinstance(raw, dict) and req_single.issubset(raw.keys()):
        report_type = "prediction_report"
    elif isinstance(raw, dict) and "results" in raw and "summary" in raw:
        report_type = "batch_prediction_result"
    elif isinstance(raw, dict) and "selected_model" in raw and "oof_metrics" in raw:
        report_type = "training_summary"
    else:
        raise ReportSerializationError(
            "Input JSON does not conform to STDS Appendix B PredictionReport or BatchPredictionResult schema.",
            details={"missing_single_keys": sorted(req_single - set(raw.keys())) if isinstance(raw, dict) else []},
        )

    eff_lang = resolve_lang(getattr(args, "lang", None))
    fmt = getattr(args, "format", "json")
    out_path = Path(args.output)
    pdf_path: Path | None = None
    json_written: Path | None = None

    if fmt in ("json", "both"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        json_written = out_path

    if fmt in ("pdf", "both"):
        explicit_pdf = getattr(args, "pdf_output", None)
        if explicit_pdf:
            target_pdf = Path(explicit_pdf)
        elif fmt == "pdf" and out_path.suffix.lower() == ".pdf":
            target_pdf = out_path
        else:
            target_pdf = out_path.with_suffix(".pdf")
        pdf_path = generate_prediction_pdf_report(raw, target_pdf, lang=eff_lang)

    size_bytes = int(
        json_written.stat().st_size if json_written is not None else (pdf_path.stat().st_size if pdf_path is not None else 0)
    )
    summary_payload: dict[str, Any] = {
        "command": "report export",
        "report_type": report_type,
        "format": fmt,
        "lang": eff_lang,
        "size_bytes": size_bytes,
    }
    if json_written is not None:
        summary_payload["exported_to"] = str(json_written)
    if pdf_path is not None:
        summary_payload["pdf_report_path"] = str(pdf_path)
    _emit_json(summary_payload)
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    profile_id = getattr(args, "instrument_profile", None)
    ds = _load_dataset_auto(args.input, profile_id=profile_id, target_columns=[args.target])
    wet_ds = _load_dataset_auto(args.wet, profile_id=profile_id) if args.wet else None
    dry_ds = _load_dataset_auto(args.dry, profile_id=profile_id) if args.dry else None

    runtime_cfg = get_runtime_config(lang=getattr(args, "lang", None))
    dpi_val = int(args.dpi) if getattr(args, "dpi", None) is not None else runtime_cfg.dpi
    include_pdf = runtime_cfg.pdf_export and (not bool(getattr(args, "no_pdf", False)))

    formats = tuple(fmt.strip() for fmt in str(args.format).split(",") if fmt.strip())
    cov_names = _parse_covariates_arg(getattr(args, "covariates", None))
    cfg = VisualizationConfig(
        target_name=args.target,
        model_name=getattr(args, "model", "auto"),
        covariate_names=cov_names,
        n_folds=int(args.folds),
        formats=formats,
        dpi=dpi_val,
        include_html_dashboard=not bool(args.no_html),
        include_interactive_3d=not bool(args.no_interactive_3d),
        include_pdf_report=include_pdf,
        include_optional_models=bool(args.include_optional),
        lang=runtime_cfg.lang,
        seed=int(args.seed),
    )
    viz_report = generate_full_visualization_suite(
        ds,
        args.output,
        artifact_path=args.artifact,
        wet_dataset=wet_ds,
        dry_dataset=dry_ds,
        config=cfg,
    )
    _emit_json(
        {
            "command": "visualize",
            "output_dir": viz_report.output_dir,
            "manifest_path": viz_report.manifest_path,
            "dashboard_path": viz_report.dashboard_path,
            "pdf_report_path": viz_report.pdf_report_path,
            "lang": viz_report.lang,
            "n_plots": viz_report.n_plots,
            "n_3d_plots": viz_report.n_3d_plots,
            "n_tables": viz_report.n_tables,
            "n_interactive_3d": len(viz_report.interactive_3d_html),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the unified `st-ds` CLI argument parser (Section 27)."""
    parser = argparse.ArgumentParser(
        prog="st-ds",
        description="SpaceTeknopoliDataScience (STDS) Standalone Soil Carbon Engine CLI",
    )
    parser.add_argument("--version", action="version", version=f"st-ds {__version__}")
    parser.add_argument("--debug", action="store_true", help="Print full Python traceback on error.")
    parser.add_argument(
        "--lang",
        choices=["tr", "en"],
        default=None,
        help="UI/report language (overrides OS env STDS_LANG and optional .env; default: tr).",
    )

    sub = parser.add_subparsers(dest="group", required=True)

    # dataset validate
    p_ds = sub.add_parser("dataset", help="Dataset operations.")
    p_ds_sub = p_ds.add_subparsers(dest="action", required=True)
    p_val = p_ds_sub.add_parser("validate", help="Validate dataset schema, units, and lineage.")
    p_val.add_argument("--input", required=True, help="Path to dataset JSON or CSV.")
    p_val.add_argument("--instrument-profile", default=None, help="Optional builtin profile ID or JSON path for CSV loading.")
    p_val.add_argument("--for-training", action="store_true", help="Require SoilReference target labels.")
    p_val.add_argument("--no-require-sic", action="store_true", help="Allow missing SIC in targets.")
    p_val.add_argument("--target", default="soc_pct", help="Target column/field name to verify during training validation.")
    p_val.add_argument("--output", default=None, help="Optional JSON output file.")
    p_val.set_defaults(func=_cmd_dataset_validate)

    # calibration fit-epo
    p_cal = sub.add_parser("calibration", help="Chemometric calibration workflows.")
    p_cal_sub = p_cal.add_subparsers(dest="action", required=True)
    p_epo = p_cal_sub.add_parser("fit-epo", help="Fit External Parameter Orthogonalization from wet/dry pairs.")
    p_epo.add_argument("--wet", required=True, help="Path to paired wet SoilDataset.")
    p_epo.add_argument("--dry", required=True, help="Path to paired dry SoilDataset.")
    p_epo.add_argument("--instrument-profile", default=None, help="Optional builtin profile ID or JSON path for raw CSV pairs.")
    p_epo.add_argument("--output", required=True, help="Output directory or .json file.")
    p_epo.add_argument("--q", type=int, default=None, help="Optional fixed EPO rank q.")
    p_epo.add_argument("--no-absorbance", action="store_true")
    p_epo.add_argument("--no-snv", action="store_true")
    p_epo.add_argument("--no-savgol", action="store_true")
    p_epo.set_defaults(func=_cmd_calibration_fit_epo)

    # train
    p_train = sub.add_parser("train", help="Train a leakage-free model and export a signed artifact bundle.")
    p_train.add_argument("--input", required=True, help="Training SoilDataset JSON or CSV.")
    p_train.add_argument("--output", required=True, help="Output artifact bundle directory.")
    p_train.add_argument(
        "--model",
        default="auto",
        choices=["auto", *sorted(set(MODEL_CLI_ALIASES.keys()))],
        help="Model family or 'auto' to pick best OOF RMSE.",
    )
    p_train.add_argument("--target", default="soc_pct", help="Target label/column name (default: soc_pct; also sic_pct, tc_pct, or custom CSV/JSON target).")
    p_train.add_argument("--instrument-profile", default=None, help="Optional builtin profile ID or JSON path for raw CSV inputs.")
    p_train.add_argument("--covariates", default=None, help="Comma-separated covariate columns or 'none' (default: ph,moisture_volumetric_pct,temperature_c,ec_ds_m).")
    p_train.add_argument("--folds", type=int, default=3)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--suppression-mode", default="none", choices=["none", "epo", "glsw"])
    p_train.add_argument("--epo-q", type=int, default=None)
    p_train.add_argument("--wet", default=None, help="Optional paired wet calibration SoilDataset for EPO/GLSW.")
    p_train.add_argument("--dry", default=None, help="Optional paired dry calibration SoilDataset for EPO/GLSW.")
    p_train.add_argument("--multi-target", action="store_true", help="Jointly train all available carbon fractions (soc_pct, sic_pct, tc_pct).")
    p_train.add_argument("--hybrid-epochs", type=int, default=35)
    p_train.add_argument("--artifact-id", default=None)
    p_train.add_argument("--semantic-version", default="0.1.0")
    p_train.add_argument("--pdf", default=None, help="Optional PDF training summary report output path.")
    p_train.add_argument("--lang", choices=["tr", "en"], default=argparse.SUPPRESS, help="Output language (tr|en).")
    p_train.set_defaults(func=_cmd_train)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run grouped CV benchmark across candidate models.")
    p_eval.add_argument("--input", required=True, help="SoilDataset JSON or CSV.")
    p_eval.add_argument("--target", default="soc_pct", help="Target label/column name (default: soc_pct; also sic_pct, tc_pct, or custom target).")
    p_eval.add_argument("--instrument-profile", default=None, help="Optional builtin profile ID or JSON path for raw CSV inputs.")
    p_eval.add_argument("--covariates", default=None, help="Comma-separated covariate columns or 'none'.")
    p_eval.add_argument("--folds", type=int, default=3)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--suppression-mode", default="none", choices=["none", "epo", "glsw"])
    p_eval.add_argument("--epo-q", type=int, default=None)
    p_eval.add_argument("--wet", default=None, help="Optional paired wet calibration SoilDataset.")
    p_eval.add_argument("--dry", default=None, help="Optional paired dry calibration SoilDataset.")
    p_eval.add_argument("--include-optional", action="store_true")
    p_eval.add_argument("--output", default=None, help="Optional benchmark report JSON output.")
    p_eval.set_defaults(func=_cmd_evaluate)

    # artifact inspect
    p_art = sub.add_parser("artifact", help="Deployment artifact bundle tools.")
    p_art_sub = p_art.add_subparsers(dest="action", required=True)
    p_ins = p_art_sub.add_parser("inspect", help="Verify SHA-256 integrity and inspect manifest & model card.")
    p_ins.add_argument("--artifact", required=True, help="Artifact bundle directory.")
    p_ins.add_argument("--output", default=None, help="Optional output JSON path.")
    p_ins.set_defaults(func=_cmd_artifact_inspect)

    # predict
    p_pred = sub.add_parser("predict", help="Run inference from a verified artifact bundle.")
    p_pred.add_argument("--artifact", required=True, help="Artifact bundle directory.")
    p_pred.add_argument("--input", required=True, help="Measurement or SoilDataset JSON/CSV path.")
    p_pred.add_argument("--instrument-profile", default=None, help="Optional builtin profile ID or JSON path for raw CSV inputs.")
    p_pred.add_argument("--output", default=None, help="Optional PredictionReport JSON output path.")
    p_pred.add_argument("--include-xai", action="store_true", help="Include XAI sensitivity/IG/HSIC reports.")
    p_pred.add_argument("--no-stock", action="store_true", help="Disable carbon stock computation.")
    p_pred.add_argument("--fail-fast", action="store_true", help="Abort batch inference on first invalid item.")
    p_pred.add_argument("--pdf", default=None, help="Optional PDF prediction report output path.")
    p_pred.add_argument("--lang", choices=["tr", "en"], default=argparse.SUPPRESS, help="Output language (tr|en).")
    p_pred.set_defaults(func=_cmd_predict)

    # report export
    p_rep = sub.add_parser("report", help="Validate and export canonical prediction reports.")
    p_rep_sub = p_rep.add_subparsers(dest="action", required=True)
    p_exp = p_rep_sub.add_parser("export", help="Validate and write canonical Appendix B report JSON and/or PDF.")
    p_exp.add_argument("--input", required=True, help="Input report JSON.")
    p_exp.add_argument("--output", required=True, help="Validated exported JSON (or PDF if --format pdf) path.")
    p_exp.add_argument("--format", choices=["json", "pdf", "both"], default="json", help="Export format (default: json).")
    p_exp.add_argument("--pdf-output", default=None, help="Explicit PDF path when --format is pdf or both.")
    p_exp.add_argument("--lang", choices=["tr", "en"], default=argparse.SUPPRESS, help="Output language (tr|en).")
    p_exp.set_defaults(func=_cmd_report_export)

    # visualize
    p_viz = sub.add_parser(
        "visualize",
        help="Generate 26+ scientific 2D/3D plots, analytical CSV/MD tables, an offline HTML dashboard, and a multi-page PDF report.",
    )
    p_viz.add_argument("--input", required=True, help="Input SoilDataset JSON/CSV path.")
    p_viz.add_argument("--output", required=True, help="Output directory for plots/, tables/, interactive/, reports/, and index.html.")
    p_viz.add_argument("--artifact", default=None, help="Optional saved artifact bundle directory (trains in-memory if omitted).")
    p_viz.add_argument("--instrument-profile", default=None, help="Optional builtin profile ID or JSON path for raw CSV inputs.")
    p_viz.add_argument("--wet", default=None, help="Optional paired wet SoilDataset JSON/CSV for EPO comparison plot.")
    p_viz.add_argument("--dry", default=None, help="Optional paired dry SoilDataset JSON/CSV for EPO comparison plot.")
    p_viz.add_argument("--target", default="soc_pct", help="Target label/column name (default: soc_pct; also sic_pct, tc_pct, or custom target).")
    p_viz.add_argument(
        "--model",
        default="auto",
        choices=["auto", *sorted(set(MODEL_CLI_ALIASES.keys()))],
        help="Model family to highlight/train in-memory when --artifact is omitted (default: auto).",
    )
    p_viz.add_argument("--covariates", default=None, help="Comma-separated covariate columns or 'none' (defaults to artifact config or standard covariates).")
    p_viz.add_argument("--folds", type=int, default=4, help="Grouped K-Fold splits for benchmark comparison plots.")
    p_viz.add_argument("--format", default="png", help="Comma-separated figure formats (e.g., png or png,svg).")
    p_viz.add_argument("--dpi", type=int, default=None, help="Raster DPI resolution (defaults to .env STDS_DPI or 160).")
    p_viz.add_argument("--no-html", action="store_true", help="Skip HTML dashboard generation (index.html).")
    p_viz.add_argument("--no-interactive-3d", action="store_true", help="Skip optional Plotly 3D HTML files.")
    p_viz.add_argument("--no-pdf", action="store_true", help="Skip multi-page PDF analysis report generation.")
    p_viz.add_argument("--include-optional", action="store_true", help="Include RandomForest/LocalPLSR in model benchmark plot.")
    p_viz.add_argument("--lang", choices=["tr", "en"], default=argparse.SUPPRESS, help="Output language (tr|en).")
    p_viz.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    p_viz.set_defaults(func=_cmd_visualize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning POSIX exit code (0 on success, non-zero on structured error)."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except STDSError as exc:
        err_payload = exc.to_dict()
        if args.debug:
            err_payload["traceback"] = traceback.format_exc()
        print(json.dumps(err_payload, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:
        err_payload = {
            "error_code": "STDS-CLI-ERROR",
            "message": str(exc),
            "remediation": "Re-run with `--debug` to inspect diagnostics.",
        }
        if args.debug:
            err_payload["traceback"] = traceback.format_exc()
        print(json.dumps(err_payload, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
