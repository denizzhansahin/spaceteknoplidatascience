"""Leakage-free grouped model training and evidence-based benchmarking (`STSoilTrainer`, Section 19 & 33, ADR-005).

Ensures:
- Every fold verifies zero `sample_id` / `site_id` leakage via `GroupedSplitBuilder`.
- `ChemometricPipeline`, `FeatureFusionPipeline`, and every candidate model are fitted
  strictly on each training fold (never on validation/test records).
- All models (`PLSR`, `KernelRidge`, optional `RandomForest`, `LocalPLSR`, `STSoilHybridNet`)
  compete on identical folds and identical preprocessing before any selection is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from st_ds.chemometrics.pipeline import ChemometricPipeline, PreprocessConfig
from st_ds.domain.errors import SchemaError
from st_ds.domain.schemas import Measurement, TargetLabelSchema
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io.dataset import SoilDataset
from st_ds.models.base import BaseSoilRegressor
from st_ds.models.baselines import (
    KernelRidgeRegressor,
    LocalPLSRRegressor,
    PLSRRegressor,
    RandomForestSoilRegressor,
)
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.training.metrics import (
    RegressionMetrics,
    check_closure_diagnostics,
    compute_regression_metrics,
    evaluate_by_subgroup,
)
from st_ds.training.splits import GroupedSplitBuilder, GroupKeyType, SplitFold


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for grouped cross-validation and model benchmarking."""

    target_name: str = "soc_pct"
    preprocess_config: PreprocessConfig = field(
        default_factory=lambda: PreprocessConfig(
            use_absorbance=True,
            use_snv=True,
            use_savgol=True,
            sg_window=7,
            sg_polyorder=2,
            sg_deriv=1,
            suppression_mode="none",
        )
    )
    fusion_config: FeatureFusionConfig = field(
        default_factory=lambda: FeatureFusionConfig(
            covariate_names=("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m"),
            missing_strategy="median_with_indicator",
            spectral_scaler="standard",
            covariate_scaler="robust",
        )
    )
    hybrid_epochs: int = 50
    hybrid_learning_rate: float = 2e-3
    hybrid_batch_size: int = 32
    seed: int = 42
    target_schema: TargetLabelSchema = field(default_factory=TargetLabelSchema)


@dataclass(frozen=True)
class ModelBenchmarkEntry:
    """Evidence summary for a single candidate model across identical grouped folds."""

    model_name: str
    model_family: str
    oof_metrics: RegressionMetrics
    fold_metrics: tuple[RegressionMetrics, ...]
    subgroup_metrics: dict[str, dict[str, RegressionMetrics]]
    closure_diagnostics: dict[str, Any] | None
    last_model_card: dict[str, Any]
    oof_predictions: np.ndarray
    oof_uncertainties: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "oof_metrics": self.oof_metrics.to_dict(),
            "fold_metrics": [m.to_dict() for m in self.fold_metrics],
            "subgroup_metrics": {
                field_k: {grp_k: m.to_dict() for grp_k, m in groups.items()}
                for field_k, groups in self.subgroup_metrics.items()
            },
            "closure_diagnostics": self.closure_diagnostics,
            "last_model_card": self.last_model_card,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Evidence-based comparison table across candidate baseline and hybrid models (ADR-005)."""

    dataset_id: str
    dataset_source: str
    target_name: str
    n_samples: int
    n_folds: int
    is_synthetic_smoke_only: bool
    warnings: tuple[str, ...]
    evidence_ranking: tuple[str, ...]
    entries: dict[str, ModelBenchmarkEntry]

    def summary_table(self) -> list[dict[str, Any]]:
        """Return ordered comparison rows sorted by OOF RMSE ascending."""
        rows: list[dict[str, Any]] = []
        for name in self.evidence_ranking:
            e = self.entries[name]
            rows.append(
                {
                    "model_name": e.model_name,
                    "rmse": e.oof_metrics.rmse,
                    "mae": e.oof_metrics.mae,
                    "bias": e.oof_metrics.bias,
                    "r2": e.oof_metrics.r2,
                    "rpiq": e.oof_metrics.rpiq,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_source": self.dataset_source,
            "target_name": self.target_name,
            "n_samples": self.n_samples,
            "n_folds": self.n_folds,
            "is_synthetic_smoke_only": self.is_synthetic_smoke_only,
            "warnings": list(self.warnings),
            "evidence_ranking": list(self.evidence_ranking),
            "summary_table": self.summary_table(),
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }


def _extract_target_vector(records: Sequence[Measurement], target_name: str) -> np.ndarray:
    vals: list[float] = []
    for r in records:
        if r.reference is None:
            raise SchemaError(
                f"Measurement '{r.sample_id}' lacks a SoilReference; cannot train/evaluate supervised model."
            )
        v = getattr(r.reference, target_name, None)
        if v is None:
            raise SchemaError(
                f"Measurement '{r.sample_id}' has missing target '{target_name}' in SoilReference."
            )
        vals.append(float(v))
    return np.asarray(vals, dtype=float)


def _extract_optional_target(records: Sequence[Measurement], field_name: str) -> np.ndarray:
    vals: list[float] = []
    for r in records:
        if r.reference is None:
            vals.append(float("nan"))
        else:
            v = getattr(r.reference, field_name, None)
            vals.append(float(v) if v is not None else float("nan"))
    return np.asarray(vals, dtype=float)


def _fit_fold_chemometrics(
    config: TrainerConfig,
    train_recs: Sequence[Measurement],
) -> ChemometricPipeline:
    wl = train_recs[0].spectrum.wavelength_grid_nm
    chemo = ChemometricPipeline(config.preprocess_config)
    if config.preprocess_config.suppression_mode == "none":
        chemo.fit(wl)
        return chemo

    # Pair wet and dry spectra matching by sample_id strictly inside train_recs
    dry_by_id = {r.sample_id: r.spectrum.values for r in train_recs if r.moisture_scenario == "dry"}
    wet_list: list[np.ndarray] = []
    dry_list: list[np.ndarray] = []
    for r in train_recs:
        if r.moisture_scenario == "wet" and r.sample_id in dry_by_id:
            wet_list.append(r.spectrum.values)
            dry_list.append(dry_by_id[r.sample_id])

    if not wet_list:
        raise SchemaError(
            f"ChemometricPipeline suppression_mode='{config.preprocess_config.suppression_mode}' "
            "requires paired wet/dry training samples inside each fold."
        )
    chemo.fit(wl, X_wet=np.vstack(wet_list), X_dry=np.vstack(dry_list))
    return chemo


class STSoilTrainer:
    """Reproducible, leakage-verified trainer and multi-model comparator."""

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()

    def default_model_factories(self, *, include_optional: bool = True) -> dict[str, Callable[[int], BaseSoilRegressor]]:
        """Build default baseline and neural model factories parameterized by n_spectral_features."""
        factories: dict[str, Callable[[int], BaseSoilRegressor]] = {
            "plsr": lambda _ns: PLSRRegressor(target_names=(self.config.target_name,)),
            "kernel_ridge_rbf": lambda _ns: KernelRidgeRegressor(target_names=(self.config.target_name,)),
        }
        if include_optional:
            factories["random_forest"] = lambda _ns: RandomForestSoilRegressor(
                random_state=self.config.seed,
                target_names=(self.config.target_name,),
            )
            factories["local_plsr_mbl"] = lambda _ns: LocalPLSRRegressor(
                target_names=(self.config.target_name,),
            )
            factories["st_soil_hybrid_net"] = lambda n_spec: STSoilHybridNetRegressor(
                n_spectral_features=n_spec,
                epochs=self.config.hybrid_epochs,
                batch_size=self.config.hybrid_batch_size,
                learning_rate=self.config.hybrid_learning_rate,
                seed=self.config.seed,
                target_names=(self.config.target_name,),
            )
        return factories

    def run_benchmark(
        self,
        dataset: SoilDataset,
        *,
        split_folds: Sequence[SplitFold] | None = None,
        n_splits: int = 4,
        group_by: GroupKeyType = "site_id",
        model_factories: dict[str, Callable[[int], BaseSoilRegressor]] | None = None,
        include_optional_models: bool = True,
    ) -> BenchmarkReport:
        """Train and compare all candidate models across identical leakage-free folds (ADR-005)."""
        records = dataset.records
        y_all = _extract_target_vector(records, self.config.target_name)
        folds = (
            list(split_folds)
            if split_folds is not None
            else GroupedSplitBuilder.group_kfold_split(dataset, n_splits=n_splits, group_by=group_by)
        )
        if not folds:
            raise SchemaError("STSoilTrainer.run_benchmark requires at least one SplitFold.")

        factories = model_factories or self.default_model_factories(include_optional=include_optional_models)
        if not factories:
            raise SchemaError("No model factories provided to STSoilTrainer.run_benchmark.")

        # Collect predictions per model indexed identically to dataset.records
        n_total = len(records)
        eval_counts = np.zeros(n_total, dtype=int)
        oof_preds_by_model: dict[str, np.ndarray] = {name: np.zeros(n_total, dtype=float) for name in factories}
        oof_stds_by_model: dict[str, np.ndarray] = {name: np.zeros(n_total, dtype=float) for name in factories}
        fold_metrics_by_model: dict[str, list[RegressionMetrics]] = {name: [] for name in factories}
        last_models: dict[str, BaseSoilRegressor] = {}

        # Optional closure tracking for multi-task STSoilHybridNet
        hybrid_soc = np.zeros(n_total, dtype=float)
        hybrid_sic = np.zeros(n_total, dtype=float)
        hybrid_tc_aux = np.zeros(n_total, dtype=float)

        for fold in folds:
            tr_idx = np.asarray(fold.train_indices, dtype=int)
            va_idx = np.asarray(fold.val_indices, dtype=int)
            tr_recs = [records[int(i)] for i in tr_idx]
            va_recs = [records[int(i)] for i in va_idx]

            GroupedSplitBuilder.verify_no_leakage(
                tr_recs,
                va_recs,
                enforce_site_disjoint=(fold.group_key == "site_id"),
            )
            eval_counts[va_idx] += 1

            # Fit preprocessing + feature fusion strictly inside training fold
            chemo = _fit_fold_chemometrics(self.config, tr_recs)
            fusion = FeatureFusionPipeline(self.config.fusion_config)
            fusion.fit(tr_recs, chemometric_pipeline=chemo)

            tr_fused = fusion.build_feature_vector(tr_recs, chemometric_pipeline=chemo)
            va_fused = fusion.build_feature_vector(va_recs, chemometric_pipeline=chemo)

            n_spec = tr_fused.X_spectral.shape[1]
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            tr_groups = np.asarray([r.site_id for r in tr_recs], dtype=str)

            soc_tr = _extract_optional_target(tr_recs, "soc_pct")
            sic_tr = _extract_optional_target(tr_recs, "sic_pct")
            tc_tr = _extract_optional_target(tr_recs, "tc_pct")
            cls_tr = _extract_optional_target(tr_recs, "soil_class_id")

            for name, factory in factories.items():
                model = factory(n_spec)
                model.set_metadata(
                    feature_names=tr_fused.fused_feature_names,
                    target_names=(self.config.target_name,),
                    dataset_source=dataset.dataset_source,
                )

                if isinstance(model, STSoilHybridNetRegressor) and np.all(np.isfinite(soc_tr)):
                    model.fit_multitask(
                        tr_fused.X_spectral,
                        tr_fused.X_covariates,
                        soc=soc_tr,
                        sic=sic_tr if np.any(np.isfinite(sic_tr)) else None,
                        tc=tc_tr if np.any(np.isfinite(tc_tr)) else None,
                        soil_class=cls_tr if np.any(np.isfinite(cls_tr)) else None,
                    )
                    struct = model.predict_structured(va_fused.X_spectral, va_fused.X_covariates)
                    assert struct.soc_std_pct is not None
                    assert struct.sic_mean_pct is not None and struct.sic_std_pct is not None
                    assert struct.tc_mean_pct is not None and struct.tc_std_pct is not None
                    assert struct.closure_residual_pct is not None

                    if self.config.target_name == "soc_pct":
                        p_va, s_va = struct.soc_mean_pct, struct.soc_std_pct
                    elif self.config.target_name == "sic_pct":
                        p_va, s_va = struct.sic_mean_pct, struct.sic_std_pct
                    elif self.config.target_name == "tc_pct":
                        p_va, s_va = struct.tc_mean_pct, struct.tc_std_pct
                    else:
                        raise SchemaError(f"Unsupported target_name '{self.config.target_name}'.")

                    hybrid_soc[va_idx] = struct.soc_mean_pct
                    hybrid_sic[va_idx] = struct.sic_mean_pct
                    hybrid_tc_aux[va_idx] = struct.tc_mean_pct + struct.closure_residual_pct
                    fold_cr = struct.closure_residual_pct
                else:
                    if isinstance(model, (PLSRRegressor, KernelRidgeRegressor)):
                        model.fit(tr_fused.X_fused, y_tr, groups=tr_groups)
                    else:
                        model.fit(tr_fused.X_fused, y_tr)
                    p_va, s_va = model.predict_distribution(va_fused.X_fused)
                    fold_cr = None

                oof_preds_by_model[name][va_idx] = p_va
                oof_stds_by_model[name][va_idx] = s_va
                fold_metrics_by_model[name].append(
                    compute_regression_metrics(y_va, p_va, closure_residuals=fold_cr)
                )
                last_models[name] = model

        eval_mask = eval_counts > 0
        eval_records = [records[i] for i in range(n_total) if eval_mask[i]]
        y_eval = y_all[eval_mask]

        entries: dict[str, ModelBenchmarkEntry] = {}
        for name in factories:
            preds_eval = oof_preds_by_model[name][eval_mask]
            stds_eval = oof_stds_by_model[name][eval_mask]
            model = last_models[name]

            clos_diag: dict[str, Any] | None = None
            eval_cr: np.ndarray | None = None
            if isinstance(model, STSoilHybridNetRegressor) and np.any(hybrid_soc[eval_mask] >= 0):
                eval_cr = hybrid_tc_aux[eval_mask] - (hybrid_soc[eval_mask] + hybrid_sic[eval_mask])
                clos_diag = check_closure_diagnostics(
                    hybrid_soc[eval_mask],
                    hybrid_sic[eval_mask],
                    hybrid_tc_aux[eval_mask],
                    tolerance_pct=self.config.target_schema.closure_tolerance_pct,
                )

            entries[name] = ModelBenchmarkEntry(
                model_name=model.model_name,
                model_family=model.model_family,
                oof_metrics=compute_regression_metrics(y_eval, preds_eval, closure_residuals=eval_cr),
                fold_metrics=tuple(fold_metrics_by_model[name]),
                subgroup_metrics=evaluate_by_subgroup(
                    eval_records,
                    y_eval,
                    preds_eval,
                    closure_residuals=eval_cr,
                ),
                closure_diagnostics=clos_diag,
                last_model_card=model.model_card(),
                oof_predictions=preds_eval,
                oof_uncertainties=stds_eval,
            )

        ranking = tuple(sorted(entries.keys(), key=lambda k: entries[k].oof_metrics.rmse))
        warnings: list[str] = []
        is_synth = dataset.dataset_source == "synthetic"
        if is_synth:
            warnings.append(
                "Dataset has dataset_source='synthetic': benchmark metrics validate pipeline mechanics "
                "only and MUST NOT be cited as real field generalization performance."
            )

        return BenchmarkReport(
            dataset_id=dataset.dataset_id,
            dataset_source=dataset.dataset_source,
            target_name=self.config.target_name,
            n_samples=len(eval_records),
            n_folds=len(folds),
            is_synthetic_smoke_only=is_synth,
            warnings=tuple(warnings),
            evidence_ranking=ranking,
            entries=entries,
        )
