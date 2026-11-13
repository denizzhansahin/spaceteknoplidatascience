"""Consumer-facing `STSoilAnalyzer` facade with batch partial-failure policy and Appendix B reporting.

Implements Sections 25, 27 & Appendix B, ADR-001, ADR-004, ADR-005, ADR-006, ADR-007,
and closes DEF-005 & DEF-006 on inference outputs.
"""

from __future__ import annotations

import json
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from st_ds import SCHEMA_VERSION, __version__
from st_ds.api.versioning import LEGACY_WRAPPER_DEPRECATION_NOTICE, PUBLIC_API_VERSION
from st_ds.artifact.bundle import LoadedArtifact, inspect_artifact, load_artifact
from st_ds.carbon.fractions import classify_soil_provisional, partition_carbon
from st_ds.carbon.stock import compute_layer_stock
from st_ds.domain.errors import (
    ArtifactCompatibilityError,
    ReportSerializationError,
    STDSError,
    SchemaError,
)
from st_ds.domain.instrument import InstrumentProfile
from st_ds.domain.quality import QualityFlag, QualityMetadata, QualityStatus
from st_ds.domain.schemas import Measurement, PredictionReport
from st_ds.features.fusion import FusedFeatures
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.signal.qc import evaluate_signal_qc
from st_ds.xai.engine import SoilXAIEngine, XAIExplanationReport


@dataclass(frozen=True)
class BatchFailureRecord:
    """Structured error entry for one failed item in `predict_batch` (Section 27)."""

    index: int
    sample_id: str | None
    error_code: str
    message: str
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sample_id": self.sample_id,
            "error_code": self.error_code,
            "message": self.message,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class BatchPredictionResult:
    """Order-preserving batch inference result container supporting partial failures."""

    correlation_id: str
    results: list[PredictionReport | None]
    errors: list[BatchFailureRecord]
    n_total: int
    n_succeeded: int
    n_failed: int
    total_elapsed_ms: float
    batch_xai_hsic: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_version": PUBLIC_API_VERSION,
            "correlation_id": self.correlation_id,
            "summary": {
                "n_total": self.n_total,
                "n_succeeded": self.n_succeeded,
                "n_failed": self.n_failed,
                "total_elapsed_ms": self.total_elapsed_ms,
            },
            "results": [r.to_dict() if r is not None else None for r in self.results],
            "errors": [e.to_dict() for e in self.errors],
            "batch_xai_hsic": dict(self.batch_xai_hsic) if self.batch_xai_hsic is not None else None,
        }

    def to_json(self, *, indent: int = 2) -> str:
        try:
            return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ReportSerializationError(
                f"Failed to serialize BatchPredictionResult to JSON: {exc}",
                details={"correlation_id": self.correlation_id},
            ) from exc


class STSoilAnalyzer:
    """Stable consumer-facing facade over verified STDS soil carbon artifacts."""

    api_version: str = PUBLIC_API_VERSION

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        loaded_artifact: LoadedArtifact | None = None,
        device_profile: InstrumentProfile | str | None = None,
        calibration_id: str | None = None,
        strict_compatibility: bool = True,
        xai_engine: SoilXAIEngine | None = None,
    ) -> None:
        if loaded_artifact is not None:
            self._artifact = loaded_artifact
        elif weights_path is not None:
            warnings.warn(LEGACY_WRAPPER_DEPRECATION_NOTICE, DeprecationWarning, stacklevel=2)
            self._artifact = load_artifact(
                weights_path,
                expected_device_profile=device_profile,
                expected_calibration_id=calibration_id,
                strict_compatibility=strict_compatibility,
            )
        else:
            raise ArtifactCompatibilityError(
                "STSoilAnalyzer requires a verified artifact bundle via STSoilAnalyzer.from_artifact(path).",
                remediation="Train and export a bundle with save_artifact, then call STSoilAnalyzer.from_artifact.",
            )

        self._xai_engine = xai_engine or SoilXAIEngine()

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        device_profile: InstrumentProfile | str | None = None,
        calibration_id: str | None = None,
        strict_compatibility: bool = True,
        xai_engine: SoilXAIEngine | None = None,
    ) -> STSoilAnalyzer:
        """Load and verify a deployment artifact bundle and return a configured analyzer."""
        loaded = load_artifact(
            artifact_path,
            expected_device_profile=device_profile,
            expected_calibration_id=calibration_id,
            strict_compatibility=strict_compatibility,
        )
        return cls(loaded_artifact=loaded, xai_engine=xai_engine)

    @property
    def artifact(self) -> LoadedArtifact:
        return self._artifact

    def inspect_artifact(self) -> dict[str, Any]:
        """Return verified manifest metadata, model card, and profile compatibility info."""
        return inspect_artifact(self._artifact.bundle_dir)

    def _run_xai_on_fused(
        self,
        fused: FusedFeatures,
        *,
        measurement: Measurement | None = None,
        reference_X: np.ndarray | None = None,
        reference_y: np.ndarray | None = None,
        sample_index: int = 0,
    ) -> XAIExplanationReport:
        primary_target = (
            self._artifact.model.target_names[0]
            if self._artifact.model.target_names
            else "soc_pct"
        )
        bg_x = reference_X if reference_X is not None else self._artifact.background_features
        bg_y = reference_y if reference_y is not None else self._artifact.background_targets

        # If caller invokes single-sample explain without stored background data, pass only sample vector (B=1 -> insufficient_sample_count).
        report = self._xai_engine.explain(
            self._artifact.model,
            fused.X_fused[sample_index],
            feature_profile=fused.feature_profile,
            reference_X=bg_x,
            reference_y=bg_y,
            target_name=primary_target,
            target_index=0,
            artifact_id=self._artifact.metadata.artifact_id,
        )
        if measurement is not None:
            report.provenance["sample_id"] = measurement.sample_id
        return report

    def explain(
        self,
        measurements: Measurement | Sequence[Measurement],
        report: PredictionReport | None = None,
    ) -> XAIExplanationReport:
        """Compute local sensitivity, Integrated Gradients, HSIC, interactions, and grouped ranking."""
        del report
        if isinstance(measurements, Measurement):
            recs = [measurements]
            is_single = True
        else:
            recs = list(measurements)
            if not recs:
                raise SchemaError("At least one Measurement is required for STSoilAnalyzer.explain.")
            is_single = len(recs) == 1

        for rec in recs:
            rec.spectrum.validate_against_profile(self._artifact.instrument_profile)

        fused = self._artifact.fusion_pipeline.build_feature_vector(
            recs,
            chemometric_pipeline=self._artifact.chemometric_pipeline,
        )
        if is_single:
            return self._run_xai_on_fused(
                fused,
                measurement=recs[0],
                reference_X=self._artifact.background_features,
                reference_y=self._artifact.background_targets,
                sample_index=0,
            )

        batch_preds = np.asarray(self._artifact.model.predict(fused.X_fused), dtype=float)
        if batch_preds.ndim == 2:
            batch_preds = batch_preds[:, 0]
        return self._run_xai_on_fused(
            fused,
            measurement=recs[0],
            reference_X=fused.X_fused,
            reference_y=batch_preds,
            sample_index=0,
        )

    def predict(
        self,
        measurement: Measurement,
        *,
        include_xai: bool = False,
        include_stock: bool = True,
        stock_cov_matrix: np.ndarray | list[list[float]] | None = None,
        alpha: float | None = None,
        correlation_id: str | None = None,
    ) -> PredictionReport:
        """Analyze one canonical Measurement and return a complete Appendix B PredictionReport."""
        if not isinstance(measurement, Measurement):
            raise SchemaError(
                f"Expected a Measurement instance, got {type(measurement).__name__}.",
            )

        corr_id = correlation_id or f"st-{uuid.uuid4().hex[:12]}"
        timings: dict[str, float] = {}

        # Stage 1: Profile & Spectral Signal Quality Validation
        t0 = time.perf_counter()
        measurement.spectrum.validate_against_profile(self._artifact.instrument_profile)
        qc_report = evaluate_signal_qc(measurement.spectrum)
        timings["validation_ms"] = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Chemometric Preprocessing (EPO/GLSW) & Named Feature Fusion
        t1 = time.perf_counter()
        fused = self._artifact.fusion_pipeline.build_feature_vector(
            measurement,
            chemometric_pipeline=self._artifact.chemometric_pipeline,
        )
        timings["preprocessing_fusion_ms"] = (time.perf_counter() - t1) * 1000.0

        # Stage 3: Model Inference
        t2 = time.perf_counter()
        model = self._artifact.model
        primary_target = model.target_names[0] if model.target_names else "soc_pct"
        sic_pct: float | None = None
        sic_std: float | None = None
        tc_pct: float | None = None
        tc_std: float | None = None
        closure_residual_pct: float | None = None
        hybrid_class_probs: list[float] | None = None

        if isinstance(model, STSoilHybridNetRegressor):
            struct = model.predict_structured(fused.X_spectral, fused.X_covariates)
            soc_pct = float(struct.soc_mean_pct[0])
            soc_std = float(struct.soc_std_pct[0]) if struct.soc_std_pct is not None else 0.1
            sic_pct = float(struct.sic_mean_pct[0]) if struct.sic_mean_pct is not None else None
            sic_std = float(struct.sic_std_pct[0]) if struct.sic_std_pct is not None else None
            tc_pct = float(struct.tc_mean_pct[0]) if struct.tc_mean_pct is not None else None
            tc_std = float(struct.tc_std_pct[0]) if struct.tc_std_pct is not None else None
            closure_residual_pct = (
                float(struct.closure_residual_pct[0])
                if struct.closure_residual_pct is not None
                else None
            )
            if struct.class_probs is not None:
                hybrid_class_probs = [float(p) for p in struct.class_probs[0].tolist()]
            if primary_target == "sic_pct" and sic_pct is not None:
                primary_mean, primary_std = sic_pct, (sic_std or soc_std)
            elif primary_target == "tc_pct" and tc_pct is not None:
                primary_mean, primary_std = tc_pct, (tc_std or soc_std)
            else:
                primary_mean, primary_std = soc_pct, soc_std
        else:
            mean_arr, std_arr = model.predict_distribution(fused.X_fused)
            mean_flat = np.asarray(mean_arr, dtype=np.float64).reshape(-1)
            std_flat = np.asarray(std_arr, dtype=np.float64).reshape(-1)
            primary_mean = float(mean_flat[0])
            primary_std = float(std_flat[0])
            if primary_target == "sic_pct":
                sic_pct, sic_std = primary_mean, primary_std
                soc_pct, soc_std = 0.0, primary_std
            elif primary_target == "tc_pct":
                tc_pct, tc_std = primary_mean, primary_std
                soc_pct, soc_std = primary_mean, primary_std
            else:
                soc_pct, soc_std = primary_mean, primary_std
        timings["inference_ms"] = (time.perf_counter() - t2) * 1000.0

        # Stage 4: Uncertainty Interval Calibration & Out-of-Domain Assessment
        t3 = time.perf_counter()
        if self._artifact.conformal_calibrator is not None:
            interval = self._artifact.conformal_calibrator.predict_interval(
                np.array([primary_mean], dtype=np.float64),
                pred_std=np.array([primary_std], dtype=np.float64),
                alpha=alpha,
            )
            ci_block: dict[str, Any] = {
                "target": primary_target,
                "lower_pct": float(interval.lower_bound[0]),
                "upper_pct": float(interval.upper_bound[0]),
                "half_width_pct": float(interval.half_width[0]),
                "nominal_coverage": float(interval.nominal_coverage),
                "calibration_method": f"conformal_{self._artifact.conformal_calibrator.config.method}",
            }
        else:
            eff_alpha = 0.10 if alpha is None else float(alpha)
            ci_hw = 1.64485 * primary_std
            ci_block = {
                "target": primary_target,
                "lower_pct": float(max(0.0, primary_mean - ci_hw)),
                "upper_pct": float(primary_mean + ci_hw),
                "half_width_pct": float(ci_hw),
                "nominal_coverage": 1.0 - eff_alpha,
                "calibration_method": "uncalibrated_model_std",
            }

        ood_assessment = None
        if self._artifact.ood_detector is not None:
            ood_batch = self._artifact.ood_detector.detect(
                fused.X_fused,
                measurements=[measurement],
                fail_on_ood=False,
            )
            ood_assessment = ood_batch.assessments[0]
        timings["uncertainty_ood_ms"] = (time.perf_counter() - t3) * 1000.0

        # Stage 5: Carbon Accounting & Stock (Strict DEF-005 / ADR-004 Compliance)
        t4 = time.perf_counter()
        fraction_res = partition_carbon(
            soc=soc_pct if primary_target != "sic_pct" else None,
            sic=sic_pct,
            tc=tc_pct,
            unit="pct",
            dry_matter_basis=True,
            closure_tolerance_pct=self._artifact.target_schema.closure_tolerance_pct,
        )

        # Compute single-layer stock only if bulk_density is present; never fallback to 1.35
        stock_carbon_val = tc_pct if (primary_target == "tc_pct" and tc_pct is not None) else soc_pct
        stock_carbon_std = tc_std if (primary_target == "tc_pct" and tc_std is not None) else soc_std
        if include_stock:
            stock_res = compute_layer_stock(
                carbon_value=stock_carbon_val,
                carbon_unit="pct",
                bulk_density=measurement.covariates.bulk_density_g_cm3,
                bulk_density_unit="g/cm3",
                depth_top=measurement.depth_interval.top_cm,
                depth_bottom=measurement.depth_interval.bottom_cm,
                depth_unit="cm",
                coarse_fraction=measurement.covariates.coarse_fragments_fraction or 0.0,
                coarse_unit="fraction",
                carbon_std=stock_carbon_std,
                bulk_density_std=measurement.covariates.bulk_density_uncertainty_g_cm3,
                cov_matrix=stock_cov_matrix,
            )
            carbon_stock_block = stock_res.to_dict()
            stock_flags = list(stock_res.quality_flags)
        else:
            carbon_stock_block = {
                "stock_mg_ha": None,
                "stock_std_mg_ha": None,
                "quality_status": QualityStatus.OK.value,
                "quality_flags": [],
                "notes": ["Carbon stock computation disabled (`include_stock=False`)."],
            }
            stock_flags = []

        carbon_metrics_block: dict[str, Any] = {
            "primary_target": primary_target,
            "soc_pct": soc_pct if primary_target != "sic_pct" else None,
            "soc_std_pct": soc_std if primary_target != "sic_pct" else None,
            "sic_pct": sic_pct,
            "sic_std_pct": sic_std,
            "tc_pct": tc_pct,
            "tc_std_pct": tc_std,
            "unit": "pct",
            "dry_matter_basis": True,
            "confidence_interval": ci_block,
            "closure_residual_pct": (
                closure_residual_pct
                if closure_residual_pct is not None
                else fraction_res.closure_residual_pct
            ),
        }

        # Stage 6: Provisional Soil Classification (OD-005)
        cls_res = classify_soil_provisional(
            soc=soc_pct if primary_target != "sic_pct" else 0.0,
            sic=sic_pct or 0.0,
            unit="pct",
        )
        classification_block: dict[str, Any] = cls_res.to_dict()
        if hybrid_class_probs is not None:
            classification_block["probabilities"] = hybrid_class_probs
        timings["carbon_accounting_ms"] = (time.perf_counter() - t4) * 1000.0

        # Stage 7: Optional XAI
        xai_block: dict[str, Any] | None = None
        if include_xai:
            t5 = time.perf_counter()
            xai_report = self._run_xai_on_fused(
                fused,
                measurement=measurement,
                reference_X=self._artifact.background_features,
                reference_y=self._artifact.background_targets,
                sample_index=0,
            )
            xai_block = xai_report.to_dict()
            timings["xai_ms"] = (time.perf_counter() - t5) * 1000.0

        # Aggregate QualityMetadata and overall QualityStatus
        all_flags: list[QualityFlag] = list(measurement.quality.flags)
        all_notes: list[str] = list(measurement.quality.notes)
        for fl in list(qc_report.flags) + list(fraction_res.quality_flags) + stock_flags:
            if fl not in all_flags:
                all_flags.append(fl)
        missing_covs = dict(fused.missing_reasons[0]) if fused.missing_reasons else {}
        if missing_covs:
            for reason in missing_covs.values():
                cov_flag = (
                    QualityFlag.COVARIATE_QC_FAILED
                    if reason == "qc_failed"
                    else QualityFlag.MISSING_COVARIATE
                )
                if cov_flag not in all_flags:
                    all_flags.append(cov_flag)

        if (
            closure_residual_pct is not None
            and abs(closure_residual_pct) > self._artifact.target_schema.closure_tolerance_pct
            and QualityFlag.CLOSURE_RESIDUAL_HIGH not in all_flags
        ):
            all_flags.append(QualityFlag.CLOSURE_RESIDUAL_HIGH)

        if ood_assessment is not None:
            for fl in ood_assessment.quality_flags:
                if fl not in all_flags:
                    all_flags.append(fl)

        if not qc_report.is_valid_for_reporting:
            overall_status = QualityStatus.NOT_VALID_FOR_REPORTING
        elif ood_assessment is not None and ood_assessment.is_ood:
            overall_status = QualityStatus.OUT_OF_DOMAIN
        elif all_flags:
            overall_status = QualityStatus.WARNING
        else:
            overall_status = QualityStatus.OK

        q_score = float(qc_report.qc_score)
        if ood_assessment is not None:
            q_score = min(q_score, float(ood_assessment.confidence_score))

        quality_meta = QualityMetadata(
            status=overall_status,
            quality_score=q_score,
            flags=all_flags,
            rejected_bands=list(qc_report.saturated_band_indices),
            rejected_shots_count=int(measurement.quality.rejected_shots_count),
            missing_covariates=missing_covs,
            notes=all_notes + list(qc_report.rejection_reasons),
        )

        timings["total_ms"] = float(sum(timings.values()))

        identity_block = {
            "correlation_id": corr_id,
            "sample_id": measurement.sample_id,
            "site_id": measurement.site_id,
            "timestamp_utc": measurement.timestamp_utc,
            "instrument_id": measurement.instrument_id,
            "instrument_profile_id": measurement.instrument_profile_id,
            "calibration_id": measurement.calibration_id,
        }

        provenance_block = {
            "stds_package_version": __version__,
            "api_version": self.api_version,
            "schema_version": SCHEMA_VERSION,
            "artifact_id": self._artifact.metadata.artifact_id,
            "artifact_semantic_version": self._artifact.metadata.semantic_version,
            "artifact_sha256": self._artifact.bundle_sha256,
            "pipeline_version": self._artifact.metadata.pipeline_version,
            "model_name": model.model_name,
            "model_family": model.model_family,
            "dataset_source": model._training_dataset_source,
        }

        diagnostics_block: dict[str, Any] = {
            "correlation_id": corr_id,
            "closure_residual_pct": carbon_metrics_block["closure_residual_pct"],
            "ood_score": ood_assessment.ood_score if ood_assessment is not None else None,
            "is_ood": ood_assessment.is_ood if ood_assessment is not None else None,
            "confidence_score": ood_assessment.confidence_score if ood_assessment is not None else q_score,
            "snr_estimate": qc_report.snr_estimate,
            "stage_timings_ms": timings,
            "warning_codes": [fl.value for fl in all_flags],
        }

        return PredictionReport(
            identity=identity_block,
            status=overall_status,
            carbon_metrics=carbon_metrics_block,
            carbon_stock=carbon_stock_block,
            classification=classification_block,
            quality=quality_meta,
            xai=xai_block,
            provenance=provenance_block,
            diagnostics=diagnostics_block,
        )

    def predict_batch(
        self,
        measurements: Sequence[Measurement],
        *,
        fail_fast: bool = False,
        include_xai: bool = False,
        include_stock: bool = True,
        alpha: float | None = None,
        correlation_id: str | None = None,
    ) -> BatchPredictionResult:
        """Analyze a sequence of Measurements, isolating invalid entries when fail_fast=False."""
        corr_id = correlation_id or f"st-batch-{uuid.uuid4().hex[:10]}"
        t_start = time.perf_counter()

        recs = list(measurements)
        results: list[PredictionReport | None] = []
        errors: list[BatchFailureRecord] = []
        valid_recs: list[Measurement] = []

        for idx, item in enumerate(recs):
            sid = item.sample_id if isinstance(item, Measurement) else None
            try:
                rep = self.predict(
                    item,
                    include_xai=include_xai,
                    include_stock=include_stock,
                    alpha=alpha,
                    correlation_id=f"{corr_id}:{idx}",
                )
                results.append(rep)
                valid_recs.append(item)
            except STDSError as exc:
                if fail_fast:
                    raise
                results.append(None)
                errors.append(
                    BatchFailureRecord(
                        index=idx,
                        sample_id=sid,
                        error_code=exc.error_code,
                        message=exc.message,
                        remediation=exc.remediation,
                    )
                )
            except Exception as exc:
                if fail_fast:
                    raise
                results.append(None)
                errors.append(
                    BatchFailureRecord(
                        index=idx,
                        sample_id=sid,
                        error_code="STDS-UNEXPECTED-ERROR",
                        message=str(exc),
                        remediation="Inspect measurement fields against st_ds.domain.schemas.Measurement.",
                    )
                )

        batch_xai_hsic: dict[str, Any] | None = None
        if include_xai and len(valid_recs) >= 4:
            xai_batch_report = self.explain(valid_recs)
            batch_xai_hsic = xai_batch_report.dataset_dependence_hsic.to_dict()

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return BatchPredictionResult(
            correlation_id=corr_id,
            results=results,
            errors=errors,
            n_total=len(recs),
            n_succeeded=len(valid_recs),
            n_failed=len(errors),
            total_elapsed_ms=float(elapsed_ms),
            batch_xai_hsic=batch_xai_hsic,
        )
