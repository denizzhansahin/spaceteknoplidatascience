"""Multi-signal Out-of-Domain (OOD) detection and domain-envelope checking (Section 20, 33, OD-006).

Combines three complementary signals fitted strictly on training/calibration distributions:
1. Shrinkage-regularized Mahalanobis distance in standardized feature space
2. k-Nearest-Neighbor distance (local training neighborhood density)
3. Physical covariate & hardware/taxonomy envelope bounds (`OD-006`)

Prevents out-of-domain soil, moisture, pH, or device combinations from receiving
misleadingly high confidence or `QualityStatus.OK` (Risk 5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from st_ds.domain.errors import CalibrationError, OutOfDomainError, SchemaError
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.domain.schemas import Measurement
from st_ds.models.base import validate_xy_arrays


@dataclass(frozen=True)
class OODConfig:
    """Configuration for multi-signal out-of-domain detection."""

    mahalanobis_quantile: float = 0.99
    knn_k: int = 5
    knn_quantile: float = 0.99
    shrinkage: float = 0.15
    envelope_lower_quantile: float = 0.0
    envelope_upper_quantile: float = 1.0
    envelope_margin_fraction: float = 0.05
    fail_on_ood: bool = False

    def __post_init__(self) -> None:
        if not (0.5 <= self.mahalanobis_quantile <= 1.0):
            raise SchemaError("OODConfig.mahalanobis_quantile must lie in [0.5, 1.0].")
        if self.knn_k < 1:
            raise SchemaError("OODConfig.knn_k must be >= 1.")
        if not (0.5 <= self.knn_quantile <= 1.0):
            raise SchemaError("OODConfig.knn_quantile must lie in [0.5, 1.0].")
        if not (0.0 <= self.shrinkage <= 1.0):
            raise SchemaError("OODConfig.shrinkage must lie in [0.0, 1.0].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DomainEnvelope:
    """Empirical operational & physical domain learned from training measurements (OD-006)."""

    covariate_bounds: dict[str, tuple[float, float]]
    seen_instrument_profiles: tuple[str, ...]
    seen_moisture_scenarios: tuple[str, ...]
    seen_texture_classes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "covariate_bounds": {k: list(v) for k, v in self.covariate_bounds.items()},
            "seen_instrument_profiles": list(self.seen_instrument_profiles),
            "seen_moisture_scenarios": list(self.seen_moisture_scenarios),
            "seen_texture_classes": list(self.seen_texture_classes),
        }


@dataclass(frozen=True)
class OODSampleAssessment:
    """Per-sample domain compatibility and calibrated confidence assessment."""

    is_ood: bool
    ood_score: float
    mahalanobis_distance: float
    mahalanobis_ratio: float
    knn_distance: float
    knn_ratio: float
    out_of_range_covariates: tuple[str, ...]
    unseen_categories: tuple[str, ...]
    confidence_score: float
    quality_status: QualityStatus
    quality_flags: tuple[QualityFlag, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ood": self.is_ood,
            "ood_score": self.ood_score,
            "mahalanobis_distance": self.mahalanobis_distance,
            "mahalanobis_ratio": self.mahalanobis_ratio,
            "knn_distance": self.knn_distance,
            "knn_ratio": self.knn_ratio,
            "out_of_range_covariates": list(self.out_of_range_covariates),
            "unseen_categories": list(self.unseen_categories),
            "confidence_score": self.confidence_score,
            "quality_status": self.quality_status.value,
            "quality_flags": [f.value for f in self.quality_flags],
        }


@dataclass(frozen=True)
class OODBatchResult:
    """Batch-level container of OOD assessments."""

    assessments: tuple[OODSampleAssessment, ...]
    ood_rate: float
    n_ood: int
    n_samples: int
    mahalanobis_threshold: float
    knn_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_ood": self.n_ood,
            "ood_rate": self.ood_rate,
            "mahalanobis_threshold": self.mahalanobis_threshold,
            "knn_threshold": self.knn_threshold,
            "assessments": [a.to_dict() for a in self.assessments],
        }


def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a_sq = np.sum(A * A, axis=1, keepdims=True)
    b_sq = np.sum(B * B, axis=1, keepdims=True).T
    d2 = a_sq + b_sq - 2.0 * np.dot(A, B.T)
    return np.maximum(d2, 0.0)


class OutOfDomainDetector:
    """Detector measuring feature-space Mahalanobis, k-NN density, and covariate envelopes."""

    ENVELOPE_COVARIATES: tuple[str, ...] = (
        "ph",
        "moisture_volumetric_pct",
        "temperature_c",
        "ec_ds_m",
    )

    def __init__(self, config: OODConfig | None = None) -> None:
        self.config = config or OODConfig()
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.z_train_: np.ndarray | None = None
        self.cho_factor_: tuple[np.ndarray, bool] | None = None
        self.mahalanobis_threshold_: float = 0.0
        self.knn_threshold_: float = 0.0
        self.domain_envelope_: DomainEnvelope | None = None
        self._is_fitted: bool = False

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        assert self.feature_mean_ is not None and self.feature_std_ is not None
        return (X - self.feature_mean_[None, :]) / self.feature_std_[None, :]

    def _mahalanobis_distances(self, Z: np.ndarray) -> np.ndarray:
        assert self.cho_factor_ is not None
        solved = cho_solve(self.cho_factor_, Z.T, check_finite=False)
        quad = np.sum(Z.T * solved, axis=0)
        return np.sqrt(np.maximum(quad, 0.0))

    def _build_envelope(self, measurements: Sequence[Measurement]) -> DomainEnvelope:
        bounds: dict[str, tuple[float, float]] = {}
        for cname in self.ENVELOPE_COVARIATES:
            vals = [
                float(getattr(m.covariates, cname))
                for m in measurements
                if getattr(m.covariates, cname, None) is not None
            ]
            if vals:
                arr = np.asarray(vals, dtype=float)
                lo = float(np.quantile(arr, self.config.envelope_lower_quantile))
                hi = float(np.quantile(arr, self.config.envelope_upper_quantile))
                span = max(hi - lo, 1e-3)
                margin = span * self.config.envelope_margin_fraction
                bounds[cname] = (lo - margin, hi + margin)

        profiles = tuple(sorted({m.instrument_profile_id for m in measurements}))
        scenarios = tuple(sorted({m.moisture_scenario for m in measurements}))
        textures = tuple(
            sorted({m.soil_texture_class for m in measurements if m.soil_texture_class})
        )
        return DomainEnvelope(
            covariate_bounds=bounds,
            seen_instrument_profiles=profiles,
            seen_moisture_scenarios=scenarios,
            seen_texture_classes=textures,
        )

    def fit(
        self,
        X_train: np.ndarray,
        *,
        measurements: Sequence[Measurement] | None = None,
    ) -> OutOfDomainDetector:
        """Fit feature-space distribution statistics, thresholds, and covariate envelope on training data."""
        X_arr, _, _ = validate_xy_arrays(X_train)
        n_samples, n_features = X_arr.shape
        if n_samples < 4:
            raise SchemaError(
                f"OutOfDomainDetector.fit requires at least 4 training samples, got {n_samples}."
            )

        mean = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0, ddof=1)
        std = np.where(std < 1e-6, 1.0, std)
        self.feature_mean_ = mean
        self.feature_std_ = std

        Z = (X_arr - mean[None, :]) / std[None, :]
        self.z_train_ = Z

        cov_emp = np.dot(Z.T, Z) / max(1, n_samples - 1)
        diag_cov = np.diag(np.diag(cov_emp))
        rho = self.config.shrinkage
        cov_reg = (1.0 - rho) * cov_emp + rho * diag_cov + 1e-3 * np.eye(n_features, dtype=float)
        self.cho_factor_ = cho_factor(cov_reg, lower=True, check_finite=False)

        train_maha = self._mahalanobis_distances(Z)
        self.mahalanobis_threshold_ = max(
            float(np.quantile(train_maha, self.config.mahalanobis_quantile)) * 1.15,
            1e-4,
        )

        # Leave-one-out k-NN distance within training set
        d2_tr = _pairwise_sq_dists(Z, Z)
        np.fill_diagonal(d2_tr, np.inf)
        k_eff = min(self.config.knn_k, n_samples - 1)
        sorted_d2 = np.sort(d2_tr, axis=1)[:, :k_eff]
        train_knn = np.mean(np.sqrt(np.maximum(sorted_d2, 0.0)), axis=1)
        self.knn_threshold_ = max(
            float(np.quantile(train_knn, self.config.knn_quantile)) * 1.15,
            1e-4,
        )

        self.domain_envelope_ = self._build_envelope(measurements) if measurements is not None else None
        self._is_fitted = True
        return self

    def detect(
        self,
        X: np.ndarray,
        *,
        measurements: Sequence[Measurement] | None = None,
        fail_on_ood: bool | None = None,
    ) -> OODBatchResult:
        """Assess whether input samples fall outside the calibrated training domain."""
        if not self._is_fitted or self.z_train_ is None:
            raise CalibrationError(
                "OutOfDomainDetector must be fitted before calling detect_out_of_domain."
            )

        X_arr, _, _ = validate_xy_arrays(X, expected_n_features=self.z_train_.shape[1])
        if measurements is not None and len(measurements) != X_arr.shape[0]:
            raise SchemaError(
                f"measurements count ({len(measurements)}) must match X row count ({X_arr.shape[0]})."
            )

        Z = self._standardize(X_arr)
        maha_dists = self._mahalanobis_distances(Z)
        d2_cross = _pairwise_sq_dists(Z, self.z_train_)
        k_eff = min(self.config.knn_k, self.z_train_.shape[0])
        knn_sorted = np.sort(d2_cross, axis=1)[:, :k_eff]
        knn_dists = np.mean(np.sqrt(np.maximum(knn_sorted, 0.0)), axis=1)

        assessments: list[OODSampleAssessment] = []
        for idx in range(X_arr.shape[0]):
            m_dist = float(maha_dists[idx])
            k_dist = float(knn_dists[idx])
            m_ratio = m_dist / self.mahalanobis_threshold_
            k_ratio = k_dist / self.knn_threshold_

            out_covs: list[str] = []
            unseen_cats: list[str] = []
            if measurements is not None and self.domain_envelope_ is not None:
                meas = measurements[idx]
                env = self.domain_envelope_
                for cname, (lo, hi) in env.covariate_bounds.items():
                    val = getattr(meas.covariates, cname, None)
                    if val is not None and (float(val) < lo or float(val) > hi):
                        out_covs.append(cname)
                if meas.instrument_profile_id not in env.seen_instrument_profiles:
                    unseen_cats.append(f"instrument_profile_id:{meas.instrument_profile_id}")
                if (
                    meas.soil_texture_class
                    and env.seen_texture_classes
                    and meas.soil_texture_class not in env.seen_texture_classes
                ):
                    unseen_cats.append(f"soil_texture_class:{meas.soil_texture_class}")

            env_penalty = 0.5 * len(out_covs) + 1.0 * len(unseen_cats)
            score = max(m_ratio, k_ratio, 1.0 + env_penalty if (out_covs or unseen_cats) else 0.0)
            is_ood = bool(m_ratio > 1.0 or k_ratio > 1.0 or out_covs or unseen_cats)

            if is_ood:
                # Guarantee OOD samples never receive misleading high confidence (>= 0.5)
                confidence = float(np.clip(0.45 * np.exp(-0.7 * max(score - 1.0, 0.0)), 0.01, 0.45))
                q_status = QualityStatus.OUT_OF_DOMAIN
                q_flags = (QualityFlag.OUT_OF_CALIBRATION_DOMAIN,)
            else:
                confidence = float(np.clip(1.0 - 0.35 * min(1.0, score), 0.55, 0.99))
                q_status = QualityStatus.OK
                q_flags = ()

            assessments.append(
                OODSampleAssessment(
                    is_ood=is_ood,
                    ood_score=float(score),
                    mahalanobis_distance=m_dist,
                    mahalanobis_ratio=float(m_ratio),
                    knn_distance=k_dist,
                    knn_ratio=float(k_ratio),
                    out_of_range_covariates=tuple(out_covs),
                    unseen_categories=tuple(unseen_cats),
                    confidence_score=confidence,
                    quality_status=q_status,
                    quality_flags=q_flags,
                )
            )

        n_ood = sum(1 for a in assessments if a.is_ood)
        batch = OODBatchResult(
            assessments=tuple(assessments),
            ood_rate=float(n_ood / len(assessments)),
            n_ood=n_ood,
            n_samples=len(assessments),
            mahalanobis_threshold=self.mahalanobis_threshold_,
            knn_threshold=self.knn_threshold_,
        )

        strict = self.config.fail_on_ood if fail_on_ood is None else bool(fail_on_ood)
        if strict and n_ood > 0:
            raise OutOfDomainError(
                f"{n_ood} of {len(assessments)} sample(s) flagged as out_of_domain (fail_on_ood=True).",
                details={"n_ood": n_ood, "ood_rate": batch.ood_rate},
            )
        return batch


def detect_out_of_domain(
    detector: OutOfDomainDetector,
    X: np.ndarray,
    *,
    measurements: Sequence[Measurement] | None = None,
    fail_on_ood: bool | None = None,
) -> OODBatchResult:
    """Functional entry point (Section 20) to run OOD checks using a fitted detector."""
    return detector.detect(X, measurements=measurements, fail_on_ood=fail_on_ood)
