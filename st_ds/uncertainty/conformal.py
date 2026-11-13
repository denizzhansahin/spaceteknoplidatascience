"""Split/OOF Conformal Prediction, calibration curves, and subgroup coverage (Section 20 & 33).

Implements:
- `fit_uncertainty_calibrator`: fits normalized or absolute nonconformity scores on
  grouped OOF/calibration residuals (never on test data)
- `predict_interval`: generates finite-sample coverage-calibrated intervals with
  separate aleatoric/epistemic attribution and non-negative carbon bounds
- `compute_calibration_curve` & `evaluate_subgroup_coverage`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np

from st_ds.domain.errors import CalibrationError, ModelInferenceError, SchemaError
from st_ds.domain.schemas import Measurement

ConformalMethod = Literal["normalized", "absolute"]


@dataclass(frozen=True)
class ConformalConfig:
    """Configuration for finite-sample conformal interval calibration."""

    method: ConformalMethod = "normalized"
    default_alpha: float = 0.10
    min_scale: float = 1e-3
    non_negative: bool = True

    def __post_init__(self) -> None:
        if self.method not in ("normalized", "absolute"):
            raise SchemaError(f"Unsupported conformal method '{self.method}'.")
        if not (0.0 < self.default_alpha < 1.0):
            raise SchemaError("ConformalConfig.default_alpha must lie strictly in (0, 1).")
        if self.min_scale <= 0.0:
            raise SchemaError("ConformalConfig.min_scale must be strictly positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UncertaintyInterval:
    """Coverage-calibrated prediction interval with aleatoric/epistemic decomposition."""

    point_prediction: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    half_width: np.ndarray
    nominal_coverage: float
    alpha: float
    conformal_quantile_q: float
    aleatoric_std: np.ndarray | None = None
    epistemic_std: np.ndarray | None = None
    total_std: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_coverage": self.nominal_coverage,
            "alpha": self.alpha,
            "conformal_quantile_q": self.conformal_quantile_q,
            "point_prediction": self.point_prediction.tolist(),
            "lower_bound": self.lower_bound.tolist(),
            "upper_bound": self.upper_bound.tolist(),
            "half_width": self.half_width.tolist(),
            "aleatoric_std": self.aleatoric_std.tolist() if self.aleatoric_std is not None else None,
            "epistemic_std": self.epistemic_std.tolist() if self.epistemic_std is not None else None,
            "total_std": self.total_std.tolist() if self.total_std is not None else None,
        }


@dataclass(frozen=True)
class CalibrationCurvePoint:
    """Single point on an empirical coverage calibration curve."""

    nominal_coverage: float
    empirical_coverage: float
    coverage_error: float
    mean_interval_width: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageReport:
    """Summary of multi-level calibration curves and optional subgroup coverage."""

    curve: tuple[CalibrationCurvePoint, ...]
    mean_abs_calibration_error: float
    nominal_alpha: float
    subgroup_coverage: dict[str, dict[str, CalibrationCurvePoint]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_abs_calibration_error": self.mean_abs_calibration_error,
            "nominal_alpha": self.nominal_alpha,
            "curve": [p.to_dict() for p in self.curve],
            "subgroup_coverage": {
                field: {grp: pt.to_dict() for grp, pt in grp_map.items()}
                for field, grp_map in self.subgroup_coverage.items()
            },
        }


class ConformalUncertaintyCalibrator:
    """Conformal predictor storing nonconformity distribution calibrated on grouped OOF/cal sets."""

    def __init__(self, config: ConformalConfig | None = None) -> None:
        self.config = config or ConformalConfig()
        self.nonconformity_scores_: np.ndarray | None = None
        self.n_calibration_samples_: int = 0
        self._is_fitted: bool = False

    def fit(
        self,
        y_cal: np.ndarray | Sequence[float],
        y_pred_cal: np.ndarray | Sequence[float],
        *,
        pred_std_cal: np.ndarray | Sequence[float] | None = None,
    ) -> ConformalUncertaintyCalibrator:
        yt = np.asarray(y_cal, dtype=float).reshape(-1)
        yp = np.asarray(y_pred_cal, dtype=float).reshape(-1)
        if yt.shape != yp.shape or yt.size < 5:
            raise ModelInferenceError(
                "Conformal calibration requires at least 5 matched samples in y_cal and y_pred_cal.",
                details={"n_y_cal": int(yt.size), "n_y_pred_cal": int(yp.size)},
            )
        if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
            raise ModelInferenceError("Calibration targets/predictions must contain only finite values.")

        abs_res = np.abs(yt - yp)
        if self.config.method == "normalized":
            if pred_std_cal is None:
                raise ModelInferenceError(
                    "Normalized conformal calibration requires `pred_std_cal`."
                )
            sd = np.asarray(pred_std_cal, dtype=float).reshape(-1)
            if sd.shape != yt.shape or not np.all(np.isfinite(sd)):
                raise ModelInferenceError("pred_std_cal must be finite and match y_cal length.")
            scale = np.maximum(sd, self.config.min_scale)
            scores = abs_res / scale
        else:
            scores = abs_res

        self.nonconformity_scores_ = np.sort(scores)
        self.n_calibration_samples_ = int(yt.size)
        self._is_fitted = True
        return self

    def quantile_for(self, alpha: float | None = None) -> float:
        """Compute finite-sample conformal quantile at level (1 - alpha)."""
        if not self._is_fitted or self.nonconformity_scores_ is None:
            raise CalibrationError(
                "ConformalUncertaintyCalibrator must be fitted before querying quantiles."
            )
        eff_alpha = self.config.default_alpha if alpha is None else float(alpha)
        if not (0.0 < eff_alpha < 1.0):
            raise SchemaError(f"alpha must lie strictly in (0, 1), got {eff_alpha}.")

        n = self.n_calibration_samples_
        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - eff_alpha)) / n))
        return float(np.quantile(self.nonconformity_scores_, q_level, method="higher"))

    def predict_interval(
        self,
        y_pred: np.ndarray | Sequence[float],
        *,
        pred_std: np.ndarray | Sequence[float] | None = None,
        aleatoric_std: np.ndarray | Sequence[float] | None = None,
        epistemic_std: np.ndarray | Sequence[float] | None = None,
        alpha: float | None = None,
    ) -> UncertaintyInterval:
        """Produce coverage-calibrated prediction intervals for new samples."""
        if not self._is_fitted:
            raise CalibrationError(
                "ConformalUncertaintyCalibrator must be fitted before calling predict_interval."
            )
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        if yp.size == 0 or not np.all(np.isfinite(yp)):
            raise ModelInferenceError("y_pred must be a non-empty finite array.")

        eff_alpha = self.config.default_alpha if alpha is None else float(alpha)
        q = self.quantile_for(eff_alpha)

        if self.config.method == "normalized":
            if pred_std is None:
                raise ModelInferenceError(
                    "Normalized conformal prediction requires `pred_std`."
                )
            sd = np.asarray(pred_std, dtype=float).reshape(-1)
            if sd.shape != yp.shape or not np.all(np.isfinite(sd)):
                raise ModelInferenceError("pred_std must be finite and match y_pred length.")
            half_width = q * np.maximum(sd, self.config.min_scale)
            tot_sd: np.ndarray | None = sd
        else:
            half_width = np.full_like(yp, q, dtype=float)
            tot_sd = np.asarray(pred_std, dtype=float).reshape(-1) if pred_std is not None else None

        lower = np.maximum(0.0, yp - half_width) if self.config.non_negative else (yp - half_width)
        upper = yp + half_width

        alea = np.asarray(aleatoric_std, dtype=float).reshape(-1) if aleatoric_std is not None else None
        epis = np.asarray(epistemic_std, dtype=float).reshape(-1) if epistemic_std is not None else None

        return UncertaintyInterval(
            point_prediction=yp,
            lower_bound=lower,
            upper_bound=upper,
            half_width=half_width,
            nominal_coverage=1.0 - eff_alpha,
            alpha=eff_alpha,
            conformal_quantile_q=q,
            aleatoric_std=alea,
            epistemic_std=epis,
            total_std=tot_sd,
        )


def fit_uncertainty_calibrator(
    y_cal: np.ndarray | Sequence[float],
    y_pred_cal: np.ndarray | Sequence[float],
    *,
    pred_std_cal: np.ndarray | Sequence[float] | None = None,
    config: ConformalConfig | None = None,
) -> ConformalUncertaintyCalibrator:
    """Functional entry point (Section 20) to fit a ConformalUncertaintyCalibrator."""
    return ConformalUncertaintyCalibrator(config=config).fit(
        y_cal,
        y_pred_cal,
        pred_std_cal=pred_std_cal,
    )


def predict_interval(
    calibrator: ConformalUncertaintyCalibrator,
    y_pred: np.ndarray | Sequence[float],
    *,
    pred_std: np.ndarray | Sequence[float] | None = None,
    aleatoric_std: np.ndarray | Sequence[float] | None = None,
    epistemic_std: np.ndarray | Sequence[float] | None = None,
    alpha: float | None = None,
) -> UncertaintyInterval:
    """Functional entry point (Section 20) to predict conformal uncertainty intervals."""
    return calibrator.predict_interval(
        y_pred,
        pred_std=pred_std,
        aleatoric_std=aleatoric_std,
        epistemic_std=epistemic_std,
        alpha=alpha,
    )


def _coverage_point(
    y_true: np.ndarray,
    interval: UncertaintyInterval,
) -> CalibrationCurvePoint:
    covered = (y_true >= interval.lower_bound) & (y_true <= interval.upper_bound)
    emp = float(np.mean(covered))
    widths = interval.upper_bound - interval.lower_bound
    return CalibrationCurvePoint(
        nominal_coverage=interval.nominal_coverage,
        empirical_coverage=emp,
        coverage_error=emp - interval.nominal_coverage,
        mean_interval_width=float(np.mean(widths)),
        n_samples=int(y_true.size),
    )


def compute_calibration_curve(
    calibrator: ConformalUncertaintyCalibrator,
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    *,
    pred_std: np.ndarray | Sequence[float] | None = None,
    coverage_levels: Sequence[float] = (0.50, 0.80, 0.90, 0.95),
) -> tuple[CalibrationCurvePoint, ...]:
    """Compute nominal vs empirical coverage across multiple confidence levels."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape != yp.shape or yt.size == 0:
        raise SchemaError("y_true and y_pred must be non-empty arrays of matching shape.")

    points: list[CalibrationCurvePoint] = []
    for cov in sorted(float(c) for c in coverage_levels):
        if not (0.0 < cov < 1.0):
            raise SchemaError(f"Coverage level must lie in (0, 1), got {cov}.")
        alpha = 1.0 - cov
        interval = calibrator.predict_interval(yp, pred_std=pred_std, alpha=alpha)
        points.append(_coverage_point(yt, interval))
    return tuple(points)


def evaluate_subgroup_coverage(
    calibrator: ConformalUncertaintyCalibrator,
    measurements: Sequence[Measurement],
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    *,
    pred_std: np.ndarray | Sequence[float] | None = None,
    alpha: float | None = None,
    fields: Sequence[str] = ("site_id", "moisture_scenario", "instrument_profile_id", "soil_class_name"),
    coverage_levels: Sequence[float] = (0.50, 0.80, 0.90, 0.95),
) -> CoverageReport:
    """Evaluate overall calibration curve and stratified empirical coverage per subgroup."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    sd = np.asarray(pred_std, dtype=float).reshape(-1) if pred_std is not None else None
    if len(measurements) != yt.size or yt.size != yp.size:
        raise SchemaError("measurements, y_true, and y_pred must have identical lengths.")

    curve = compute_calibration_curve(
        calibrator,
        yt,
        yp,
        pred_std=sd,
        coverage_levels=coverage_levels,
    )
    mace = float(np.mean([abs(pt.coverage_error) for pt in curve]))
    eff_alpha = calibrator.config.default_alpha if alpha is None else float(alpha)

    subgroups: dict[str, dict[str, CalibrationCurvePoint]] = {}
    for field_name in fields:
        buckets: dict[str, list[int]] = {}
        for idx, m in enumerate(measurements):
            if field_name == "soil_class_name":
                key = (
                    m.reference.soil_class_name
                    if m.reference is not None and m.reference.soil_class_name
                    else (m.soil_texture_class or "unspecified")
                )
            else:
                val = getattr(m, field_name, None)
                key = str(val) if val is not None else "unspecified"
            buckets.setdefault(key, []).append(idx)

        field_map: dict[str, CalibrationCurvePoint] = {}
        for grp_key, idx_list in sorted(buckets.items()):
            idx_arr = np.asarray(idx_list, dtype=int)
            interval_grp = calibrator.predict_interval(
                yp[idx_arr],
                pred_std=sd[idx_arr] if sd is not None else None,
                alpha=eff_alpha,
            )
            field_map[grp_key] = _coverage_point(yt[idx_arr], interval_grp)
        subgroups[field_name] = field_map

    return CoverageReport(
        curve=curve,
        mean_abs_calibration_error=mace,
        nominal_alpha=eff_alpha,
        subgroup_coverage=subgroups,
    )
