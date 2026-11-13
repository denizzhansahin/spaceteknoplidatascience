"""Chemometric evaluation metrics, Bland-Altman agreement, and subgroup diagnostics (Section 16, ADR-005).

Provides:
- `compute_regression_metrics`: MAE, RMSE, Bias, R^2, RPIQ, and Bland-Altman 95% LoA
- `evaluate_by_subgroup`: metrics stratified by site_id, moisture_scenario, soil_class, instrument_profile_id
- `check_closure_diagnostics`: physical TC = SOC + SIC closure residual & non-negativity audit
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from st_ds.domain.errors import SchemaError
from st_ds.domain.schemas import Measurement


@dataclass(frozen=True)
class RegressionMetrics:
    """Comprehensive scientific regression & agreement metrics for soil carbon evaluation."""

    n_samples: int
    mae: float
    rmse: float
    bias: float
    r2: float | None
    rpiq: float | None
    bland_altman_mean_diff: float
    bland_altman_std_diff: float
    bland_altman_loa_lower: float
    bland_altman_loa_upper: float
    max_abs_closure_residual: float | None = None
    mean_abs_closure_residual: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_regression_metrics(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    *,
    closure_residuals: np.ndarray | Sequence[float] | None = None,
) -> RegressionMetrics:
    """Compute MAE, RMSE, Bias, R^2, RPIQ, and Bland-Altman limits of agreement."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.size == 0 or yt.shape != yp.shape:
        raise SchemaError(
            "y_true and y_pred must be non-empty 1D arrays of identical length.",
            details={"y_true_shape": yt.shape, "y_pred_shape": yp.shape},
        )
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        raise SchemaError("y_true and y_pred must contain only finite values.")

    diff = yp - yt  # Positive bias means prediction overestimates reference
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    bias = float(np.mean(diff))

    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2: float | None = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None

    q75, q25 = np.percentile(yt, [75.0, 25.0])
    iqr = float(q75 - q25)
    rpiq: float | None = float(iqr / rmse) if rmse > 1e-12 else None

    ba_mean = bias
    ba_std = float(np.std(diff, ddof=1)) if yt.size > 1 else 0.0
    ba_lower = ba_mean - 1.96 * ba_std
    ba_upper = ba_mean + 1.96 * ba_std

    max_clos: float | None = None
    mean_clos: float | None = None
    if closure_residuals is not None:
        cr = np.asarray(closure_residuals, dtype=float).reshape(-1)
        if cr.shape != yt.shape:
            raise SchemaError("closure_residuals length must match y_true length.")
        max_clos = float(np.max(np.abs(cr)))
        mean_clos = float(np.mean(np.abs(cr)))

    return RegressionMetrics(
        n_samples=int(yt.size),
        mae=mae,
        rmse=rmse,
        bias=bias,
        r2=r2,
        rpiq=rpiq,
        bland_altman_mean_diff=ba_mean,
        bland_altman_std_diff=ba_std,
        bland_altman_loa_lower=ba_lower,
        bland_altman_loa_upper=ba_upper,
        max_abs_closure_residual=max_clos,
        mean_abs_closure_residual=mean_clos,
    )


def evaluate_by_subgroup(
    measurements: Sequence[Measurement],
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    *,
    fields: Sequence[str] = ("site_id", "moisture_scenario", "instrument_profile_id", "soil_class_name"),
    closure_residuals: np.ndarray | Sequence[float] | None = None,
) -> dict[str, dict[str, RegressionMetrics]]:
    """Evaluate regression metrics stratified across physical and operational subgroups."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    cr = np.asarray(closure_residuals, dtype=float).reshape(-1) if closure_residuals is not None else None
    if len(measurements) != yt.size or yt.size != yp.size:
        raise SchemaError("measurements, y_true, and y_pred must have matching lengths.")

    result: dict[str, dict[str, RegressionMetrics]] = {}
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

        field_metrics: dict[str, RegressionMetrics] = {}
        for group_val, indices in sorted(buckets.items()):
            idx_arr = np.asarray(indices, dtype=int)
            field_metrics[group_val] = compute_regression_metrics(
                yt[idx_arr],
                yp[idx_arr],
                closure_residuals=cr[idx_arr] if cr is not None else None,
            )
        result[field_name] = field_metrics

    return result


def check_closure_diagnostics(
    soc_pred: np.ndarray | Sequence[float],
    sic_pred: np.ndarray | Sequence[float],
    tc_pred: np.ndarray | Sequence[float] | None = None,
    *,
    tolerance_pct: float = 0.05,
) -> dict[str, Any]:
    """Audit predicted SOC/SIC/TC physical non-negativity and closure residual."""
    soc = np.asarray(soc_pred, dtype=float).reshape(-1)
    sic = np.asarray(sic_pred, dtype=float).reshape(-1)
    if soc.shape != sic.shape or soc.size == 0:
        raise SchemaError("soc_pred and sic_pred must have identical non-empty shapes.")

    tc_sum = soc + sic
    if tc_pred is not None:
        tc = np.asarray(tc_pred, dtype=float).reshape(-1)
        if tc.shape != soc.shape:
            raise SchemaError("tc_pred shape must match soc_pred.")
        res = tc - tc_sum
    else:
        res = np.zeros_like(soc)

    neg_soc = int(np.sum(soc < 0.0))
    neg_sic = int(np.sum(sic < 0.0))
    violations = int(np.sum(np.abs(res) > tolerance_pct))
    return {
        "n_samples": int(soc.size),
        "negative_soc_count": neg_soc,
        "negative_sic_count": neg_sic,
        "max_abs_closure_residual_pct": float(np.max(np.abs(res))),
        "mean_abs_closure_residual_pct": float(np.mean(np.abs(res))),
        "closure_violation_count": violations,
        "closure_violation_rate": float(violations / soc.size),
        "tolerance_pct": float(tolerance_pct),
    }
