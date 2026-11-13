"""Normalized Hilbert-Schmidt Independence Criterion (HSIC) dependence metrics (Section 21, DEF-004, ADR-006).

Implements centered RBF Gram matrix HSIC:
    HSIC(K, L) = tr(K H L H) / (n - 1)^2,  H = I - (1 / n) 1 1^T
    nHSIC(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

Strictly rejects sample counts B < 4 by returning `insufficient_sample_count` instead of
fabricated uniform feature weights (`DEF-004b`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from st_ds.domain.errors import SchemaError
from st_ds.domain.quality import QualityFlag

HSICStatus = Literal["ok", "low_sample_warning", "insufficient_sample_count"]

CAUSAL_DISCLAIMER = (
    "HSIC measures statistical dependence across the analyzed dataset/batch, not causal effect; "
    "confounding from site_id, soil texture, and moisture regime must be examined."
)


@dataclass(frozen=True)
class HSICResult:
    """Structured result from dataset/batch HSIC statistical dependence calculation."""

    status: HSICStatus
    n_samples: int
    quality_flags: tuple[QualityFlag, ...]
    raw_hsic: float | None
    normalized_hsic: float | None
    feature_dependence: dict[str, float] | None = None
    disclaimer: str = CAUSAL_DISCLAIMER
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_samples": self.n_samples,
            "quality_flags": [f.value for f in self.quality_flags],
            "raw_hsic": self.raw_hsic,
            "normalized_hsic": self.normalized_hsic,
            "feature_dependence": dict(self.feature_dependence) if self.feature_dependence is not None else None,
            "disclaimer": self.disclaimer,
            "notes": list(self.notes),
        }


def _pairwise_sq_dists_1d_or_2d(Z: np.ndarray) -> np.ndarray:
    arr = np.asarray(Z, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    a_sq = np.sum(arr * arr, axis=1, keepdims=True)
    d2 = a_sq + a_sq.T - 2.0 * np.dot(arr, arr.T)
    return np.maximum(d2, 0.0)


def _centered_rbf_gram(Z: np.ndarray, sigma: float | None = None) -> tuple[np.ndarray, float]:
    """Compute H K H where K is an RBF Gram matrix, without explicit O(n^3) matmul."""
    d2 = _pairwise_sq_dists_1d_or_2d(Z)
    n = d2.shape[0]
    if sigma is None:
        triu = d2[np.triu_indices(n, k=1)]
        pos = np.sqrt(triu[triu > 0.0])
        eff_sigma = float(np.median(pos)) if pos.size > 0 else 1.0
    else:
        eff_sigma = float(sigma)

    if eff_sigma <= 1e-12:
        return np.zeros((n, n), dtype=float), 0.0

    K = np.exp(-d2 / (2.0 * eff_sigma * eff_sigma))
    K_centered = K - np.mean(K, axis=1, keepdims=True) - np.mean(K, axis=0, keepdims=True) + np.mean(K)
    return K_centered, eff_sigma


def _hsic_from_centered(K_c: np.ndarray, L_c: np.ndarray) -> tuple[float, float]:
    n = K_c.shape[0]
    denom = float((n - 1) ** 2)
    raw = float(np.sum(K_c * L_c) / denom)
    kk = float(np.sum(K_c * K_c) / denom)
    ll = float(np.sum(L_c * L_c) / denom)
    if kk <= 1e-15 or ll <= 1e-15:
        return 0.0, 0.0
    norm = float(np.clip(raw / np.sqrt(kk * ll), 0.0, 1.0))
    return max(0.0, raw), norm


def compute_hsic(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    *,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
) -> HSICResult:
    """Compute normalized RBF HSIC between variables `x` (N, [D]) and `y` (N, [Dy]).

    Strictly returns `status='insufficient_sample_count'` when `n_samples < 4` (DEF-004b).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim == 0 or y_arr.ndim == 0 or x_arr.shape[0] != y_arr.shape[0]:
        raise SchemaError(
            "x and y must be non-scalar arrays with identical leading sample counts.",
            details={"x_shape": x_arr.shape, "y_shape": y_arr.shape},
        )
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise SchemaError("compute_hsic inputs contain NaN or Inf values.")

    n = int(x_arr.shape[0])
    if n < 4:
        return HSICResult(
            status="insufficient_sample_count",
            n_samples=n,
            quality_flags=(QualityFlag.INSUFFICIENT_SAMPLE_COUNT,),
            raw_hsic=None,
            normalized_hsic=None,
            feature_dependence=None,
            notes=(
                f"HSIC requires at least 4 samples (recommended >= 8); received n={n}. "
                "Returned None instead of fabricated equal weights (DEF-004b).",
            ),
        )

    # Constant feature/target carries zero statistical dependence
    if float(np.ptp(x_arr)) <= 1e-12 or float(np.ptp(y_arr)) <= 1e-12:
        status: HSICStatus = "low_sample_warning" if n < 8 else "ok"
        notes = (
            (f"Low sample count (n={n} < 8); interpret HSIC with caution.",)
            if n < 8
            else ()
        )
        return HSICResult(
            status=status,
            n_samples=n,
            quality_flags=(),
            raw_hsic=0.0,
            normalized_hsic=0.0,
            notes=notes,
        )

    K_c, _ = _centered_rbf_gram(x_arr, sigma=sigma_x)
    L_c, _ = _centered_rbf_gram(y_arr, sigma=sigma_y)
    raw_val, norm_val = _hsic_from_centered(K_c, L_c)

    if n < 8:
        return HSICResult(
            status="low_sample_warning",
            n_samples=n,
            quality_flags=(),
            raw_hsic=raw_val,
            normalized_hsic=norm_val,
            notes=(f"Low sample count (n={n} < 8); statistical power is limited.",),
        )
    return HSICResult(
        status="ok",
        n_samples=n,
        quality_flags=(),
        raw_hsic=raw_val,
        normalized_hsic=norm_val,
    )


def compute_feature_hsic_ranking(
    X: np.ndarray,
    y: np.ndarray | Sequence[float],
    *,
    feature_names: Sequence[str] | None = None,
) -> HSICResult:
    """Compute per-feature normalized HSIC dependence against target `y` across a batch X (N, D)."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.size:
        raise SchemaError(
            "X must be 2D (n_samples, n_features) with n_samples matching y.",
            details={"X_shape": X_arr.shape, "y_len": int(y_arr.size)},
        )
    if not (np.all(np.isfinite(X_arr)) and np.all(np.isfinite(y_arr))):
        raise SchemaError("compute_feature_hsic_ranking inputs contain NaN or Inf values.")

    n, d = X_arr.shape
    names = tuple(feature_names) if feature_names is not None else tuple(f"feature_{j}" for j in range(d))
    if len(names) != d:
        raise SchemaError(f"feature_names length ({len(names)}) must match X columns ({d}).")

    if n < 4:
        return HSICResult(
            status="insufficient_sample_count",
            n_samples=n,
            quality_flags=(QualityFlag.INSUFFICIENT_SAMPLE_COUNT,),
            raw_hsic=None,
            normalized_hsic=None,
            feature_dependence=None,
            notes=(
                f"Batch feature HSIC requires at least 4 samples (recommended >= 8); received n={n}. "
                "Returned None instead of equal importance weights (DEF-004b).",
            ),
        )

    joint_res = compute_hsic(X_arr, y_arr)
    L_c, _ = _centered_rbf_gram(y_arr)
    scores: dict[str, float] = {}
    for j, fname in enumerate(names):
        col = X_arr[:, j]
        if float(np.ptp(col)) <= 1e-12 or float(np.ptp(y_arr)) <= 1e-12:
            scores[fname] = 0.0
        else:
            K_c, _ = _centered_rbf_gram(col)
            _, n_val = _hsic_from_centered(K_c, L_c)
            scores[fname] = n_val

    return HSICResult(
        status=joint_res.status,
        n_samples=n,
        quality_flags=joint_res.quality_flags,
        raw_hsic=joint_res.raw_hsic,
        normalized_hsic=joint_res.normalized_hsic,
        feature_dependence=scores,
        notes=joint_res.notes,
    )
