"""Base contracts, array validation, and physical non-negativity guards for STDS models (ADR-004, ADR-005).

All baseline and neural models inherit from `BaseSoilRegressor` and adhere to
`st_ds.domain.protocols.RegressorProtocol`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from st_ds.domain.errors import ModelInferenceError


@dataclass(frozen=True)
class ModelPrediction:
    """Structured multi-target soil carbon and classification prediction container."""

    soc_mean_pct: np.ndarray
    sic_mean_pct: np.ndarray | None = None
    tc_mean_pct: np.ndarray | None = None
    soc_std_pct: np.ndarray | None = None
    sic_std_pct: np.ndarray | None = None
    tc_std_pct: np.ndarray | None = None
    closure_residual_pct: np.ndarray | None = None
    class_logits: np.ndarray | None = None
    class_probs: np.ndarray | None = None
    clipped_negative_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def validate_xy_arrays(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    expected_n_features: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Validate finite 2D feature matrix X and optional 1D/2D target array y.

    Returns ``(X_arr, y_arr, is_1d_target)``.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ModelInferenceError(
            "Feature matrix X must be a non-empty 2D array of shape (n_samples, n_features).",
            details={"shape": X_arr.shape},
        )
    if not np.all(np.isfinite(X_arr)):
        raise ModelInferenceError("Feature matrix X contains NaN or Inf values.")
    if expected_n_features is not None and X_arr.shape[1] != expected_n_features:
        raise ModelInferenceError(
            f"Expected {expected_n_features} features, got {X_arr.shape[1]}.",
            details={"expected_n_features": expected_n_features, "actual_n_features": int(X_arr.shape[1])},
        )

    if y is None:
        return X_arr, None, True

    y_raw = np.asarray(y, dtype=float)
    if y_raw.ndim not in (1, 2):
        raise ModelInferenceError(
            "Target array y must be 1D (n_samples,) or 2D (n_samples, n_targets).",
            details={"shape": y_raw.shape},
        )
    if y_raw.shape[0] != X_arr.shape[0]:
        raise ModelInferenceError(
            f"Sample count mismatch between X ({X_arr.shape[0]}) and y ({y_raw.shape[0]}).",
            details={"n_x": int(X_arr.shape[0]), "n_y": int(y_raw.shape[0])},
        )
    if not np.all(np.isfinite(y_raw)):
        raise ModelInferenceError("Target array y contains NaN or Inf values.")

    is_1d = y_raw.ndim == 1
    y_2d = y_raw[:, None] if is_1d else y_raw
    if y_2d.shape[1] == 0:
        raise ModelInferenceError("Target array y must contain at least one target column.")
    return X_arr, y_2d, is_1d


def enforce_non_negative_carbon(
    predictions: np.ndarray,
    *,
    clip_negative: bool = True,
) -> tuple[np.ndarray, int]:
    """Clamp physically impossible negative carbon concentrations to 0.0 when enabled (ADR-004).

    Returns ``(adjusted_predictions, negative_value_count)``.
    """
    arr = np.asarray(predictions, dtype=float)
    neg_mask = arr < 0.0
    neg_count = int(np.sum(neg_mask))
    if clip_negative and neg_count > 0:
        arr = np.maximum(arr, 0.0)
    return arr, neg_count


def compute_closure_residual(
    soc_pct: np.ndarray,
    sic_pct: np.ndarray,
    tc_pct: np.ndarray,
) -> np.ndarray:
    """Compute physical carbon closure residual r_TC = TC - SOC - SIC (ADR-004)."""
    return np.asarray(tc_pct, dtype=float) - np.asarray(soc_pct, dtype=float) - np.asarray(sic_pct, dtype=float)


class BaseSoilRegressor:
    """Shared base implementation for RegressorProtocol models in STDS."""

    model_name: str = "base_soil_regressor"
    model_family: str = "unknown"

    def __init__(
        self,
        *,
        target_names: Sequence[str] = ("soc_pct",),
        clip_negative: bool = True,
        min_std: float = 1e-4,
    ) -> None:
        self.target_names: tuple[str, ...] = tuple(target_names)
        self.clip_negative: bool = bool(clip_negative)
        self.min_std: float = float(min_std)
        self._is_fitted: bool = False
        self._is_1d_target: bool = True
        self._n_features_in: int = 0
        self._n_targets: int = 1
        self._last_negative_clip_count: int = 0
        self._feature_names: tuple[str, ...] = ()
        self._training_dataset_source: str = "unknown"

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ModelInferenceError(
                f"{self.__class__.__name__} must be fitted before calling predict or predict_distribution.",
                remediation="Call `.fit(X, y)` on training data before inference.",
            )

    def _format_output(self, preds_2d: np.ndarray, stds_2d: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        clipped, n_neg = enforce_non_negative_carbon(preds_2d, clip_negative=self.clip_negative)
        self._last_negative_clip_count = n_neg
        if self._is_1d_target and clipped.shape[1] == 1:
            mean_out = clipped[:, 0]
            std_out = np.maximum(stds_2d[:, 0], self.min_std) if stds_2d is not None else None
            return mean_out, std_out
        std_out = np.maximum(stds_2d, self.min_std) if stds_2d is not None else None
        return clipped, std_out

    def set_metadata(
        self,
        *,
        feature_names: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
        dataset_source: str | None = None,
    ) -> None:
        """Attach optional provenance metadata for `model_card()`."""
        if feature_names is not None:
            self._feature_names = tuple(feature_names)
        if target_names is not None:
            self.target_names = tuple(target_names)
        if dataset_source is not None:
            self._training_dataset_source = str(dataset_source)

    def get_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "target_names": list(self.target_names),
            "clip_negative": self.clip_negative,
            "min_std": self.min_std,
        }

    def model_card(self) -> dict[str, Any]:
        card: dict[str, Any] = {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "is_fitted": self._is_fitted,
            "n_features_in": self._n_features_in,
            "n_targets": self._n_targets,
            "target_names": list(self.target_names),
            "feature_names_count": len(self._feature_names),
            "clip_negative": self.clip_negative,
            "last_negative_clip_count": self._last_negative_clip_count,
            "training_dataset_source": self._training_dataset_source,
            "params": self.get_params(),
        }
        if self._training_dataset_source == "synthetic":
            card["warning"] = (
                "Model trained on synthetic data (`dataset_source='synthetic'`); "
                "valid for pipeline smoke tests only, not field validation evidence."
            )
        return card
