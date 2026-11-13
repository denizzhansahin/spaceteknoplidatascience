"""Ensemble disagreement and aleatoric vs epistemic uncertainty decomposition (Section 20 & 33).

Implements:
- `estimate_ensemble_spread`: separates model disagreement (epistemic) from mean
  predicted intra-model variance (aleatoric) into `EnsembleSpreadResult`
- `predict_ensemble_distribution`: aggregates predictions from RegressorProtocol members
- `GroupedBootstrapEnsemble`: cluster-level bootstrap resampling by site/sample group
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from st_ds.domain.errors import CalibrationError, ModelInferenceError
from st_ds.domain.protocols import RegressorProtocol
from st_ds.models.base import validate_xy_arrays


@dataclass(frozen=True)
class EnsembleSpreadResult:
    """Decomposed uncertainty estimates across an ensemble of regression models."""

    mean_prediction: np.ndarray
    aleatoric_std: np.ndarray
    epistemic_std: np.ndarray
    total_std: np.ndarray
    n_members: int
    has_aleatoric_component: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_members": self.n_members,
            "has_aleatoric_component": self.has_aleatoric_component,
            "mean_prediction": self.mean_prediction.tolist(),
            "aleatoric_std": self.aleatoric_std.tolist(),
            "epistemic_std": self.epistemic_std.tolist(),
            "total_std": self.total_std.tolist(),
        }


def estimate_ensemble_spread(
    member_means: np.ndarray | Sequence[np.ndarray],
    member_stds: np.ndarray | Sequence[np.ndarray] | None = None,
    *,
    min_std: float = 1e-4,
) -> EnsembleSpreadResult:
    """Decompose ensemble predictions into epistemic (between-member) and aleatoric spread.

    Parameters
    ----------
    member_means:
        Array of shape ``(n_members, n_samples)`` or ``(n_members, n_samples, n_targets)``.
    member_stds:
        Optional array of per-member predictive standard deviations with matching shape.
        When omitted, `aleatoric_std` defaults to `0.0` (only epistemic spread is estimated).
    """
    means = np.asarray(member_means, dtype=float)
    if means.ndim not in (2, 3) or means.shape[0] < 2 or means.shape[1] == 0:
        raise ModelInferenceError(
            "member_means must have shape (n_members >= 2, n_samples, [n_targets]).",
            details={"shape": means.shape},
        )
    if not np.all(np.isfinite(means)):
        raise ModelInferenceError("member_means contains NaN or Inf values.")

    mean_pred = np.mean(means, axis=0)
    epistemic_var = np.var(means, axis=0, ddof=0)
    epistemic_std = np.sqrt(np.maximum(epistemic_var, 0.0))

    has_aleatoric = member_stds is not None
    if member_stds is not None:
        stds = np.asarray(member_stds, dtype=float)
        if stds.shape != means.shape or not np.all(np.isfinite(stds)):
            raise ModelInferenceError(
                "member_stds must be finite and match member_means shape exactly.",
                details={"means_shape": means.shape, "stds_shape": stds.shape},
            )
        aleatoric_var = np.mean(stds * stds, axis=0)
        aleatoric_std = np.sqrt(np.maximum(aleatoric_var, 0.0))
    else:
        aleatoric_var = np.zeros_like(epistemic_var)
        aleatoric_std = np.zeros_like(epistemic_std)

    total_std = np.maximum(np.sqrt(aleatoric_var + epistemic_var), float(min_std))
    return EnsembleSpreadResult(
        mean_prediction=mean_pred,
        aleatoric_std=aleatoric_std,
        epistemic_std=epistemic_std,
        total_std=total_std,
        n_members=int(means.shape[0]),
        has_aleatoric_component=has_aleatoric,
    )


def predict_ensemble_distribution(
    models: Sequence[RegressorProtocol],
    X: np.ndarray,
    *,
    min_std: float = 1e-4,
) -> EnsembleSpreadResult:
    """Collect `predict_distribution` outputs from RegressorProtocol models and decompose spread."""
    if len(models) < 2:
        raise ModelInferenceError(
            f"Ensemble spread requires at least 2 fitted models, got {len(models)}."
        )
    means_list: list[np.ndarray] = []
    stds_list: list[np.ndarray] = []
    for model in models:
        mu, sigma = model.predict_distribution(X)
        means_list.append(np.asarray(mu, dtype=float))
        stds_list.append(np.asarray(sigma, dtype=float))
    return estimate_ensemble_spread(means_list, stds_list, min_std=min_std)


class GroupedBootstrapEnsemble:
    """Cluster-level bootstrap ensemble preserving sample/site group integrity (Section 20)."""

    def __init__(
        self,
        *,
        n_bootstraps: int = 10,
        seed: int = 42,
        min_std: float = 1e-4,
    ) -> None:
        if n_bootstraps < 2:
            raise ModelInferenceError("GroupedBootstrapEnsemble.n_bootstraps must be >= 2.")
        self.n_bootstraps = int(n_bootstraps)
        self.seed = int(seed)
        self.min_std = float(min_std)
        self.members_: list[RegressorProtocol] = []
        self._is_fitted: bool = False

    def fit(
        self,
        model_factory: Callable[[int], RegressorProtocol],
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | Sequence[str] | None = None,
    ) -> GroupedBootstrapEnsemble:
        """Fit bootstrap members by resampling clusters (`groups`) rather than individual replicates."""
        X_arr, y_2d, is_1d = validate_xy_arrays(X, y)
        assert y_2d is not None
        y_fit = y_2d[:, 0] if is_1d else y_2d
        n_samples = X_arr.shape[0]

        rng = np.random.default_rng(self.seed)
        self.members_ = []

        if groups is not None:
            grp_arr = np.asarray(groups, dtype=str)
            if grp_arr.shape[0] != n_samples:
                raise ModelInferenceError("groups length must match X sample count.")
            unique_groups = np.unique(grp_arr)
            if unique_groups.size < 2:
                raise ModelInferenceError("Grouped bootstrap requires at least 2 distinct groups.")
            group_to_indices = {g: np.flatnonzero(grp_arr == g) for g in unique_groups}
        else:
            unique_groups = None
            group_to_indices = {}

        for b in range(self.n_bootstraps):
            if unique_groups is not None:
                # Resample whole groups with replacement until reaching >= 2 distinct groups
                sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
                if np.unique(sampled_groups).size < 2:
                    sampled_groups = np.concatenate([sampled_groups[:-1], [unique_groups[(b + 1) % unique_groups.size]]])
                boot_idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
            else:
                boot_idx = rng.integers(0, n_samples, size=n_samples)

            model = model_factory(self.seed + b)
            model.fit(X_arr[boot_idx], y_fit[boot_idx])
            self.members_.append(model)

        self._is_fitted = True
        return self

    def predict_distribution(self, X: np.ndarray) -> EnsembleSpreadResult:
        """Predict decomposed mean, aleatoric std, epistemic std, and total std."""
        if not self._is_fitted or not self.members_:
            raise CalibrationError(
                "GroupedBootstrapEnsemble must be fitted before calling predict_distribution."
            )
        return predict_ensemble_distribution(self.members_, X, min_std=self.min_std)
