"""True second-order variable interaction metrics (Section 21, DEF-004c, ADR-006).

Replaces the defective `coupling_strength = max|grad_s| * max|grad_c|` product with
the mixed partial derivative (autograd Hessian off-diagonal or symmetric 2D perturbation):
    I_ij(x) = |d^2 f / (dx_i dx_j)| * sigma_i * sigma_j

Guarantees that purely additive models f(x_1, x_2) = a*x_1 + b*x_2 yield 0.0 interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from st_ds.domain.errors import SchemaError
from st_ds.domain.protocols import RegressorProtocol
from st_ds.models.hybrid_net import STSoilHybridNet
from st_ds.xai.sensitivity import (
    TARGET_TO_HYBRID_HEAD,
    eval_scalar_predictions,
    resolve_torch_network,
)


@dataclass(frozen=True)
class PairwiseInteraction:
    """Pairwise second-order interaction strength between two named features."""

    feature_i: str
    feature_j: str
    index_i: int
    index_j: int
    interaction_score: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_i": self.feature_i,
            "feature_j": self.feature_j,
            "index_i": self.index_i,
            "index_j": self.index_j,
            "interaction_score": float(self.interaction_score),
            "method": self.method,
        }


def _torch_mixed_partial(
    net: STSoilHybridNet,
    x: np.ndarray,
    idx_i: int,
    idx_j: int,
    *,
    target_name: str,
) -> float:
    head_key = TARGET_TO_HYBRID_HEAD.get(target_name, "soc_mean")
    n_spec = net.config.n_spectral_features
    n_cov = net.config.n_covariate_features
    if x.size != n_spec + n_cov:
        raise SchemaError(f"Expected {n_spec + n_cov} features, got {x.size}.")

    was_training = net.training
    net.eval()
    t_x = torch.tensor(x[None, :], dtype=torch.float32, requires_grad=True)
    t_s = t_x[:, :n_spec]
    t_c = t_x[:, n_spec:] if n_cov > 0 else None
    out = net(t_s, t_c)
    y = out[head_key].sum()
    grad1 = torch.autograd.grad(y, t_x, create_graph=True)[0][0, idx_i]
    grad2 = torch.autograd.grad(grad1, t_x, retain_graph=False)[0][0, idx_j]
    if was_training:
        net.train()
    return float(abs(grad2.detach().cpu().item()))


def _finite_diff_mixed_partial(
    predict_or_model: Any,
    x: np.ndarray,
    idx_i: int,
    idx_j: int,
    *,
    delta_i: float,
    delta_j: float,
    target_name: str,
    target_index: int,
) -> float:
    """Symmetric 4-point central 2D mixed partial derivative |d^2 f / (dx_i dx_j)|."""
    X_pert = np.repeat(x[None, :], 4, axis=0)
    # (+, +), (+, -), (-, +), (-, -)
    X_pert[0, idx_i] += delta_i
    X_pert[0, idx_j] += delta_j

    X_pert[1, idx_i] += delta_i
    X_pert[1, idx_j] -= delta_j

    X_pert[2, idx_i] -= delta_i
    X_pert[2, idx_j] += delta_j

    X_pert[3, idx_i] -= delta_i
    X_pert[3, idx_j] -= delta_j

    f_vals = eval_scalar_predictions(
        predict_or_model,
        X_pert,
        target_name=target_name,
        target_index=target_index,
    )
    mixed = (f_vals[0] - f_vals[1] - f_vals[2] + f_vals[3]) / (4.0 * delta_i * delta_j)
    return float(abs(mixed))


def interaction_score(
    predict_or_model: RegressorProtocol | Callable[[np.ndarray], np.ndarray] | STSoilHybridNet,
    x: np.ndarray | Sequence[float],
    idx_i: int,
    idx_j: int,
    *,
    feature_scales: np.ndarray | Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    target_name: str = "soc_pct",
    target_index: int = 0,
    step: float = 1e-2,
) -> PairwiseInteraction:
    """Compute true scale-normalized pairwise interaction |d^2 f / (dx_i dx_j)| * sigma_i * sigma_j."""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 2 and x_arr.shape[0] == 1:
        x_arr = x_arr[0]
    if x_arr.ndim != 1 or x_arr.size < 2 or not np.all(np.isfinite(x_arr)):
        raise SchemaError("x must be a finite 1D vector with at least 2 features.")

    d = x_arr.size
    if idx_i == idx_j:
        raise SchemaError("idx_i and idx_j must be distinct feature indices.")
    if not (0 <= idx_i < d and 0 <= idx_j < d):
        raise SchemaError(f"Feature indices ({idx_i}, {idx_j}) out of bounds for dimension {d}.")

    if feature_scales is None:
        scales = np.ones(d, dtype=float)
    else:
        scales = np.asarray(feature_scales, dtype=float).reshape(-1)
        if scales.size != d or np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
            raise SchemaError("feature_scales must be positive finite values matching x dimension.")

    names = tuple(feature_names) if feature_names is not None else tuple(f"feature_{k}" for k in range(d))
    if len(names) != d:
        raise SchemaError("feature_names length must match x dimension.")

    scale_i = float(scales[idx_i])
    scale_j = float(scales[idx_j])
    net = resolve_torch_network(predict_or_model)
    if net is not None:
        raw_mixed = _torch_mixed_partial(net, x_arr, idx_i, idx_j, target_name=target_name)
        method = "mixed_partial_autograd"
    else:
        delta_i = float(step * max(scale_i, 1.0))
        delta_j = float(step * max(scale_j, 1.0))
        raw_mixed = _finite_diff_mixed_partial(
            predict_or_model,
            x_arr,
            idx_i,
            idx_j,
            delta_i=delta_i,
            delta_j=delta_j,
            target_name=target_name,
            target_index=target_index,
        )
        method = "2d_perturbation"

    return PairwiseInteraction(
        feature_i=names[idx_i],
        feature_j=names[idx_j],
        index_i=int(idx_i),
        index_j=int(idx_j),
        interaction_score=float(raw_mixed * scale_i * scale_j),
        method=method,
    )


def compute_top_interactions(
    predict_or_model: RegressorProtocol | Callable[[np.ndarray], np.ndarray] | STSoilHybridNet,
    x: np.ndarray | Sequence[float],
    *,
    candidate_pairs: Sequence[tuple[int, int]] | None = None,
    feature_scales: np.ndarray | Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    target_name: str = "soc_pct",
    target_index: int = 0,
    max_pairs: int = 8,
    step: float = 1e-2,
) -> tuple[PairwiseInteraction, ...]:
    """Evaluate candidate pairwise second-order interactions sorted descending by score."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    d = x_arr.size
    if d < 2:
        return ()

    if candidate_pairs is None:
        limit = min(4, d)
        pairs = [(i, j) for i in range(limit) for j in range(i + 1, limit)]
    else:
        pairs = list(candidate_pairs)

    results = [
        interaction_score(
            predict_or_model,
            x_arr,
            i,
            j,
            feature_scales=feature_scales,
            feature_names=feature_names,
            target_name=target_name,
            target_index=target_index,
            step=step,
        )
        for i, j in pairs
    ]
    results.sort(key=lambda r: r.interaction_score, reverse=True)
    return tuple(results[:max_pairs])
