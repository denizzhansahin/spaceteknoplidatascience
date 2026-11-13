"""Local gradient sensitivity and Integrated Gradients for STDS models (Section 21, ADR-006).

Supports both differentiable PyTorch models (`STSoilHybridNet` / `STSoilHybridNetRegressor`)
via exact `torch.autograd` and arbitrary `RegressorProtocol` / Python callables via batched
central finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from st_ds.domain.errors import ModelInferenceError, SchemaError
from st_ds.domain.protocols import RegressorProtocol
from st_ds.models.hybrid_net import STSoilHybridNet, STSoilHybridNetRegressor


TARGET_TO_HYBRID_HEAD: dict[str, str] = {
    "soc_pct": "soc_mean",
    "sic_pct": "sic_mean",
    "tc_pct": "tc_mean",
}


@dataclass(frozen=True)
class SensitivityResult:
    """Result container for local sensitivity and Integrated Gradients attributions."""

    method: str
    target_name: str
    feature_names: tuple[str, ...]
    raw_gradients: np.ndarray
    normalized_sensitivity: np.ndarray
    attributions: np.ndarray | None
    sample_prediction: float
    baseline_prediction: float | None = None
    completeness_residual: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "target_name": self.target_name,
            "feature_names": list(self.feature_names),
            "raw_gradients": self.raw_gradients.tolist(),
            "normalized_sensitivity": self.normalized_sensitivity.tolist(),
            "attributions": self.attributions.tolist() if self.attributions is not None else None,
            "sample_prediction": float(self.sample_prediction),
            "baseline_prediction": float(self.baseline_prediction) if self.baseline_prediction is not None else None,
            "completeness_residual": float(self.completeness_residual) if self.completeness_residual is not None else None,
        }


def resolve_torch_network(model: Any) -> STSoilHybridNet | None:
    """Return underlying `STSoilHybridNet` if available and initialized, else None."""
    if isinstance(model, STSoilHybridNet):
        return model
    if isinstance(model, STSoilHybridNetRegressor):
        if not model._is_fitted or model.net_ is None:
            raise ModelInferenceError("STSoilHybridNetRegressor must be fitted before XAI computation.")
        return model.net_
    return None


def eval_scalar_predictions(
    predict_or_model: RegressorProtocol | Callable[[np.ndarray], np.ndarray] | STSoilHybridNet,
    X: np.ndarray,
    *,
    target_name: str = "soc_pct",
    target_index: int = 0,
) -> np.ndarray:
    """Evaluate a 1D vector of scalar target predictions for batch `X` of shape (N, D)."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or not np.all(np.isfinite(X_arr)):
        raise ModelInferenceError("X must be a non-empty finite 2D array for XAI evaluation.")

    net = resolve_torch_network(predict_or_model)
    if net is not None:
        head_key = TARGET_TO_HYBRID_HEAD.get(target_name, "soc_mean")
        n_spec = net.config.n_spectral_features
        n_cov = net.config.n_covariate_features
        if X_arr.shape[1] != n_spec + n_cov:
            raise SchemaError(
                f"Expected {n_spec + n_cov} features for STSoilHybridNet, got {X_arr.shape[1]}."
            )
        was_training = net.training
        net.eval()
        with torch.no_grad():
            t_s = torch.tensor(X_arr[:, :n_spec], dtype=torch.float32)
            t_c = torch.tensor(X_arr[:, n_spec:], dtype=torch.float32) if n_cov > 0 else None
            out = net(t_s, t_c)
            vals = out[head_key].detach().cpu().numpy().astype(float)
        if was_training:
            net.train()
        return vals.reshape(-1)

    if hasattr(predict_or_model, "predict") and callable(predict_or_model.predict):
        raw = np.asarray(predict_or_model.predict(X_arr), dtype=float)
    elif callable(predict_or_model):
        raw = np.asarray(predict_or_model(X_arr), dtype=float)
    else:
        raise ModelInferenceError("predict_or_model must implement `.predict(X)` or be callable.")

    if raw.ndim == 1:
        return raw
    if raw.ndim == 2:
        if not (0 <= target_index < raw.shape[1]):
            raise SchemaError(
                f"target_index={target_index} out of bounds for prediction shape {raw.shape}."
            )
        return raw[:, target_index]
    raise ModelInferenceError(f"Unsupported prediction output shape {raw.shape}.")


def _torch_input_gradients(
    net: STSoilHybridNet,
    X: np.ndarray,
    *,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact autograd gradients df/dx for a batch (N, D) on `STSoilHybridNet`."""
    head_key = TARGET_TO_HYBRID_HEAD.get(target_name)
    if head_key is None:
        raise SchemaError(
            f"Unsupported hybrid net target_name '{target_name}'. Expected one of {list(TARGET_TO_HYBRID_HEAD)}."
        )
    n_spec = net.config.n_spectral_features
    n_cov = net.config.n_covariate_features
    if X.shape[1] != n_spec + n_cov:
        raise SchemaError(f"Expected {n_spec + n_cov} features, got {X.shape[1]}.")

    was_training = net.training
    net.eval()
    t_x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    t_s = t_x[:, :n_spec]
    t_c = t_x[:, n_spec:] if n_cov > 0 else None
    out = net(t_s, t_c)
    preds = out[head_key]
    grads = torch.autograd.grad(preds.sum(), t_x)[0]
    if was_training:
        net.train()
    return (
        grads.detach().cpu().numpy().astype(float),
        preds.detach().cpu().numpy().astype(float).reshape(-1),
    )


def _finite_diff_gradients(
    predict_or_model: Any,
    x: np.ndarray,
    *,
    scales: np.ndarray,
    eps: float,
    target_name: str,
    target_index: int,
) -> np.ndarray:
    """Compute central finite-difference gradient vector for a single sample `x` of shape (D,)."""
    d = x.size
    steps = eps * np.maximum(scales, 1.0)
    X_pert = np.repeat(x[None, :], 2 * d, axis=0)
    idx = np.arange(d)
    X_pert[2 * idx, idx] += steps
    X_pert[2 * idx + 1, idx] -= steps
    y_pert = eval_scalar_predictions(
        predict_or_model,
        X_pert,
        target_name=target_name,
        target_index=target_index,
    )
    return (y_pert[0::2] - y_pert[1::2]) / (2.0 * steps)


def _prepare_sample_and_scales(
    x: np.ndarray | Sequence[float],
    feature_scales: np.ndarray | Sequence[float] | None,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 2 and x_arr.shape[0] == 1:
        x_arr = x_arr[0]
    if x_arr.ndim != 1 or x_arr.size == 0 or not np.all(np.isfinite(x_arr)):
        raise SchemaError("Sample x for single-instance XAI must be a non-empty finite 1D vector.")
    d = x_arr.size

    if feature_scales is None:
        scales = np.ones(d, dtype=float)
    else:
        scales = np.asarray(feature_scales, dtype=float).reshape(-1)
        if scales.size != d or np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
            raise SchemaError("feature_scales must be positive finite values matching x dimension.")

    if feature_names is None:
        names = tuple(f"feature_{i}" for i in range(d))
    else:
        names = tuple(feature_names)
        if len(names) != d:
            raise SchemaError(f"feature_names length ({len(names)}) must match feature dimension ({d}).")
    return x_arr, scales, names


def local_sensitivity(
    predict_or_model: RegressorProtocol | Callable[[np.ndarray], np.ndarray] | STSoilHybridNet,
    x: np.ndarray | Sequence[float],
    *,
    feature_scales: np.ndarray | Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    target_name: str = "soc_pct",
    target_index: int = 0,
    eps: float = 1e-4,
) -> SensitivityResult:
    """Compute raw gradients df/dx and unit/scale-normalized local sensitivity sigma_i * |df/dx_i|."""
    x_arr, scales, names = _prepare_sample_and_scales(x, feature_scales, feature_names)
    net = resolve_torch_network(predict_or_model)
    if net is not None:
        grads_2d, preds = _torch_input_gradients(net, x_arr[None, :], target_name=target_name)
        grads = grads_2d[0]
        pred_val = float(preds[0])
        method = "autograd"
    else:
        grads = _finite_diff_gradients(
            predict_or_model,
            x_arr,
            scales=scales,
            eps=eps,
            target_name=target_name,
            target_index=target_index,
        )
        pred_val = float(
            eval_scalar_predictions(
                predict_or_model,
                x_arr[None, :],
                target_name=target_name,
                target_index=target_index,
            )[0]
        )
        method = "central_difference"

    norm_sens = scales * np.abs(grads)
    return SensitivityResult(
        method=method,
        target_name=target_name,
        feature_names=names,
        raw_gradients=grads,
        normalized_sensitivity=norm_sens,
        attributions=None,
        sample_prediction=pred_val,
    )


def integrated_gradients(
    predict_or_model: RegressorProtocol | Callable[[np.ndarray], np.ndarray] | STSoilHybridNet,
    x: np.ndarray | Sequence[float],
    *,
    baseline: np.ndarray | Sequence[float] | None = None,
    n_steps: int = 32,
    feature_scales: np.ndarray | Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    target_name: str = "soc_pct",
    target_index: int = 0,
    eps: float = 1e-4,
) -> SensitivityResult:
    """Compute path-integrated gradients IG_i(x) = (x_i - x0_i) * int_0^1 df(x0 + a(x-x0))/dx_i da."""
    if n_steps < 4:
        raise SchemaError(f"integrated_gradients requires n_steps >= 4, got {n_steps}.")
    x_arr, scales, names = _prepare_sample_and_scales(x, feature_scales, feature_names)

    if baseline is None:
        x0 = np.zeros_like(x_arr)
    else:
        x0 = np.asarray(baseline, dtype=float).reshape(-1)
        if x0.shape != x_arr.shape or not np.all(np.isfinite(x0)):
            raise SchemaError("baseline must be finite and match sample dimension.")

    diff = x_arr - x0
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_steps)
    alphas = 0.5 * (gl_nodes + 1.0)
    weights = 0.5 * gl_weights
    path_X = x0[None, :] + alphas[:, None] * diff[None, :]

    net = resolve_torch_network(predict_or_model)
    if net is not None:
        path_grads, _ = _torch_input_gradients(net, path_X, target_name=target_name)
        method = "integrated_gradients_autograd"
    else:
        path_grads = np.vstack(
            [
                _finite_diff_gradients(
                    predict_or_model,
                    path_X[k],
                    scales=scales,
                    eps=eps,
                    target_name=target_name,
                    target_index=target_index,
                )
                for k in range(n_steps)
            ]
        )
        method = "integrated_gradients_central_difference"

    avg_grad = np.dot(weights, path_grads)
    ig = diff * avg_grad
    y_pair = eval_scalar_predictions(
        predict_or_model,
        np.vstack([x0, x_arr]),
        target_name=target_name,
        target_index=target_index,
    )
    f_x0 = float(y_pair[0])
    f_x = float(y_pair[1])
    completeness_res = float(np.sum(ig) - (f_x - f_x0))

    return SensitivityResult(
        method=method,
        target_name=target_name,
        feature_names=names,
        raw_gradients=avg_grad,
        normalized_sensitivity=scales * np.abs(avg_grad),
        attributions=ig,
        sample_prediction=f_x,
        baseline_prediction=f_x0,
        completeness_residual=completeness_res,
    )
