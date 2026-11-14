"""Classical chemometric and machine-learning baseline models (Sections 16 & 17, ADR-005).

Implements:
- `PLSRRegressor` (PLSRegression with training-only inner CV for n_components)
- `KernelRidgeRegressor` (RBF Kernel Ridge with training-only inner CV for sigma & alpha)
- `RandomForestSoilRegressor` (tree-ensemble nonlinear baseline with spread estimates)
- `LocalPLSRRegressor` (Memory-Based Learning / locally weighted PLSR over spectral neighbors)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

from st_ds.domain.errors import ModelInferenceError
from st_ds.models.base import BaseSoilRegressor, validate_xy_arrays


def _make_inner_splitter(n_samples: int, n_folds: int, groups: np.ndarray | None) -> Any:
    """Create a deterministic inner CV splitter (GroupKFold when groups allow, else KFold)."""
    eff_folds = max(2, min(int(n_folds), n_samples))
    if groups is not None:
        g_arr = np.asarray(groups, dtype=str)
        n_unique = int(np.unique(g_arr).size)
        if n_unique >= 2:
            return GroupKFold(n_splits=min(eff_folds, n_unique)), g_arr
    return KFold(n_splits=eff_folds, shuffle=False), None


def _pls_predict(estimator: Any, X_arr: np.ndarray) -> np.ndarray:
    """Evaluate PLSRegression via np.dot to match sklearn bit-for-bit without macOS BLAS warnings."""
    if isinstance(estimator, PLSRegression):
        pred = np.dot(X_arr - estimator._x_mean, estimator.coef_.T) + estimator.intercept_
    else:
        pred = estimator.predict(X_arr)
    arr = np.asarray(pred, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


class PLSRRegressor(BaseSoilRegressor):
    """Partial Least Squares Regression with training-only nested CV component selection."""

    model_name = "plsr"
    model_family = "linear_latent_chemometric"

    def __init__(
        self,
        *,
        max_components: int = 10,
        inner_cv_folds: int = 5,
        scale: bool = True,
        target_names: Sequence[str] = ("soc_pct",),
        clip_negative: bool = True,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__(target_names=target_names, clip_negative=clip_negative, min_std=min_std)
        if max_components < 1:
            raise ModelInferenceError("PLSRRegressor.max_components must be >= 1.")
        if inner_cv_folds < 2:
            raise ModelInferenceError("PLSRRegressor.inner_cv_folds must be >= 2.")
        self.max_components = int(max_components)
        self.inner_cv_folds = int(inner_cv_folds)
        self.scale = bool(scale)
        self.selected_n_components_: int = 1
        self._cv_residual_std_: np.ndarray | None = None
        self._estimator: PLSRegression | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> PLSRRegressor:
        del sample_weight  # sklearn PLSRegression does not support sample_weight natively
        X_arr, y_2d, is_1d = validate_xy_arrays(X, y)
        assert y_2d is not None
        n_samples, n_features = X_arr.shape
        if n_samples < 3:
            raise ModelInferenceError("PLSRRegressor requires at least 3 training samples.")

        self._is_1d_target = is_1d
        self._n_features_in = n_features
        self._n_targets = y_2d.shape[1]

        splitter, g_arr = _make_inner_splitter(n_samples, self.inner_cv_folds, groups)
        folds = list(splitter.split(X_arr, y_2d, g_arr))
        min_train_fold = min(len(tr) for tr, _ in folds)
        if min_train_fold < 2:
            best_k = 1
            best_oof_res_var = np.maximum(np.var(y_2d, axis=0), self.min_std * self.min_std)
        else:
            max_k = max(1, min(self.max_components, n_features, min_train_fold - 1))

            best_k = 1
            best_mse = float("inf")
            best_oof_res_var = np.var(y_2d, axis=0)

            for k in range(1, max_k + 1):
                oof_preds = np.zeros_like(y_2d, dtype=float)
                for tr_idx, va_idx in folds:
                    est = PLSRegression(n_components=k, scale=self.scale)
                    est.fit(X_arr[tr_idx], y_2d[tr_idx])
                    oof_preds[va_idx] = _pls_predict(est, X_arr[va_idx])
                diff = y_2d - oof_preds
                mse = float(np.mean(diff * diff))
                if mse < best_mse:
                    best_mse = mse
                    best_k = k
                    best_oof_res_var = np.mean(diff * diff, axis=0)

        self.selected_n_components_ = best_k
        self._cv_residual_std_ = np.maximum(np.sqrt(best_oof_res_var), self.min_std)
        self._estimator = PLSRegression(n_components=best_k, scale=self.scale)
        self._estimator.fit(X_arr, y_2d)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean_out, _ = self.predict_distribution(X)
        return mean_out

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert self._estimator is not None and self._cv_residual_std_ is not None
        X_arr, _, _ = validate_xy_arrays(X, expected_n_features=self._n_features_in)
        preds = _pls_predict(self._estimator, X_arr)
        stds = np.broadcast_to(self._cv_residual_std_[None, :], preds.shape).copy()
        mean_out, std_out = self._format_output(preds, stds)
        assert std_out is not None
        return mean_out, std_out

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "max_components": self.max_components,
                "inner_cv_folds": self.inner_cv_folds,
                "scale": self.scale,
                "selected_n_components": self.selected_n_components_,
            }
        )
        return params


class KernelRidgeRegressor(BaseSoilRegressor):
    """RBF Kernel Ridge Regression: alpha* = (K + lambda I)^(-1) y with inner CV tuning."""

    model_name = "kernel_ridge_rbf"
    model_family = "kernel_method"

    def __init__(
        self,
        *,
        alpha_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
        sigma_multipliers: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        inner_cv_folds: int = 5,
        target_names: Sequence[str] = ("soc_pct",),
        clip_negative: bool = True,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__(target_names=target_names, clip_negative=clip_negative, min_std=min_std)
        if inner_cv_folds < 2:
            raise ModelInferenceError("KernelRidgeRegressor.inner_cv_folds must be >= 2.")
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.sigma_multipliers = tuple(float(s) for s in sigma_multipliers)
        self.inner_cv_folds = int(inner_cv_folds)
        self.selected_alpha_: float = 1.0
        self.selected_sigma_: float = 1.0
        self._X_train_: np.ndarray | None = None
        self._dual_coef_: np.ndarray | None = None
        self._cho_factor_: tuple[np.ndarray, bool] | None = None
        self._residual_std_: np.ndarray | None = None

    @staticmethod
    def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        a_sq = np.sum(A * A, axis=1, keepdims=True)
        b_sq = np.sum(B * B, axis=1, keepdims=True).T
        d2 = a_sq + b_sq - 2.0 * np.dot(A, B.T)
        return np.maximum(d2, 0.0)

    def _rbf_kernel(self, A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
        d2 = self._pairwise_sq_dists(A, B)
        return np.exp(-d2 / (2.0 * sigma * sigma))

    @staticmethod
    def _solve_krr_dual(
        K_train: np.ndarray,
        y_2d: np.ndarray,
        alpha: float,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, bool]]:
        """Solve (K + alpha I) alpha* = y via Cholesky factorization (sample-weight aware)."""
        n = K_train.shape[0]
        reg = max(float(alpha), 1e-8)
        if sample_weight is None:
            K_reg = K_train + reg * np.eye(n, dtype=float)
            cf = cho_factor(K_reg, lower=True, check_finite=False)
            dual = cho_solve(cf, y_2d, check_finite=False)
            return dual, cf

        sw = np.sqrt(np.maximum(np.asarray(sample_weight, dtype=float), 1e-12))
        K_w = K_train * (sw[:, None] * sw[None, :])
        K_reg = K_w + reg * np.eye(n, dtype=float)
        cf = cho_factor(K_reg, lower=True, check_finite=False)
        dual_tilde = cho_solve(cf, y_2d * sw[:, None], check_finite=False)
        dual = dual_tilde * sw[:, None]
        cf_unweighted = cho_factor(K_train + reg * np.eye(n, dtype=float), lower=True, check_finite=False)
        return dual, cf_unweighted

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> KernelRidgeRegressor:
        X_arr, y_2d, is_1d = validate_xy_arrays(X, y)
        assert y_2d is not None
        n_samples, n_features = X_arr.shape
        if n_samples < 3:
            raise ModelInferenceError("KernelRidgeRegressor requires at least 3 training samples.")

        self._is_1d_target = is_1d
        self._n_features_in = n_features
        self._n_targets = y_2d.shape[1]

        # Median Euclidean distance heuristic strictly from training fold features
        d2_tr = self._pairwise_sq_dists(X_arr, X_arr)
        triu = d2_tr[np.triu_indices(n_samples, k=1)]
        pos_d = np.sqrt(triu[triu > 0.0])
        base_sigma = float(np.median(pos_d)) if pos_d.size > 0 else float(np.sqrt(n_features))
        base_sigma = max(base_sigma, 1e-3)

        splitter, g_arr = _make_inner_splitter(n_samples, self.inner_cv_folds, groups)
        folds = list(splitter.split(X_arr, y_2d, g_arr))

        best_alpha = self.alpha_grid[0]
        best_sigma = base_sigma
        best_mse = float("inf")
        best_res_var = np.var(y_2d, axis=0)

        for mult in self.sigma_multipliers:
            sigma = max(base_sigma * mult, 1e-4)
            K_all = np.exp(-d2_tr / (2.0 * sigma * sigma))
            for alpha in self.alpha_grid:
                oof_preds = np.zeros_like(y_2d, dtype=float)
                for tr_idx, va_idx in folds:
                    K_tr = K_all[np.ix_(tr_idx, tr_idx)]
                    K_va_tr = K_all[np.ix_(va_idx, tr_idx)]
                    sw = sample_weight[tr_idx] if sample_weight is not None else None
                    dual_fold, _ = self._solve_krr_dual(K_tr, y_2d[tr_idx], alpha, sample_weight=sw)
                    oof_preds[va_idx] = np.dot(K_va_tr, dual_fold)
                diff = y_2d - oof_preds
                mse = float(np.mean(diff * diff))
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(alpha)
                    best_sigma = float(sigma)
                    best_res_var = np.mean(diff * diff, axis=0)

        self.selected_alpha_ = best_alpha
        self.selected_sigma_ = best_sigma
        self._residual_std_ = np.maximum(np.sqrt(best_res_var), self.min_std)
        self._X_train_ = X_arr.copy()

        K_train = np.exp(-d2_tr / (2.0 * best_sigma * best_sigma))
        self._dual_coef_, self._cho_factor_ = self._solve_krr_dual(
            K_train,
            y_2d,
            best_alpha,
            sample_weight=sample_weight,
        )
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean_out, _ = self.predict_distribution(X)
        return mean_out

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert (
            self._X_train_ is not None
            and self._dual_coef_ is not None
            and self._cho_factor_ is not None
            and self._residual_std_ is not None
        )
        X_arr, _, _ = validate_xy_arrays(X, expected_n_features=self._n_features_in)
        K_cross = self._rbf_kernel(self._X_train_, X_arr, self.selected_sigma_)  # (N_train, M)
        preds = np.dot(K_cross.T, self._dual_coef_)

        # Posterior RBF kernel uncertainty: 1 - k(x, X)^T (K + alpha I)^-1 k(x, X)
        V = cho_solve(self._cho_factor_, K_cross, check_finite=False)
        explained_norm = np.sum(K_cross * V, axis=0)  # (M,)
        unc_scale = np.sqrt(np.clip(1.0 - explained_norm, 0.05, 1.0))[:, None]
        stds = unc_scale * self._residual_std_[None, :]
        mean_out, std_out = self._format_output(preds, stds)
        assert std_out is not None
        return mean_out, std_out

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "alpha_grid": list(self.alpha_grid),
                "sigma_multipliers": list(self.sigma_multipliers),
                "inner_cv_folds": self.inner_cv_folds,
                "selected_alpha": self.selected_alpha_,
                "selected_sigma": self.selected_sigma_,
            }
        )
        return params


class RandomForestSoilRegressor(BaseSoilRegressor):
    """Random Forest nonlinear baseline with inter-tree ensemble spread."""

    model_name = "random_forest"
    model_family = "tree_ensemble"

    def __init__(
        self,
        *,
        n_estimators: int = 150,
        max_depth: int | None = 12,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        target_names: Sequence[str] = ("soc_pct",),
        clip_negative: bool = True,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__(target_names=target_names, clip_negative=clip_negative, min_std=min_std)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self._estimator: RandomForestRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> RandomForestSoilRegressor:
        X_arr, y_2d, is_1d = validate_xy_arrays(X, y)
        assert y_2d is not None
        self._is_1d_target = is_1d
        self._n_features_in = X_arr.shape[1]
        self._n_targets = y_2d.shape[1]

        y_fit = y_2d[:, 0] if y_2d.shape[1] == 1 else y_2d
        est = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        est.fit(X_arr, y_fit, sample_weight=sample_weight)
        self._estimator = est
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean_out, _ = self.predict_distribution(X)
        return mean_out

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert self._estimator is not None
        X_arr, _, _ = validate_xy_arrays(X, expected_n_features=self._n_features_in)

        tree_preds = np.asarray(
            [tree.predict(X_arr) for tree in self._estimator.estimators_],
            dtype=float,
        )
        if tree_preds.ndim == 2:
            # (n_trees, n_samples) -> expand to (n_trees, n_samples, 1)
            tree_preds = tree_preds[:, :, None]

        mean_2d = np.mean(tree_preds, axis=0)
        std_2d = np.maximum(np.std(tree_preds, axis=0), self.min_std)
        mean_out, std_out = self._format_output(mean_2d, std_2d)
        assert std_out is not None
        return mean_out, std_out

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
            }
        )
        return params


class LocalPLSRRegressor(BaseSoilRegressor):
    """Memory-Based Learning (MBL / Local PLSR) over Gaussian-weighted spectral neighbors (Section 17).

    Weights neighbors using w_i prop exp(-d(x, x_i)^2 / (2 h^2)) over the top-k nearest references.
    """

    model_name = "local_plsr_mbl"
    model_family = "memory_based_learning"

    def __init__(
        self,
        *,
        n_neighbors: int = 20,
        n_components: int = 3,
        bandwidth: float | None = None,
        min_library_size: int = 8,
        target_names: Sequence[str] = ("soc_pct",),
        clip_negative: bool = True,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__(target_names=target_names, clip_negative=clip_negative, min_std=min_std)
        if n_neighbors < 3:
            raise ModelInferenceError("LocalPLSRRegressor.n_neighbors must be >= 3.")
        if n_components < 1:
            raise ModelInferenceError("LocalPLSRRegressor.n_components must be >= 1.")
        if min_library_size < 4:
            raise ModelInferenceError("LocalPLSRRegressor.min_library_size must be >= 4.")
        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        self.bandwidth = float(bandwidth) if bandwidth is not None else None
        self.min_library_size = int(min_library_size)
        self._X_lib_: np.ndarray | None = None
        self._y_lib_: np.ndarray | None = None
        self._effective_bandwidth_: float = 1.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> LocalPLSRRegressor:
        del sample_weight
        X_arr, y_2d, is_1d = validate_xy_arrays(X, y)
        assert y_2d is not None
        if X_arr.shape[0] < self.min_library_size:
            raise ModelInferenceError(
                f"LocalPLSRRegressor requires a library of at least {self.min_library_size} samples, "
                f"got {X_arr.shape[0]}.",
                details={"n_samples": int(X_arr.shape[0]), "min_library_size": self.min_library_size},
            )
        self._is_1d_target = is_1d
        self._n_features_in = X_arr.shape[1]
        self._n_targets = y_2d.shape[1]
        self._X_lib_ = X_arr.copy()
        self._y_lib_ = y_2d.copy()

        if self.bandwidth is not None:
            self._effective_bandwidth_ = max(self.bandwidth, 1e-6)
        else:
            d2 = KernelRidgeRegressor._pairwise_sq_dists(X_arr, X_arr)
            triu = d2[np.triu_indices(X_arr.shape[0], k=1)]
            pos = np.sqrt(triu[triu > 0.0])
            self._effective_bandwidth_ = max(float(np.median(pos)) if pos.size > 0 else 1.0, 1e-4)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean_out, _ = self.predict_distribution(X)
        return mean_out

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert self._X_lib_ is not None and self._y_lib_ is not None
        X_arr, _, _ = validate_xy_arrays(X, expected_n_features=self._n_features_in)

        n_lib = self._X_lib_.shape[0]
        k = min(self.n_neighbors, n_lib)
        h2 = 2.0 * self._effective_bandwidth_ * self._effective_bandwidth_
        d2_all = KernelRidgeRegressor._pairwise_sq_dists(X_arr, self._X_lib_)  # (M, n_lib)

        preds = np.zeros((X_arr.shape[0], self._n_targets), dtype=float)
        stds = np.zeros((X_arr.shape[0], self._n_targets), dtype=float)

        for i in range(X_arr.shape[0]):
            d2_row = d2_all[i]
            nbr_idx = np.argpartition(d2_row, k - 1)[:k]
            nbr_d2 = d2_row[nbr_idx]
            weights = np.exp(-nbr_d2 / h2)
            w_sum = float(np.sum(weights))
            if w_sum <= 1e-12:
                weights = np.full(k, 1.0 / k, dtype=float)
            else:
                weights = weights / w_sum

            X_nbr = self._X_lib_[nbr_idx]
            y_nbr = self._y_lib_[nbr_idx]

            x_mean_w = np.dot(weights, X_nbr)
            y_mean_w = np.dot(weights, y_nbr)

            sqrt_w = np.sqrt(weights)[:, None]
            X_centered_w = (X_nbr - x_mean_w[None, :]) * sqrt_w
            y_centered_w = (y_nbr - y_mean_w[None, :]) * sqrt_w

            # Skip PLSR fit if neighbors are constant in features or targets
            if float(np.max(np.std(X_nbr, axis=0))) < 1e-10 or k < 3:
                preds[i] = y_mean_w
                res_var = np.dot(weights, (y_nbr - y_mean_w[None, :]) ** 2)
                stds[i] = np.maximum(np.sqrt(res_var), self.min_std)
                continue

            n_comp = min(self.n_components, k - 1, X_nbr.shape[1])
            local_est = PLSRegression(n_components=max(1, n_comp), scale=False)
            local_est.fit(X_centered_w, y_centered_w)

            q_centered = (X_arr[i] - x_mean_w)[None, :]
            delta_y = np.asarray(local_est.predict(q_centered), dtype=float)
            if delta_y.ndim == 1:
                delta_y = delta_y[:, None]
            preds[i] = y_mean_w + delta_y[0]

            nbr_fit = np.asarray(local_est.predict(X_nbr - x_mean_w[None, :]), dtype=float)
            if nbr_fit.ndim == 1:
                nbr_fit = nbr_fit[:, None]
            nbr_res = y_nbr - (y_mean_w[None, :] + nbr_fit)
            local_var = np.dot(weights, nbr_res * nbr_res)
            stds[i] = np.maximum(np.sqrt(local_var), self.min_std)

        mean_out, std_out = self._format_output(preds, stds)
        assert std_out is not None
        return mean_out, std_out

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_neighbors": self.n_neighbors,
                "n_components": self.n_components,
                "bandwidth": self._effective_bandwidth_,
                "min_library_size": self.min_library_size,
            }
        )
        return params
