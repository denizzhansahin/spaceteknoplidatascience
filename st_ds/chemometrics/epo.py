"""External Parameter Orthogonalization (EPO) and GLSW (Section 13-14, DEF-001, ADR-003).

Mathematical contracts:
- EPO:
    D = X_wet - X_dry                              (n x p, matched pairs in identical space)
    C_D = D^T D / (n - 1) = V Sigma V^T
    V_s = V[:, :q]                                 (p x q orthonormal moisture subspace)
    Q = V_s V_s^T
    P_epo = I_p - Q                                (symmetric, idempotent projection)
    X_clean = X P_epo
  Invariants: P^T = P, P^2 ~= P, V_s^T V_s ~= I_q.

- GLSW (Generalized Least Squares Weighting, Section 14):
    P_GLSW = V R^{-1} V^T,   R = sqrt(S^2 / alpha + I)
  where S are singular values of scaled difference matrix D / sqrt(n - 1).

Fixes DEF-001:
- Both `fit` and `transform` strictly require an identical `space_signature`
  (SHA-256 hash of preprocessing steps + wavelength grid). Calling `transform`
  with spectra from a different preprocessing space raises `CalibrationError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from st_ds.domain.errors import CalibrationError, SchemaError


@dataclass(frozen=True)
class EPOConfig:
    """Configuration for External Parameter Orthogonalization."""

    q: int | None = None
    max_q: int = 6
    variance_capture_threshold: float = 0.90


class EPO:
    """External Parameter Orthogonalization transformer (calibration-only fit)."""

    def __init__(self, config: EPOConfig | None = None) -> None:
        self.config = config or EPOConfig()
        self.n_bands: int | None = None
        self.q: int | None = None
        self.V_s: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.explained_diff_variance_ratio: np.ndarray | None = None
        self.space_signature: str | None = None
        self.calibration_dataset_hash: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.P is not None and self.space_signature is not None

    @staticmethod
    def _select_rank_q(
        singular_values: np.ndarray,
        max_q: int,
        variance_threshold: float,
    ) -> int:
        """Select EPO rank q dynamically (never hardcoded to 2) via difference variance spectrum."""
        eigvals = singular_values**2
        total_var = float(np.sum(eigvals))
        if total_var <= 1e-18:
            return 1
        cum_ratio = np.cumsum(eigvals) / total_var
        max_allowed = min(max_q, eigvals.size)
        for idx in range(max_allowed):
            if float(cum_ratio[idx]) >= variance_threshold:
                return idx + 1
        return max_allowed

    def fit(
        self,
        X_wet: np.ndarray,
        X_dry: np.ndarray,
        *,
        space_signature: str,
        q: int | None = None,
        calibration_dataset_hash: str = "untracked",
    ) -> EPO:
        """Estimate moisture interference subspace from paired (X_wet, X_dry) matrices."""
        if not space_signature or not space_signature.strip():
            raise CalibrationError("EPO.fit requires a non-empty `space_signature` (ADR-003).")

        xw = np.asarray(X_wet, dtype=float)
        xd = np.asarray(X_dry, dtype=float)
        if xw.ndim != 2 or xd.ndim != 2:
            raise SchemaError("EPO.fit requires 2D matrices X_wet and X_dry of shape (n_pairs, n_bands).")
        if xw.shape != xd.shape:
            raise CalibrationError(
                f"X_wet shape {xw.shape} does not match X_dry shape {xd.shape}.",
                details={"X_wet_shape": xw.shape, "X_dry_shape": xd.shape},
            )
        n_pairs, n_bands = xw.shape
        if n_pairs < 2:
            raise CalibrationError(f"EPO.fit requires at least 2 paired samples, got {n_pairs}.")
        if not (np.all(np.isfinite(xw)) and np.all(np.isfinite(xd))):
            raise CalibrationError("X_wet or X_dry contains NaN or Inf values.")

        # Difference matrix D = X_wet - X_dry; covariance C_D = D^T D / (n - 1)
        D = xw - xd
        D_scaled = D / np.sqrt(float(n_pairs - 1))
        _, s_vals, vt = np.linalg.svd(D_scaled, full_matrices=False)

        max_possible_q = min(n_pairs, n_bands)
        chosen_q = q if q is not None else self.config.q
        if chosen_q is None:
            chosen_q = self._select_rank_q(
                s_vals,
                max_q=min(self.config.max_q, max_possible_q),
                variance_threshold=self.config.variance_capture_threshold,
            )

        if chosen_q < 1 or chosen_q > max_possible_q:
            raise CalibrationError(
                f"EPO rank q={chosen_q} is out of valid bounds [1, {max_possible_q}].",
                details={"q": chosen_q, "max_possible_q": max_possible_q},
            )

        V_s = vt[:chosen_q, :].T  # shape (p, q)
        Q = np.dot(V_s, V_s.T)    # shape (p, p)
        P = np.eye(n_bands, dtype=float) - Q
        # Symmetrize floating-point rounding residuals
        P = 0.5 * (P + P.T)

        eigvals = s_vals**2
        total_var = float(np.sum(eigvals)) if np.sum(eigvals) > 0 else 1.0

        self.n_bands = n_bands
        self.q = int(chosen_q)
        self.V_s = V_s
        self.Q = Q
        self.P = P
        self.singular_values = s_vals
        self.explained_diff_variance_ratio = eigvals / total_var
        self.space_signature = space_signature
        self.calibration_dataset_hash = calibration_dataset_hash
        return self

    def transform(self, X: np.ndarray, *, space_signature: str) -> np.ndarray:
        """Project preprocessed spectra X onto the moisture-orthogonal subspace X_clean = X P."""
        if not self.is_fitted or self.P is None:
            raise CalibrationError("EPO must be fitted before calling transform.")
        if space_signature != self.space_signature:
            raise CalibrationError(
                "EPO space mismatch (DEF-001): transform input preprocessing/wavelength signature "
                f"'{space_signature}' does not match fit signature '{self.space_signature}'.",
                details={"fit_signature": self.space_signature, "transform_signature": space_signature},
            )

        arr = np.asarray(X, dtype=float)
        if arr.ndim not in (1, 2):
            raise SchemaError("EPO.transform expects a 1D or 2D array.")
        if arr.shape[-1] != self.n_bands:
            raise CalibrationError(
                f"Input band count ({arr.shape[-1]}) does not match fitted EPO bands ({self.n_bands})."
            )
        if not np.all(np.isfinite(arr)):
            raise CalibrationError("Input to EPO.transform contains NaN or Inf values.")

        return np.dot(arr, self.P)

    def check_invariants(self) -> dict[str, float]:
        """Compute matrix invariant errors: ||P^T - P||, ||P^2 - P||, ||V_s^T V_s - I_q||."""
        if not self.is_fitted or self.P is None or self.V_s is None or self.q is None:
            raise CalibrationError("EPO is not fitted; cannot evaluate matrix invariants.")
        sym_err = float(np.linalg.norm(self.P.T - self.P, ord="fro"))
        idem_err = float(np.linalg.norm(np.dot(self.P, self.P) - self.P, ord="fro"))
        ortho_err = float(np.linalg.norm(np.dot(self.V_s.T, self.V_s) - np.eye(self.q), ord="fro"))
        return {
            "symmetry_frobenius_error": sym_err,
            "idempotence_frobenius_error": idem_err,
            "orthonormality_frobenius_error": ortho_err,
        }

    def to_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise CalibrationError("Cannot serialize unfitted EPO.")
        return {
            "n_bands": self.n_bands,
            "q": self.q,
            "space_signature": self.space_signature,
            "calibration_dataset_hash": self.calibration_dataset_hash,
            "V_s": self.V_s.tolist(),  # type: ignore[union-attr]
            "P": self.P.tolist(),      # type: ignore[union-attr]
        }


class GLSW:
    """Generalized Least Squares Weighting alternative moisture suppression (Section 14, Page 16).

    Computes P_GLSW = V R^{-1} V^T where R = sqrt(S^2 / alpha + I).
    """

    def __init__(self, *, alpha: float = 0.1) -> None:
        if alpha <= 0:
            raise SchemaError(f"GLSW alpha must be positive, got {alpha}.")
        self.alpha = float(alpha)
        self.n_bands: int | None = None
        self.P: np.ndarray | None = None
        self.space_signature: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.P is not None and self.space_signature is not None

    def fit(
        self,
        X_wet: np.ndarray,
        X_dry: np.ndarray,
        *,
        space_signature: str,
        alpha: float | None = None,
    ) -> GLSW:
        if not space_signature or not space_signature.strip():
            raise CalibrationError("GLSW.fit requires a non-empty `space_signature`.")
        if alpha is not None:
            if alpha <= 0:
                raise SchemaError(f"GLSW alpha must be positive, got {alpha}.")
            self.alpha = float(alpha)

        xw = np.asarray(X_wet, dtype=float)
        xd = np.asarray(X_dry, dtype=float)
        if xw.ndim != 2 or xd.ndim != 2 or xw.shape != xd.shape:
            raise CalibrationError("GLSW.fit requires matching 2D matrices X_wet and X_dry.")
        n_pairs, n_bands = xw.shape
        if n_pairs < 2:
            raise CalibrationError("GLSW.fit requires at least 2 paired samples.")

        D_scaled = (xw - xd) / np.sqrt(float(n_pairs - 1))
        _, s_vals, vt = np.linalg.svd(D_scaled, full_matrices=False)

        # R = sqrt(S^2 / alpha + I); outside spanned right singular vectors S=0 => R^{-1}=1
        r_inv = 1.0 / np.sqrt((s_vals**2) / self.alpha + 1.0)
        V_r = vt.T  # shape (p, r)
        P = np.eye(n_bands, dtype=float) + np.dot(V_r * (r_inv - 1.0), V_r.T)
        P = 0.5 * (P + P.T)

        self.n_bands = n_bands
        self.P = P
        self.space_signature = space_signature
        return self

    def transform(self, X: np.ndarray, *, space_signature: str) -> np.ndarray:
        if not self.is_fitted or self.P is None:
            raise CalibrationError("GLSW must be fitted before calling transform.")
        if space_signature != self.space_signature:
            raise CalibrationError(
                f"GLSW space mismatch: '{space_signature}' != '{self.space_signature}'."
            )
        arr = np.asarray(X, dtype=float)
        if arr.shape[-1] != self.n_bands:
            raise CalibrationError(
                f"Input band count ({arr.shape[-1]}) does not match GLSW bands ({self.n_bands})."
            )
        return np.dot(arr, self.P)
