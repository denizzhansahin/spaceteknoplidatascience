"""Wavelength-aware Savitzky-Golay smoothing and physical derivatives (Section 11, DEF-002).

Fixes DEF-002:
- Never treats `axis=-1` index steps as physical derivatives.
- For uniform grids: passes exact `delta=delta_nm` to `scipy.signal.savgol_filter`
  to compute d^k p(lambda) / d(lambda)^k in units of nm^-k.
- For non-uniform grids: fits an exact local least-squares polynomial p(xi) in
  shifted physical wavelength coordinates xi = lambda - lambda_c inside each sliding
  window and evaluates k! * c_k analytically at the center wavelength lambda_c.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np
from scipy.signal import savgol_filter

from st_ds.chemometrics.transforms import _clone_spectrum_with_values, validate_spectral_grid
from st_ds.domain.errors import SchemaError, SpectralQualityError
from st_ds.domain.schemas import Spectrum


def _is_uniform_grid(wl: np.ndarray, rtol: float = 1e-3) -> tuple[bool, float]:
    if wl.size <= 2:
        delta = float(wl[1] - wl[0]) if wl.size == 2 else 1.0
        return True, delta
    diffs = np.diff(wl)
    mean_diff = float(np.mean(diffs))
    is_unif = bool(np.allclose(diffs, mean_diff, rtol=rtol, atol=1e-6))
    return is_unif, mean_diff


def _nonuniform_savgol_weights(
    wl: np.ndarray,
    window: int,
    polyorder: int,
    deriv: int,
) -> np.ndarray:
    """Compute exact linear derivative operator W of shape (P, P) for non-uniform wavelengths.

    For each band center j, selects a local window of `window` contiguous wavelengths,
    solves the Vandermonde least-squares polynomial p(xi) = sum_{m=0}^polyorder c_m xi^m
    with xi = wl_win - wl[j], and extracts row j of W = k! * (V^T V)^{-1} V^T [row k].
    """
    n_bands = wl.size
    half = window // 2
    weights = np.zeros((n_bands, n_bands), dtype=float)
    fact_k = float(math.factorial(deriv))

    for j in range(n_bands):
        start = max(0, j - half)
        end = start + window
        if end > n_bands:
            end = n_bands
            start = end - window

        wl_win = wl[start:end]
        # Scale xi by local half-span for numerical conditioning, then rescale derivative by span^-deriv
        xi_raw = wl_win - wl[j]
        scale = float(np.max(np.abs(xi_raw)))
        if scale <= 0:
            scale = 1.0
        xi_scaled = xi_raw / scale

        # Vandermonde matrix V[r, m] = xi_scaled[r]^m
        vander = np.vander(xi_scaled, N=polyorder + 1, increasing=True)
        pinv = np.linalg.pinv(vander)  # shape (polyorder + 1, window)
        # Derivative k at xi=0 is (k! / scale^k) * c_k
        weights[j, start:end] = (fact_k / (scale**deriv)) * pinv[deriv, :]

    return weights


@overload
def savgol_derivative(
    data: Spectrum,
    wavelengths: np.ndarray | tuple[float, ...] | list[float] | None = None,
    *,
    window: int = 7,
    polyorder: int = 2,
    deriv: int = 1,
    uniform_rtol: float = 1e-3,
) -> Spectrum: ...


@overload
def savgol_derivative(
    data: np.ndarray,
    wavelengths: np.ndarray | tuple[float, ...] | list[float],
    *,
    window: int = 7,
    polyorder: int = 2,
    deriv: int = 1,
    uniform_rtol: float = 1e-3,
) -> np.ndarray: ...


def savgol_derivative(
    data: Spectrum | np.ndarray,
    wavelengths: np.ndarray | tuple[float, ...] | list[float] | None = None,
    *,
    window: int = 7,
    polyorder: int = 2,
    deriv: int = 1,
    uniform_rtol: float = 1e-3,
) -> Spectrum | np.ndarray:
    """Compute physical Savitzky-Golay smoothing or derivative d^k p(lambda) / d(lambda)^k."""
    if window < 3 or window % 2 == 0:
        raise SchemaError(f"Savitzky-Golay `window` must be an odd integer >= 3, got {window}.")
    if polyorder < 0 or polyorder >= window:
        raise SchemaError(
            f"Savitzky-Golay `polyorder` ({polyorder}) must satisfy 0 <= polyorder < window ({window})."
        )
    if deriv < 0 or deriv > polyorder:
        raise SchemaError(
            f"Savitzky-Golay `deriv` ({deriv}) must satisfy 0 <= deriv <= polyorder ({polyorder})."
        )

    is_spec = isinstance(data, Spectrum)
    arr = np.asarray(data.values if is_spec else data, dtype=float)
    if arr.ndim not in (1, 2) or arr.size == 0:
        raise SchemaError("Input to `savgol_derivative` must be a non-empty 1D or 2D array.")
    if not np.all(np.isfinite(arr)):
        raise SpectralQualityError("Input to `savgol_derivative` contains NaN or Inf.")

    if wavelengths is None:
        if is_spec:
            wl_raw: np.ndarray | tuple[float, ...] | list[float] = data.wavelength_grid_nm
        else:
            raise SchemaError(
                "`wavelengths` must be explicitly supplied when passing a numpy array to `savgol_derivative` (DEF-002)."
            )
    else:
        wl_raw = wavelengths

    wl = validate_spectral_grid(wl_raw)
    n_bands = arr.shape[-1]
    if wl.size != n_bands:
        raise SchemaError(
            f"Wavelength grid length ({wl.size}) does not match spectral band count ({n_bands})."
        )
    if n_bands < window:
        raise SchemaError(
            f"Spectral band count ({n_bands}) is smaller than Savitzky-Golay window ({window}). "
            "Reduce window size or use an appropriate instrument profile.",
            details={"n_bands": n_bands, "window": window},
        )

    arr_2d = arr[np.newaxis, :] if arr.ndim == 1 else arr
    is_unif, delta_nm = _is_uniform_grid(wl, rtol=uniform_rtol)

    if is_unif:
        out_2d = savgol_filter(
            arr_2d,
            window_length=window,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta_nm,
            axis=-1,
            mode="interp",
        )
    else:
        w_mat = _nonuniform_savgol_weights(wl, window=window, polyorder=polyorder, deriv=deriv)
        out_2d = np.dot(arr_2d, w_mat.T)

    out = out_2d[0] if arr.ndim == 1 else out_2d
    if is_spec:
        return _clone_spectrum_with_values(data, out)
    return out
