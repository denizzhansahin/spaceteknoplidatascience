"""Pseudo-Voigt peak fitting and LIBS emission feature extraction (Section 12, Page 14).

Models emission peaks as a convex combination of Lorentzian and Gaussian profiles:
    V(lambda) = A * [eta * L(lambda) + (1 - eta) * G(lambda)] + b0
where:
    L(lambda) = 1 / (1 + ((lambda - mu) / gamma)^2)
    G(lambda) = exp(-0.5 * ((lambda - mu) / sigma)^2)

Includes explicit window bounds, initial parameters, fit residual RMSE, and a
non-silent `success` flag for failed fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import curve_fit

from st_ds.domain.errors import SchemaError
from st_ds.domain.schemas import Spectrum


@dataclass(frozen=True)
class PeakWindowSpec:
    """Spectral fitting window for an atomic emission peak."""

    name: str
    nominal_center_nm: float
    min_nm: float
    max_nm: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.min_nm >= self.max_nm:
            raise SchemaError(f"PeakWindowSpec '{self.name}' requires min_nm < max_nm.")
        if not (self.min_nm <= self.nominal_center_nm <= self.max_nm):
            raise SchemaError(
                f"PeakWindowSpec '{self.name}' nominal_center_nm must lie inside [min_nm, max_nm]."
            )


DEFAULT_LIBS_CARBON_WINDOWS: tuple[PeakWindowSpec, ...] = (
    PeakWindowSpec("C_I_193_1nm", 193.1, 192.1, 194.1, "Carbon UV atomic emission line"),
    PeakWindowSpec("C_I_229_7nm", 229.7, 228.7, 230.7, "Secondary Carbon emission line"),
    PeakWindowSpec(
        "C_I_247_8nm",
        247.8,
        246.8,
        248.8,
        "Primary Carbon line; monitor potential Fe matrix interference (Section 12)",
    ),
    PeakWindowSpec("H_I_656_3nm", 656.3, 655.3, 657.3, "Hydrogen alpha line (moisture/organic indicator)"),
)


@dataclass(frozen=True)
class PseudoVoigtFitResult:
    """Fitted Pseudo-Voigt peak parameters and diagnostics."""

    name: str
    window_min_nm: float
    window_max_nm: float
    center_nm: float
    amplitude: float
    sigma_gaussian_nm: float
    gamma_lorentzian_nm: float
    eta_mixing: float
    baseline_offset: float
    fwhm_nm: float
    area: float
    fit_residual_rmse: float
    initial_params: tuple[float, ...]
    success: bool
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "window_nm": [self.window_min_nm, self.window_max_nm],
            "center_nm": float(self.center_nm),
            "amplitude": float(self.amplitude),
            "sigma_gaussian_nm": float(self.sigma_gaussian_nm),
            "gamma_lorentzian_nm": float(self.gamma_lorentzian_nm),
            "eta_mixing": float(self.eta_mixing),
            "baseline_offset": float(self.baseline_offset),
            "fwhm_nm": float(self.fwhm_nm),
            "area": float(self.area),
            "fit_residual_rmse": float(self.fit_residual_rmse),
            "initial_params": list(self.initial_params),
            "success": self.success,
            "failure_reason": self.failure_reason,
        }


def pseudovoigt_profile(
    wavelengths: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
    eta: float,
    offset: float,
) -> np.ndarray:
    """Evaluate a 6-parameter Pseudo-Voigt line profile."""
    x = np.asarray(wavelengths, dtype=float)
    gauss = np.exp(-0.5 * ((x - center) / max(sigma, 1e-6)) ** 2)
    lorentz = 1.0 / (1.0 + ((x - center) / max(gamma, 1e-6)) ** 2)
    return offset + amplitude * (eta * lorentz + (1.0 - eta) * gauss)


def fit_pseudovoigt_peaks(
    wavelengths: np.ndarray | Sequence[float],
    intensity: np.ndarray | Sequence[float],
    windows: Sequence[PeakWindowSpec] | None = None,
    *,
    max_nfev: int = 4000,
) -> list[PseudoVoigtFitResult]:
    """Fit Pseudo-Voigt profiles inside each specified spectral peak window."""
    wl = np.asarray(wavelengths, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if wl.ndim != 1 or y.ndim != 1 or wl.size != y.size:
        raise SchemaError("`fit_pseudovoigt_peaks` requires matching 1D wavelengths and intensity arrays.")

    target_windows = windows if windows is not None else DEFAULT_LIBS_CARBON_WINDOWS
    results: list[PseudoVoigtFitResult] = []

    for win in target_windows:
        mask = (wl >= win.min_nm) & (wl <= win.max_nm)
        x_w = wl[mask]
        y_w = y[mask]

        if x_w.size < 7:
            results.append(
                PseudoVoigtFitResult(
                    name=win.name,
                    window_min_nm=win.min_nm,
                    window_max_nm=win.max_nm,
                    center_nm=win.nominal_center_nm,
                    amplitude=0.0,
                    sigma_gaussian_nm=0.0,
                    gamma_lorentzian_nm=0.0,
                    eta_mixing=0.5,
                    baseline_offset=0.0,
                    fwhm_nm=0.0,
                    area=0.0,
                    fit_residual_rmse=float("nan"),
                    initial_params=(),
                    success=False,
                    failure_reason=f"Insufficient points ({x_w.size} < 7) inside window [{win.min_nm}, {win.max_nm}].",
                )
            )
            continue

        b0_init = float(np.percentile(y_w, 10))
        peak_idx = int(np.argmax(y_w))
        amp_init = float(max(y_w[peak_idx] - b0_init, 1e-3))
        center_init = float(x_w[peak_idx])
        span = float(win.max_nm - win.min_nm)
        sigma_init = max(span / 8.0, 1e-3)
        gamma_init = max(span / 8.0, 1e-3)
        p0 = (amp_init, center_init, sigma_init, gamma_init, 0.5, b0_init)

        y_range = float(max(np.max(y_w) - np.min(y_w), 1.0))
        lower_bounds = (0.0, win.min_nm, 1e-3, 1e-3, 0.0, float(np.min(y_w)) - y_range)
        upper_bounds = (y_range * 3.0 + 1e-3, win.max_nm, span, span, 1.0, float(np.max(y_w)) + y_range)
        # Ensure p0 strictly inside bounds
        p0_clamped = tuple(
            float(np.clip(val, lo + 1e-6, hi - 1e-6))
            for val, lo, hi in zip(p0, lower_bounds, upper_bounds)
        )

        try:
            popt, _ = curve_fit(
                pseudovoigt_profile,
                x_w,
                y_w,
                p0=p0_clamped,
                bounds=(lower_bounds, upper_bounds),
                maxfev=max_nfev,
            )
            amp, center, sigma, gamma, eta, offset = (float(v) for v in popt)
            y_pred = pseudovoigt_profile(x_w, amp, center, sigma, gamma, eta, offset)
            rmse = float(np.sqrt(np.mean((y_w - y_pred) ** 2)))

            # Analytical area under baseline-free Pseudo-Voigt curve:
            # int Gaussian = sigma * sqrt(2*pi); int Lorentzian = pi * gamma
            area = float(amp * (eta * np.pi * gamma + (1.0 - eta) * sigma * np.sqrt(2.0 * np.pi)))
            fwhm_g = 2.354820045 * sigma
            fwhm_l = 2.0 * gamma
            fwhm = float(eta * fwhm_l + (1.0 - eta) * fwhm_g)

            results.append(
                PseudoVoigtFitResult(
                    name=win.name,
                    window_min_nm=win.min_nm,
                    window_max_nm=win.max_nm,
                    center_nm=center,
                    amplitude=amp,
                    sigma_gaussian_nm=sigma,
                    gamma_lorentzian_nm=gamma,
                    eta_mixing=eta,
                    baseline_offset=offset,
                    fwhm_nm=fwhm,
                    area=area,
                    fit_residual_rmse=rmse,
                    initial_params=p0_clamped,
                    success=True,
                    failure_reason=None,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive convergence failure reporting
            results.append(
                PseudoVoigtFitResult(
                    name=win.name,
                    window_min_nm=win.min_nm,
                    window_max_nm=win.max_nm,
                    center_nm=center_init,
                    amplitude=amp_init,
                    sigma_gaussian_nm=sigma_init,
                    gamma_lorentzian_nm=gamma_init,
                    eta_mixing=0.5,
                    baseline_offset=b0_init,
                    fwhm_nm=0.0,
                    area=0.0,
                    fit_residual_rmse=float("nan"),
                    initial_params=p0_clamped,
                    success=False,
                    failure_reason=str(exc),
                )
            )
    return results


def extract_libs_peak_features(
    spectrum: Spectrum,
    windows: Sequence[PeakWindowSpec] | None = None,
) -> dict[str, Any]:
    """Extract named LIBS peak amplitude, area, FWHM, and residual features from a 1D Spectrum."""
    vals = np.asarray(spectrum.values, dtype=float)
    if vals.ndim != 1:
        raise SchemaError("extract_libs_peak_features expects a 1D aggregated LIBS Spectrum.")

    fit_results = fit_pseudovoigt_peaks(spectrum.wavelength_grid_nm, vals, windows=windows)
    features: dict[str, float] = {}
    for res in fit_results:
        prefix = f"libs_{res.name}"
        features[f"{prefix}_area"] = res.area if res.success else float("nan")
        features[f"{prefix}_amplitude"] = res.amplitude if res.success else float("nan")
        features[f"{prefix}_fwhm_nm"] = res.fwhm_nm if res.success else float("nan")
        features[f"{prefix}_fit_rmse"] = res.fit_residual_rmse

    return {
        "features": features,
        "peak_fits": [r.to_dict() for r in fit_results],
        "all_converged": all(r.success for r in fit_results),
    }
