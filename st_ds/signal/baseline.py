"""ARPLS (Asymmetrically Reweighted Penalized Least Squares) baseline estimation (Section 12, Page 14).

Solves the sparse second-difference penalized weighted least squares system:
    (W + lam * D2^T D2) z = W y
with asymmetric logistic weight updates that prevent emission peaks from pulling
the estimated continuum baseline upward.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from st_ds.domain.errors import SchemaError, SpectralQualityError
from st_ds.domain.schemas import Spectrum


def estimate_arpls_baseline(
    intensity: np.ndarray,
    *,
    lam: float = 1e4,
    ratio: float = 0.01,
    max_iter: int = 100,
) -> np.ndarray:
    """Estimate smooth spectral continuum baseline using ARPLS."""
    y = np.asarray(intensity, dtype=float)
    if y.ndim != 1 or y.size < 4:
        raise SchemaError(
            f"ARPLS baseline requires a 1D intensity vector of length >= 4, got shape {y.shape}."
        )
    if not np.all(np.isfinite(y)):
        raise SpectralQualityError("Input intensity to `estimate_arpls_baseline` contains NaN or Inf.")
    if lam <= 0 or ratio <= 0 or max_iter < 1:
        raise SchemaError("ARPLS parameters must satisfy lam > 0, ratio > 0, max_iter >= 1.")

    n = y.size
    # Second-order difference operator D2 of shape (n-2, n)
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    penalty = lam * (d2.T @ d2)

    w = np.ones(n, dtype=float)
    z = y.copy()

    for _ in range(max_iter):
        w_diag = sparse.diags(w, 0, shape=(n, n), format="csc")
        z = spsolve(w_diag + penalty, w * y)
        d = y - z
        neg_d = d[d < 0]
        if neg_d.size < 2:
            # No significant negative residuals below baseline -> converged
            break
        mean_neg = float(np.mean(neg_d))
        std_neg = float(np.std(neg_d))
        if std_neg <= 1e-12:
            break

        # Asymmetric logistic reweighting (Baek et al., 2015)
        exponent = np.clip(2.0 * (d - (-mean_neg + 2.0 * std_neg)) / std_neg, -60.0, 60.0)
        w_new = 1.0 / (1.0 + np.exp(exponent))

        rel_change = float(np.linalg.norm(w - w_new) / max(np.linalg.norm(w), 1e-12))
        w = w_new
        if rel_change < ratio:
            break

    return np.asarray(z, dtype=float)


def correct_baseline(
    spectrum: Spectrum,
    *,
    lam: float = 1e4,
    ratio: float = 0.01,
    max_iter: int = 100,
) -> tuple[Spectrum, np.ndarray]:
    """Subtract ARPLS baseline from a 1D or 2D Spectrum and return (corrected_spectrum, baseline)."""
    vals = np.asarray(spectrum.values, dtype=float)
    if vals.ndim == 1:
        baseline = estimate_arpls_baseline(vals, lam=lam, ratio=ratio, max_iter=max_iter)
        corrected_vals = vals - baseline
    else:
        baselines = np.vstack(
            [estimate_arpls_baseline(row, lam=lam, ratio=ratio, max_iter=max_iter) for row in vals]
        )
        baseline = baselines
        corrected_vals = vals - baselines

    corrected_spec = Spectrum(
        values=corrected_vals,
        wavelength_grid_nm=spectrum.wavelength_grid_nm,
        channel_ids=spectrum.channel_ids,
        mode=spectrum.mode,
        dark_reference=spectrum.dark_reference,
        white_reference=spectrum.white_reference,
        integration_time_ms=spectrum.integration_time_ms,
        gain=spectrum.gain,
        sensor_temperature_c=spectrum.sensor_temperature_c,
        reference_age_s=spectrum.reference_age_s,
        replicate_id=spectrum.replicate_id,
        saturation_flags=spectrum.saturation_flags,
        valid_mask=spectrum.valid_mask,
    )
    return corrected_spec, baseline
