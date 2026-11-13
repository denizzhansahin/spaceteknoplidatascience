"""General spectral signal quality control (Section 12 & 32, Page 14 & 34).

Evaluates saturation, valid band ratio, flatline/zero-variance sensor disconnects,
and signal-to-noise ratio (SNR). Returns a structured SignalQCReport with explicit
`qc_score`, `rejection_reasons`, and `NOT_VALID_FOR_REPORTING` status when rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from st_ds.domain.errors import SpectralQualityError
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.domain.schemas import SpectralSignalMode, Spectrum


@dataclass(frozen=True)
class SignalQCConfig:
    """Configuration thresholds for spectral signal quality control."""

    saturation_value: float = 1.05
    raw_saturation_limit: float = 65535.0
    min_snr: float = 5.0
    max_saturated_fraction: float = 0.10
    min_valid_band_ratio: float = 0.80
    min_signal_std: float = 1e-6
    reject_policy: Literal["flag", "raise"] = "flag"


@dataclass(frozen=True)
class SignalQCReport:
    """Structured quality control report for a spectral measurement (Section 32)."""

    qc_score: float
    is_valid_for_reporting: bool
    status: QualityStatus
    flags: tuple[QualityFlag, ...]
    rejection_reasons: tuple[str, ...]
    snr_estimate: float
    saturated_band_indices: tuple[int, ...]
    valid_band_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "qc_score": float(self.qc_score),
            "is_valid_for_reporting": self.is_valid_for_reporting,
            "status": self.status.value,
            "flags": [f.value for f in self.flags],
            "rejection_reasons": list(self.rejection_reasons),
            "snr_estimate": float(self.snr_estimate),
            "saturated_band_indices": list(self.saturated_band_indices),
            "valid_band_ratio": float(self.valid_band_ratio),
        }


def estimate_spectral_snr(values_1d: np.ndarray) -> float:
    """Estimate SNR cleanly without needing repeated scans.

    Uses second-difference noise estimation when P >= 5, falling back to robust
    signal-to-MAD/std ratio for very short discrete LED arrays.
    """
    arr = np.asarray(values_1d, dtype=float)
    signal_span = float(np.max(arr) - np.min(arr))
    signal_level = float(np.mean(np.abs(arr)))
    if arr.size >= 5:
        # Second-difference noise estimator: var(d2x) = 6 * sigma_noise^2
        d2 = np.diff(arr, n=2)
        noise_sigma = float(np.median(np.abs(d2 - np.median(d2))) / (0.6745 * np.sqrt(6.0)))
        if noise_sigma <= 1e-12:
            # Flatline check: if signal_span is also ~0, SNR is 0; otherwise smooth clean signal
            return 0.0 if signal_span <= 1e-12 else 1000.0
        return float(max(signal_level, signal_span) / noise_sigma)
    std_val = float(np.std(arr))
    if std_val <= 1e-12:
        return 0.0
    return float(signal_level / std_val)


def evaluate_signal_qc(
    spectrum: Spectrum,
    *,
    config: SignalQCConfig | None = None,
    valid_mask: tuple[bool, ...] | None = None,
) -> SignalQCReport:
    """Evaluate spectral signal quality and assign `qc_score` and `rejection_reasons`."""
    cfg = config or SignalQCConfig()
    vals = np.asarray(spectrum.values, dtype=float)
    vals_1d = np.mean(vals, axis=0) if vals.ndim == 2 else vals

    flags: list[QualityFlag] = []
    reasons: list[str] = []

    # 1. Valid band mask ratio
    mask = np.asarray(valid_mask or spectrum.valid_mask or tuple(True for _ in vals_1d), dtype=bool)
    valid_ratio = float(np.mean(mask))
    if valid_ratio < cfg.min_valid_band_ratio:
        flags.append(QualityFlag.DARK_WHITE_INVALID)
        reasons.append(
            f"Valid band ratio {valid_ratio:.2f} is below minimum {cfg.min_valid_band_ratio:.2f}."
        )

    # 2. Saturation check (respecting explicit saturation_flags + mode-specific thresholds)
    sat_limit = (
        cfg.raw_saturation_limit
        if spectrum.mode in (SpectralSignalMode.RAW_INTENSITY, SpectralSignalMode.LIBS_EMISSION)
        else cfg.saturation_value
    )
    flagged_sat = np.asarray(spectrum.saturation_flags or tuple(False for _ in vals_1d), dtype=bool)
    threshold_sat = vals_1d >= sat_limit
    sat_mask = flagged_sat | threshold_sat
    sat_indices = tuple(int(i) for i in np.where(sat_mask)[0])
    sat_frac = float(np.mean(sat_mask))

    if sat_indices:
        flags.append(QualityFlag.SATURATED_BANDS)
    if sat_frac > cfg.max_saturated_fraction:
        reasons.append(
            f"Saturated band fraction {sat_frac:.2f} exceeds allowed {cfg.max_saturated_fraction:.2f}."
        )

    # 3. Flatline / disconnected optical fiber check
    valid_vals = vals_1d[mask] if np.any(mask) else vals_1d
    signal_std = float(np.std(valid_vals))
    if signal_std < cfg.min_signal_std:
        flags.append(QualityFlag.LOW_SNR)
        reasons.append(
            f"Spectrum exhibits near-zero variance (std={signal_std:.2e} < {cfg.min_signal_std:.2e}), "
            "indicating optical disconnect or flatline."
        )

    # 4. SNR estimation (for 2D replicate series, compute inter-shot SNR when possible)
    if vals.ndim == 2 and vals.shape[0] >= 3:
        shot_noise = float(np.mean(np.std(vals[:, mask], axis=0)))
        if shot_noise <= 1e-12:
            snr_est = 0.0 if signal_std < cfg.min_signal_std else 1000.0
        else:
            snr_est = float(np.mean(np.abs(valid_vals)) / shot_noise)
    else:
        snr_est = estimate_spectral_snr(valid_vals)

    if snr_est < cfg.min_snr and QualityFlag.LOW_SNR not in flags:
        flags.append(QualityFlag.LOW_SNR)
        reasons.append(f"Estimated SNR ({snr_est:.2f}) is below threshold ({cfg.min_snr:.2f}).")
    elif snr_est < cfg.min_snr and "optical disconnect" not in " ".join(reasons):
        reasons.append(f"Estimated SNR ({snr_est:.2f}) is below threshold ({cfg.min_snr:.2f}).")

    # Compute bounded QC score in [0, 1]
    snr_term = float(np.clip(snr_est / max(cfg.min_snr * 3.0, 1.0), 0.0, 1.0))
    sat_penalty = float(np.clip(sat_frac / max(cfg.max_saturated_fraction, 1e-6), 0.0, 1.0)) * 0.4
    qc_score = float(np.clip(0.5 * valid_ratio + 0.5 * snr_term - sat_penalty, 0.0, 1.0))

    is_valid = len(reasons) == 0
    if is_valid:
        status = QualityStatus.WARNING if flags else QualityStatus.OK
    else:
        status = QualityStatus.NOT_VALID_FOR_REPORTING

    report = SignalQCReport(
        qc_score=qc_score,
        is_valid_for_reporting=is_valid,
        status=status,
        flags=tuple(flags),
        rejection_reasons=tuple(reasons),
        snr_estimate=snr_est,
        saturated_band_indices=sat_indices,
        valid_band_ratio=valid_ratio,
    )

    if not is_valid and cfg.reject_policy == "raise":
        raise SpectralQualityError(
            "Spectral measurement rejected by Signal QC: " + "; ".join(reasons),
            details=report.to_dict(),
        )

    return report
