"""Digital Lock-In Amplifier (DLIA) readout parsing and summary utilities (Section 7, Page 9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from st_ds.domain.errors import SpectralQualityError


@dataclass(frozen=True)
class DLIAReadout:
    """Raw in-phase (I) and quadrature (Q) lock-in amplifier samples for a channel."""

    channel_id: str
    wavelength_nm: float
    in_phase: tuple[float, ...]
    quadrature: tuple[float, ...]
    reference_freq_hz: float

    def __post_init__(self) -> None:
        if len(self.in_phase) == 0 or len(self.in_phase) != len(self.quadrature):
            raise SpectralQualityError(
                f"DLIA channel '{self.channel_id}' in_phase and quadrature lengths must match and be non-empty."
            )
        if not np.all(np.isfinite(self.in_phase)) or not np.all(np.isfinite(self.quadrature)):
            raise SpectralQualityError(f"DLIA channel '{self.channel_id}' contains NaN or Inf.")
        if self.reference_freq_hz <= 0:
            raise SpectralQualityError("DLIA reference_freq_hz must be positive.")


@dataclass(frozen=True)
class DLIAChannelSummary:
    """Demodulated amplitude R = sqrt(I^2 + Q^2), phase, and lock-in stability."""

    channel_id: str
    wavelength_nm: float
    mean_amplitude: float
    amplitude_std: float
    mean_phase_rad: float
    lock_snr: float
    is_locked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "wavelength_nm": float(self.wavelength_nm),
            "mean_amplitude": float(self.mean_amplitude),
            "amplitude_std": float(self.amplitude_std),
            "mean_phase_rad": float(self.mean_phase_rad),
            "lock_snr": float(self.lock_snr),
            "is_locked": self.is_locked,
        }


def summarize_dlia_outputs(
    readouts: Sequence[DLIAReadout],
    *,
    min_lock_snr: float = 10.0,
) -> dict[str, Any]:
    """Summarize DLIA I/Q demodulated channels into amplitudes, phases, and lock quality."""
    if not readouts:
        raise SpectralQualityError("DLIA readout list cannot be empty.")

    summaries: list[DLIAChannelSummary] = []
    for ro in readouts:
        i_arr = np.asarray(ro.in_phase, dtype=float)
        q_arr = np.asarray(ro.quadrature, dtype=float)
        amp = np.sqrt(i_arr**2 + q_arr**2)
        mean_amp = float(np.mean(amp))
        std_amp = float(np.std(amp))
        mean_phase = float(np.arctan2(np.mean(q_arr), np.mean(i_arr)))
        lock_snr = float(mean_amp / std_amp) if std_amp > 1e-12 else (1000.0 if mean_amp > 0 else 0.0)
        summaries.append(
            DLIAChannelSummary(
                channel_id=ro.channel_id,
                wavelength_nm=ro.wavelength_nm,
                mean_amplitude=mean_amp,
                amplitude_std=std_amp,
                mean_phase_rad=mean_phase,
                lock_snr=lock_snr,
                is_locked=lock_snr >= min_lock_snr,
            )
        )

    return {
        "n_channels": len(summaries),
        "all_locked": all(s.is_locked for s in summaries),
        "wavelengths_nm": [s.wavelength_nm for s in summaries],
        "amplitudes": [s.mean_amplitude for s in summaries],
        "channels": [s.to_dict() for s in summaries],
    }
