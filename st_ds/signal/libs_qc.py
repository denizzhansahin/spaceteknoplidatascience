"""LIBS multi-shot quality control and outlier shot filtering (Section 12 & 32, Page 14 & 34).

Filters corrupted LIBS laser shots using physical QC rules FIRST:
1. Detector saturation (>= saturation_limit)
2. Low signal-to-noise ratio (SNR) or optical disconnect (flatline / near-zero intensity)
3. Unexpected spectral peak shift beyond `max_peak_shift_nm`
4. Laser pulse energy deviation outside acceptable bounds

Percentile trimming (e.g., top 5% / bottom 10% carbon peak intensity) is strictly
OFF by default (`enable_percentile_trim=False`) and must only be enabled when
validated on labeled data, because high carbon peaks alone can represent genuine
soil carbon enrichment rather than biological contamination (Section 12, Page 14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from st_ds.domain.errors import SpectralQualityError
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.domain.schemas import SpectralSignalMode, Spectrum
from st_ds.signal.qc import SignalQCReport, estimate_spectral_snr


@dataclass(frozen=True)
class LIBSShotQCConfig:
    """QC rules for multi-shot LIBS measurements."""

    saturation_limit: float = 65535.0
    min_snr: float = 4.0
    min_peak_intensity: float = 50.0
    max_peak_shift_nm: float = 0.5
    energy_rel_tol: float = 0.30
    min_surviving_shots: int = 5
    enable_percentile_trim: bool = False
    top_trim_pct: float = 0.05
    bottom_trim_pct: float = 0.10


@dataclass(frozen=True)
class LIBSShotFilterResult:
    """Result of filtering multi-shot LIBS spectra and averaging surviving shots."""

    averaged_spectrum: Spectrum
    filtered_shots: np.ndarray
    kept_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    rejection_reasons: dict[int, tuple[str, ...]]
    qc_report: SignalQCReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total_shots": len(self.kept_indices) + len(self.rejected_indices),
            "n_kept_shots": len(self.kept_indices),
            "n_rejected_shots": len(self.rejected_indices),
            "kept_indices": list(self.kept_indices),
            "rejected_indices": list(self.rejected_indices),
            "rejection_reasons": {str(k): list(v) for k, v in self.rejection_reasons.items()},
            "qc_report": self.qc_report.to_dict(),
        }


def filter_libs_shots(
    spectrum: Spectrum,
    *,
    shot_energies: Sequence[float] | np.ndarray | None = None,
    reference_peak_window_nm: tuple[float, float] = (246.8, 248.8),
    config: LIBSShotQCConfig | None = None,
) -> LIBSShotFilterResult:
    """Apply physical QC rules to 2D LIBS shots (N_shots, P) and aggregate surviving shots."""
    cfg = config or LIBSShotQCConfig()
    vals = np.asarray(spectrum.values, dtype=float)
    if vals.ndim != 2 or vals.shape[0] < 2:
        raise SpectralQualityError(
            "filter_libs_shots requires a 2D Spectrum of shape (N_shots >= 2, P).",
            details={"shape": vals.shape},
        )

    n_shots, n_bands = vals.shape
    wl = np.asarray(spectrum.wavelength_grid_nm, dtype=float)

    energies_arr: np.ndarray | None = None
    median_energy: float | None = None
    if shot_energies is not None:
        energies_arr = np.asarray(shot_energies, dtype=float)
        if energies_arr.shape != (n_shots,) or not np.all(np.isfinite(energies_arr)):
            raise SpectralQualityError(
                f"shot_energies must be a finite 1D array of length {n_shots}."
            )
        median_energy = float(np.median(energies_arr))

    # Locate window indices for peak-shift monitoring
    win_mask = (wl >= reference_peak_window_nm[0]) & (wl <= reference_peak_window_nm[1])
    if not np.any(win_mask):
        win_mask = np.ones(n_bands, dtype=bool)
    win_wl = wl[win_mask]

    # Precompute median peak wavelength across non-saturated, non-flatline shots
    candidate_peaks: list[float] = []
    for i in range(n_shots):
        row = vals[i]
        if np.max(row) < cfg.saturation_limit and np.max(row) >= cfg.min_peak_intensity:
            candidate_peaks.append(float(win_wl[int(np.argmax(row[win_mask]))]))
    nominal_peak_wl = float(np.median(candidate_peaks)) if candidate_peaks else None

    kept: list[int] = []
    rejected: list[int] = []
    reasons_by_shot: dict[int, tuple[str, ...]] = {}

    for i in range(n_shots):
        row = vals[i]
        shot_reasons: list[str] = []

        if np.any(row >= cfg.saturation_limit):
            shot_reasons.append(f"detector_saturation (>= {cfg.saturation_limit})")

        if float(np.max(row)) < cfg.min_peak_intensity:
            shot_reasons.append(
                f"optical_disconnect_or_low_signal (max={float(np.max(row)):.2f} < {cfg.min_peak_intensity})"
            )

        shot_snr = estimate_spectral_snr(row)
        if shot_snr < cfg.min_snr:
            shot_reasons.append(f"low_snr ({shot_snr:.2f} < {cfg.min_snr})")

        if nominal_peak_wl is not None and float(np.max(row[win_mask])) >= cfg.min_peak_intensity:
            peak_wl = float(win_wl[int(np.argmax(row[win_mask]))])
            if abs(peak_wl - nominal_peak_wl) > cfg.max_peak_shift_nm:
                shot_reasons.append(
                    f"peak_wavelength_shift ({peak_wl:.3f} nm vs nominal {nominal_peak_wl:.3f} nm)"
                )

        if energies_arr is not None and median_energy is not None and median_energy > 0:
            rel_dev = abs(float(energies_arr[i]) - median_energy) / median_energy
            if rel_dev > cfg.energy_rel_tol:
                shot_reasons.append(
                    f"laser_energy_deviation ({rel_dev:.2%} > {cfg.energy_rel_tol:.2%})"
                )

        if shot_reasons:
            rejected.append(i)
            reasons_by_shot[i] = tuple(shot_reasons)
        else:
            kept.append(i)

    # Optional validated percentile trimming ONLY on shots that already passed physical QC
    if cfg.enable_percentile_trim and len(kept) >= 4:
        kept_peaks = np.array([float(np.max(vals[idx][win_mask])) for idx in kept])
        order = np.argsort(kept_peaks)
        n_kept = len(kept)
        n_drop_bot = int(np.floor(n_kept * cfg.bottom_trim_pct))
        n_drop_top = int(np.floor(n_kept * cfg.top_trim_pct))
        if n_drop_bot + n_drop_top < n_kept:
            trimmed_order = order[n_drop_bot : n_kept - n_drop_top if n_drop_top > 0 else n_kept]
            trimmed_set = {kept[int(pos)] for pos in trimmed_order}
            new_kept: list[int] = []
            for idx in kept:
                if idx in trimmed_set:
                    new_kept.append(idx)
                else:
                    rejected.append(idx)
                    reasons_by_shot[idx] = ("percentile_trim_excluded",)
            kept = new_kept

    if len(kept) == 0:
        raise SpectralQualityError(
            "All LIBS shots were rejected by physical QC rules.",
            details={"n_total_shots": n_shots, "rejection_reasons": reasons_by_shot},
        )

    surviving_shots = vals[kept]
    mean_spectrum_vals = np.mean(surviving_shots, axis=0)
    surviving_ratio = len(kept) / float(n_shots)

    flags: list[QualityFlag] = []
    summary_reasons: list[str] = []
    if len(kept) < cfg.min_surviving_shots:
        flags.append(QualityFlag.LOW_SNR)
        summary_reasons.append(
            f"Surviving LIBS shots ({len(kept)}) below minimum required ({cfg.min_surviving_shots})."
        )

    is_valid = len(summary_reasons) == 0
    avg_snr = float(np.mean([estimate_spectral_snr(row) for row in surviving_shots]))
    qc_score = float(np.clip(surviving_ratio * min(avg_snr / max(cfg.min_snr * 2.0, 1.0), 1.0), 0.0, 1.0))

    qc_report = SignalQCReport(
        qc_score=qc_score,
        is_valid_for_reporting=is_valid,
        status=QualityStatus.OK if is_valid else QualityStatus.NOT_VALID_FOR_REPORTING,
        flags=tuple(flags),
        rejection_reasons=tuple(summary_reasons),
        snr_estimate=avg_snr,
        saturated_band_indices=(),
        valid_band_ratio=1.0,
    )

    avg_spec = Spectrum(
        values=mean_spectrum_vals,
        wavelength_grid_nm=spectrum.wavelength_grid_nm,
        channel_ids=spectrum.channel_ids,
        mode=SpectralSignalMode.LIBS_EMISSION,
        integration_time_ms=spectrum.integration_time_ms,
        gain=spectrum.gain,
        sensor_temperature_c=spectrum.sensor_temperature_c,
        replicate_id=f"{spectrum.replicate_id}_avg_{len(kept)}shots",
    )

    return LIBSShotFilterResult(
        averaged_spectrum=avg_spec,
        filtered_shots=surviving_shots,
        kept_indices=tuple(kept),
        rejected_indices=tuple(sorted(rejected)),
        rejection_reasons=reasons_by_shot,
        qc_report=qc_report,
    )
