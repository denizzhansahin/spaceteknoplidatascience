"""Wavelength alignment and validated spectral resampling (Section 9 & 32).

Enforces the strict no-extrapolation rule (Section 9, Page 11):
- Software must NEVER fill wavelengths outside the instrument's measured range.
- Records an explicit `interpolated_mask` and `measured_band_ratio`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.interpolate import CubicSpline

from st_ds.domain.errors import CalibrationError
from st_ds.domain.instrument import InstrumentProfile
from st_ds.domain.schemas import Spectrum


@dataclass(frozen=True)
class AlignedSpectrum:
    """Output of wavelength alignment onto a target InstrumentProfile grid."""

    spectrum: Spectrum
    target_profile_id: str
    interpolated_mask: tuple[bool, ...]
    measured_band_ratio: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_profile_id": self.target_profile_id,
            "interpolated_mask": list(self.interpolated_mask),
            "measured_band_ratio": float(self.measured_band_ratio),
            "warnings": list(self.warnings),
            "spectrum": self.spectrum.to_dict(),
        }


def align_wavelengths(
    spectrum: Spectrum,
    target_profile: InstrumentProfile,
    *,
    method: Literal["linear", "cubic"] = "linear",
    tol_nm: float = 1e-3,
    min_measured_ratio: float = 0.0,
) -> AlignedSpectrum:
    """Resample `spectrum` onto `target_profile.wavelength_grid_nm` without extrapolation."""
    src_wl = np.asarray(spectrum.wavelength_grid_nm, dtype=float)
    dst_wl = np.asarray(target_profile.wavelength_grid_nm, dtype=float)

    src_min, src_max = float(src_wl[0]), float(src_wl[-1])
    dst_min, dst_max = float(dst_wl[0]), float(dst_wl[-1])

    if dst_min < src_min - tol_nm or dst_max > src_max + tol_nm:
        raise CalibrationError(
            f"Target profile '{target_profile.profile_id}' range [{dst_min}, {dst_max}] nm "
            f"exceeds measured spectrum range [{src_min}, {src_max}] nm. Extrapolation is forbidden (Section 9).",
            details={
                "source_range_nm": [src_min, src_max],
                "target_range_nm": [dst_min, dst_max],
                "target_profile_id": target_profile.profile_id,
            },
        )

    # Identify exact vs interpolated bands
    exact_matches = np.array(
        [bool(np.min(np.abs(src_wl - w)) <= tol_nm) for w in dst_wl],
        dtype=bool,
    )
    interpolated_mask = ~exact_matches
    measured_ratio = float(np.mean(exact_matches))

    if measured_ratio < min_measured_ratio:
        raise CalibrationError(
            f"Measured band ratio ({measured_ratio:.3f}) is below minimum required ({min_measured_ratio:.3f}).",
            details={"measured_band_ratio": measured_ratio, "min_measured_ratio": min_measured_ratio},
        )

    vals = np.asarray(spectrum.values, dtype=float)
    is_1d = vals.ndim == 1
    vals_2d = vals[np.newaxis, :] if is_1d else vals

    # Clamp endpoint numerical tolerance for interpolation
    dst_clamped = np.clip(dst_wl, src_min, src_max)

    if np.all(exact_matches) and src_wl.size == dst_wl.size:
        aligned_2d = vals_2d.copy()
    elif method == "linear":
        aligned_2d = np.vstack([np.interp(dst_clamped, src_wl, row) for row in vals_2d])
    elif method == "cubic":
        if src_wl.size < 4:
            raise CalibrationError("Cubic spline wavelength alignment requires at least 4 source bands.")
        cs = CubicSpline(src_wl, vals_2d, axis=1, extrapolate=False)
        aligned_2d = cs(dst_clamped)
    else:
        raise CalibrationError(f"Unsupported wavelength alignment method '{method}'.")

    # Propagate valid_mask conservatively: band is valid if nearest bounding source bands are valid
    src_valid = np.asarray(spectrum.valid_mask or tuple(True for _ in src_wl), dtype=bool)
    dst_valid = np.ones(dst_wl.size, dtype=bool)
    for idx, w in enumerate(dst_clamped):
        pos = int(np.searchsorted(src_wl, w, side="left"))
        if pos == 0:
            dst_valid[idx] = bool(src_valid[0])
        elif pos >= src_wl.size:
            dst_valid[idx] = bool(src_valid[-1])
        elif np.isclose(src_wl[pos], w, atol=tol_nm):
            dst_valid[idx] = bool(src_valid[pos])
        else:
            dst_valid[idx] = bool(src_valid[pos - 1] and src_valid[pos])

    warnings: list[str] = []
    if np.any(interpolated_mask):
        warnings.append(
            f"{int(np.sum(interpolated_mask))}/{dst_wl.size} bands interpolated onto '{target_profile.profile_id}'."
        )

    out_vals = aligned_2d[0] if is_1d else aligned_2d
    out_spectrum = Spectrum(
        values=out_vals,
        wavelength_grid_nm=target_profile.wavelength_grid_nm,
        channel_ids=target_profile.channel_ids,
        mode=spectrum.mode,
        dark_reference=None,
        white_reference=None,
        integration_time_ms=spectrum.integration_time_ms,
        gain=spectrum.gain,
        sensor_temperature_c=spectrum.sensor_temperature_c,
        reference_age_s=spectrum.reference_age_s,
        replicate_id=spectrum.replicate_id,
        valid_mask=tuple(bool(v) for v in dst_valid),
    )

    return AlignedSpectrum(
        spectrum=out_spectrum,
        target_profile_id=target_profile.profile_id,
        interpolated_mask=tuple(bool(x) for x in interpolated_mask),
        measured_band_ratio=measured_ratio,
        warnings=tuple(warnings),
    )
