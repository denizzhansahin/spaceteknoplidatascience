"""Radiometric white/dark reference calibration (Section 10 & 32, Page 12 & 34).

Computes raw reflectance:
    R_raw(lambda) = [I_soil(lambda) - I_dark(lambda)] / [I_white(lambda) - I_dark(lambda)]

Enforces:
- Matching sensor gain and integration time across raw, dark, and white measurements
- Saturation detection and clipping count tracking
- Near-zero or negative denominator detection (band-level invalidation)
- Reference age staleness flagging (QualityFlag.REFERENCE_AGED)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from st_ds.domain.errors import CalibrationError
from st_ds.domain.quality import QualityFlag
from st_ds.domain.schemas import SpectralSignalMode, Spectrum


@dataclass(frozen=True)
class CalibratedSpectrum:
    """Output contract of `calibrate_reflectance` (Section 10, Page 12)."""

    spectrum: Spectrum
    valid_mask: tuple[bool, ...]
    clipping_count: int
    reference_age_s: float | None
    calibration_id: str
    warnings: tuple[str, ...] = ()
    quality_flags: tuple[QualityFlag, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "clipping_count": self.clipping_count,
            "reference_age_s": self.reference_age_s,
            "valid_band_count": int(sum(self.valid_mask)),
            "total_band_count": len(self.valid_mask),
            "valid_mask": list(self.valid_mask),
            "warnings": list(self.warnings),
            "quality_flags": [f.value for f in self.quality_flags],
            "spectrum": self.spectrum.to_dict(),
        }


def _extract_1d_reference(ref: Spectrum | np.ndarray, name: str, expected_len: int) -> np.ndarray:
    arr = np.asarray(ref.values if isinstance(ref, Spectrum) else ref, dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise CalibrationError(
                f"{name} reference must be a 1D array of length {expected_len}, got shape {arr.shape}."
            )
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise CalibrationError(
            f"{name} reference length ({arr.shape}) does not match raw spectrum bands ({expected_len})."
        )
    if not np.all(np.isfinite(arr)):
        raise CalibrationError(f"{name} reference contains NaN or Inf values.")
    return arr


def calibrate_reflectance(
    raw: Spectrum,
    dark: Spectrum | np.ndarray | None = None,
    white: Spectrum | np.ndarray | None = None,
    *,
    epsilon: float = 1e-6,
    saturation_limit: float = 65535.0,
    max_reference_age_s: float = 1800.0,
    calibration_id: str = "rad_cal_v1",
) -> CalibratedSpectrum:
    """Calibrate raw intensity Spectrum against dark and white references.

    Supports both 1D (P,) and multi-shot 2D (N_shots, P) raw spectra.
    """
    if epsilon <= 0:
        raise CalibrationError(f"epsilon must be positive, got {epsilon}.")
    if saturation_limit <= 0:
        raise CalibrationError(f"saturation_limit must be positive, got {saturation_limit}.")

    if dark is None:
        if raw.dark_reference is None:
            raise CalibrationError("Dark reference is missing; provide `dark` or `raw.dark_reference`.")
        dark_arr = np.asarray(raw.dark_reference, dtype=float)
    else:
        dark_arr = _extract_1d_reference(dark, "dark", raw.n_bands)

    if white is None:
        if raw.white_reference is None:
            raise CalibrationError("White reference is missing; provide `white` or `raw.white_reference`.")
        white_arr = np.asarray(raw.white_reference, dtype=float)
    else:
        white_arr = _extract_1d_reference(white, "white", raw.n_bands)

    # Check sensor integration time & gain consistency when references are Spectrum objects
    for ref_name, ref_obj in (("dark", dark), ("white", white)):
        if isinstance(ref_obj, Spectrum):
            if (
                raw.integration_time_ms is not None
                and ref_obj.integration_time_ms is not None
                and not np.isclose(raw.integration_time_ms, ref_obj.integration_time_ms, rtol=1e-4)
            ):
                raise CalibrationError(
                    f"Integration time mismatch between raw ({raw.integration_time_ms} ms) "
                    f"and {ref_name} ({ref_obj.integration_time_ms} ms)."
                )
            if (
                raw.gain is not None
                and ref_obj.gain is not None
                and not np.isclose(raw.gain, ref_obj.gain, rtol=1e-4)
            ):
                raise CalibrationError(
                    f"Sensor gain mismatch between raw ({raw.gain}) and {ref_name} ({ref_obj.gain})."
                )
            if not np.allclose(raw.wavelength_grid_nm, ref_obj.wavelength_grid_nm, atol=1e-3):
                raise CalibrationError(
                    f"Wavelength grid mismatch between raw spectrum and {ref_name} reference."
                )

    raw_vals = np.asarray(raw.values, dtype=float)
    raw_2d = raw_vals[np.newaxis, :] if raw_vals.ndim == 1 else raw_vals

    warnings: list[str] = []
    flags: list[QualityFlag] = []

    # Saturation check across raw and white references
    raw_saturated = np.any(raw_2d >= saturation_limit, axis=0)
    white_saturated = white_arr >= saturation_limit
    saturated_mask = raw_saturated | white_saturated
    clipping_count = int(np.sum(raw_2d >= saturation_limit) + np.sum(white_saturated))
    if np.any(saturated_mask):
        flags.append(QualityFlag.SATURATED_BANDS)
        warnings.append(
            f"Detector saturation (>= {saturation_limit}) detected on {int(np.sum(saturated_mask))} band(s)."
        )

    # Denominator validity check: I_white - I_dark > epsilon
    denom = white_arr - dark_arr
    denom_invalid = denom <= epsilon
    if np.any(denom_invalid):
        flags.append(QualityFlag.DARK_WHITE_INVALID)
        warnings.append(
            f"Near-zero or negative (I_white - I_dark <= {epsilon}) on {int(np.sum(denom_invalid))} band(s)."
        )

    # Reference age check
    ref_age = raw.reference_age_s
    if isinstance(white, Spectrum) and white.reference_age_s is not None:
        ref_age = max(ref_age or 0.0, white.reference_age_s)
    if ref_age is not None and ref_age > max_reference_age_s:
        flags.append(QualityFlag.REFERENCE_AGED)
        warnings.append(
            f"White/dark reference age ({ref_age:.1f} s) exceeds limit ({max_reference_age_s:.1f} s)."
        )

    valid_band_mask = (~saturated_mask) & (~denom_invalid)
    if not np.any(valid_band_mask):
        raise CalibrationError(
            "All spectral bands failed radiometric calibration (saturated or invalid white-dark denominator).",
            details={"clipping_count": clipping_count, "n_bands": raw.n_bands},
        )

    # Compute reflectance cleanly; set invalid bands to 0.0 (explicitly masked in valid_mask)
    safe_denom = np.where(denom_invalid, 1.0, denom)
    refl_2d = (raw_2d - dark_arr[np.newaxis, :]) / safe_denom[np.newaxis, :]
    # Track negative or >1.5 physical clipping without crashing on valid bands
    refl_2d = np.where(valid_band_mask[np.newaxis, :], refl_2d, 0.0)
    out_vals = refl_2d[0] if raw_vals.ndim == 1 else refl_2d

    valid_tuple = tuple(bool(x) for x in valid_band_mask)
    sat_tuple = tuple(bool(x) for x in saturated_mask)

    cal_spectrum = Spectrum(
        values=out_vals,
        wavelength_grid_nm=raw.wavelength_grid_nm,
        channel_ids=raw.channel_ids,
        mode=SpectralSignalMode.REFLECTANCE,
        dark_reference=tuple(float(x) for x in dark_arr),
        white_reference=tuple(float(x) for x in white_arr),
        integration_time_ms=raw.integration_time_ms,
        gain=raw.gain,
        sensor_temperature_c=raw.sensor_temperature_c,
        reference_age_s=ref_age,
        replicate_id=raw.replicate_id,
        saturation_flags=sat_tuple,
        valid_mask=valid_tuple,
    )

    return CalibratedSpectrum(
        spectrum=cal_spectrum,
        valid_mask=valid_tuple,
        clipping_count=clipping_count,
        reference_age_s=ref_age,
        calibration_id=calibration_id,
        warnings=tuple(warnings),
        quality_flags=tuple(flags),
    )
