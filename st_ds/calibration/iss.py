"""Internal Standard / Instrument Standardization (ISS) calibration (Section 10, DEF-007).

Fixes DEF-007 by explicitly freezing the sign and multiply/divide convention of the source formula:
    CF(lambda_i) = 1 - [S(lambda_i) - M(lambda_i)] / S(lambda_i) = M(lambda_i) / S(lambda_i)

where S(lambda) is the primary/standard reference spectrum and M(lambda) is the secondary/measured
instrument spectrum of the same transfer standard. Therefore, correcting a new measurement X_measured
to the standard instrument scale is:
    X_corrected(lambda_i) = X_measured(lambda_i) / CF(lambda_i)
so that apply_iss(M, fit_iss(S, M)) == S identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from st_ds.domain.errors import CalibrationError
from st_ds.domain.schemas import Spectrum


@dataclass(frozen=True)
class ISSModel:
    """Fitted Instrument Standardization multiplicative/ratio correction profile."""

    wavelength_grid_nm: tuple[float, ...]
    correction_factors: tuple[float, ...]
    calibration_id: str = "iss_v1"
    reference_standard_id: str = "transfer_standard_v1"

    def __post_init__(self) -> None:
        if len(self.wavelength_grid_nm) != len(self.correction_factors):
            raise CalibrationError("ISSModel wavelength_grid_nm and correction_factors lengths must match.")
        cf = np.asarray(self.correction_factors, dtype=float)
        if not np.all(np.isfinite(cf)) or np.any(np.abs(cf) <= 1e-12):
            raise CalibrationError("ISSModel correction_factors must be finite and non-zero.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "reference_standard_id": self.reference_standard_id,
            "wavelength_grid_nm": list(self.wavelength_grid_nm),
            "correction_factors": list(self.correction_factors),
        }


def fit_iss(
    standard_reference: Spectrum | np.ndarray,
    measured_reference: Spectrum | np.ndarray,
    wavelength_grid_nm: tuple[float, ...] | None = None,
    *,
    min_signal: float = 1e-6,
    calibration_id: str = "iss_v1",
    reference_standard_id: str = "transfer_standard_v1",
) -> ISSModel:
    """Fit ISS correction factors CF(lambda_i) = 1 - [S(lambda_i) - M(lambda_i)] / S(lambda_i)."""
    s_arr = np.asarray(
        standard_reference.values if isinstance(standard_reference, Spectrum) else standard_reference,
        dtype=float,
    )
    m_arr = np.asarray(
        measured_reference.values if isinstance(measured_reference, Spectrum) else measured_reference,
        dtype=float,
    )

    if s_arr.ndim == 2:
        s_arr = np.mean(s_arr, axis=0)
    if m_arr.ndim == 2:
        m_arr = np.mean(m_arr, axis=0)

    if s_arr.ndim != 1 or s_arr.shape != m_arr.shape:
        raise CalibrationError(
            f"ISS fit requires matching 1D reference shapes; got S={s_arr.shape}, M={m_arr.shape}."
        )

    if wavelength_grid_nm is None:
        if isinstance(standard_reference, Spectrum):
            wavelength_grid_nm = standard_reference.wavelength_grid_nm
        elif isinstance(measured_reference, Spectrum):
            wavelength_grid_nm = measured_reference.wavelength_grid_nm
        else:
            raise CalibrationError("wavelength_grid_nm must be provided when passing raw numpy arrays.")

    if (
        isinstance(standard_reference, Spectrum)
        and isinstance(measured_reference, Spectrum)
        and not np.allclose(standard_reference.wavelength_grid_nm, measured_reference.wavelength_grid_nm, atol=1e-3)
    ):
        raise CalibrationError("ISS fit requires identical wavelength grids for standard and measured spectra.")

    if len(wavelength_grid_nm) != s_arr.size:
        raise CalibrationError("wavelength_grid_nm length does not match reference spectrum bands.")

    if np.any(np.abs(s_arr) <= min_signal) or np.any(np.abs(m_arr) <= min_signal):
        raise CalibrationError(
            f"Reference signals in fit_iss must exceed min_signal={min_signal} across all bands."
        )

    # Exact source PDF formula: CF(λ_i) = 1 - [S(λ_i) - M(λ_i)] / S(λ_i) == M(λ_i) / S(λ_i)
    cf = 1.0 - (s_arr - m_arr) / s_arr
    return ISSModel(
        wavelength_grid_nm=tuple(float(w) for w in wavelength_grid_nm),
        correction_factors=tuple(float(c) for c in cf),
        calibration_id=calibration_id,
        reference_standard_id=reference_standard_id,
    )


def apply_iss(spectrum: Spectrum, iss_model: ISSModel, *, tol_nm: float = 1e-3) -> Spectrum:
    """Apply fitted ISSModel to transform a measured spectrum onto the standard reference scale.

    Applies X_corrected = X_measured / CF so that measured_reference -> standard_reference.
    """
    if spectrum.n_bands != len(iss_model.wavelength_grid_nm):
        raise CalibrationError(
            f"Spectrum band count ({spectrum.n_bands}) does not match ISSModel "
            f"({len(iss_model.wavelength_grid_nm)})."
        )
    if not np.allclose(spectrum.wavelength_grid_nm, iss_model.wavelength_grid_nm, atol=tol_nm, rtol=0.0):
        raise CalibrationError("Spectrum wavelength_grid_nm does not match ISSModel wavelength grid.")

    cf = np.asarray(iss_model.correction_factors, dtype=float)
    corrected = spectrum.values / cf
    return Spectrum(
        values=corrected,
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
