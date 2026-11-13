"""Spectral transforms: grid validation, reflectance-to-absorbance, and SNV (Section 11, Page 13).

Implements:
- `validate_spectral_grid(wavelengths)`
- `to_absorbance(X)` -> A(lambda) = -log10(R(lambda))
- `snv_transform(X)` -> row-wise deterministic Standard Normal Variate normalization
"""

from __future__ import annotations

from typing import overload

import numpy as np

from st_ds.domain.errors import SchemaError, SpectralQualityError
from st_ds.domain.schemas import SpectralSignalMode, Spectrum
from st_ds.domain.units import UnitRegistry


def validate_spectral_grid(wavelengths: np.ndarray | tuple[float, ...] | list[float]) -> np.ndarray:
    """Validate that a wavelength vector is 1D, finite, within physical bounds, and strictly increasing."""
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1 or wl.size == 0:
        raise SchemaError("Wavelength grid must be a non-empty 1D array.", details={"shape": wl.shape})
    if not np.all(np.isfinite(wl)):
        raise SchemaError("Wavelength grid contains NaN or Inf values.")
    for val in wl:
        UnitRegistry.validate_physical_range("wavelength", float(val))
    if wl.size > 1 and not np.all(np.diff(wl) > 0):
        raise SchemaError("Wavelength grid must be strictly increasing (lambda_i < lambda_{i+1}).")
    return wl


def _clone_spectrum_with_values(
    spec: Spectrum,
    values: np.ndarray,
    *,
    mode: SpectralSignalMode | None = None,
) -> Spectrum:
    return Spectrum(
        values=values,
        wavelength_grid_nm=spec.wavelength_grid_nm,
        channel_ids=spec.channel_ids,
        mode=mode or spec.mode,
        dark_reference=spec.dark_reference,
        white_reference=spec.white_reference,
        integration_time_ms=spec.integration_time_ms,
        gain=spec.gain,
        sensor_temperature_c=spec.sensor_temperature_c,
        reference_age_s=spec.reference_age_s,
        replicate_id=spec.replicate_id,
        saturation_flags=spec.saturation_flags,
        valid_mask=spec.valid_mask,
    )


@overload
def to_absorbance(data: Spectrum, *, min_reflectance: float = 1e-6) -> Spectrum: ...


@overload
def to_absorbance(data: np.ndarray, *, min_reflectance: float = 1e-6) -> np.ndarray: ...


def to_absorbance(
    data: Spectrum | np.ndarray,
    *,
    min_reflectance: float = 1e-6,
) -> Spectrum | np.ndarray:
    """Convert positive reflectance R(lambda) to absorbance A(lambda) = -log10(R(lambda)).

    Raises SpectralQualityError on non-finite or non-positive reflectance values on valid bands.
    """
    if min_reflectance <= 0:
        raise SchemaError(f"min_reflectance must be positive, got {min_reflectance}.")

    is_spec = isinstance(data, Spectrum)
    arr = np.asarray(data.values if is_spec else data, dtype=float)
    if arr.ndim not in (1, 2) or arr.size == 0:
        raise SchemaError("Input to `to_absorbance` must be a non-empty 1D or 2D array.")
    if not np.all(np.isfinite(arr)):
        raise SpectralQualityError("Input reflectance contains NaN or Inf values.")

    if is_spec and data.mode == SpectralSignalMode.ABSORBANCE:
        return data

    valid_mask = (
        np.asarray(data.valid_mask, dtype=bool)
        if is_spec and data.valid_mask is not None
        else np.ones(arr.shape[-1], dtype=bool)
    )
    arr_2d = arr[np.newaxis, :] if arr.ndim == 1 else arr
    valid_vals = arr_2d[:, valid_mask]

    if np.any(valid_vals <= 0.0) or np.any(valid_vals < min_reflectance):
        min_found = float(np.min(valid_vals))
        raise SpectralQualityError(
            f"Reflectance-to-absorbance requires strictly positive reflectance (>= {min_reflectance}); "
            f"found min={min_found:.6g}.",
            details={"min_reflectance_found": min_found, "threshold": min_reflectance},
        )

    safe_vals = np.where(valid_mask[np.newaxis, :], arr_2d, 1.0)
    abs_2d = -np.log10(safe_vals)
    abs_2d = np.where(valid_mask[np.newaxis, :], abs_2d, 0.0)
    out = abs_2d[0] if arr.ndim == 1 else abs_2d

    if is_spec:
        return _clone_spectrum_with_values(data, out, mode=SpectralSignalMode.ABSORBANCE)
    return out


@overload
def snv_transform(data: Spectrum, *, eps: float = 1e-12) -> Spectrum: ...


@overload
def snv_transform(data: np.ndarray, *, eps: float = 1e-12) -> np.ndarray: ...


def snv_transform(
    data: Spectrum | np.ndarray,
    *,
    eps: float = 1e-12,
) -> Spectrum | np.ndarray:
    """Apply deterministic row-wise Standard Normal Variate (SNV) normalization:
        SNV(x_i) = [x_i - mean(x)] / [std(x) + eps]
    """
    if eps <= 0:
        raise SchemaError(f"eps must be positive, got {eps}.")

    is_spec = isinstance(data, Spectrum)
    arr = np.asarray(data.values if is_spec else data, dtype=float)
    if arr.ndim not in (1, 2) or arr.size == 0:
        raise SchemaError("Input to `snv_transform` must be a non-empty 1D or 2D array.")
    if not np.all(np.isfinite(arr)):
        raise SpectralQualityError("Input spectrum to `snv_transform` contains NaN or Inf.")

    arr_2d = arr[np.newaxis, :] if arr.ndim == 1 else arr
    valid_mask = (
        np.asarray(data.valid_mask, dtype=bool)
        if is_spec and data.valid_mask is not None
        else np.ones(arr.shape[-1], dtype=bool)
    )
    if not np.any(valid_mask):
        raise SpectralQualityError("Cannot compute SNV with zero valid spectral bands.")

    valid_slice = arr_2d[:, valid_mask]
    means = np.mean(valid_slice, axis=1, keepdims=True)
    stds = np.std(valid_slice, axis=1, keepdims=True)
    norm_2d = (arr_2d - means) / (stds + eps)
    norm_2d = np.where(valid_mask[np.newaxis, :], norm_2d, 0.0)

    out = norm_2d[0] if arr.ndim == 1 else norm_2d
    if is_spec:
        return _clone_spectrum_with_values(data, out)
    return out
