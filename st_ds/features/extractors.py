"""Modality-specific feature extractors: Vis-NIR chemical bands, EIS frequency response, and derived covariates (Section 7, 9, 15)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from st_ds.domain.errors import SchemaError
from st_ds.domain.instrument import InstrumentProfile
from st_ds.domain.schemas import CovariateSet, EISMeasurement, Spectrum

# Chemically meaningful Vis-NIR / SWIR absorption windows (Section 9, Page 11)
VISNIR_CHEMICAL_WINDOWS_NM: dict[str, tuple[float, float]] = {
    "band_water_1430nm": (1410.0, 1450.0),
    "band_water_1925nm": (1900.0, 1950.0),
    "band_organic_2170nm": (2130.0, 2207.0),
    "band_clay_2205nm": (2185.0, 2225.0),
    "band_carbonate_2330nm": (2310.0, 2345.0),
    "band_carbonate_2520nm": (2500.0, 2550.0),
}


def available_visnir_band_names(profile: InstrumentProfile) -> tuple[str, ...]:
    """Return only the chemical band feature names physically covered by `profile`."""
    return tuple(
        name
        for name, (w_min, w_max) in VISNIR_CHEMICAL_WINDOWS_NM.items()
        if profile.bands_in_window(w_min, w_max)
    )


def extract_visnir_band_features(
    spectrum: Spectrum,
    profile: InstrumentProfile,
) -> dict[str, float]:
    """Extract mean signal inside chemically meaningful Vis-NIR/SWIR bands covered by `profile`.

    Never fabricates features for bands outside the instrument's spectral range.
    """
    spectrum.validate_against_profile(profile)
    vals = np.asarray(spectrum.values, dtype=float)
    vals_1d = np.mean(vals, axis=0) if vals.ndim == 2 else vals

    features: dict[str, float] = {}
    for band_name, (w_min, w_max) in VISNIR_CHEMICAL_WINDOWS_NM.items():
        indices = profile.bands_in_window(w_min, w_max)
        if indices:
            features[band_name] = float(np.mean(vals_1d[indices]))
    return features


def extract_eis_features(
    eis: EISMeasurement | None,
    *,
    expected_frequencies_hz: Sequence[float] | None = None,
    expected_protocol_id: str | None = None,
    freq_rtol: float = 1e-4,
) -> dict[str, float | None]:
    """Extract real, imaginary, magnitude |Z|, and phase angle features from EIS sweeps.

    Enforces frequency axis and protocol consistency (Section 15, Page 17).
    """
    if expected_frequencies_hz is None and eis is None:
        return {}

    target_freqs = (
        tuple(float(f) for f in expected_frequencies_hz)
        if expected_frequencies_hz is not None
        else tuple(float(f) for f in eis.frequencies_hz)  # type: ignore[union-attr]
    )

    features: dict[str, float | None] = {}
    if eis is None:
        for f in target_freqs:
            f_tag = f"{f:g}hz"
            features[f"eis_zreal_{f_tag}"] = None
            features[f"eis_zimag_{f_tag}"] = None
            features[f"eis_mag_{f_tag}"] = None
            features[f"eis_phase_rad_{f_tag}"] = None
        return features

    if expected_protocol_id is not None and eis.protocol_id != expected_protocol_id:
        raise SchemaError(
            f"EIS protocol mismatch: got '{eis.protocol_id}', expected '{expected_protocol_id}'.",
            details={"got_protocol": eis.protocol_id, "expected_protocol": expected_protocol_id},
        )

    if len(eis.frequencies_hz) != len(target_freqs) or not np.allclose(
        eis.frequencies_hz, target_freqs, rtol=freq_rtol, atol=1e-6
    ):
        raise SchemaError(
            "EIS frequency vector does not match registered feature profile.",
            details={"got_freqs": list(eis.frequencies_hz), "expected_freqs": list(target_freqs)},
        )

    for f, zr, zi in zip(target_freqs, eis.z_real_ohm, eis.z_imag_ohm):
        f_tag = f"{f:g}hz"
        mag = float(np.hypot(zr, zi))
        phase = float(np.arctan2(zi, zr))
        features[f"eis_zreal_{f_tag}"] = float(zr)
        features[f"eis_zimag_{f_tag}"] = float(zi)
        features[f"eis_mag_{f_tag}"] = mag
        features[f"eis_phase_rad_{f_tag}"] = phase

    return features


def extract_derived_covariates(covariates: CovariateSet) -> dict[str, float | None]:
    """Compute physically meaningful derived covariates (returns None when inputs are missing)."""
    ec_25c: float | None = None
    if covariates.ec_ds_m is not None and covariates.temperature_c is not None:
        # Standard temperature compensation to 25 degC (2% per degC)
        denom = 1.0 + 0.02 * (covariates.temperature_c - 25.0)
        if denom > 0.1:
            ec_25c = float(covariates.ec_ds_m / denom)

    return {
        "ec_25c_ds_m": ec_25c,
    }
