"""Covariate validation, time-tolerant alignment, and multi-sensor feature fusion (Section 15 & 33).

Implements:
- `validate_covariates`: distinguishes `not_measured` vs `qc_failed` missingness
- `align_spectral_and_covariates`: joins separate spectral & covariate records by `sample_id`
  subject to an explicit UTC timestamp tolerance policy
- `FeatureFusionPipeline`: fits separate scalers for spectral and covariate branches,
  learns training-fold imputation medians + missingness indicators, and exposes
  `transform_covariates` and `build_feature_vector`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Sequence

import numpy as np

from st_ds.chemometrics.pipeline import ChemometricPipeline
from st_ds.domain.errors import CalibrationError, SchemaError
from st_ds.domain.instrument import InstrumentProfile, get_builtin_profile
from st_ds.domain.quality import QualityFlag
from st_ds.domain.schemas import CovariateSet, Measurement, Spectrum
from st_ds.features.extractors import (
    available_visnir_band_names,
    extract_derived_covariates,
    extract_eis_features,
    extract_visnir_band_features,
)
from st_ds.features.registry import FeatureProfileSpec, FeatureRegistry
from st_ds.io.dataset import SoilDataset


@dataclass(frozen=True)
class CovariateValidationResult:
    """Result of validating a CovariateSet against required fields and QC flags."""

    is_valid: bool
    canonical_values: dict[str, float | None]
    missing_flags: dict[str, str]
    quality_flags: tuple[QualityFlag, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "canonical_values": dict(self.canonical_values),
            "missing_flags": dict(self.missing_flags),
            "quality_flags": [f.value for f in self.quality_flags],
            "warnings": list(self.warnings),
        }


def validate_covariates(
    covariates: CovariateSet,
    *,
    covariate_names: Sequence[str] = ("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m"),
    allow_missing: bool = True,
) -> CovariateValidationResult:
    """Validate canonical covariates and distinguish `not_measured` vs `qc_failed` states."""
    canonical_values: dict[str, float | None] = {}
    missing_flags: dict[str, str] = {}
    q_flags: list[QualityFlag] = []
    warnings: list[str] = []

    # Map CovariateSet missing_reasons keys (which may use short quantity names like 'bulk_density')
    short_key_map = {
        "ph": "ph",
        "moisture_volumetric_pct": "moisture_volumetric",
        "moisture_gravimetric_pct": "moisture_gravimetric",
        "temperature_c": "temperature",
        "ec_ds_m": "ec",
        "bulk_density_g_cm3": "bulk_density",
        "coarse_fragments_fraction": "coarse_fragments",
    }

    for name in covariate_names:
        if not hasattr(covariates, name):
            raise SchemaError(f"Unknown covariate attribute '{name}' in validate_covariates.")
        val = getattr(covariates, name)
        canonical_values[name] = float(val) if val is not None else None
        if val is None:
            short_key = short_key_map.get(name, name)
            reason = covariates.missing_reasons.get(
                name, covariates.missing_reasons.get(short_key, "not_measured")
            )
            missing_flags[name] = reason
            if reason == "qc_failed":
                if QualityFlag.COVARIATE_QC_FAILED not in q_flags:
                    q_flags.append(QualityFlag.COVARIATE_QC_FAILED)
                warnings.append(f"Covariate '{name}' failed sensor QC (`qc_failed`).")
            else:
                if QualityFlag.MISSING_COVARIATE not in q_flags:
                    q_flags.append(QualityFlag.MISSING_COVARIATE)
                warnings.append(f"Covariate '{name}' was not measured (`not_measured`).")

    is_valid = len(missing_flags) == 0 or allow_missing
    if not is_valid:
        raise SchemaError(
            f"Missing or QC-failed required covariates under fail-closed policy: {missing_flags}",
            details={"missing_flags": missing_flags},
        )

    return CovariateValidationResult(
        is_valid=is_valid,
        canonical_values=canonical_values,
        missing_flags=missing_flags,
        quality_flags=tuple(q_flags),
        warnings=tuple(warnings),
    )


def _parse_utc_timestamp(ts: str) -> datetime:
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise SchemaError(f"Invalid ISO-8601 timestamp '{ts}': {exc}") from exc


def align_spectral_and_covariates(
    spectral_records: Sequence[Measurement],
    covariate_records: Sequence[tuple[str, str, CovariateSet]] | Sequence[Measurement] | None = None,
    *,
    max_time_delta_s: float = 300.0,
) -> list[Measurement]:
    """Align spectral measurements and covariate scans by `sample_id` within `max_time_delta_s` (Section 15).

    If `covariate_records` is None, verifies internal timestamp validity of `spectral_records`.
    When `covariate_records` is provided as either `Measurement` objects or
    `(sample_id, timestamp_utc, CovariateSet)` tuples, matches each spectral measurement
    to its covariate record and enforces `|t_spec - t_cov| <= max_time_delta_s`.
    """
    if max_time_delta_s < 0:
        raise SchemaError("max_time_delta_s must be non-negative.")

    if covariate_records is None:
        for m in spectral_records:
            _parse_utc_timestamp(m.timestamp_utc)
        return list(spectral_records)

    cov_lookup: dict[str, tuple[datetime, CovariateSet]] = {}
    for item in covariate_records:
        if isinstance(item, Measurement):
            sid, ts_str, cov = item.sample_id, item.timestamp_utc, item.covariates
        else:
            sid, ts_str, cov = item
        if sid in cov_lookup:
            raise SchemaError(f"Duplicate covariate record for sample_id '{sid}'.")
        cov_lookup[sid] = (_parse_utc_timestamp(ts_str), cov)

    aligned: list[Measurement] = []
    for m in spectral_records:
        if m.sample_id not in cov_lookup:
            raise SchemaError(
                f"No matching covariate record found for spectral sample_id '{m.sample_id}'.",
                details={"sample_id": m.sample_id},
            )
        t_spec = _parse_utc_timestamp(m.timestamp_utc)
        t_cov, cov = cov_lookup[m.sample_id]
        delta_s = abs((t_spec - t_cov).total_seconds())
        if delta_s > max_time_delta_s:
            raise SchemaError(
                f"Timestamp difference ({delta_s:.1f} s) between spectrum and covariates for "
                f"sample_id '{m.sample_id}' exceeds tolerance ({max_time_delta_s:.1f} s).",
                details={"sample_id": m.sample_id, "delta_s": delta_s, "max_time_delta_s": max_time_delta_s},
            )
        aligned.append(replace(m, covariates=cov))
    return aligned


@dataclass(frozen=True)
class FeatureFusionConfig:
    """Configuration for separate-branch scaling, imputation, and feature fusion."""

    covariate_names: tuple[str, ...] = ("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m")
    missing_strategy: Literal["fail_closed", "median_with_indicator"] = "median_with_indicator"
    spectral_scaler: Literal["none", "standard"] = "standard"
    covariate_scaler: Literal["standard", "robust"] = "robust"
    include_eis: bool = False
    include_derived: bool = False
    include_visnir_bands: bool = False
    eis_frequencies_hz: tuple[float, ...] = ()
    eis_protocol_id: str | None = None


@dataclass
class FusedFeatures:
    """Output container holding separate spectral/covariate branches and fused matrix."""

    X_spectral: np.ndarray
    X_covariates: np.ndarray
    X_fused: np.ndarray
    spectral_feature_names: tuple[str, ...]
    covariate_feature_names: tuple[str, ...]
    fused_feature_names: tuple[str, ...]
    missingness_mask: np.ndarray
    missing_reasons: list[dict[str, str]]
    feature_profile: FeatureProfileSpec


class _BranchScaler:
    """Lightweight deterministic standard or robust scaler with zero-variance protection."""

    def __init__(self, mode: Literal["none", "standard", "robust"]) -> None:
        self.mode = mode
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> _BranchScaler:
        arr = np.asarray(X, dtype=float)
        if self.mode == "none" or arr.shape[1] == 0:
            self.center = np.zeros(arr.shape[1], dtype=float)
            self.scale = np.ones(arr.shape[1], dtype=float)
            return self
        if self.mode == "standard":
            self.center = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            self.scale = np.where(std > 1e-12, std, 1.0)
        elif self.mode == "robust":
            self.center = np.median(arr, axis=0)
            q75, q25 = np.percentile(arr, [75, 25], axis=0)
            iqr = q75 - q25
            std = np.std(arr, axis=0)
            # Fall back to std if IQR is zero on discrete-like columns, then 1.0 if constant
            self.scale = np.where(iqr > 1e-12, iqr, np.where(std > 1e-12, std, 1.0))
        else:
            raise SchemaError(f"Unsupported scaler mode '{self.mode}'.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise CalibrationError("Scaler must be fitted before transform.")
        arr = np.asarray(X, dtype=float)
        if self.mode == "none" or arr.shape[1] == 0:
            return arr.copy()
        return (arr - self.center[np.newaxis, :]) / self.scale[np.newaxis, :]


class FeatureFusionPipeline:
    """Builds named spectral + covariate feature vectors with separate branch scaling."""

    def __init__(
        self,
        config: FeatureFusionConfig | None = None,
        *,
        registry: FeatureRegistry | None = None,
    ) -> None:
        self.config = config or FeatureFusionConfig()
        self.registry = registry or FeatureRegistry()
        self.instrument_profile: InstrumentProfile | None = None
        self.feature_profile: FeatureProfileSpec | None = None
        self._spectral_scaler: _BranchScaler | None = None
        self._covariate_scaler: _BranchScaler | None = None
        self._covariate_medians: np.ndarray | None = None
        self._numeric_cov_names: tuple[str, ...] = ()
        self._indicator_names: tuple[str, ...] = ()
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _extract_raw_spectral_block(
        self,
        measurements: Sequence[Measurement],
        *,
        spectral_matrix: np.ndarray | None = None,
        chemometric_pipeline: ChemometricPipeline | None = None,
    ) -> np.ndarray:
        assert self.instrument_profile is not None
        if spectral_matrix is not None:
            base_spec = np.asarray(spectral_matrix, dtype=float)
        elif chemometric_pipeline is not None:
            base_spec = np.vstack([chemometric_pipeline.transform_spectrum(m.spectrum) for m in measurements])
        else:
            base_spec = np.vstack([m.spectrum.values for m in measurements])

        if base_spec.ndim != 2 or base_spec.shape[0] != len(measurements):
            raise SchemaError(
                f"spectral_matrix shape {base_spec.shape} does not match {len(measurements)} measurements."
            )
        if base_spec.shape[1] != self.instrument_profile.n_channels:
            raise SchemaError(
                f"Spectral band count ({base_spec.shape[1]}) does not match instrument profile "
                f"'{self.instrument_profile.profile_id}' ({self.instrument_profile.n_channels})."
            )

        if not self.config.include_visnir_bands:
            return base_spec

        band_names = available_visnir_band_names(self.instrument_profile)
        band_rows: list[list[float]] = []
        for idx, m in enumerate(measurements):
            spec_obj = Spectrum(
                values=base_spec[idx],
                wavelength_grid_nm=m.spectrum.wavelength_grid_nm,
                channel_ids=m.spectrum.channel_ids,
                mode=m.spectrum.mode,
            )
            b_feats = extract_visnir_band_features(spec_obj, self.instrument_profile)
            band_rows.append([b_feats[bn] for bn in band_names])
        return np.hstack([base_spec, np.asarray(band_rows, dtype=float)]) if band_names else base_spec

    def _extract_raw_covariate_block(
        self,
        measurements: Sequence[Measurement],
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
        """Extract raw numeric covariates (with NaN for missing) and missingness reason metadata."""
        rows: list[list[float]] = []
        missing_masks: list[list[bool]] = []
        reasons_list: list[dict[str, str]] = []

        allow_missing = self.config.missing_strategy == "median_with_indicator"

        for m in measurements:
            val_res = validate_covariates(
                m.covariates,
                covariate_names=self.config.covariate_names,
                allow_missing=allow_missing,
            )
            row_vals: list[float] = []
            row_miss: list[bool] = []

            for cname in self.config.covariate_names:
                v = val_res.canonical_values[cname]
                row_vals.append(float(v) if v is not None else float("nan"))
                row_miss.append(v is None)

            if self.config.include_derived:
                derived = extract_derived_covariates(m.covariates)
                for dname in ("ec_25c_ds_m",):
                    dv = derived[dname]
                    if dv is None and not allow_missing:
                        raise SchemaError(f"Derived covariate '{dname}' is missing under fail_closed policy.")
                    row_vals.append(float(dv) if dv is not None else float("nan"))
                    row_miss.append(dv is None)

            if self.config.include_eis:
                eis_feats = extract_eis_features(
                    m.covariates.eis,
                    expected_frequencies_hz=self.config.eis_frequencies_hz,
                    expected_protocol_id=self.config.eis_protocol_id,
                )
                for f in self.config.eis_frequencies_hz:
                    f_tag = f"{float(f):g}hz"
                    for key in (
                        f"eis_zreal_{f_tag}",
                        f"eis_zimag_{f_tag}",
                        f"eis_mag_{f_tag}",
                        f"eis_phase_rad_{f_tag}",
                    ):
                        ev = eis_feats.get(key)
                        if ev is None and not allow_missing:
                            raise SchemaError(f"EIS feature '{key}' is missing under fail_closed policy.")
                        row_vals.append(float(ev) if ev is not None else float("nan"))
                        row_miss.append(ev is None)

            rows.append(row_vals)
            missing_masks.append(row_miss)
            reasons_list.append(val_res.missing_flags)

        return (
            np.asarray(rows, dtype=float),
            np.asarray(missing_masks, dtype=bool),
            reasons_list,
        )

    def fit(
        self,
        measurements: Sequence[Measurement] | SoilDataset,
        *,
        spectral_matrix: np.ndarray | None = None,
        chemometric_pipeline: ChemometricPipeline | None = None,
        profile: InstrumentProfile | None = None,
    ) -> FeatureFusionPipeline:
        """Fit imputers and separate spectral/covariate scalers on training records only."""
        is_ds = isinstance(measurements, SoilDataset)
        recs = measurements.records if is_ds else list(measurements)
        if len(recs) == 0:
            raise SchemaError("FeatureFusionPipeline.fit requires at least 1 measurement.")

        prof_id = recs[0].instrument_profile_id
        resolved_profile = profile or (measurements.instrument_profile if is_ds else None)
        self.instrument_profile = resolved_profile or get_builtin_profile(prof_id)
        for r in recs:
            r.spectrum.validate_against_profile(self.instrument_profile)

        visnir_names = (
            available_visnir_band_names(self.instrument_profile)
            if self.config.include_visnir_bands
            else ()
        )
        derived_names = ("ec_25c_ds_m",) if self.config.include_derived else ()

        eis_freqs = self.config.eis_frequencies_hz
        eis_proto = self.config.eis_protocol_id
        if self.config.include_eis and not eis_freqs:
            first_eis = next((r.covariates.eis for r in recs if r.covariates.eis is not None), None)
            if first_eis is None:
                raise SchemaError("include_eis=True but no EISMeasurement found and eis_frequencies_hz is empty.")
            eis_freqs = first_eis.frequencies_hz
            eis_proto = eis_proto or first_eis.protocol_id
            # Persist inferred frequencies back onto config snapshot via instance attributes
            self.config = FeatureFusionConfig(
                covariate_names=self.config.covariate_names,
                missing_strategy=self.config.missing_strategy,
                spectral_scaler=self.config.spectral_scaler,
                covariate_scaler=self.config.covariate_scaler,
                include_eis=self.config.include_eis,
                include_derived=self.config.include_derived,
                include_visnir_bands=self.config.include_visnir_bands,
                eis_frequencies_hz=eis_freqs,
                eis_protocol_id=eis_proto,
            )

        include_indicators = self.config.missing_strategy == "median_with_indicator"
        self.feature_profile = self.registry.build_profile_from_instrument(
            self.instrument_profile,
            covariate_names=self.config.covariate_names,
            visnir_band_names=visnir_names,
            derived_names=derived_names,
            eis_frequencies_hz=eis_freqs if self.config.include_eis else (),
            eis_protocol_id=eis_proto,
            include_missing_indicators=include_indicators,
        )

        # Fit spectral branch scaler
        X_spec_raw = self._extract_raw_spectral_block(
            recs,
            spectral_matrix=spectral_matrix,
            chemometric_pipeline=chemometric_pipeline,
        )
        self._spectral_scaler = _BranchScaler(self.config.spectral_scaler).fit(X_spec_raw)

        # Fit covariate medians (for imputation) and covariate branch scaler
        X_cov_raw, miss_mask, _ = self._extract_raw_covariate_block(recs)
        n_cov_cols = X_cov_raw.shape[1]
        medians = np.zeros(n_cov_cols, dtype=float)
        for j in range(n_cov_cols):
            valid_col = X_cov_raw[~np.isnan(X_cov_raw[:, j]), j]
            if valid_col.size == 0:
                raise SchemaError(
                    f"Covariate column index {j} is 100% missing across training samples; "
                    "cannot fit imputer without inventing silent defaults (ADR-002)."
                )
            medians[j] = float(np.median(valid_col))
        self._covariate_medians = medians

        X_cov_imputed = X_cov_raw.copy()
        for j in range(n_cov_cols):
            nan_rows = np.isnan(X_cov_imputed[:, j])
            X_cov_imputed[nan_rows, j] = medians[j]

        self._covariate_scaler = _BranchScaler(self.config.covariate_scaler).fit(X_cov_imputed)
        self._is_fitted = True
        return self

    def transform_covariates(
        self,
        measurements: Measurement | Sequence[Measurement] | SoilDataset,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
        """Transform covariates using fitted medians, separate covariate scaler, and missingness indicators."""
        if not self._is_fitted or self._covariate_medians is None or self._covariate_scaler is None:
            raise CalibrationError("FeatureFusionPipeline must be fitted before `transform_covariates`.")

        if isinstance(measurements, SoilDataset):
            recs = measurements.records
        elif isinstance(measurements, Measurement):
            recs = [measurements]
        else:
            recs = list(measurements)

        X_cov_raw, miss_mask, reasons_list = self._extract_raw_covariate_block(recs)
        X_cov_imputed = X_cov_raw.copy()
        for j in range(X_cov_raw.shape[1]):
            nan_rows = np.isnan(X_cov_imputed[:, j])
            if np.any(nan_rows):
                X_cov_imputed[nan_rows, j] = self._covariate_medians[j]

        X_cov_scaled = self._covariate_scaler.transform(X_cov_imputed)

        if self.config.missing_strategy == "median_with_indicator":
            # Missing indicators stay strictly binary {0.0, 1.0} corresponding to config.covariate_names
            n_primary = len(self.config.covariate_names)
            indicators = miss_mask[:, :n_primary].astype(float)
            X_cov_out = np.hstack([X_cov_scaled, indicators])
        else:
            X_cov_out = X_cov_scaled

        return X_cov_out, miss_mask, reasons_list

    def build_feature_vector(
        self,
        measurements: Measurement | Sequence[Measurement] | SoilDataset,
        *,
        spectral_matrix: np.ndarray | None = None,
        chemometric_pipeline: ChemometricPipeline | None = None,
    ) -> FusedFeatures:
        """Build aligned, separately scaled spectral, covariate, and fused feature vectors (Section 15)."""
        if (
            not self._is_fitted
            or self.instrument_profile is None
            or self.feature_profile is None
            or self._spectral_scaler is None
        ):
            raise CalibrationError("FeatureFusionPipeline must be fitted before `build_feature_vector`.")

        if isinstance(measurements, SoilDataset):
            recs = measurements.records
        elif isinstance(measurements, Measurement):
            recs = [measurements]
        else:
            recs = list(measurements)

        for r in recs:
            r.spectrum.validate_against_profile(self.instrument_profile)

        X_spec_raw = self._extract_raw_spectral_block(
            recs,
            spectral_matrix=spectral_matrix,
            chemometric_pipeline=chemometric_pipeline,
        )
        X_spec_scaled = self._spectral_scaler.transform(X_spec_raw)
        X_cov_out, miss_mask, reasons_list = self.transform_covariates(recs)
        X_fused = np.hstack([X_spec_scaled, X_cov_out])

        spec_names = self.feature_profile.spectral_feature_names
        cov_names = self.feature_profile.covariate_feature_names

        return FusedFeatures(
            X_spectral=X_spec_scaled,
            X_covariates=X_cov_out,
            X_fused=X_fused,
            spectral_feature_names=spec_names,
            covariate_feature_names=cov_names,
            fused_feature_names=spec_names + cov_names,
            missingness_mask=miss_mask,
            missing_reasons=reasons_list,
            feature_profile=self.feature_profile,
        )
