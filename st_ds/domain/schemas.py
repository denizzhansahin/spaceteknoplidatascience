"""Canonical domain data models and measurement/report contracts (Section 8, 22, 23, Appendix B).

Implements:
- Spectrum (wavelength-coupled, with white/dark references, integration time, saturation flags)
- EISMeasurement (frequency-coupled complex impedance)
- CovariateSet (canonical unit conversion, explicit missingness reasons, zero hidden defaults)
- SoilReference (laboratory ground truth with dry-matter basis and method provenance)
- Measurement (complete canonical input record)
- PredictionReport (Appendix B JSON-serializable output contract)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from st_ds import SCHEMA_VERSION
from st_ds.domain.errors import ReportSerializationError, SchemaError, UnitError
from st_ds.domain.instrument import BUILTIN_PROFILES, InstrumentProfile
from st_ds.domain.quality import QualityFlag, QualityMetadata, QualityStatus
from st_ds.domain.units import UnitRegistry


class SpectralSignalMode(str, Enum):
    """Representation mode of the spectral values."""

    RAW_INTENSITY = "raw_intensity"
    REFLECTANCE = "reflectance"
    ABSORBANCE = "absorbance"
    LIBS_EMISSION = "libs_emission"


@dataclass
class DepthInterval:
    """Soil sampling layer depth interval normalized to centimeters (cm)."""

    top_cm: float
    bottom_cm: float

    @classmethod
    def from_units(cls, top: float, bottom: float, unit: str = "cm") -> DepthInterval:
        top_c = UnitRegistry.to_canonical("depth", top, unit)
        bot_c = UnitRegistry.to_canonical("depth", bottom, unit)
        return cls(top_cm=top_c, bottom_cm=bot_c)

    def __post_init__(self) -> None:
        UnitRegistry.validate_physical_range("depth", float(self.top_cm))
        UnitRegistry.validate_physical_range("depth", float(self.bottom_cm))
        if self.bottom_cm <= self.top_cm:
            raise SchemaError(
                f"DepthInterval bottom_cm ({self.bottom_cm}) must be strictly greater than top_cm ({self.top_cm}).",
                details={"top_cm": self.top_cm, "bottom_cm": self.bottom_cm},
            )

    @property
    def thickness_cm(self) -> float:
        return float(self.bottom_cm - self.top_cm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_cm": float(self.top_cm),
            "bottom_cm": float(self.bottom_cm),
            "thickness_cm": self.thickness_cm,
            "unit": "cm",
        }


@dataclass
class SpatialCoordinates:
    """Optional geospatial coordinates with explicit CRS (Section 8, Page 10)."""

    lat: float
    lon: float
    crs_epsg: str = "EPSG:4326"
    elevation_m: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.lat) or not (-90.0 <= self.lat <= 90.0):
            raise SchemaError(
                f"Latitude {self.lat} is out of valid [-90, 90] bounds.",
                details={"lat": self.lat},
            )
        if not np.isfinite(self.lon) or not (-180.0 <= self.lon <= 180.0):
            raise SchemaError(
                f"Longitude {self.lon} is out of valid [-180, 180] bounds.",
                details={"lon": self.lon},
            )
        if not self.crs_epsg or not self.crs_epsg.strip():
            raise SchemaError("SpatialCoordinates.crs_epsg cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": float(self.lat),
            "lon": float(self.lon),
            "crs_epsg": self.crs_epsg,
            "elevation_m": float(self.elevation_m) if self.elevation_m is not None else None,
        }


@dataclass
class EISMeasurement:
    """Electrochemical Impedance Spectroscopy (EIS) frequency sweep (Section 15, Page 17)."""

    frequencies_hz: tuple[float, ...]
    z_real_ohm: tuple[float, ...]
    z_imag_ohm: tuple[float, ...]
    protocol_id: str = "eis_sweep_v1"

    @classmethod
    def from_units(
        cls,
        frequencies: list[float] | tuple[float, ...],
        z_real: list[float] | tuple[float, ...],
        z_imag: list[float] | tuple[float, ...],
        *,
        freq_unit: str = "Hz",
        impedance_unit: str = "ohm",
        protocol_id: str = "eis_sweep_v1",
    ) -> EISMeasurement:
        freqs_c = tuple(UnitRegistry.to_canonical("frequency", f, freq_unit) for f in frequencies)
        zr_c = tuple(UnitRegistry.to_canonical("impedance", r, impedance_unit) for r in z_real)
        zi_c = tuple(UnitRegistry.to_canonical("impedance", im, impedance_unit) for im in z_imag)
        return cls(
            frequencies_hz=freqs_c,
            z_real_ohm=zr_c,
            z_imag_ohm=zi_c,
            protocol_id=protocol_id,
        )

    def __post_init__(self) -> None:
        if len(self.frequencies_hz) == 0:
            raise SchemaError("EISMeasurement.frequencies_hz cannot be empty.")
        if len(self.z_real_ohm) != len(self.frequencies_hz) or len(self.z_imag_ohm) != len(self.frequencies_hz):
            raise SchemaError(
                "EISMeasurement z_real_ohm and z_imag_ohm lengths must match frequencies_hz.",
                details={
                    "n_freq": len(self.frequencies_hz),
                    "n_real": len(self.z_real_ohm),
                    "n_imag": len(self.z_imag_ohm),
                },
            )
        for f in self.frequencies_hz:
            UnitRegistry.validate_physical_range("frequency", float(f))
        for r, im in zip(self.z_real_ohm, self.z_imag_ohm):
            UnitRegistry.validate_physical_range("impedance", float(r))
            UnitRegistry.validate_physical_range("impedance", float(im))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "frequencies_hz": list(self.frequencies_hz),
            "z_real_ohm": list(self.z_real_ohm),
            "z_imag_ohm": list(self.z_imag_ohm),
            "units": {"frequency": "Hz", "impedance": "ohm"},
        }


@dataclass
class Spectrum:
    """Wavelength-coupled spectral measurement object (Section 8, Page 10).

    Supports both single spectra (1D array of length P) and multi-shot replicate series
    such as LIBS shots (2D array of shape (N_shots, P)).
    """

    values: np.ndarray
    wavelength_grid_nm: tuple[float, ...]
    channel_ids: tuple[str, ...]
    mode: SpectralSignalMode = SpectralSignalMode.REFLECTANCE
    dark_reference: tuple[float, ...] | None = None
    white_reference: tuple[float, ...] | None = None
    integration_time_ms: float | None = None
    gain: float | None = None
    sensor_temperature_c: float | None = None
    reference_age_s: float | None = None
    replicate_id: str = "rep_1"
    saturation_flags: tuple[bool, ...] | None = None
    valid_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=float)
        if arr.ndim not in (1, 2) or arr.size == 0:
            raise SchemaError(
                "Spectrum.values must be a non-empty 1D (P,) or 2D (N_shots, P) array.",
                details={"shape": arr.shape},
            )
        if not np.all(np.isfinite(arr)):
            raise SchemaError("Spectrum.values contains NaN or Inf values.")

        object.__setattr__(self, "values", arr)

        n_bands = arr.shape[-1]
        if len(self.wavelength_grid_nm) != n_bands:
            raise SchemaError(
                "Spectrum.wavelength_grid_nm length must match the spectral band dimension.",
                details={"n_bands": n_bands, "n_wavelengths": len(self.wavelength_grid_nm)},
            )
        if len(self.channel_ids) != n_bands:
            raise SchemaError(
                "Spectrum.channel_ids length must match the spectral band dimension.",
                details={"n_bands": n_bands, "n_channel_ids": len(self.channel_ids)},
            )

        wl = np.asarray(self.wavelength_grid_nm, dtype=float)
        if not np.all(np.isfinite(wl)) or (wl.size > 1 and not np.all(np.diff(wl) > 0)):
            raise SchemaError("Spectrum.wavelength_grid_nm must be finite and strictly increasing.")
        for w in wl:
            UnitRegistry.validate_physical_range("wavelength", float(w))

        if self.dark_reference is not None and len(self.dark_reference) != n_bands:
            raise SchemaError("Spectrum.dark_reference length must match spectral band count.")
        if self.white_reference is not None and len(self.white_reference) != n_bands:
            raise SchemaError("Spectrum.white_reference length must match spectral band count.")

        if self.saturation_flags is None:
            object.__setattr__(self, "saturation_flags", tuple(False for _ in range(n_bands)))
        elif len(self.saturation_flags) != n_bands:
            raise SchemaError("Spectrum.saturation_flags length must match spectral band count.")

        if self.valid_mask is None:
            object.__setattr__(self, "valid_mask", tuple(True for _ in range(n_bands)))
        elif len(self.valid_mask) != n_bands:
            raise SchemaError("Spectrum.valid_mask length must match spectral band count.")

        if self.sensor_temperature_c is not None:
            UnitRegistry.validate_physical_range("temperature", float(self.sensor_temperature_c))

    @property
    def n_bands(self) -> int:
        return int(self.values.shape[-1])

    @property
    def n_shots(self) -> int:
        return 1 if self.values.ndim == 1 else int(self.values.shape[0])

    def validate_against_profile(self, profile: InstrumentProfile, tol_nm: float = 1e-3) -> None:
        """Verify that this spectrum matches the given InstrumentProfile."""
        if self.n_bands != profile.n_channels:
            raise SchemaError(
                f"Spectrum band count ({self.n_bands}) does not match profile "
                f"'{profile.profile_id}' channel count ({profile.n_channels}).",
                details={"profile_id": profile.profile_id, "n_bands": self.n_bands},
            )
        if not np.allclose(self.wavelength_grid_nm, profile.wavelength_grid_nm, atol=tol_nm, rtol=0.0):
            raise SchemaError(
                f"Spectrum wavelength_grid_nm does not match profile '{profile.profile_id}'.",
                details={"profile_id": profile.profile_id},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "n_bands": self.n_bands,
            "n_shots": self.n_shots,
            "wavelength_grid_nm": list(self.wavelength_grid_nm),
            "channel_ids": list(self.channel_ids),
            "values": self.values.tolist(),
            "dark_reference": list(self.dark_reference) if self.dark_reference is not None else None,
            "white_reference": list(self.white_reference) if self.white_reference is not None else None,
            "integration_time_ms": self.integration_time_ms,
            "gain": self.gain,
            "sensor_temperature_c": self.sensor_temperature_c,
            "reference_age_s": self.reference_age_s,
            "replicate_id": self.replicate_id,
            "saturation_flags": list(self.saturation_flags) if self.saturation_flags is not None else None,
            "valid_mask": list(self.valid_mask) if self.valid_mask is not None else None,
        }


MissingReason = Literal["not_measured", "qc_failed"]
BulkDensitySource = Literal["core_sample", "penetrometer_estimated", "external_lab"]


@dataclass
class CovariateSet:
    """Physicochemical covariates normalized to canonical STDS units (Section 8 & 15, ADR-002, DEF-005).

    CRITICAL RULE (DEF-005): `bulk_density_g_cm3` is NEVER defaulted to 1.35.
    If not measured, it remains `None` and triggers `QualityFlag.MISSING_BULK_DENSITY`
    when carbon stock is requested.
    """

    ph: float | None = None
    moisture_volumetric_pct: float | None = None
    moisture_gravimetric_pct: float | None = None
    temperature_c: float | None = None
    ec_ds_m: float | None = None
    bulk_density_g_cm3: float | None = None
    bulk_density_uncertainty_g_cm3: float | None = None
    bulk_density_source: BulkDensitySource | None = None
    coarse_fragments_fraction: float | None = None
    eis: EISMeasurement | None = None
    missing_reasons: dict[str, MissingReason] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        raw_values: dict[str, dict[str, Any] | float | None],
        *,
        eis: EISMeasurement | None = None,
        missing_reasons: dict[str, MissingReason] | None = None,
        bulk_density_source: BulkDensitySource | None = None,
        bulk_density_uncertainty_g_cm3: float | None = None,
    ) -> CovariateSet:
        """Construct a canonical CovariateSet from explicit `{name: {"value": v, "unit": u}}` entries.

        Bare numbers without a `"unit"` key raise `UnitError` (ADR-002).
        """
        field_to_quantity = {
            "ph": "ph",
            "moisture_volumetric": "moisture_volumetric",
            "moisture_gravimetric": "moisture_gravimetric",
            "temperature": "temperature",
            "ec": "ec",
            "bulk_density": "bulk_density",
            "coarse_fragments": "coarse_fragments",
        }
        canonical_values: dict[str, float | None] = {k: None for k in field_to_quantity}
        reasons: dict[str, MissingReason] = dict(missing_reasons or {})

        for key, item in raw_values.items():
            if key not in field_to_quantity:
                raise SchemaError(
                    f"Unknown covariate key '{key}'.",
                    details={"key": key, "supported_keys": list(field_to_quantity.keys())},
                )
            if item is None:
                canonical_values[key] = None
                reasons.setdefault(key, "not_measured")
                continue
            if not isinstance(item, dict) or "value" not in item or "unit" not in item:
                raise UnitError(
                    f"Covariate '{key}' must be specified as a dict with explicit 'value' and 'unit'.",
                    details={"key": key, "received": item},
                )
            if item["value"] is None:
                canonical_values[key] = None
                reasons.setdefault(key, item.get("missing_reason", "not_measured"))
                continue
            canonical_values[key] = UnitRegistry.to_canonical(
                field_to_quantity[key],
                item["value"],
                item["unit"],
            )

        return cls(
            ph=canonical_values["ph"],
            moisture_volumetric_pct=canonical_values["moisture_volumetric"],
            moisture_gravimetric_pct=canonical_values["moisture_gravimetric"],
            temperature_c=canonical_values["temperature"],
            ec_ds_m=canonical_values["ec"],
            bulk_density_g_cm3=canonical_values["bulk_density"],
            bulk_density_uncertainty_g_cm3=bulk_density_uncertainty_g_cm3,
            bulk_density_source=bulk_density_source,
            coarse_fragments_fraction=canonical_values["coarse_fragments"],
            eis=eis,
            missing_reasons=reasons,
        )

    def __post_init__(self) -> None:
        checks = [
            ("ph", self.ph),
            ("moisture_volumetric", self.moisture_volumetric_pct),
            ("moisture_gravimetric", self.moisture_gravimetric_pct),
            ("temperature", self.temperature_c),
            ("ec", self.ec_ds_m),
            ("bulk_density", self.bulk_density_g_cm3),
            ("coarse_fragments", self.coarse_fragments_fraction),
        ]
        for q_type, val in checks:
            if val is not None:
                UnitRegistry.validate_physical_range(q_type, float(val))
            else:
                self.missing_reasons.setdefault(q_type, "not_measured")

        for k, reason in self.missing_reasons.items():
            if reason not in ("not_measured", "qc_failed"):
                raise SchemaError(
                    f"Invalid missing_reason '{reason}' for covariate '{k}'.",
                    details={"covariate": k, "reason": reason},
                )

    @property
    def has_bulk_density(self) -> bool:
        return self.bulk_density_g_cm3 is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ph": self.ph,
            "moisture_volumetric_pct": self.moisture_volumetric_pct,
            "moisture_gravimetric_pct": self.moisture_gravimetric_pct,
            "temperature_c": self.temperature_c,
            "ec_ds_m": self.ec_ds_m,
            "bulk_density_g_cm3": self.bulk_density_g_cm3,
            "bulk_density_uncertainty_g_cm3": self.bulk_density_uncertainty_g_cm3,
            "bulk_density_source": self.bulk_density_source,
            "coarse_fragments_fraction": self.coarse_fragments_fraction,
            "eis": self.eis.to_dict() if self.eis is not None else None,
            "missing_reasons": dict(self.missing_reasons),
            "canonical_units": {
                "ph": "dimensionless",
                "moisture_volumetric_pct": "pct",
                "moisture_gravimetric_pct": "pct",
                "temperature_c": "degC",
                "ec_ds_m": "dS/m",
                "bulk_density_g_cm3": "g/cm3",
                "coarse_fragments_fraction": "fraction",
            },
        }


@dataclass
class SoilReference:
    """Laboratory reference ground-truth for SOC, SIC, and TC (Section 8 & 22, ADR-004).

    Enforces dry-matter basis, method traceability, and physical closure checking:
    TC = SOC + SIC.
    """

    soc_pct: float | None = None
    sic_pct: float | None = None
    tc_pct: float | None = None
    dry_matter_basis: bool = True
    soc_method: str | None = "dry_combustion_acid_pretreated_iso10694"
    sic_method: str | None = "calcimeter_iso10693"
    tc_method: str | None = "dry_combustion_dumas_iso10694"
    is_soc_derived_from_difference: bool = False
    lab_uncertainty_soc_pct: float | None = None
    lab_uncertainty_sic_pct: float | None = None
    lab_uncertainty_tc_pct: float | None = None
    soil_class_id: int | None = None
    soil_class_name: str | None = None
    taxonomy_version: str = "provisional_taxonomy_v0"

    @classmethod
    def from_units(
        cls,
        *,
        soc: float | None = None,
        sic: float | None = None,
        tc: float | None = None,
        unit: str = "pct",
        dry_matter_basis: bool = True,
        soc_method: str | None = "dry_combustion_acid_pretreated_iso10694",
        sic_method: str | None = "calcimeter_iso10693",
        tc_method: str | None = "dry_combustion_dumas_iso10694",
        derive_soc_from_difference: bool = False,
        soil_class_id: int | None = None,
        soil_class_name: str | None = None,
    ) -> SoilReference:
        soc_c = UnitRegistry.to_canonical("carbon_concentration", soc, unit) if soc is not None else None
        sic_c = UnitRegistry.to_canonical("carbon_concentration", sic, unit) if sic is not None else None
        tc_c = UnitRegistry.to_canonical("carbon_concentration", tc, unit) if tc is not None else None

        if derive_soc_from_difference:
            if tc_c is None or sic_c is None:
                raise SchemaError("Cannot derive SOC = TC - SIC when TC or SIC is None.")
            diff = tc_c - sic_c
            if diff < 0.0:
                raise SchemaError(
                    f"Negative derived SOC ({diff:.4f}%) from TC ({tc_c:.4f}%) - SIC ({sic_c:.4f}%). "
                    "Silent clipping to zero is forbidden (Section 22, Page 24).",
                    details={"tc_pct": tc_c, "sic_pct": sic_c, "derived_soc_pct": diff},
                )
            soc_c = diff

        return cls(
            soc_pct=soc_c,
            sic_pct=sic_c,
            tc_pct=tc_c,
            dry_matter_basis=dry_matter_basis,
            soc_method=soc_method,
            sic_method=sic_method,
            tc_method=tc_method,
            is_soc_derived_from_difference=derive_soc_from_difference,
            soil_class_id=soil_class_id,
            soil_class_name=soil_class_name,
        )

    def __post_init__(self) -> None:
        if not self.dry_matter_basis:
            raise SchemaError(
                "SoilReference requires dry_matter_basis=True for all carbon fractions (Section 22)."
            )
        for name, val in (("soc_pct", self.soc_pct), ("sic_pct", self.sic_pct), ("tc_pct", self.tc_pct)):
            if val is not None:
                UnitRegistry.validate_physical_range("carbon_concentration", float(val))

        if self.tc_pct is None and self.soc_pct is not None and self.sic_pct is not None:
            object.__setattr__(self, "tc_pct", float(self.soc_pct + self.sic_pct))

    @property
    def closure_residual_pct(self) -> float | None:
        """Compute r_TC = TC - SOC - SIC if all three are present."""
        if self.tc_pct is not None and self.soc_pct is not None and self.sic_pct is not None:
            return float(self.tc_pct - self.soc_pct - self.sic_pct)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "soc_pct": self.soc_pct,
            "sic_pct": self.sic_pct,
            "tc_pct": self.tc_pct,
            "unit": "pct",
            "dry_matter_basis": self.dry_matter_basis,
            "soc_method": self.soc_method,
            "sic_method": self.sic_method,
            "tc_method": self.tc_method,
            "is_soc_derived_from_difference": self.is_soc_derived_from_difference,
            "closure_residual_pct": self.closure_residual_pct,
            "lab_uncertainty_soc_pct": self.lab_uncertainty_soc_pct,
            "lab_uncertainty_sic_pct": self.lab_uncertainty_sic_pct,
            "lab_uncertainty_tc_pct": self.lab_uncertainty_tc_pct,
            "soil_class_id": self.soil_class_id,
            "soil_class_name": self.soil_class_name,
            "taxonomy_version": self.taxonomy_version,
        }


DatasetSourceType = Literal["real_lab", "synthetic"]
MoistureScenarioType = Literal["dry", "wet", "field_moist"]


@dataclass
class Measurement:
    """Canonical single-sample measurement contract (Section 8, Page 10)."""

    sample_id: str
    timestamp_utc: str
    instrument_id: str
    instrument_profile_id: str
    calibration_id: str
    spectrum: Spectrum
    covariates: CovariateSet
    depth_interval: DepthInterval = field(default_factory=lambda: DepthInterval(0.0, 30.0))
    site_id: str = "unknown_site"
    session_id: str = "session_1"
    soil_texture_class: str | None = None
    sample_preparation: str = "air_dried_sieved_2mm"
    moisture_scenario: MoistureScenarioType = "dry"
    coordinates: SpatialCoordinates | None = None
    reference: SoilReference | None = None
    dataset_source: DatasetSourceType = "real_lab"
    quality: QualityMetadata = field(default_factory=QualityMetadata)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("sample_id", "timestamp_utc", "instrument_id", "instrument_profile_id", "calibration_id"):
            val = getattr(self, field_name)
            if not val or not isinstance(val, str) or not val.strip():
                raise SchemaError(f"Measurement.{field_name} cannot be empty.")

        if self.dataset_source not in ("real_lab", "synthetic"):
            raise SchemaError(
                f"Invalid dataset_source '{self.dataset_source}'. Must be 'real_lab' or 'synthetic'."
            )

        if self.instrument_profile_id in BUILTIN_PROFILES:
            profile = BUILTIN_PROFILES[self.instrument_profile_id]
            self.spectrum.validate_against_profile(profile)
            if not profile.has_carbonate_bands:
                self.quality.add_flag(
                    QualityFlag.CARBONATE_BAND_UNAVAILABLE,
                    f"Profile '{profile.profile_id}' does not cover SWIR/LIBS carbonate bands.",
                )

        if not self.covariates.has_bulk_density:
            self.quality.add_flag(
                QualityFlag.MISSING_BULK_DENSITY,
                "bulk_density_g_cm3 is missing; carbon stock cannot be computed without measured bulk density.",
            )
            self.quality.missing_covariates["bulk_density"] = self.covariates.missing_reasons.get(
                "bulk_density", "not_measured"
            )

        if self.dataset_source == "synthetic":
            self.quality.add_flag(
                QualityFlag.SYNTHETIC_DATA_SOURCE,
                "Synthetic measurement generated for pipeline testing; not valid as field evidence.",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "site_id": self.site_id,
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
            "instrument_id": self.instrument_id,
            "instrument_profile_id": self.instrument_profile_id,
            "calibration_id": self.calibration_id,
            "soil_texture_class": self.soil_texture_class,
            "sample_preparation": self.sample_preparation,
            "moisture_scenario": self.moisture_scenario,
            "dataset_source": self.dataset_source,
            "depth_interval": self.depth_interval.to_dict(),
            "coordinates": self.coordinates.to_dict() if self.coordinates is not None else None,
            "spectrum": self.spectrum.to_dict(),
            "covariates": self.covariates.to_dict(),
            "reference": self.reference.to_dict() if self.reference is not None else None,
            "quality": self.quality.to_dict(),
        }


@dataclass
class PredictionReport:
    """Canonical output report contract (Appendix B, Page 44).

    Contains all 9 mandatory logical sections with explicit units and basis metadata.
    """

    identity: dict[str, Any]
    status: QualityStatus
    carbon_metrics: dict[str, Any]
    carbon_stock: dict[str, Any]
    classification: dict[str, Any] | None
    quality: QualityMetadata
    xai: dict[str, Any] | None
    provenance: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": dict(self.identity),
            "status": self.status.value if isinstance(self.status, QualityStatus) else str(self.status),
            "carbon_metrics": dict(self.carbon_metrics),
            "carbon_stock": dict(self.carbon_stock),
            "classification": dict(self.classification) if self.classification is not None else None,
            "quality": self.quality.to_dict() if isinstance(self.quality, QualityMetadata) else dict(self.quality),
            "xai": dict(self.xai) if self.xai is not None else None,
            "provenance": dict(self.provenance),
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self, *, indent: int = 2) -> str:
        try:
            return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ReportSerializationError(
                f"Failed to serialize PredictionReport to JSON: {exc}",
                details={"identity": self.identity},
            ) from exc


@dataclass(frozen=True)
class TargetLabelSchema:
    """Explicit contract for target carbon labels and classification taxonomy (Section 31)."""

    targets: tuple[str, ...] = ("soc_pct", "sic_pct", "tc_pct")
    carbon_unit: str = "pct"
    dry_matter_basis: bool = True
    enforce_non_negative: bool = True
    enforce_tc_closure: bool = True
    closure_tolerance_pct: float = 0.05
    taxonomy_version: str = "provisional_taxonomy_v0"
    class_labels: tuple[str, ...] = (
        "low_organic_low_carbonate",
        "low_organic_calcareous",
        "humus_rich_non_calcareous",
        "humus_rich_calcareous",
    )

    def __post_init__(self) -> None:
        if not self.dry_matter_basis:
            raise SchemaError("TargetLabelSchema requires dry_matter_basis=True.")
        if self.closure_tolerance_pct < 0:
            raise SchemaError("TargetLabelSchema.closure_tolerance_pct must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "carbon_unit": self.carbon_unit,
            "dry_matter_basis": self.dry_matter_basis,
            "enforce_non_negative": self.enforce_non_negative,
            "enforce_tc_closure": self.enforce_tc_closure,
            "closure_tolerance_pct": self.closure_tolerance_pct,
            "taxonomy_version": self.taxonomy_version,
            "class_labels": list(self.class_labels),
        }


@dataclass(frozen=True)
class ArtifactMetadata:
    """Immutable provenance and compatibility metadata for a model/pipeline artifact (Section 7 & 26)."""

    artifact_id: str
    semantic_version: str
    schema_version: str
    instrument_profile_id: str
    calibration_id: str
    pipeline_version: str
    dataset_hash: str
    dependency_versions: dict[str, str]
    artifact_sha256: str = ""
    code_commit: str = "uncommitted"
    seed: int = 42
    created_utc: str = "2026-11-12T00:00:00Z"

    def __post_init__(self) -> None:
        for req in ("artifact_id", "semantic_version", "instrument_profile_id", "calibration_id", "pipeline_version"):
            if not getattr(self, req).strip():
                raise SchemaError(f"ArtifactMetadata.{req} cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "semantic_version": self.semantic_version,
            "schema_version": self.schema_version,
            "instrument_profile_id": self.instrument_profile_id,
            "calibration_id": self.calibration_id,
            "pipeline_version": self.pipeline_version,
            "dataset_hash": self.dataset_hash,
            "dependency_versions": dict(self.dependency_versions),
            "artifact_sha256": self.artifact_sha256,
            "code_commit": self.code_commit,
            "seed": self.seed,
            "created_utc": self.created_utc,
        }
