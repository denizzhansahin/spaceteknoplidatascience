"""Named Feature Registry and unit-tracked feature profile contracts (Section 15, ADR-002).

Eliminates hardcoded positional indices (DEF-003): every feature in spectral,
Vis-NIR band, LIBS peak, EIS, covariate, derived, and missing-indicator groups
carries an explicit name, physical unit, and provenance descriptor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from st_ds.domain.errors import SchemaError
from st_ds.domain.instrument import InstrumentProfile

FeatureGroup = Literal[
    "spectral",
    "visnir_band",
    "libs_peak",
    "eis",
    "covariate",
    "derived",
    "missing_indicator",
]

COVARIATE_CANONICAL_UNITS: dict[str, str] = {
    "ph": "dimensionless",
    "moisture_volumetric_pct": "pct",
    "moisture_gravimetric_pct": "pct",
    "temperature_c": "degC",
    "ec_ds_m": "dS/m",
    "bulk_density_g_cm3": "g/cm3",
    "coarse_fragments_fraction": "fraction",
}


@dataclass(frozen=True)
class FeatureDescriptor:
    """Metadata descriptor for a single named feature column."""

    name: str
    group: FeatureGroup
    unit: str
    description: str = ""
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SchemaError("FeatureDescriptor.name cannot be empty.")
        if not self.unit or not self.unit.strip():
            raise SchemaError(f"FeatureDescriptor '{self.name}' must define an explicit unit.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "unit": self.unit,
            "description": self.description,
            "required": self.required,
        }


@dataclass(frozen=True)
class FeatureProfileSpec:
    """Ordered, immutable specification of all features produced for an instrument profile."""

    profile_id: str
    instrument_profile_id: str
    descriptors: tuple[FeatureDescriptor, ...]
    eis_frequencies_hz: tuple[float, ...] = ()
    eis_protocol_id: str | None = None
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id or not self.profile_id.strip():
            raise SchemaError("FeatureProfileSpec.profile_id cannot be empty.")
        if len(self.descriptors) == 0:
            raise SchemaError("FeatureProfileSpec.descriptors cannot be empty.")
        names = [d.name for d in self.descriptors]
        if len(set(names)) != len(names):
            raise SchemaError("FeatureProfileSpec feature names must be unique.")

        if not self.schema_hash:
            payload = json.dumps([d.to_dict() for d in self.descriptors], sort_keys=True)
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            object.__setattr__(self, "schema_hash", digest)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.descriptors)

    @property
    def spectral_feature_names(self) -> tuple[str, ...]:
        return tuple(
            d.name for d in self.descriptors if d.group in ("spectral", "visnir_band", "libs_peak")
        )

    @property
    def covariate_feature_names(self) -> tuple[str, ...]:
        return tuple(
            d.name
            for d in self.descriptors
            if d.group in ("covariate", "derived", "eis", "missing_indicator")
        )

    def index_of(self, feature_name: str) -> int:
        for idx, d in enumerate(self.descriptors):
            if d.name == feature_name:
                return idx
        raise SchemaError(
            f"Feature '{feature_name}' is not registered in profile '{self.profile_id}'.",
            details={"feature_name": feature_name, "profile_id": self.profile_id},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "instrument_profile_id": self.instrument_profile_id,
            "schema_hash": self.schema_hash,
            "n_features": len(self.descriptors),
            "eis_frequencies_hz": list(self.eis_frequencies_hz),
            "eis_protocol_id": self.eis_protocol_id,
            "descriptors": [d.to_dict() for d in self.descriptors],
        }


class FeatureRegistry:
    """Registry managing named FeatureProfileSpec definitions."""

    def __init__(self) -> None:
        self._profiles: dict[str, FeatureProfileSpec] = {}

    def register_profile(self, spec: FeatureProfileSpec, *, overwrite: bool = False) -> None:
        if not overwrite and spec.profile_id in self._profiles:
            raise SchemaError(
                f"FeatureProfileSpec '{spec.profile_id}' is already registered.",
                details={"profile_id": spec.profile_id},
            )
        self._profiles[spec.profile_id] = spec

    def get_profile(self, profile_id: str) -> FeatureProfileSpec:
        """Retrieve a registered FeatureProfileSpec by ID (Section 15 API)."""
        if profile_id not in self._profiles:
            raise SchemaError(
                f"Unknown feature profile '{profile_id}'.",
                details={"profile_id": profile_id, "available": sorted(self._profiles.keys())},
            )
        return self._profiles[profile_id]

    def list_profiles(self) -> list[str]:
        return sorted(self._profiles.keys())

    def build_profile_from_instrument(
        self,
        instrument_profile: InstrumentProfile,
        *,
        profile_id: str | None = None,
        covariate_names: Sequence[str] = ("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m"),
        visnir_band_names: Sequence[str] = (),
        derived_names: Sequence[str] = (),
        eis_frequencies_hz: Sequence[float] = (),
        eis_protocol_id: str | None = None,
        include_missing_indicators: bool = True,
        spectral_unit: str = "preprocessed_signal",
        register: bool = True,
    ) -> FeatureProfileSpec:
        """Construct a named FeatureProfileSpec dynamically from an InstrumentProfile."""
        descriptors: list[FeatureDescriptor] = []

        for ch_id, wl_nm in zip(instrument_profile.channel_ids, instrument_profile.wavelength_grid_nm):
            descriptors.append(
                FeatureDescriptor(
                    name=f"spec_{ch_id}",
                    group="spectral",
                    unit=spectral_unit,
                    description=f"Spectral channel {ch_id} centered at {wl_nm:.2f} nm",
                )
            )

        for b_name in visnir_band_names:
            descriptors.append(
                FeatureDescriptor(
                    name=b_name,
                    group="visnir_band",
                    unit=spectral_unit,
                    description=f"Aggregated Vis-NIR/SWIR chemical window '{b_name}'",
                )
            )

        for cov_name in covariate_names:
            if cov_name not in COVARIATE_CANONICAL_UNITS:
                raise SchemaError(
                    f"Unknown canonical covariate '{cov_name}'.",
                    details={"supported": list(COVARIATE_CANONICAL_UNITS.keys())},
                )
            descriptors.append(
                FeatureDescriptor(
                    name=cov_name,
                    group="covariate",
                    unit=COVARIATE_CANONICAL_UNITS[cov_name],
                    description=f"Canonical soil covariate '{cov_name}'",
                )
            )

        for d_name in derived_names:
            unit = "dS/m" if d_name == "ec_25c_ds_m" else "dimensionless"
            descriptors.append(
                FeatureDescriptor(
                    name=d_name,
                    group="derived",
                    unit=unit,
                    description=f"Derived physicochemical feature '{d_name}'",
                )
            )

        for freq in eis_frequencies_hz:
            f_tag = f"{float(freq):g}hz"
            descriptors.extend(
                [
                    FeatureDescriptor(name=f"eis_zreal_{f_tag}", group="eis", unit="ohm"),
                    FeatureDescriptor(name=f"eis_zimag_{f_tag}", group="eis", unit="ohm"),
                    FeatureDescriptor(name=f"eis_mag_{f_tag}", group="eis", unit="ohm"),
                    FeatureDescriptor(name=f"eis_phase_rad_{f_tag}", group="eis", unit="rad"),
                ]
            )

        if include_missing_indicators:
            for cov_name in covariate_names:
                descriptors.append(
                    FeatureDescriptor(
                        name=f"is_missing_{cov_name}",
                        group="missing_indicator",
                        unit="binary_flag",
                        description=f"Missingness indicator (1.0 if '{cov_name}' was missing or failed QC)",
                    )
                )

        spec = FeatureProfileSpec(
            profile_id=profile_id or f"feat_{instrument_profile.profile_id}",
            instrument_profile_id=instrument_profile.profile_id,
            descriptors=tuple(descriptors),
            eis_frequencies_hz=tuple(float(f) for f in eis_frequencies_hz),
            eis_protocol_id=eis_protocol_id,
        )
        if register:
            self.register_profile(spec, overwrite=True)
        return spec
