"""Independent measurement and dataset validation utilities (Section 31, Page 33)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from st_ds.domain.errors import SchemaError
from st_ds.domain.instrument import BUILTIN_PROFILES, InstrumentProfile
from st_ds.domain.schemas import Measurement, TargetLabelSchema
from st_ds.io.dataset import SoilDataset


@dataclass
class ValidationSummary:
    """Result summary returned by DataValidator."""

    is_valid: bool
    n_records: int
    n_errors: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """Validates Measurement and SoilDataset objects against domain, lineage, and target contracts."""

    @staticmethod
    def validate_measurement(
        measurement: Measurement,
        *,
        expected_profile: InstrumentProfile | None = None,
        require_reference: bool = False,
        require_bulk_density: bool = False,
        target_schema: TargetLabelSchema | None = None,
    ) -> list[str]:
        """Validate a single Measurement; raises SchemaError on failure and returns non-fatal warnings."""
        warnings: list[str] = []
        profile = expected_profile or BUILTIN_PROFILES.get(measurement.instrument_profile_id)
        if profile is not None:
            if measurement.instrument_profile_id != profile.profile_id:
                raise SchemaError(
                    f"Profile mismatch: measurement has '{measurement.instrument_profile_id}', "
                    f"expected '{profile.profile_id}'."
                )
            measurement.spectrum.validate_against_profile(profile)

        if require_bulk_density and not measurement.covariates.has_bulk_density:
            raise SchemaError(
                f"Measurement '{measurement.sample_id}' lacks bulk_density_g_cm3 required for stock calculation."
            )

        if require_reference:
            if measurement.reference is None:
                raise SchemaError(f"Measurement '{measurement.sample_id}' lacks SoilReference labels.")
            schema = target_schema or TargetLabelSchema()
            for target_name in schema.targets:
                if getattr(measurement.reference, target_name, None) is None:
                    raise SchemaError(
                        f"Measurement '{measurement.sample_id}' is missing required target '{target_name}'."
                    )
            res = measurement.reference.closure_residual_pct
            if res is not None and abs(res) > schema.closure_tolerance_pct:
                raise SchemaError(
                    f"Measurement '{measurement.sample_id}' closure residual {res:.4f}% exceeds tolerance "
                    f"{schema.closure_tolerance_pct:.4f}%."
                )

        if not measurement.covariates.has_bulk_density:
            warnings.append(f"{measurement.sample_id}: missing bulk_density_g_cm3 (stock disabled).")
        return warnings

    @classmethod
    def validate_dataset(
        cls,
        dataset: SoilDataset,
        *,
        for_training: bool = False,
        require_sic: bool = True,
        target_schema: TargetLabelSchema | None = None,
    ) -> ValidationSummary:
        """Validate an entire SoilDataset and return a structured ValidationSummary."""
        errors: list[dict[str, Any]] = []
        warnings: list[str] = []

        if for_training:
            try:
                dataset.validate_for_training(require_sic=require_sic)
            except SchemaError as exc:
                errors.append(exc.to_dict())

        for rec in dataset.records:
            try:
                rec_warnings = cls.validate_measurement(
                    rec,
                    require_reference=for_training,
                    target_schema=target_schema,
                )
                warnings.extend(rec_warnings)
            except SchemaError as exc:
                errors.append(exc.to_dict())

        return ValidationSummary(
            is_valid=len(errors) == 0,
            n_records=len(dataset.records),
            n_errors=len(errors),
            errors=errors,
            warnings=warnings,
        )
