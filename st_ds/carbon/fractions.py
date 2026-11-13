"""Carbon fractionation, mass-balance closure checks, and provisional soil classification.

Implements PDF Sections 22 & 34, ADR-002 (explicit units), ADR-004 (no silent
zero-clipping on negative derived fractions), and OD-005 (explicitly versioned
provisional taxonomy until domain thresholds are finalized).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from st_ds.domain.errors import SchemaError
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.domain.units import UnitRegistry

PROVISIONAL_TAXONOMY_VERSION = "provisional_taxonomy_v0"


@dataclass(frozen=True)
class CarbonFractionResult:
    """Validated and normalized soil carbon fractions on a dry-matter % basis."""

    soc_pct: float | None
    sic_pct: float | None
    tc_pct: float | None
    closure_residual_pct: float | None
    closure_tolerance_pct: float
    dry_matter_basis: bool
    soc_is_derived: bool
    sic_is_derived: bool
    tc_is_derived: bool
    quality_status: QualityStatus
    quality_flags: list[QualityFlag] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "soc_pct": self.soc_pct,
            "sic_pct": self.sic_pct,
            "tc_pct": self.tc_pct,
            "closure_residual_pct": self.closure_residual_pct,
            "closure_tolerance_pct": self.closure_tolerance_pct,
            "dry_matter_basis": self.dry_matter_basis,
            "soc_is_derived": self.soc_is_derived,
            "sic_is_derived": self.sic_is_derived,
            "tc_is_derived": self.tc_is_derived,
            "quality_status": self.quality_status.value,
            "quality_flags": [flag.value for flag in self.quality_flags],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ProvisionalSoilClassification:
    """Provisional soil organic/inorganic class label (OD-005)."""

    label: str
    soc_pct: float
    sic_pct: float
    tc_pct: float
    taxonomy_version: str = PROVISIONAL_TAXONOMY_VERSION
    is_provisional: bool = True
    criteria: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "soc_pct": self.soc_pct,
            "sic_pct": self.sic_pct,
            "tc_pct": self.tc_pct,
            "taxonomy_version": self.taxonomy_version,
            "is_provisional": self.is_provisional,
            "criteria": dict(self.criteria),
        }


def _convert_optional_carbon(
    value: float | None,
    unit: str,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    try:
        return UnitRegistry.to_canonical(
            "carbon_concentration",
            value,
            unit,
            validate_bounds=True,
        )
    except SchemaError as exc:
        raise SchemaError(
            f"Invalid value for carbon fraction '{field_name}': {exc.message}",
            details={"field": field_name, **exc.details},
        ) from exc


def partition_carbon(
    *,
    soc: float | None = None,
    sic: float | None = None,
    tc: float | None = None,
    unit: str = "pct",
    soc_unit: str | None = None,
    sic_unit: str | None = None,
    tc_unit: str | None = None,
    dry_matter_basis: bool = True,
    derive_soc_from_difference: bool = False,
    derive_sic_from_difference: bool = False,
    derive_tc_from_sum: bool = False,
    closure_tolerance_pct: float = 0.2,
) -> CarbonFractionResult:
    """Normalize SOC/SIC/TC to canonical dry-matter `%` and verify TC = SOC + SIC.

    Raises:
        SchemaError: If ``dry_matter_basis`` is False, all fractions are None,
            derivation options conflict or lack inputs, or a derived difference
            (``TC - SIC`` or ``TC - SOC``) is negative (no silent zero-clipping).
        UnitError: If any unit string is missing or unsupported.
    """
    if not isinstance(dry_matter_basis, bool) or not dry_matter_basis:
        raise SchemaError(
            "Carbon fractions must be reported on a dry-matter basis (dry_matter_basis=True).",
            details={"dry_matter_basis": dry_matter_basis},
        )

    if (
        isinstance(closure_tolerance_pct, bool)
        or not isinstance(closure_tolerance_pct, (int, float))
        or not math.isfinite(float(closure_tolerance_pct))
        or float(closure_tolerance_pct) < 0.0
    ):
        raise SchemaError(
            "closure_tolerance_pct must be a finite non-negative number.",
            details={"closure_tolerance_pct": closure_tolerance_pct},
        )
    tol_pct = float(closure_tolerance_pct)

    if soc is None and sic is None and tc is None:
        raise SchemaError(
            "At least one carbon fraction (soc, sic, or tc) must be provided.",
            details={"soc": soc, "sic": sic, "tc": tc},
        )

    if derive_soc_from_difference and derive_sic_from_difference:
        raise SchemaError(
            "Cannot derive both SOC and SIC by difference simultaneously.",
        )

    soc_pct = _convert_optional_carbon(soc, soc_unit or unit, "soc")
    sic_pct = _convert_optional_carbon(sic, sic_unit or unit, "sic")
    tc_pct = _convert_optional_carbon(tc, tc_unit or unit, "tc")

    soc_is_derived = False
    sic_is_derived = False
    tc_is_derived = False
    notes: list[str] = []

    if derive_soc_from_difference:
        if soc_pct is not None:
            raise SchemaError(
                "derive_soc_from_difference=True cannot be used when soc is already supplied.",
                details={"soc_pct": soc_pct},
            )
        if tc_pct is None or sic_pct is None:
            raise SchemaError(
                "Deriving SOC from difference requires both tc and sic.",
                details={"tc_pct": tc_pct, "sic_pct": sic_pct},
            )
        diff = tc_pct - sic_pct
        if diff < 0.0:
            raise SchemaError(
                "Negative derived SOC (TC - SIC < 0) violates physical mass balance; "
                "silent zero-clipping is forbidden (ADR-004).",
                details={"tc_pct": tc_pct, "sic_pct": sic_pct, "tc_minus_sic_pct": diff},
            )
        UnitRegistry.validate_physical_range("carbon_concentration", diff)
        soc_pct = float(diff)
        soc_is_derived = True
        notes.append("SOC derived as TC - SIC.")

    if derive_sic_from_difference:
        if sic_pct is not None:
            raise SchemaError(
                "derive_sic_from_difference=True cannot be used when sic is already supplied.",
                details={"sic_pct": sic_pct},
            )
        if tc_pct is None or soc_pct is None:
            raise SchemaError(
                "Deriving SIC from difference requires both tc and soc.",
                details={"tc_pct": tc_pct, "soc_pct": soc_pct},
            )
        diff = tc_pct - soc_pct
        if diff < 0.0:
            raise SchemaError(
                "Negative derived SIC (TC - SOC < 0) violates physical mass balance; "
                "silent zero-clipping is forbidden (ADR-004).",
                details={"tc_pct": tc_pct, "soc_pct": soc_pct, "tc_minus_soc_pct": diff},
            )
        UnitRegistry.validate_physical_range("carbon_concentration", diff)
        sic_pct = float(diff)
        sic_is_derived = True
        notes.append("SIC derived as TC - SOC.")

    if derive_tc_from_sum:
        if tc_pct is not None:
            raise SchemaError(
                "derive_tc_from_sum=True cannot be used when tc is already supplied.",
                details={"tc_pct": tc_pct},
            )
        if soc_pct is None or sic_pct is None:
            raise SchemaError(
                "Deriving TC from sum requires both soc and sic.",
                details={"soc_pct": soc_pct, "sic_pct": sic_pct},
            )
        tc_sum = float(soc_pct + sic_pct)
        UnitRegistry.validate_physical_range("carbon_concentration", tc_sum)
        tc_pct = tc_sum
        tc_is_derived = True
        notes.append("TC derived as SOC + SIC.")

    flags: list[QualityFlag] = []
    status = QualityStatus.OK
    closure_residual_pct: float | None = None

    if tc_pct is not None and soc_pct is not None and sic_pct is not None:
        closure_residual_pct = float(tc_pct - (soc_pct + sic_pct))
        if abs(closure_residual_pct) > tol_pct:
            flags.append(QualityFlag.CLOSURE_RESIDUAL_HIGH)
            status = QualityStatus.WARNING
            notes.append(
                f"Closure residual r_TC = {closure_residual_pct:.4f}% exceeds tolerance {tol_pct:.4f}%."
            )

    return CarbonFractionResult(
        soc_pct=soc_pct,
        sic_pct=sic_pct,
        tc_pct=tc_pct,
        closure_residual_pct=closure_residual_pct,
        closure_tolerance_pct=tol_pct,
        dry_matter_basis=dry_matter_basis,
        soc_is_derived=soc_is_derived,
        sic_is_derived=sic_is_derived,
        tc_is_derived=tc_is_derived,
        quality_status=status,
        quality_flags=flags,
        notes=notes,
    )


def classify_soil_provisional(
    soc: float,
    sic: float = 0.0,
    *,
    unit: str = "pct",
) -> ProvisionalSoilClassification:
    """Assign an explicitly versioned provisional 4-class carbon regime label (OD-005).

    Classes:
    1. ``organic_rich``: ``SOC >= 12.0 %``
    2. ``calcareous_mineral``: ``SOC < 2.0 %`` and ``SIC >= 2.0 %``
    3. ``humic_mineral``: ``2.0 <= SOC < 12.0 %``
    4. ``low_carbon_mineral``: ``SOC < 2.0 %`` and ``SIC < 2.0 %``
    """
    soc_pct = UnitRegistry.to_canonical("carbon_concentration", soc, unit, validate_bounds=True)
    sic_pct = UnitRegistry.to_canonical("carbon_concentration", sic, unit, validate_bounds=True)
    tc_pct = float(soc_pct + sic_pct)

    if soc_pct >= 12.0:
        label = "organic_rich"
        rule = "soc_pct >= 12.0"
    elif soc_pct >= 2.0:
        label = "humic_mineral"
        rule = "2.0 <= soc_pct < 12.0"
    elif sic_pct >= 2.0:
        label = "calcareous_mineral"
        rule = "soc_pct < 2.0 and sic_pct >= 2.0"
    else:
        label = "low_carbon_mineral"
        rule = "soc_pct < 2.0 and sic_pct < 2.0"

    return ProvisionalSoilClassification(
        label=label,
        soc_pct=soc_pct,
        sic_pct=sic_pct,
        tc_pct=tc_pct,
        taxonomy_version=PROVISIONAL_TAXONOMY_VERSION,
        is_provisional=True,
        criteria={
            "matched_rule": rule,
            "soc_organic_threshold_pct": 12.0,
            "soc_humic_threshold_pct": 2.0,
            "sic_calcareous_threshold_pct": 2.0,
        },
    )
