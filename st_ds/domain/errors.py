"""Structured error taxonomy for STDS (Section 27, Page 29).

All domain, calibration, chemometric, model, and serialization errors inherit from
STDSError and provide an explicit error_code, human-readable message, actionable
remediation hint, and structured diagnostic details.
"""

from __future__ import annotations

from typing import Any


class STDSError(Exception):
    """Base exception for all SpaceTeknopoliDataScience (st_ds) errors."""

    default_code: str = "STDS_ERROR"
    default_remediation: str = "Inspect input data and configuration against STDS contracts."

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        remediation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.default_code
        self.remediation = remediation or self.default_remediation
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the error."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "remediation": self.remediation,
            "details": self.details,
        }


class SchemaError(STDSError):
    """Raised when input/output structures violate the canonical data contract."""

    default_code = "SCHEMA_ERROR"
    default_remediation = (
        "Ensure all required fields, dimensions, non-NaN values, and physical bounds match the schema."
    )


class UnitError(STDSError):
    """Raised when a physical unit is missing, unknown, or incompatible (ADR-002)."""

    default_code = "UNIT_ERROR"
    default_remediation = (
        "Provide an explicit, supported physical unit registered in UnitRegistry."
    )


class CalibrationError(STDSError):
    """Raised when radiometric, wavelength, ISS, or EPO calibration fails or mismatches."""

    default_code = "CALIBRATION_ERROR"
    default_remediation = (
        "Check white/dark references, wavelength grid alignment, and EPO preprocessing space signature."
    )


class SpectralQualityError(STDSError):
    """Raised when spectral measurements fail quality control (saturation, low SNR, invalid grid)."""

    default_code = "SPECTRAL_QUALITY_ERROR"
    default_remediation = (
        "Re-acquire the spectrum with proper integration time, valid references, and non-saturated channels."
    )


class OutOfDomainError(STDSError):
    """Raised when a measurement lies outside the calibrated/trained domain."""

    default_code = "OUT_OF_DOMAIN_ERROR"
    default_remediation = (
        "Verify instrument profile and soil/moisture domain or recalibrate the model with representative samples."
    )


class ArtifactCompatibilityError(STDSError):
    """Raised when an artifact bundle is corrupted, untrusted, or incompatible with the runtime/profile."""

    default_code = "ARTIFACT_COMPATIBILITY_ERROR"
    default_remediation = (
        "Use a signed/verified artifact bundle with matching schema version, SHA-256 hash, and instrument profile."
    )


class ModelInferenceError(STDSError):
    """Raised when model forward pass, uncertainty estimation, or XAI computation fails."""

    default_code = "MODEL_INFERENCE_ERROR"
    default_remediation = (
        "Check feature vector alignment, model state, and numerical stability of inputs."
    )


class ReportSerializationError(STDSError):
    """Raised when PredictionReport serialization or export fails."""

    default_code = "REPORT_SERIALIZATION_ERROR"
    default_remediation = (
        "Ensure all report fields contain JSON-serializable primitives or canonical dataclasses."
    )
