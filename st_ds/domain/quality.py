"""Quality status and flag definitions for STDS (Section 5, 6, 8, Appendix B)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityStatus(str, Enum):
    """High-level validity and reporting status of a measurement or prediction."""

    OK = "ok"
    WARNING = "warning"
    INSUFFICIENT_DATA = "insufficient_data"
    OUT_OF_DOMAIN = "out_of_domain"
    NOT_VALID_FOR_REPORTING = "not_valid_for_reporting"
    REJECTED = "rejected"


class QualityFlag(str, Enum):
    """Granular quality flags attached to measurements, preprocessing, or predictions."""

    MISSING_BULK_DENSITY = "missing_bulk_density"
    MISSING_COVARIATE = "missing_covariate"
    COVARIATE_QC_FAILED = "covariate_qc_failed"
    SATURATED_BANDS = "saturated_bands"
    LOW_SNR = "low_snr"
    DARK_WHITE_INVALID = "dark_white_invalid"
    REFERENCE_AGED = "reference_aged"
    INTERPOLATED_BANDS = "interpolated_bands"
    CLOSURE_RESIDUAL_HIGH = "closure_residual_high"
    OUT_OF_CALIBRATION_DOMAIN = "out_of_calibration_domain"
    INSUFFICIENT_SAMPLE_COUNT = "insufficient_sample_count"
    SYNTHETIC_DATA_SOURCE = "synthetic_data_source"
    CARBONATE_BAND_UNAVAILABLE = "carbonate_band_unavailable"


@dataclass
class QualityMetadata:
    """Quality metadata accompanying a measurement or prediction report."""

    status: QualityStatus = QualityStatus.OK
    quality_score: float = 1.0
    flags: list[QualityFlag] = field(default_factory=list)
    rejected_bands: list[str] = field(default_factory=list)
    rejected_shots_count: int = 0
    missing_covariates: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_flag(self, flag: QualityFlag, note: str | None = None) -> None:
        if flag not in self.flags:
            self.flags.append(flag)
        if note and note not in self.notes:
            self.notes.append(note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "quality_score": float(self.quality_score),
            "flags": [f.value for f in self.flags],
            "rejected_bands": list(self.rejected_bands),
            "rejected_shots_count": int(self.rejected_shots_count),
            "missing_covariates": dict(self.missing_covariates),
            "notes": list(self.notes),
        }
