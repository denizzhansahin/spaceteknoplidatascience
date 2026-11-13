"""Stable public consumer API for STDS."""

from st_ds.api.analyzer import (
    BatchFailureRecord,
    BatchPredictionResult,
    STSoilAnalyzer,
)
from st_ds.api.versioning import (
    API_SEMVER_POLICY,
    LEGACY_WRAPPER_DEPRECATION_NOTICE,
    PUBLIC_API_VERSION,
    SUPPORTED_REPORT_SCHEMA_VERSION,
)

__all__ = [
    "PUBLIC_API_VERSION",
    "SUPPORTED_REPORT_SCHEMA_VERSION",
    "API_SEMVER_POLICY",
    "LEGACY_WRAPPER_DEPRECATION_NOTICE",
    "BatchFailureRecord",
    "BatchPredictionResult",
    "STSoilAnalyzer",
]
