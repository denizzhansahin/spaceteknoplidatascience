"""Public API semantic versioning and deprecation policy for st_ds.api (Sections 25 & 35)."""

from __future__ import annotations

PUBLIC_API_VERSION = "1.0.0"
SUPPORTED_REPORT_SCHEMA_VERSION = "1.0.0"

API_SEMVER_POLICY: dict[str, str] = {
    "major": "Incremented only for breaking changes in STSoilAnalyzer signatures or PredictionReport schema.",
    "minor": "Incremented for backward-compatible additions such as optional report fields or CLI subcommands.",
    "patch": "Incremented for bug fixes, numeric stability improvements, and documentation updates.",
}

LEGACY_WRAPPER_DEPRECATION_NOTICE = (
    "Constructing `STSoilAnalyzer(weights_path=...)` directly is deprecated and will be removed in v2.0.0. "
    "Use `STSoilAnalyzer.from_artifact(bundle_dir, device_profile=..., calibration_id=...)` instead. "
    "The transition wrapper refuses unverified raw `.pt`/`.pkl` checkpoints (DEF-006) and forbids "
    "implicit 24-channel or default 1.35 bulk_density fallbacks (ADR-002 / ADR-004)."
)
