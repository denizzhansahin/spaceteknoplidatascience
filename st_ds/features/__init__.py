"""Feature registry, modality extractors, and sensor fusion exports (st_ds.features)."""

from st_ds.features.extractors import (
    VISNIR_CHEMICAL_WINDOWS_NM,
    available_visnir_band_names,
    extract_derived_covariates,
    extract_eis_features,
    extract_visnir_band_features,
)
from st_ds.features.fusion import (
    CovariateValidationResult,
    FeatureFusionConfig,
    FeatureFusionPipeline,
    FusedFeatures,
    align_spectral_and_covariates,
    validate_covariates,
)
from st_ds.features.registry import (
    COVARIATE_CANONICAL_UNITS,
    FeatureDescriptor,
    FeatureGroup,
    FeatureProfileSpec,
    FeatureRegistry,
)

__all__ = [
    "FeatureGroup",
    "COVARIATE_CANONICAL_UNITS",
    "FeatureDescriptor",
    "FeatureProfileSpec",
    "FeatureRegistry",
    "VISNIR_CHEMICAL_WINDOWS_NM",
    "available_visnir_band_names",
    "extract_visnir_band_features",
    "extract_eis_features",
    "extract_derived_covariates",
    "CovariateValidationResult",
    "validate_covariates",
    "align_spectral_and_covariates",
    "FeatureFusionConfig",
    "FusedFeatures",
    "FeatureFusionPipeline",
]
