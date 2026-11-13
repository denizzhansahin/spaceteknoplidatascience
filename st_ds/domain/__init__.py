"""Domain package exports for SpaceTeknopoliDataScience (st_ds.domain)."""

from st_ds.domain.errors import (
    ArtifactCompatibilityError,
    CalibrationError,
    ModelInferenceError,
    OutOfDomainError,
    ReportSerializationError,
    SchemaError,
    SpectralQualityError,
    STDSError,
    UnitError,
)
from st_ds.domain.instrument import (
    BUILTIN_PROFILES,
    InstrumentProfile,
    SpectralModality,
    get_builtin_profile,
)
from st_ds.domain.protocols import (
    AnalyzerProtocol,
    CalibratorProtocol,
    RegressorProtocol,
    SignalQCProtocol,
)
from st_ds.domain.quality import QualityFlag, QualityMetadata, QualityStatus
from st_ds.domain.schemas import (
    ArtifactMetadata,
    CovariateSet,
    DepthInterval,
    EISMeasurement,
    Measurement,
    PredictionReport,
    SoilReference,
    SpatialCoordinates,
    SpectralSignalMode,
    Spectrum,
    TargetLabelSchema,
)
from st_ds.domain.units import QuantitySpec, UnitRegistry

__all__ = [
    "STDSError",
    "SchemaError",
    "UnitError",
    "CalibrationError",
    "SpectralQualityError",
    "OutOfDomainError",
    "ArtifactCompatibilityError",
    "ModelInferenceError",
    "ReportSerializationError",
    "QuantitySpec",
    "UnitRegistry",
    "QualityStatus",
    "QualityFlag",
    "QualityMetadata",
    "SpectralModality",
    "InstrumentProfile",
    "BUILTIN_PROFILES",
    "get_builtin_profile",
    "SpectralSignalMode",
    "DepthInterval",
    "SpatialCoordinates",
    "EISMeasurement",
    "Spectrum",
    "CovariateSet",
    "SoilReference",
    "Measurement",
    "PredictionReport",
    "TargetLabelSchema",
    "ArtifactMetadata",
    "RegressorProtocol",
    "CalibratorProtocol",
    "SignalQCProtocol",
    "AnalyzerProtocol",
]
