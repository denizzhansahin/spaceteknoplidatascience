"""SpaceTeknopoliDataScience (st_ds) core package.

Standalone scientific library for soil carbon analytics (SOC / SIC / TC),
radiometric calibration, chemometrics (SNV, wavelength-aware Savitzky-Golay,
EPO, GLSW), multi-sensor fusion, uncertainty quantification, explainability (XAI),
and carbon stock accounting.

Strictly free of hardware driver, web framework, and database dependencies (ADR-001).
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0.0"

from st_ds.api import PUBLIC_API_VERSION, BatchFailureRecord, BatchPredictionResult, STSoilAnalyzer
from st_ds.artifact import inspect_artifact, load_artifact, save_artifact, verify_artifact_integrity
from st_ds.config import LangCode, STDSRuntimeConfig, get_runtime_config, resolve_lang
from st_ds.reporting import generate_analysis_pdf_report, generate_prediction_pdf_report
from st_ds.visualization import (
    PlotArtifactRecord,
    TableArtifactRecord,
    VisualizationConfig,
    VisualizationReport,
    generate_full_visualization_suite,
)

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "PUBLIC_API_VERSION",
    "LangCode",
    "STDSRuntimeConfig",
    "get_runtime_config",
    "resolve_lang",
    "STSoilAnalyzer",
    "BatchPredictionResult",
    "BatchFailureRecord",
    "save_artifact",
    "load_artifact",
    "inspect_artifact",
    "verify_artifact_integrity",
    "PlotArtifactRecord",
    "TableArtifactRecord",
    "VisualizationConfig",
    "VisualizationReport",
    "generate_full_visualization_suite",
    "generate_analysis_pdf_report",
    "generate_prediction_pdf_report",
]
