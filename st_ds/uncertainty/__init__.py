"""Uncertainty calibration, Conformal Prediction, ensemble spread, and OOD checks (`st_ds.uncertainty`)."""

from st_ds.uncertainty.conformal import (
    CalibrationCurvePoint,
    ConformalConfig,
    ConformalMethod,
    ConformalUncertaintyCalibrator,
    CoverageReport,
    UncertaintyInterval,
    compute_calibration_curve,
    evaluate_subgroup_coverage,
    fit_uncertainty_calibrator,
    predict_interval,
)
from st_ds.uncertainty.ensemble import (
    EnsembleSpreadResult,
    GroupedBootstrapEnsemble,
    estimate_ensemble_spread,
    predict_ensemble_distribution,
)
from st_ds.uncertainty.ood import (
    DomainEnvelope,
    OODBatchResult,
    OODConfig,
    OODSampleAssessment,
    OutOfDomainDetector,
    detect_out_of_domain,
)

__all__ = [
    "ConformalMethod",
    "ConformalConfig",
    "UncertaintyInterval",
    "CalibrationCurvePoint",
    "CoverageReport",
    "ConformalUncertaintyCalibrator",
    "fit_uncertainty_calibrator",
    "predict_interval",
    "compute_calibration_curve",
    "evaluate_subgroup_coverage",
    "EnsembleSpreadResult",
    "estimate_ensemble_spread",
    "predict_ensemble_distribution",
    "GroupedBootstrapEnsemble",
    "OODConfig",
    "DomainEnvelope",
    "OODSampleAssessment",
    "OODBatchResult",
    "OutOfDomainDetector",
    "detect_out_of_domain",
]
