"""Signal QC, baseline estimation, peak fitting, and DLIA exports (st_ds.signal)."""

from st_ds.signal.baseline import correct_baseline, estimate_arpls_baseline
from st_ds.signal.dlia import DLIAChannelSummary, DLIAReadout, summarize_dlia_outputs
from st_ds.signal.libs_qc import LIBSShotFilterResult, LIBSShotQCConfig, filter_libs_shots
from st_ds.signal.peaks import (
    DEFAULT_LIBS_CARBON_WINDOWS,
    PeakWindowSpec,
    PseudoVoigtFitResult,
    extract_libs_peak_features,
    fit_pseudovoigt_peaks,
    pseudovoigt_profile,
)
from st_ds.signal.qc import (
    SignalQCConfig,
    SignalQCReport,
    estimate_spectral_snr,
    evaluate_signal_qc,
)

__all__ = [
    "SignalQCConfig",
    "SignalQCReport",
    "estimate_spectral_snr",
    "evaluate_signal_qc",
    "LIBSShotQCConfig",
    "LIBSShotFilterResult",
    "filter_libs_shots",
    "DLIAReadout",
    "DLIAChannelSummary",
    "summarize_dlia_outputs",
    "estimate_arpls_baseline",
    "correct_baseline",
    "PeakWindowSpec",
    "DEFAULT_LIBS_CARBON_WINDOWS",
    "PseudoVoigtFitResult",
    "pseudovoigt_profile",
    "fit_pseudovoigt_peaks",
    "extract_libs_peak_features",
]
