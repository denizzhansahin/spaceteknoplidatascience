"""Core protocol contracts for loosely coupled STDS modules (Section 5, 16, 25, 31).

Enforces Separation of Concerns: modules communicate via typed schemas and protocols
without depending on concrete internal implementations.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from st_ds.domain.schemas import Measurement, PredictionReport, Spectrum


@runtime_checkable
class RegressorProtocol(Protocol):
    """Unified estimator contract for PLSR, KernelRidge, RandomForest, MBL, and STSoilHybridNet."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> RegressorProtocol: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def get_params(self) -> dict[str, Any]: ...

    def model_card(self) -> dict[str, Any]: ...


@runtime_checkable
class CalibratorProtocol(Protocol):
    """Contract for radiometric, ISS, and spectral preprocessing transformers."""

    def fit(self, *args: Any, **kwargs: Any) -> CalibratorProtocol: ...

    def transform(self, spectrum: Spectrum) -> Spectrum: ...


@runtime_checkable
class SignalQCProtocol(Protocol):
    """Contract for spectral and LIBS shot quality control evaluators."""

    def evaluate(self, spectrum: Spectrum) -> Any: ...


@runtime_checkable
class AnalyzerProtocol(Protocol):
    """Consumer-facing facade contract implemented by STSoilAnalyzer (Section 25)."""

    def predict(self, measurement: Measurement) -> PredictionReport: ...

    def predict_batch(self, measurements: Sequence[Measurement]) -> list[PredictionReport]: ...

    def explain(self, measurement: Measurement, result: PredictionReport | None = None) -> dict[str, Any]: ...

    def inspect_artifact(self) -> dict[str, Any]: ...
