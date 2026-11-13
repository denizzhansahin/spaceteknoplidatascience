"""Soil regression baselines, neural architectures, and physics-constrained losses (`st_ds.models`)."""

from st_ds.models.base import (
    BaseSoilRegressor,
    ModelPrediction,
    compute_closure_residual,
    enforce_non_negative_carbon,
    validate_xy_arrays,
)
from st_ds.models.baselines import (
    KernelRidgeRegressor,
    LocalPLSRRegressor,
    PLSRRegressor,
    RandomForestSoilRegressor,
)
from st_ds.models.hybrid_net import (
    HybridNetConfig,
    STSoilHybridNet,
    STSoilHybridNetRegressor,
)
from st_ds.models.loss import (
    MultiTaskLossConfig,
    STSoilMultiTaskLoss,
    gaussian_nll_loss,
)

__all__ = [
    "BaseSoilRegressor",
    "ModelPrediction",
    "validate_xy_arrays",
    "enforce_non_negative_carbon",
    "compute_closure_residual",
    "PLSRRegressor",
    "KernelRidgeRegressor",
    "RandomForestSoilRegressor",
    "LocalPLSRRegressor",
    "HybridNetConfig",
    "STSoilHybridNet",
    "STSoilHybridNetRegressor",
    "MultiTaskLossConfig",
    "STSoilMultiTaskLoss",
    "gaussian_nll_loss",
]
