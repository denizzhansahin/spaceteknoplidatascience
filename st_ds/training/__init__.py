"""Training package exports (`st_ds.training`)."""

from st_ds.training.metrics import (
    RegressionMetrics,
    check_closure_diagnostics,
    compute_regression_metrics,
    evaluate_by_subgroup,
)
from st_ds.training.splits import GroupedSplitBuilder, GroupKeyType, SplitFold
from st_ds.training.trainer import (
    BenchmarkReport,
    ModelBenchmarkEntry,
    STSoilTrainer,
    TrainerConfig,
)

__all__ = [
    "GroupKeyType",
    "SplitFold",
    "GroupedSplitBuilder",
    "RegressionMetrics",
    "compute_regression_metrics",
    "evaluate_by_subgroup",
    "check_closure_diagnostics",
    "TrainerConfig",
    "ModelBenchmarkEntry",
    "BenchmarkReport",
    "STSoilTrainer",
]
