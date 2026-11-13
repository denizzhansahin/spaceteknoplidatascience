"""Explainability (XAI): sensitivity, Integrated Gradients, HSIC, interactions, and ranking (`st_ds.xai`)."""

from st_ds.xai.engine import (
    ComponentRanking,
    RankedFeature,
    SoilXAIEngine,
    XAIExplanationReport,
)
from st_ds.xai.hsic import (
    CAUSAL_DISCLAIMER,
    HSICResult,
    HSICStatus,
    compute_feature_hsic_ranking,
    compute_hsic,
)
from st_ds.xai.interactions import (
    PairwiseInteraction,
    compute_top_interactions,
    interaction_score,
)
from st_ds.xai.sensitivity import (
    SensitivityResult,
    integrated_gradients,
    local_sensitivity,
)

__all__ = [
    "SensitivityResult",
    "local_sensitivity",
    "integrated_gradients",
    "CAUSAL_DISCLAIMER",
    "HSICStatus",
    "HSICResult",
    "compute_hsic",
    "compute_feature_hsic_ranking",
    "PairwiseInteraction",
    "interaction_score",
    "compute_top_interactions",
    "RankedFeature",
    "ComponentRanking",
    "XAIExplanationReport",
    "SoilXAIEngine",
]
