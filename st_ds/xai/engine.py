"""SoilXAIEngine and structured XAI explanation reports (Section 21 & 34, Appendix B, ADR-006).

Strictly separates the four scientific explainability concepts:
1. Local sensitivity & Integrated Gradients (per-sample perturbation behavior)
2. Dataset/batch dependence via centered RBF HSIC (fails closed with `insufficient_sample_count` if B < 4)
3. True second-order variable interactions (mixed partial derivative / 2D perturbation)
4. Grouped component rankings (spectral channels/bands vs covariates ranked in separate unit contexts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from st_ds.domain.errors import SchemaError
from st_ds.domain.protocols import RegressorProtocol
from st_ds.features.registry import FeatureDescriptor, FeatureProfileSpec
from st_ds.models.hybrid_net import STSoilHybridNet
from st_ds.xai.hsic import (
    CAUSAL_DISCLAIMER,
    HSICResult,
    compute_feature_hsic_ranking,
)
from st_ds.xai.interactions import (
    PairwiseInteraction,
    compute_top_interactions,
)
from st_ds.xai.sensitivity import (
    SensitivityResult,
    integrated_gradients,
    local_sensitivity,
)


@dataclass(frozen=True)
class RankedFeature:
    """Single ranked feature entry scoped to its physical domain and target variable."""

    name: str
    group: str
    unit: str
    score: float
    rank: int
    target_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "unit": self.unit,
            "score": float(self.score),
            "rank": int(self.rank),
            "target_name": self.target_name,
        }


@dataclass(frozen=True)
class ComponentRanking:
    """Target-specific, physical-group-separated importance rankings (DEF-004d, ADR-006)."""

    target_name: str
    spectral_ranking: tuple[RankedFeature, ...]
    covariate_ranking: tuple[RankedFeature, ...]
    derived_eis_ranking: tuple[RankedFeature, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "spectral_ranking": [r.to_dict() for r in self.spectral_ranking],
            "covariate_ranking": [r.to_dict() for r in self.covariate_ranking],
            "derived_eis_ranking": [r.to_dict() for r in self.derived_eis_ranking],
        }


@dataclass(frozen=True)
class XAIExplanationReport:
    """Appendix-B-aligned XAI report binding explanations to model/artifact provenance."""

    target_name: str
    local_sensitivity: SensitivityResult
    integrated_gradients: SensitivityResult
    dataset_dependence_hsic: HSICResult
    feature_interactions: tuple[PairwiseInteraction, ...]
    component_ranking: ComponentRanking
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "local_sensitivity": self.local_sensitivity.to_dict(),
            "integrated_gradients": self.integrated_gradients.to_dict(),
            "dataset_dependence_hsic": self.dataset_dependence_hsic.to_dict(),
            "feature_interactions": [i.to_dict() for i in self.feature_interactions],
            "component_ranking": self.component_ranking.to_dict(),
            "provenance": dict(self.provenance),
        }


def _rank_subset(
    descriptors: Sequence[FeatureDescriptor],
    scores: np.ndarray,
    indices: Sequence[int],
    *,
    target_name: str,
) -> tuple[RankedFeature, ...]:
    ordered = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
    return tuple(
        RankedFeature(
            name=descriptors[idx].name,
            group=descriptors[idx].group,
            unit=descriptors[idx].unit,
            score=float(scores[idx]),
            rank=pos + 1,
            target_name=target_name,
        )
        for pos, idx in enumerate(ordered)
    )


class SoilXAIEngine:
    """Unified XAI engine separating sensitivity, IG, HSIC, interactions, and grouped ranking."""

    def __init__(
        self,
        *,
        ig_steps: int = 32,
        max_interaction_pairs: int = 6,
        pipeline_version: str = "xai_v1",
    ) -> None:
        self.ig_steps = int(ig_steps)
        self.max_interaction_pairs = int(max_interaction_pairs)
        self.pipeline_version = str(pipeline_version)

    def explain(
        self,
        model: RegressorProtocol | Callable[[np.ndarray], np.ndarray] | STSoilHybridNet,
        x_sample: np.ndarray | Sequence[float],
        *,
        feature_profile: FeatureProfileSpec | None = None,
        feature_names: Sequence[str] | None = None,
        feature_scales: np.ndarray | Sequence[float] | None = None,
        baseline: np.ndarray | Sequence[float] | None = None,
        reference_X: np.ndarray | None = None,
        reference_y: np.ndarray | Sequence[float] | None = None,
        target_name: str = "soc_pct",
        target_index: int = 0,
        artifact_id: str | None = None,
    ) -> XAIExplanationReport:
        """Produce a structured, provenance-tagged XAI report for a single prediction."""
        x_arr = np.asarray(x_sample, dtype=float)
        if x_arr.ndim == 2 and x_arr.shape[0] == 1:
            x_arr = x_arr[0]
        if x_arr.ndim != 1 or x_arr.size == 0 or not np.all(np.isfinite(x_arr)):
            raise SchemaError("x_sample must be a finite 1D feature vector.")

        d = x_arr.size
        if feature_profile is not None:
            descriptors = feature_profile.descriptors
            if len(descriptors) != d:
                raise SchemaError(
                    f"FeatureProfileSpec '{feature_profile.profile_id}' has {len(descriptors)} features, "
                    f"but x_sample has {d}."
                )
            eff_names = tuple(d_item.name for d_item in descriptors)
        else:
            eff_names = tuple(feature_names) if feature_names is not None else tuple(f"feature_{i}" for i in range(d))
            if len(eff_names) != d:
                raise SchemaError("feature_names length must match x_sample dimension.")
            descriptors = tuple(
                FeatureDescriptor(name=n, group="spectral", unit="unknown")
                for n in eff_names
            )

        # Derive feature scales from reference_X if explicit scales were omitted
        if feature_scales is None and reference_X is not None and reference_X.shape[0] >= 2:
            ref_std = np.std(np.asarray(reference_X, dtype=float), axis=0, ddof=1)
            eff_scales: np.ndarray | Sequence[float] | None = np.where(ref_std > 1e-6, ref_std, 1.0)
        else:
            eff_scales = feature_scales

        sens_res = local_sensitivity(
            model,
            x_arr,
            feature_scales=eff_scales,
            feature_names=eff_names,
            target_name=target_name,
            target_index=target_index,
        )
        ig_res = integrated_gradients(
            model,
            x_arr,
            baseline=baseline,
            n_steps=self.ig_steps,
            feature_scales=eff_scales,
            feature_names=eff_names,
            target_name=target_name,
            target_index=target_index,
        )

        # Dataset/batch HSIC (fails closed with insufficient_sample_count when B < 4)
        if reference_X is not None and reference_y is not None:
            hsic_res = compute_feature_hsic_ranking(
                reference_X,
                reference_y,
                feature_names=eff_names,
            )
        else:
            hsic_res = compute_feature_hsic_ranking(
                x_arr[None, :],
                np.array([sens_res.sample_prediction], dtype=float),
                feature_names=eff_names,
            )

        # Partition indices strictly by physical domain (never compare % moisture vs % SOC/bands)
        spec_indices = [
            idx for idx, desc in enumerate(descriptors) if desc.group in ("spectral", "visnir_band", "libs_peak")
        ]
        cov_indices = [idx for idx, desc in enumerate(descriptors) if desc.group == "covariate"]
        derived_eis_indices = [
            idx for idx, desc in enumerate(descriptors) if desc.group in ("derived", "eis")
        ]

        scores = sens_res.normalized_sensitivity
        ranking = ComponentRanking(
            target_name=target_name,
            spectral_ranking=_rank_subset(descriptors, scores, spec_indices, target_name=target_name),
            covariate_ranking=_rank_subset(descriptors, scores, cov_indices, target_name=target_name),
            derived_eis_ranking=_rank_subset(descriptors, scores, derived_eis_indices, target_name=target_name),
        )

        # Evaluate cross-branch spectral x covariate interactions if both exist, else top pairs
        top_spec = [
            idx for idx in sorted(spec_indices, key=lambda i: float(scores[i]), reverse=True)[:3]
        ]
        top_cov = [
            idx for idx in sorted(cov_indices + derived_eis_indices, key=lambda i: float(scores[i]), reverse=True)[:3]
        ]
        if top_spec and top_cov:
            cand_pairs = [(si, ci) for si in top_spec for ci in top_cov]
        else:
            top_all = sorted(range(d), key=lambda i: float(scores[i]), reverse=True)[: min(4, d)]
            cand_pairs = [(top_all[a], top_all[b]) for a in range(len(top_all)) for b in range(a + 1, len(top_all))]

        interactions = compute_top_interactions(
            model,
            x_arr,
            candidate_pairs=cand_pairs,
            feature_scales=eff_scales,
            feature_names=eff_names,
            target_name=target_name,
            target_index=target_index,
            max_pairs=self.max_interaction_pairs,
        )

        model_name = getattr(model, "model_name", model.__class__.__name__)
        provenance: dict[str, Any] = {
            "model_name": str(model_name),
            "target_name": target_name,
            "pipeline_version": self.pipeline_version,
            "artifact_id": artifact_id or "untracked",
            "feature_profile_id": feature_profile.profile_id if feature_profile is not None else "ad_hoc",
            "feature_schema_hash": feature_profile.schema_hash if feature_profile is not None else None,
            "disclaimer": (
                "Explanations report local model sensitivity, path attributions, and dataset HSIC "
                "dependence within physical groups; they do not establish physical causality. "
                + CAUSAL_DISCLAIMER
            ),
        }

        return XAIExplanationReport(
            target_name=target_name,
            local_sensitivity=sens_res,
            integrated_gradients=ig_res,
            dataset_dependence_hsic=hsic_res,
            feature_interactions=interactions,
            component_ranking=ranking,
            provenance=provenance,
        )
