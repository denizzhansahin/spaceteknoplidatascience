"""Multi-task physics-constrained losses for STSoilHybridNet (Section 19, ADR-004).

Implements:
- `gaussian_nll_loss`: heteroscedastic Gaussian negative log-likelihood
  L_Gaussian = 0.5 * [log(sigma^2) + (y - mu)^2 / sigma^2]
- `STSoilMultiTaskLoss`: scale-normalized sum of SOC NLL, SIC NLL, optional TC/Huber terms,
  soil-class cross-entropy, and physical closure penalty:
  lambda_close * (TC_hat - SOC_hat - SIC_hat)^2
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_ds.domain.errors import ModelInferenceError


@dataclass(frozen=True)
class MultiTaskLossConfig:
    """Weights and numerical stability parameters for multi-task soil carbon loss."""

    lambda_soc: float = 1.0
    lambda_sic: float = 1.0
    lambda_tc: float = 0.5
    lambda_class: float = 0.25
    lambda_close: float = 2.0
    lambda_huber: float = 0.25
    huber_delta: float = 1.0
    min_variance: float = 1e-4

    def __post_init__(self) -> None:
        for name in (
            "lambda_soc",
            "lambda_sic",
            "lambda_tc",
            "lambda_class",
            "lambda_close",
            "lambda_huber",
        ):
            if getattr(self, name) < 0.0:
                raise ModelInferenceError(f"MultiTaskLossConfig.{name} must be non-negative.")
        if self.min_variance <= 0.0:
            raise ModelInferenceError("MultiTaskLossConfig.min_variance must be strictly positive.")


def gaussian_nll_loss(
    target: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    scale: float = 1.0,
    min_variance: float = 1e-4,
) -> torch.Tensor:
    """Compute masked heteroscedastic Gaussian NLL normalized by target scale."""
    inv_scale = 1.0 / max(float(scale), 1e-3)
    target_n = target * inv_scale
    mean_n = mean * inv_scale
    var_n = torch.clamp(var * (inv_scale * inv_scale), min=min_variance)

    element_loss = 0.5 * (torch.log(var_n) + (target_n - mean_n).pow(2) / var_n)
    if mask is None:
        return element_loss.mean()

    m = mask.to(dtype=element_loss.dtype)
    denom = m.sum()
    if denom <= 0:
        return element_loss.sum() * 0.0
    return (element_loss * m).sum() / denom


class STSoilMultiTaskLoss(nn.Module):
    """Multi-task objective balancing SOC/SIC uncertainty, taxonomy CE, and TC=SOC+SIC closure."""

    def __init__(self, config: MultiTaskLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiTaskLossConfig()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        soc_target: torch.Tensor | None = None,
        soc_mask: torch.Tensor | None = None,
        sic_target: torch.Tensor | None = None,
        sic_mask: torch.Tensor | None = None,
        tc_target: torch.Tensor | None = None,
        tc_mask: torch.Tensor | None = None,
        class_target: torch.Tensor | None = None,
        class_mask: torch.Tensor | None = None,
        soc_scale: float = 1.0,
        sic_scale: float = 1.0,
        tc_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = outputs["soc_mean"].device
        total_loss = torch.zeros((), dtype=outputs["soc_mean"].dtype, device=device)
        breakdown: dict[str, float] = {}

        if soc_target is not None and self.config.lambda_soc > 0.0:
            l_soc = gaussian_nll_loss(
                soc_target,
                outputs["soc_mean"],
                outputs["soc_var"],
                mask=soc_mask,
                scale=soc_scale,
                min_variance=self.config.min_variance,
            )
            total_loss = total_loss + self.config.lambda_soc * l_soc
            breakdown["loss_soc_nll"] = float(l_soc.detach().cpu().item())

            if self.config.lambda_huber > 0.0:
                inv_s = 1.0 / max(float(soc_scale), 1e-3)
                h_el = F.huber_loss(
                    outputs["soc_mean"] * inv_s,
                    soc_target * inv_s,
                    reduction="none",
                    delta=self.config.huber_delta,
                )
                l_h_soc = (
                    (h_el * soc_mask.to(dtype=h_el.dtype)).sum() / torch.clamp(soc_mask.sum(), min=1.0)
                    if soc_mask is not None and soc_mask.sum() > 0
                    else h_el.mean()
                )
                total_loss = total_loss + self.config.lambda_huber * l_h_soc
                breakdown["loss_soc_huber"] = float(l_h_soc.detach().cpu().item())

        if sic_target is not None and "sic_mean" in outputs and self.config.lambda_sic > 0.0:
            l_sic = gaussian_nll_loss(
                sic_target,
                outputs["sic_mean"],
                outputs["sic_var"],
                mask=sic_mask,
                scale=sic_scale,
                min_variance=self.config.min_variance,
            )
            total_loss = total_loss + self.config.lambda_sic * l_sic
            breakdown["loss_sic_nll"] = float(l_sic.detach().cpu().item())

            if self.config.lambda_huber > 0.0:
                inv_s = 1.0 / max(float(sic_scale), 1e-3)
                h_el = F.huber_loss(
                    outputs["sic_mean"] * inv_s,
                    sic_target * inv_s,
                    reduction="none",
                    delta=self.config.huber_delta,
                )
                l_h_sic = (
                    (h_el * sic_mask.to(dtype=h_el.dtype)).sum() / torch.clamp(sic_mask.sum(), min=1.0)
                    if sic_mask is not None and sic_mask.sum() > 0
                    else h_el.mean()
                )
                total_loss = total_loss + self.config.lambda_huber * l_h_sic
                breakdown["loss_sic_huber"] = float(l_h_sic.detach().cpu().item())

        if tc_target is not None and "tc_aux_mean" in outputs and self.config.lambda_tc > 0.0:
            inv_tc = 1.0 / max(float(tc_scale), 1e-3)
            tc_err = F.huber_loss(
                outputs["tc_aux_mean"] * inv_tc,
                tc_target * inv_tc,
                reduction="none",
                delta=self.config.huber_delta,
            )
            l_tc = (
                (tc_err * tc_mask.to(dtype=tc_err.dtype)).sum() / torch.clamp(tc_mask.sum(), min=1.0)
                if tc_mask is not None and tc_mask.sum() > 0
                else tc_err.mean()
            )
            total_loss = total_loss + self.config.lambda_tc * l_tc
            breakdown["loss_tc"] = float(l_tc.detach().cpu().item())

        # Physical closure consistency penalty: (TC_aux - SOC - SIC)^2
        if "tc_aux_mean" in outputs and "sic_mean" in outputs and self.config.lambda_close > 0.0:
            residual = (outputs["tc_aux_mean"] - outputs["soc_mean"] - outputs["sic_mean"]) / max(
                float(tc_scale), 1e-3
            )
            l_close = residual.pow(2).mean()
            total_loss = total_loss + self.config.lambda_close * l_close
            breakdown["loss_closure"] = float(l_close.detach().cpu().item())

        if class_target is not None and "class_logits" in outputs and self.config.lambda_class > 0.0:
            if class_mask is not None:
                valid = class_mask.bool()
                if valid.any():
                    l_ce = F.cross_entropy(outputs["class_logits"][valid], class_target[valid].long())
                else:
                    l_ce = outputs["class_logits"].sum() * 0.0
            else:
                l_ce = F.cross_entropy(outputs["class_logits"], class_target.long())
            total_loss = total_loss + self.config.lambda_class * l_ce
            breakdown["loss_class_ce"] = float(l_ce.detach().cpu().item())

        breakdown["loss_total"] = float(total_loss.detach().cpu().item())
        return total_loss, breakdown
