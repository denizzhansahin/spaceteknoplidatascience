"""Two-branch physically constrained STSoilHybridNet architecture and RegressorProtocol wrapper (Section 18, ADR-004).

Supports any instrument spectral dimension (e.g., 5-LED, 24-ch, FT-NIR) and
heterogeneous covariates/missingness indicators via separate encoders, residual
fusion blocks, softplus positive parameterization for carbon means/variances,
and multi-task training with closure penalty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from st_ds.domain.errors import ModelInferenceError
from st_ds.models.base import (
    BaseSoilRegressor,
    ModelPrediction,
    compute_closure_residual,
    validate_xy_arrays,
)
from st_ds.models.loss import MultiTaskLossConfig, STSoilMultiTaskLoss


@dataclass(frozen=True)
class HybridNetConfig:
    """Architecture configuration for STSoilHybridNet."""

    n_spectral_features: int
    n_covariate_features: int = 0
    spectral_hidden_dim: int = 64
    covariate_hidden_dim: int = 32
    fusion_hidden_dim: int = 64
    dropout: float = 0.1
    n_classes: int = 4
    enable_tc_head: bool = True
    enable_moisture_head: bool = False
    min_variance: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_spectral_features < 1:
            raise ModelInferenceError("HybridNetConfig.n_spectral_features must be >= 1.")
        if self.n_covariate_features < 0:
            raise ModelInferenceError("HybridNetConfig.n_covariate_features must be >= 0.")
        if not (0.0 <= self.dropout < 1.0):
            raise ModelInferenceError("HybridNetConfig.dropout must be in [0.0, 1.0).")
        if self.n_classes < 2:
            raise ModelInferenceError("HybridNetConfig.n_classes must be >= 2.")
        if self.min_variance <= 0.0:
            raise ModelInferenceError("HybridNetConfig.min_variance must be strictly positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ResidualDenseBlock(nn.Module):
    """Residual MLP block with LayerNorm, SiLU activation, and Dropout."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class STSoilHybridNet(nn.Module):
    """Two-branch soil spectral + covariate neural architecture with non-negative heads."""

    def __init__(self, config: HybridNetConfig) -> None:
        super().__init__()
        self.config = config

        # Branch 1: Spectral encoder
        self.spectral_encoder = nn.Sequential(
            nn.LayerNorm(config.n_spectral_features),
            nn.Linear(config.n_spectral_features, config.spectral_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            _ResidualDenseBlock(config.spectral_hidden_dim, config.dropout),
        )

        # Branch 2: Covariate encoder (optional when n_covariate_features == 0)
        if config.n_covariate_features > 0:
            self.covariate_encoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(config.n_covariate_features),
                nn.Linear(config.n_covariate_features, config.covariate_hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                _ResidualDenseBlock(config.covariate_hidden_dim, config.dropout),
            )
            fused_in_dim = config.spectral_hidden_dim + config.covariate_hidden_dim
        else:
            self.covariate_encoder = None
            fused_in_dim = config.spectral_hidden_dim

        self.fusion_backbone = nn.Sequential(
            nn.Linear(fused_in_dim, config.fusion_hidden_dim),
            nn.SiLU(),
            _ResidualDenseBlock(config.fusion_hidden_dim, config.dropout),
            nn.LayerNorm(config.fusion_hidden_dim),
        )

        h = config.fusion_hidden_dim
        self.soc_mean_head = nn.Linear(h, 1)
        self.soc_var_head = nn.Linear(h, 1)
        self.sic_mean_head = nn.Linear(h, 1)
        self.sic_var_head = nn.Linear(h, 1)
        self.class_head = nn.Linear(h, config.n_classes)

        self.tc_aux_head: nn.Linear | None = nn.Linear(h, 1) if config.enable_tc_head else None
        self.moisture_mean_head: nn.Linear | None = nn.Linear(h, 1) if config.enable_moisture_head else None
        self.moisture_var_head: nn.Linear | None = nn.Linear(h, 1) if config.enable_moisture_head else None

    def encode(
        self,
        x_spectral: torch.Tensor,
        x_covariate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint fused latent representation."""
        h_spec = self.spectral_encoder(x_spectral)
        if self.covariate_encoder is not None:
            if x_covariate is None or x_covariate.shape[1] != self.config.n_covariate_features:
                raise ModelInferenceError(
                    f"Expected x_covariate with {self.config.n_covariate_features} features.",
                    details={"n_covariate_features": self.config.n_covariate_features},
                )
            h_cov = self.covariate_encoder(x_covariate)
            h_cat = torch.cat([h_spec, h_cov], dim=-1)
        else:
            h_cat = h_spec
        return self.fusion_backbone(h_cat)

    def forward(
        self,
        x_spectral: torch.Tensor,
        x_covariate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(x_spectral, x_covariate)

        soc_mean = F.softplus(self.soc_mean_head(latent)).squeeze(-1)
        sic_mean = F.softplus(self.sic_mean_head(latent)).squeeze(-1)
        soc_var = F.softplus(self.soc_var_head(latent)).squeeze(-1) + self.config.min_variance
        sic_var = F.softplus(self.sic_var_head(latent)).squeeze(-1) + self.config.min_variance
        tc_mean = soc_mean + sic_mean

        outputs: dict[str, torch.Tensor] = {
            "latent": latent,
            "soc_mean": soc_mean,
            "soc_var": soc_var,
            "sic_mean": sic_mean,
            "sic_var": sic_var,
            "tc_mean": tc_mean,
            "class_logits": self.class_head(latent),
        }

        if self.tc_aux_head is not None:
            tc_aux_mean = F.softplus(self.tc_aux_head(latent)).squeeze(-1)
            outputs["tc_aux_mean"] = tc_aux_mean
            outputs["closure_residual"] = tc_aux_mean - tc_mean

        if self.moisture_mean_head is not None and self.moisture_var_head is not None:
            outputs["moisture_mean"] = F.softplus(self.moisture_mean_head(latent)).squeeze(-1)
            outputs["moisture_var"] = F.softplus(self.moisture_var_head(latent)).squeeze(-1) + self.config.min_variance

        return outputs


class STSoilHybridNetRegressor(BaseSoilRegressor):
    """RegressorProtocol wrapper and multi-task trainer for `STSoilHybridNet`."""

    model_name = "st_soil_hybrid_net"
    model_family = "neural_two_branch_physics_constrained"

    def __init__(
        self,
        *,
        n_spectral_features: int | None = None,
        spectral_hidden_dim: int = 64,
        covariate_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.1,
        n_classes: int = 4,
        enable_tc_head: bool = True,
        epochs: int = 60,
        batch_size: int = 32,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        loss_config: MultiTaskLossConfig | None = None,
        mc_dropout_samples: int = 1,
        seed: int = 42,
        target_names: Sequence[str] = ("soc_pct",),
        min_std: float = 1e-4,
    ) -> None:
        super().__init__(target_names=target_names, clip_negative=True, min_std=min_std)
        self.n_spectral_features = int(n_spectral_features) if n_spectral_features is not None else None
        self.spectral_hidden_dim = int(spectral_hidden_dim)
        self.covariate_hidden_dim = int(covariate_hidden_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.dropout = float(dropout)
        self.n_classes = int(n_classes)
        self.enable_tc_head = bool(enable_tc_head)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss_config = loss_config or MultiTaskLossConfig()
        self.mc_dropout_samples = max(1, int(mc_dropout_samples))
        self.seed = int(seed)

        self.net_: STSoilHybridNet | None = None
        self.config_: HybridNetConfig | None = None
        self.training_history_: list[dict[str, float]] = []

    def _split_fused(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        n_total = X.shape[1]
        n_spec = self.n_spectral_features if self.n_spectral_features is not None else n_total
        if n_spec <= 0 or n_spec > n_total:
            raise ModelInferenceError(
                f"Invalid n_spectral_features ({n_spec}) for feature matrix with {n_total} columns."
            )
        X_spec = X[:, :n_spec]
        X_cov = X[:, n_spec:] if n_spec < n_total else None
        return X_spec, X_cov

    def _build_network(self, n_spec: int, n_cov: int) -> STSoilHybridNet:
        torch.manual_seed(self.seed)
        cfg = HybridNetConfig(
            n_spectral_features=n_spec,
            n_covariate_features=n_cov,
            spectral_hidden_dim=self.spectral_hidden_dim,
            covariate_hidden_dim=self.covariate_hidden_dim,
            fusion_hidden_dim=self.fusion_hidden_dim,
            dropout=self.dropout,
            n_classes=self.n_classes,
            enable_tc_head=self.enable_tc_head,
            min_variance=self.min_std * self.min_std,
            seed=self.seed,
        )
        self.config_ = cfg
        net = STSoilHybridNet(cfg)
        self.net_ = net
        return net

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> STSoilHybridNetRegressor:
        del sample_weight
        X_arr, y_2d, is_1d = validate_xy_arrays(X, y)
        assert y_2d is not None
        self._is_1d_target = is_1d
        self._n_features_in = X_arr.shape[1]
        self._n_targets = y_2d.shape[1]

        X_spec, X_cov = self._split_fused(X_arr)
        soc = y_2d[:, 0]
        sic = y_2d[:, 1] if y_2d.shape[1] >= 2 else None
        tc = y_2d[:, 2] if y_2d.shape[1] >= 3 else None

        return self._train_internal(X_spec, X_cov, soc=soc, sic=sic, tc=tc, soil_class=None)

    def fit_multitask(
        self,
        X_spectral: np.ndarray,
        X_covariates: np.ndarray | None,
        *,
        soc: np.ndarray | None = None,
        sic: np.ndarray | None = None,
        tc: np.ndarray | None = None,
        soil_class: np.ndarray | None = None,
    ) -> STSoilHybridNetRegressor:
        """Train directly on explicit spectral + covariate branches and multi-task targets."""
        X_s, _, _ = validate_xy_arrays(X_spectral)
        n_samples = X_s.shape[0]
        if X_covariates is not None and X_covariates.shape[1] > 0:
            X_c, _, _ = validate_xy_arrays(X_covariates)
            if X_c.shape[0] != n_samples:
                raise ModelInferenceError("Sample count mismatch between X_spectral and X_covariates.")
            n_cov = X_c.shape[1]
        else:
            X_c = None
            n_cov = 0

        self.n_spectral_features = X_s.shape[1]
        self._n_features_in = X_s.shape[1] + n_cov
        self._is_1d_target = sic is None and tc is None
        self._n_targets = 1 if self._is_1d_target else (3 if tc is not None else 2)
        return self._train_internal(X_s, X_c, soc=soc, sic=sic, tc=tc, soil_class=soil_class)

    def _train_internal(
        self,
        X_spec: np.ndarray,
        X_cov: np.ndarray | None,
        *,
        soc: np.ndarray | None,
        sic: np.ndarray | None,
        tc: np.ndarray | None,
        soil_class: np.ndarray | None,
    ) -> STSoilHybridNetRegressor:
        n_samples, n_spec = X_spec.shape
        n_cov = X_cov.shape[1] if X_cov is not None else 0
        net = self._build_network(n_spec, n_cov)

        criterion = STSoilMultiTaskLoss(self.loss_config)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def _prep_target(arr: np.ndarray | None) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
            if arr is None:
                return None, None, 1.0
            a = np.asarray(arr, dtype=float).reshape(-1)
            if a.shape[0] != n_samples:
                raise ModelInferenceError("Target vector length does not match X sample count.")
            finite = np.isfinite(a)
            if not np.any(finite):
                return None, None, 1.0
            scale = max(float(np.std(a[finite])), 0.25)
            filled = np.where(finite, a, 0.0)
            return (
                torch.tensor(filled, dtype=torch.float32),
                torch.tensor(finite, dtype=torch.bool),
                scale,
            )

        t_soc, m_soc, s_soc = _prep_target(soc)
        t_sic, m_sic, s_sic = _prep_target(sic)
        t_tc, m_tc, s_tc = _prep_target(tc)

        if soil_class is not None:
            cls_arr = np.asarray(soil_class, dtype=float).reshape(-1)
            cls_valid = np.isfinite(cls_arr) & (cls_arr >= 0) & (cls_arr < self.n_classes)
            t_cls = torch.tensor(np.where(cls_valid, cls_arr, 0), dtype=torch.long) if np.any(cls_valid) else None
            m_cls = torch.tensor(cls_valid, dtype=torch.bool) if t_cls is not None else None
        else:
            t_cls, m_cls = None, None

        t_X_spec = torch.tensor(X_spec, dtype=torch.float32)
        t_X_cov = torch.tensor(X_cov, dtype=torch.float32) if X_cov is not None else None

        rng = np.random.default_rng(self.seed)
        self.training_history_ = []
        net.train()

        eff_batch = max(1, min(self.batch_size, n_samples))
        for epoch in range(self.epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, eff_batch):
                idx = torch.tensor(perm[start : start + eff_batch], dtype=torch.long)
                xb_s = t_X_spec[idx]
                xb_c = t_X_cov[idx] if t_X_cov is not None else None

                optimizer.zero_grad()
                out = net(xb_s, xb_c)
                loss, _ = criterion(
                    out,
                    soc_target=t_soc[idx] if t_soc is not None else None,
                    soc_mask=m_soc[idx] if m_soc is not None else None,
                    sic_target=t_sic[idx] if t_sic is not None else None,
                    sic_mask=m_sic[idx] if m_sic is not None else None,
                    tc_target=t_tc[idx] if t_tc is not None else None,
                    tc_mask=m_tc[idx] if m_tc is not None else None,
                    class_target=t_cls[idx] if t_cls is not None else None,
                    class_mask=m_cls[idx] if m_cls is not None else None,
                    soc_scale=s_soc,
                    sic_scale=s_sic,
                    tc_scale=s_tc,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

            self.training_history_.append(
                {"epoch": float(epoch + 1), "loss_total": epoch_loss / max(1, n_batches)}
            )

        net.eval()
        self._is_fitted = True
        return self

    def predict_structured(
        self,
        X_spectral: np.ndarray,
        X_covariates: np.ndarray | None = None,
    ) -> ModelPrediction:
        """Predict structured physical carbon quantities, uncertainties, closure residual, and class logits."""
        self._require_fitted()
        assert self.net_ is not None and self.config_ is not None

        X_s, _, _ = validate_xy_arrays(X_spectral, expected_n_features=self.config_.n_spectral_features)
        t_s = torch.tensor(X_s, dtype=torch.float32)
        if self.config_.n_covariate_features > 0:
            if X_covariates is None:
                raise ModelInferenceError("Model expects covariate features alongside spectral features.")
            X_c, _, _ = validate_xy_arrays(X_covariates, expected_n_features=self.config_.n_covariate_features)
            if X_c.shape[0] != X_s.shape[0]:
                raise ModelInferenceError("Sample count mismatch between X_spectral and X_covariates.")
            t_c: torch.Tensor | None = torch.tensor(X_c, dtype=torch.float32)
        else:
            t_c = None

        with torch.no_grad():
            if self.mc_dropout_samples > 1 and self.dropout > 0.0:
                self.net_.train()
                passes = [self.net_(t_s, t_c) for _ in range(self.mc_dropout_samples)]
                self.net_.eval()
                soc_samples = torch.stack([p["soc_mean"] for p in passes], dim=0)
                sic_samples = torch.stack([p["sic_mean"] for p in passes], dim=0)
                soc_var_alea = torch.stack([p["soc_var"] for p in passes], dim=0).mean(dim=0)
                sic_var_alea = torch.stack([p["sic_var"] for p in passes], dim=0).mean(dim=0)

                soc_mean = soc_samples.mean(dim=0).cpu().numpy()
                sic_mean = sic_samples.mean(dim=0).cpu().numpy()
                soc_std = np.sqrt((soc_var_alea + soc_samples.var(dim=0, unbiased=False)).cpu().numpy())
                sic_std = np.sqrt((sic_var_alea + sic_samples.var(dim=0, unbiased=False)).cpu().numpy())
                logits = torch.stack([p["class_logits"] for p in passes], dim=0).mean(dim=0)
                tc_aux = (
                    torch.stack([p["tc_aux_mean"] for p in passes], dim=0).mean(dim=0).cpu().numpy()
                    if self.enable_tc_head
                    else None
                )
            else:
                self.net_.eval()
                out = self.net_(t_s, t_c)
                soc_mean = out["soc_mean"].cpu().numpy()
                sic_mean = out["sic_mean"].cpu().numpy()
                soc_std = np.sqrt(out["soc_var"].cpu().numpy())
                sic_std = np.sqrt(out["sic_var"].cpu().numpy())
                logits = out["class_logits"]
                tc_aux = out["tc_aux_mean"].cpu().numpy() if "tc_aux_mean" in out else None

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            logits_np = logits.cpu().numpy()

        tc_mean = soc_mean + sic_mean
        tc_std = np.sqrt(soc_std * soc_std + sic_std * sic_std)
        closure_res = compute_closure_residual(soc_mean, sic_mean, tc_aux) if tc_aux is not None else np.zeros_like(tc_mean)

        return ModelPrediction(
            soc_mean_pct=soc_mean,
            sic_mean_pct=sic_mean,
            tc_mean_pct=tc_mean,
            soc_std_pct=soc_std,
            sic_std_pct=sic_std,
            tc_std_pct=tc_std,
            closure_residual_pct=closure_res,
            class_logits=logits_np,
            class_probs=probs,
            clipped_negative_count=0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean_out, _ = self.predict_distribution(X)
        return mean_out

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        X_arr, _, _ = validate_xy_arrays(X, expected_n_features=self._n_features_in)
        X_spec, X_cov = self._split_fused(X_arr)
        pred = self.predict_structured(X_spec, X_cov)

        assert pred.soc_std_pct is not None and pred.sic_mean_pct is not None and pred.sic_std_pct is not None
        assert pred.tc_mean_pct is not None and pred.tc_std_pct is not None

        if self._n_targets == 1:
            means = pred.soc_mean_pct[:, None]
            stds = pred.soc_std_pct[:, None]
        elif self._n_targets == 2:
            means = np.column_stack([pred.soc_mean_pct, pred.sic_mean_pct])
            stds = np.column_stack([pred.soc_std_pct, pred.sic_std_pct])
        else:
            means = np.column_stack([pred.soc_mean_pct, pred.sic_mean_pct, pred.tc_mean_pct])
            stds = np.column_stack([pred.soc_std_pct, pred.sic_std_pct, pred.tc_std_pct])

        mean_out, std_out = self._format_output(means, stds)
        assert std_out is not None
        return mean_out, std_out

    def state_dict_tensors(self) -> dict[str, torch.Tensor]:
        """Return safe tensor-only state_dict for Faz 10 artifact packaging."""
        self._require_fitted()
        assert self.net_ is not None
        return {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_spectral_features": self.n_spectral_features,
                "spectral_hidden_dim": self.spectral_hidden_dim,
                "covariate_hidden_dim": self.covariate_hidden_dim,
                "fusion_hidden_dim": self.fusion_hidden_dim,
                "dropout": self.dropout,
                "n_classes": self.n_classes,
                "enable_tc_head": self.enable_tc_head,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "mc_dropout_samples": self.mc_dropout_samples,
                "seed": self.seed,
            }
        )
        return params
