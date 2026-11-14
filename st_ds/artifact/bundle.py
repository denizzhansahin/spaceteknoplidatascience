"""Signed, versioned, zero-pickle deployment artifact bundling and verification (Section 26, DEF-006).

Bundle layout inside `bundle_dir/`:
    manifest.json            # SemVer, ArtifactMetadata, InstrumentProfile, TargetLabelSchema, hashes
    model_card.json          # ModelCard, domain envelope, metrics, limitations, OOD & recalibration policy
    preprocess_state.npz     # Numeric-only arrays (no pickle): wavelengths, EPO/GLSW, scalers, OOD, conformal
    model_weights.safetensors # Safe weight tensor archive (zero pickle code execution)

Every integrity check runs BEFORE any weight or `.npz` file is opened. Tampered SHA-256
digests, single untrusted `.pt`/`.pkl` files, or pickle payloads requiring
`weights_only=False` fail closed with `ArtifactCompatibilityError`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import safetensors
import safetensors.numpy as st_np
import safetensors.torch as st_torch
import sklearn
import torch

from st_ds import SCHEMA_VERSION
from st_ds.chemometrics.epo import EPO, EPOConfig, GLSW
from st_ds.chemometrics.pipeline import ChemometricPipeline, PreprocessConfig
from st_ds.domain.errors import ArtifactCompatibilityError, SchemaError
from st_ds.domain.instrument import InstrumentProfile, SpectralModality
from st_ds.domain.schemas import ArtifactMetadata, TargetLabelSchema
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline, _BranchScaler
from st_ds.features.registry import FeatureDescriptor, FeatureProfileSpec, FeatureRegistry
from st_ds.models.base import BaseSoilRegressor
from st_ds.models.baselines import (
    KernelRidgeRegressor,
    LocalPLSRRegressor,
    PLSRRegressor,
    RandomForestSoilRegressor,
)
from st_ds.models.hybrid_net import HybridNetConfig, STSoilHybridNet, STSoilHybridNetRegressor
from st_ds.models.loss import MultiTaskLossConfig
from st_ds.uncertainty.conformal import ConformalConfig, ConformalUncertaintyCalibrator
from st_ds.uncertainty.ood import DomainEnvelope, OODConfig, OutOfDomainDetector

ARTIFACT_FORMAT_VERSION = "1.0.0"
DEFAULT_WEIGHT_FILENAME = "model_weights.safetensors"


def compute_file_sha256(path: str | Path) -> str:
    """Compute hexadecimal SHA-256 digest of a file using fixed chunked reads."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_dependency_versions() -> dict[str, str]:
    """Capture runtime library versions stored in artifact manifests."""
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "sklearn": str(sklearn.__version__),
        "torch": str(torch.__version__),
        "safetensors": str(safetensors.__version__),
    }


class _FrozenPLSEstimator:
    """Zero-pickle PLS prediction helper matching sklearn PLSRegression bit-for-bit."""

    def __init__(
        self,
        *,
        x_mean: np.ndarray,
        y_mean: np.ndarray,
        coef: np.ndarray,
    ) -> None:
        self._x_mean = np.asarray(x_mean, dtype=np.float64)
        self._y_mean = np.asarray(y_mean, dtype=np.float64)
        self.coef_ = np.asarray(coef, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        return np.dot(arr - self._x_mean[None, :], self.coef_.T) + self._y_mean[None, :]


class _FrozenDecisionTree:
    """Zero-pickle iterative evaluator for a serialized sklearn DecisionTreeRegressor."""

    def __init__(
        self,
        *,
        children_left: np.ndarray,
        children_right: np.ndarray,
        feature: np.ndarray,
        threshold: np.ndarray,
        value: np.ndarray,
    ) -> None:
        self.children_left = np.asarray(children_left, dtype=np.int64)
        self.children_right = np.asarray(children_right, dtype=np.int64)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        self.value = np.asarray(value, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        n_samples = arr.shape[0]
        nodes = np.zeros(n_samples, dtype=np.int64)
        active = self.children_left[nodes] != self.children_right[nodes]
        while np.any(active):
            act_idx = np.where(active)[0]
            act_nodes = nodes[act_idx]
            feats = self.feature[act_nodes]
            thresh = self.threshold[act_nodes]
            go_left = arr[act_idx, feats] <= thresh
            nodes[act_idx] = np.where(
                go_left,
                self.children_left[act_nodes],
                self.children_right[act_nodes],
            )
            active = self.children_left[nodes] != self.children_right[nodes]

        out = self.value[nodes]
        if out.ndim == 2 and out.shape[1] == 1:
            return out[:, 0]
        return out


class _FrozenRandomForestEstimator:
    """Zero-pickle ensemble holder providing `.estimators_` compatible with RandomForestSoilRegressor."""

    def __init__(self, trees: Sequence[_FrozenDecisionTree]) -> None:
        self.estimators_ = list(trees)


@dataclass(frozen=True)
class LoadedArtifact:
    """Verified, ready-for-inference deployment artifact bundle."""

    bundle_dir: Path
    manifest: dict[str, Any]
    model_card: dict[str, Any]
    metadata: ArtifactMetadata
    instrument_profile: InstrumentProfile
    target_schema: TargetLabelSchema
    chemometric_pipeline: ChemometricPipeline
    fusion_pipeline: FeatureFusionPipeline
    model: BaseSoilRegressor
    conformal_calibrator: ConformalUncertaintyCalibrator | None = None
    ood_detector: OutOfDomainDetector | None = None
    feature_names: tuple[str, ...] = ()
    background_features: np.ndarray | None = None
    background_targets: np.ndarray | None = None
    bundle_sha256: str = ""


def _profile_from_dict(data: Mapping[str, Any]) -> InstrumentProfile:
    return InstrumentProfile(
        profile_id=str(data["profile_id"]),
        sensor_model=str(data["sensor_model"]),
        modality=SpectralModality(str(data["modality"])),
        wavelength_grid_nm=tuple(float(w) for w in data["wavelength_grid_nm"]),
        channel_ids=tuple(str(c) for c in data["channel_ids"]),
        fwhm_nm=tuple(float(f) for f in data["fwhm_nm"]) if data.get("fwhm_nm") is not None else None,
        calibration_file_id=str(data.get("calibration_file_id", "default_cal_v1")),
        description=str(data.get("description", "")),
        metadata=dict(data.get("metadata", {})),
    )


def _target_schema_from_dict(data: Mapping[str, Any]) -> TargetLabelSchema:
    return TargetLabelSchema(
        targets=tuple(str(t) for t in data.get("targets", ("soc_pct", "sic_pct", "tc_pct"))),
        carbon_unit=str(data.get("carbon_unit", "pct")),
        dry_matter_basis=bool(data.get("dry_matter_basis", True)),
        enforce_non_negative=bool(data.get("enforce_non_negative", True)),
        enforce_tc_closure=bool(data.get("enforce_tc_closure", True)),
        closure_tolerance_pct=float(data.get("closure_tolerance_pct", 0.05)),
        taxonomy_version=str(data.get("taxonomy_version", "provisional_taxonomy_v0")),
        class_labels=tuple(str(c) for c in data.get("class_labels", TargetLabelSchema().class_labels)),
    )


def _feature_profile_to_dict(spec: FeatureProfileSpec) -> dict[str, Any]:
    return {
        "profile_id": spec.profile_id,
        "instrument_profile_id": spec.instrument_profile_id,
        "descriptors": [d.to_dict() for d in spec.descriptors],
        "eis_frequencies_hz": list(spec.eis_frequencies_hz),
        "eis_protocol_id": spec.eis_protocol_id,
        "schema_hash": spec.schema_hash,
    }


def _feature_profile_from_dict(data: Mapping[str, Any]) -> FeatureProfileSpec:
    descs = tuple(
        FeatureDescriptor(
            name=str(d["name"]),
            group=d["group"],
            unit=str(d["unit"]),
            description=str(d.get("description", "")),
            required=bool(d.get("required", False)),
        )
        for d in data["descriptors"]
    )
    return FeatureProfileSpec(
        profile_id=str(data["profile_id"]),
        instrument_profile_id=str(data["instrument_profile_id"]),
        descriptors=descs,
        eis_frequencies_hz=tuple(float(f) for f in data.get("eis_frequencies_hz", [])),
        eis_protocol_id=data.get("eis_protocol_id"),
        schema_hash=str(data.get("schema_hash", "")),
    )


def _serialize_model_safetensors(model: BaseSoilRegressor, out_path: Path) -> dict[str, Any]:
    """Serialize any STDS RegressorProtocol model strictly without pickle."""
    if not getattr(model, "_is_fitted", False):
        raise ArtifactCompatibilityError(
            f"Cannot export unfitted model '{model.__class__.__name__}'."
        )

    base_meta = {
        "class_name": model.__class__.__name__,
        "model_name": model.model_name,
        "model_family": model.model_family,
        "is_1d_target": bool(model._is_1d_target),
        "n_features_in": int(model._n_features_in),
        "n_targets": int(model._n_targets),
        "target_names": list(model.target_names),
        "clip_negative": bool(model.clip_negative),
        "min_std": float(model.min_std),
        "feature_names": list(model._feature_names),
        "training_dataset_source": str(model._training_dataset_source),
        "params": model.get_params(),
    }

    if isinstance(model, STSoilHybridNetRegressor):
        if model.net_ is None or model.config_ is None:
            raise ArtifactCompatibilityError("STSoilHybridNetRegressor network weights are missing.")
        state = {k: v.detach().cpu().contiguous() for k, v in model.net_.state_dict().items()}
        st_torch.save_file(state, str(out_path))
        base_meta["weight_format"] = "safetensors_torch"
        base_meta["hybrid_config"] = model.config_.to_dict()
        base_meta["n_spectral_features"] = model.n_spectral_features
        return base_meta

    tensors: dict[str, np.ndarray] = {}
    if isinstance(model, PLSRRegressor):
        est = model._estimator
        if est is None or model._cv_residual_std_ is None:
            raise ArtifactCompatibilityError("PLSRRegressor state is incomplete.")
        if isinstance(est, _FrozenPLSEstimator):
            x_mean, y_mean, coef = est._x_mean, est._y_mean, est.coef_
        else:
            x_mean = np.asarray(est._x_mean, dtype=np.float64)
            y_mean = np.asarray(est._y_mean, dtype=np.float64)
            coef = np.asarray(est.coef_, dtype=np.float64)
        tensors = {
            "x_mean": x_mean,
            "y_mean": y_mean,
            "coef": coef,
            "cv_residual_std": np.asarray(model._cv_residual_std_, dtype=np.float64),
        }
        base_meta["selected_n_components"] = int(model.selected_n_components_)
    elif isinstance(model, KernelRidgeRegressor):
        if (
            model._X_train_ is None
            or model._dual_coef_ is None
            or model._cho_factor_ is None
            or model._residual_std_ is None
        ):
            raise ArtifactCompatibilityError("KernelRidgeRegressor state is incomplete.")
        tensors = {
            "X_train": np.asarray(model._X_train_, dtype=np.float64),
            "dual_coef": np.asarray(model._dual_coef_, dtype=np.float64),
            "cho_factor_L": np.asarray(model._cho_factor_[0], dtype=np.float64),
            "residual_std": np.asarray(model._residual_std_, dtype=np.float64),
        }
        base_meta["cho_lower"] = bool(model._cho_factor_[1])
        base_meta["selected_alpha"] = float(model.selected_alpha_)
        base_meta["selected_sigma"] = float(model.selected_sigma_)
    elif isinstance(model, LocalPLSRRegressor):
        if model._X_lib_ is None or model._y_lib_ is None:
            raise ArtifactCompatibilityError("LocalPLSRRegressor library is incomplete.")
        tensors = {
            "X_lib": np.asarray(model._X_lib_, dtype=np.float64),
            "y_lib": np.asarray(model._y_lib_, dtype=np.float64),
        }
        base_meta["effective_bandwidth"] = float(model._effective_bandwidth_)
    elif isinstance(model, RandomForestSoilRegressor):
        if model._estimator is None:
            raise ArtifactCompatibilityError("RandomForestSoilRegressor estimator is missing.")
        lefts: list[np.ndarray] = []
        rights: list[np.ndarray] = []
        feats: list[np.ndarray] = []
        threshs: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        offsets = [0]
        for tree_est in model._estimator.estimators_:
            if isinstance(tree_est, _FrozenDecisionTree):
                cl = tree_est.children_left
                cr = tree_est.children_right
                ft = tree_est.feature
                th = tree_est.threshold
                vl = tree_est.value
            else:
                tr = tree_est.tree_
                cl = np.asarray(tr.children_left, dtype=np.int64)
                cr = np.asarray(tr.children_right, dtype=np.int64)
                ft = np.asarray(tr.feature, dtype=np.int64)
                th = np.asarray(tr.threshold, dtype=np.float64)
                raw_v = np.asarray(tr.value, dtype=np.float64)
                vl = raw_v[:, :, 0] if raw_v.ndim == 3 else raw_v
            lefts.append(cl)
            rights.append(cr)
            feats.append(ft)
            threshs.append(th)
            vals.append(vl if vl.ndim == 2 else vl[:, None])
            offsets.append(offsets[-1] + int(cl.shape[0]))
        tensors = {
            "tree_offsets": np.asarray(offsets, dtype=np.int64),
            "children_left": np.concatenate(lefts),
            "children_right": np.concatenate(rights),
            "feature": np.concatenate(feats),
            "threshold": np.concatenate(threshs),
            "value": np.vstack(vals),
        }
    else:
        raise ArtifactCompatibilityError(
            f"Unsupported model class '{model.__class__.__name__}' for safe artifact export."
        )

    st_np.save_file(tensors, str(out_path))
    base_meta["weight_format"] = "safetensors_numpy"
    return base_meta


def _deserialize_model_from_bundle(
    model_spec: Mapping[str, Any],
    weight_path: Path,
) -> BaseSoilRegressor:
    """Load model weights strictly without arbitrary pickle execution."""
    wfmt = str(model_spec.get("weight_format", ""))
    cls_name = str(model_spec.get("class_name", ""))
    params = dict(model_spec.get("params", {}))
    target_names = tuple(model_spec.get("target_names", ("soc_pct",)))
    clip_neg = bool(model_spec.get("clip_negative", True))
    min_std = float(model_spec.get("min_std", 1e-4))

    model: BaseSoilRegressor
    if cls_name == "STSoilHybridNetRegressor":
        hc = dict(model_spec["hybrid_config"])
        cfg = HybridNetConfig(
            n_spectral_features=int(hc["n_spectral_features"]),
            n_covariate_features=int(hc.get("n_covariate_features", 0)),
            spectral_hidden_dim=int(hc.get("spectral_hidden_dim", 64)),
            covariate_hidden_dim=int(hc.get("covariate_hidden_dim", 32)),
            fusion_hidden_dim=int(hc.get("fusion_hidden_dim", 64)),
            dropout=float(hc.get("dropout", 0.1)),
            n_classes=int(hc.get("n_classes", 4)),
            enable_tc_head=bool(hc.get("enable_tc_head", True)),
            enable_moisture_head=bool(hc.get("enable_moisture_head", False)),
            min_variance=float(hc.get("min_variance", 1e-4)),
            seed=int(hc.get("seed", 42)),
        )
        net = STSoilHybridNet(cfg)
        if wfmt == "safetensors_torch":
            state_dict = st_torch.load_file(str(weight_path), device="cpu")
        elif wfmt == "torch_state_dict_weights_only":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ArtifactCompatibilityError(
                    f"Rejected unsafe or unreadable PyTorch checkpoint '{weight_path.name}': {exc}",
                    remediation="Re-export using save_artifact with safetensors or weights_only state_dict.",
                ) from exc
        else:
            raise ArtifactCompatibilityError(f"Unsupported weight_format '{wfmt}' for STSoilHybridNetRegressor.")
        net.load_state_dict(state_dict, strict=True)
        net.eval()

        reg = STSoilHybridNetRegressor(
            n_spectral_features=model_spec.get("n_spectral_features", cfg.n_spectral_features),
            spectral_hidden_dim=cfg.spectral_hidden_dim,
            covariate_hidden_dim=cfg.covariate_hidden_dim,
            fusion_hidden_dim=cfg.fusion_hidden_dim,
            dropout=cfg.dropout,
            n_classes=cfg.n_classes,
            enable_tc_head=cfg.enable_tc_head,
            epochs=int(params.get("epochs", 60)),
            batch_size=int(params.get("batch_size", 32)),
            learning_rate=float(params.get("learning_rate", 2e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
            loss_config=MultiTaskLossConfig(),
            mc_dropout_samples=int(params.get("mc_dropout_samples", 1)),
            seed=cfg.seed,
            target_names=target_names,
            min_std=min_std,
        )
        reg.config_ = cfg
        reg.net_ = net
        model = reg
    elif cls_name == "PLSRRegressor":
        if wfmt != "safetensors_numpy":
            raise ArtifactCompatibilityError(f"Unsupported weight_format '{wfmt}' for PLSRRegressor.")
        tensors = st_np.load_file(str(weight_path))
        reg = PLSRRegressor(
            max_components=int(params.get("max_components", 10)),
            inner_cv_folds=int(params.get("inner_cv_folds", 5)),
            scale=bool(params.get("scale", True)),
            target_names=target_names,
            clip_negative=clip_neg,
            min_std=min_std,
        )
        reg.selected_n_components_ = int(model_spec["selected_n_components"])
        reg._cv_residual_std_ = np.asarray(tensors["cv_residual_std"], dtype=np.float64)
        reg._estimator = _FrozenPLSEstimator(  # type: ignore[assignment]
            x_mean=tensors["x_mean"],
            y_mean=tensors["y_mean"],
            coef=tensors["coef"],
        )
        model = reg
    elif cls_name == "KernelRidgeRegressor":
        if wfmt != "safetensors_numpy":
            raise ArtifactCompatibilityError(f"Unsupported weight_format '{wfmt}' for KernelRidgeRegressor.")
        tensors = st_np.load_file(str(weight_path))
        reg = KernelRidgeRegressor(
            alpha_grid=tuple(float(a) for a in params.get("alpha_grid", (1e-3, 1e-2, 1e-1, 1.0, 10.0))),
            sigma_multipliers=tuple(float(s) for s in params.get("sigma_multipliers", (0.5, 1.0, 2.0, 4.0))),
            inner_cv_folds=int(params.get("inner_cv_folds", 5)),
            target_names=target_names,
            clip_negative=clip_neg,
            min_std=min_std,
        )
        reg.selected_alpha_ = float(model_spec["selected_alpha"])
        reg.selected_sigma_ = float(model_spec["selected_sigma"])
        reg._X_train_ = np.asarray(tensors["X_train"], dtype=np.float64)
        reg._dual_coef_ = np.asarray(tensors["dual_coef"], dtype=np.float64)
        reg._cho_factor_ = (
            np.asarray(tensors["cho_factor_L"], dtype=np.float64),
            bool(model_spec.get("cho_lower", True)),
        )
        reg._residual_std_ = np.asarray(tensors["residual_std"], dtype=np.float64)
        model = reg
    elif cls_name == "LocalPLSRRegressor":
        if wfmt != "safetensors_numpy":
            raise ArtifactCompatibilityError(f"Unsupported weight_format '{wfmt}' for LocalPLSRRegressor.")
        tensors = st_np.load_file(str(weight_path))
        reg = LocalPLSRRegressor(
            n_neighbors=int(params.get("n_neighbors", 20)),
            n_components=int(params.get("n_components", 3)),
            bandwidth=float(model_spec["effective_bandwidth"]),
            min_library_size=int(params.get("min_library_size", 8)),
            target_names=target_names,
            clip_negative=clip_neg,
            min_std=min_std,
        )
        reg._X_lib_ = np.asarray(tensors["X_lib"], dtype=np.float64)
        reg._y_lib_ = np.asarray(tensors["y_lib"], dtype=np.float64)
        reg._effective_bandwidth_ = float(model_spec["effective_bandwidth"])
        model = reg
    elif cls_name == "RandomForestSoilRegressor":
        if wfmt != "safetensors_numpy":
            raise ArtifactCompatibilityError(f"Unsupported weight_format '{wfmt}' for RandomForestSoilRegressor.")
        tensors = st_np.load_file(str(weight_path))
        reg = RandomForestSoilRegressor(
            n_estimators=int(params.get("n_estimators", 150)),
            max_depth=params.get("max_depth", 12),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            random_state=int(params.get("random_state", 42)),
            target_names=target_names,
            clip_negative=clip_neg,
            min_std=min_std,
        )
        offsets = tensors["tree_offsets"]
        cl_all = tensors["children_left"]
        cr_all = tensors["children_right"]
        ft_all = tensors["feature"]
        th_all = tensors["threshold"]
        vl_all = tensors["value"]
        trees = [
            _FrozenDecisionTree(
                children_left=cl_all[int(offsets[i]) : int(offsets[i + 1])],
                children_right=cr_all[int(offsets[i]) : int(offsets[i + 1])],
                feature=ft_all[int(offsets[i]) : int(offsets[i + 1])],
                threshold=th_all[int(offsets[i]) : int(offsets[i + 1])],
                value=vl_all[int(offsets[i]) : int(offsets[i + 1])],
            )
            for i in range(len(offsets) - 1)
        ]
        reg._estimator = _FrozenRandomForestEstimator(trees)  # type: ignore[assignment]
        model = reg
    else:
        raise ArtifactCompatibilityError(
            f"Unsupported model class '{cls_name}' in artifact manifest."
        )

    model._is_1d_target = bool(model_spec["is_1d_target"])
    model._n_features_in = int(model_spec["n_features_in"])
    model._n_targets = int(model_spec["n_targets"])
    model._feature_names = tuple(str(n) for n in model_spec.get("feature_names", []))
    model._training_dataset_source = str(model_spec.get("training_dataset_source", "unknown"))
    model._is_fitted = True
    return model


def _compute_manifest_digest(core_payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(core_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_artifact(
    bundle_dir: str | Path,
    *,
    metadata: ArtifactMetadata,
    instrument_profile: InstrumentProfile,
    model: BaseSoilRegressor,
    chemometric_pipeline: ChemometricPipeline,
    fusion_pipeline: FeatureFusionPipeline,
    conformal_calibrator: ConformalUncertaintyCalibrator | None = None,
    ood_detector: OutOfDomainDetector | None = None,
    validation_metrics: Mapping[str, Any] | None = None,
    subgroup_metrics: Mapping[str, Any] | None = None,
    uncertainty_coverage: Mapping[str, Any] | None = None,
    reference_methods: Mapping[str, Any] | None = None,
    background_features: np.ndarray | None = None,
    background_targets: np.ndarray | None = None,
    target_schema: TargetLabelSchema | None = None,
    model_card_overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a verified, zero-pickle deployment artifact bundle to `bundle_dir`."""
    if not chemometric_pipeline.is_fitted:
        raise ArtifactCompatibilityError("ChemometricPipeline must be fitted before save_artifact.")
    if not fusion_pipeline.is_fitted or fusion_pipeline.feature_profile is None:
        raise ArtifactCompatibilityError("FeatureFusionPipeline must be fitted before save_artifact.")
    if (
        fusion_pipeline._spectral_scaler is None
        or fusion_pipeline._spectral_scaler.center is None
        or fusion_pipeline._spectral_scaler.scale is None
        or fusion_pipeline._covariate_scaler is None
        or fusion_pipeline._covariate_scaler.center is None
        or fusion_pipeline._covariate_scaler.scale is None
        or fusion_pipeline._covariate_medians is None
    ):
        raise ArtifactCompatibilityError("FeatureFusionPipeline internal scaler arrays are incomplete.")

    out_dir = Path(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Serialize model weights cleanly via safetensors
    weight_filename = DEFAULT_WEIGHT_FILENAME
    weight_path = out_dir / weight_filename
    model_spec = _serialize_model_safetensors(model, weight_path)

    # 2. Serialize numeric-only preprocessing, uncertainty, OOD, and XAI background state
    npz_filename = "preprocess_state.npz"
    npz_path = out_dir / npz_filename
    npz_arrays: dict[str, np.ndarray] = {
        "wavelength_grid_nm": np.asarray(chemometric_pipeline.wavelength_grid_nm, dtype=np.float64),
        "fusion_spectral_center": np.asarray(fusion_pipeline._spectral_scaler.center, dtype=np.float64),
        "fusion_spectral_scale": np.asarray(fusion_pipeline._spectral_scaler.scale, dtype=np.float64),
        "fusion_covariate_center": np.asarray(fusion_pipeline._covariate_scaler.center, dtype=np.float64),
        "fusion_covariate_scale": np.asarray(fusion_pipeline._covariate_scaler.scale, dtype=np.float64),
        "fusion_covariate_medians": np.asarray(fusion_pipeline._covariate_medians, dtype=np.float64),
    }

    epo_meta: dict[str, Any] | None = None
    if chemometric_pipeline.epo is not None and chemometric_pipeline.epo.is_fitted:
        epo = chemometric_pipeline.epo
        assert epo.V_s is not None and epo.Q is not None and epo.P is not None
        npz_arrays["epo_V_s"] = np.asarray(epo.V_s, dtype=np.float64)
        npz_arrays["epo_Q"] = np.asarray(epo.Q, dtype=np.float64)
        npz_arrays["epo_P"] = np.asarray(epo.P, dtype=np.float64)
        if epo.singular_values is not None:
            npz_arrays["epo_singular_values"] = np.asarray(epo.singular_values, dtype=np.float64)
        if epo.explained_diff_variance_ratio is not None:
            npz_arrays["epo_explained_diff_variance_ratio"] = np.asarray(
                epo.explained_diff_variance_ratio, dtype=np.float64
            )
        epo_meta = {
            "n_bands": int(epo.n_bands or 0),
            "q": int(epo.q or 1),
            "max_q": int(epo.config.max_q),
            "variance_capture_threshold": float(epo.config.variance_capture_threshold),
            "space_signature": epo.space_signature,
            "calibration_dataset_hash": epo.calibration_dataset_hash,
        }

    glsw_meta: dict[str, Any] | None = None
    if chemometric_pipeline.glsw is not None and chemometric_pipeline.glsw.is_fitted:
        glsw = chemometric_pipeline.glsw
        assert glsw.P is not None
        npz_arrays["glsw_P"] = np.asarray(glsw.P, dtype=np.float64)
        glsw_meta = {
            "alpha": float(glsw.alpha),
            "n_bands": int(glsw.n_bands or 0),
            "space_signature": glsw.space_signature,
        }

    conformal_meta: dict[str, Any] | None = None
    if conformal_calibrator is not None and conformal_calibrator._is_fitted:
        assert conformal_calibrator.nonconformity_scores_ is not None
        npz_arrays["conformal_nonconformity_scores"] = np.asarray(
            conformal_calibrator.nonconformity_scores_, dtype=np.float64
        )
        conformal_meta = {
            "config": conformal_calibrator.config.to_dict(),
            "n_calibration_samples": int(conformal_calibrator.n_calibration_samples_),
        }

    ood_meta: dict[str, Any] | None = None
    if ood_detector is not None and ood_detector._is_fitted:
        assert (
            ood_detector.feature_mean_ is not None
            and ood_detector.feature_std_ is not None
            and ood_detector.z_train_ is not None
            and ood_detector.cho_factor_ is not None
        )
        npz_arrays["ood_feature_mean"] = np.asarray(ood_detector.feature_mean_, dtype=np.float64)
        npz_arrays["ood_feature_std"] = np.asarray(ood_detector.feature_std_, dtype=np.float64)
        npz_arrays["ood_z_train"] = np.asarray(ood_detector.z_train_, dtype=np.float64)
        npz_arrays["ood_cho_L"] = np.asarray(ood_detector.cho_factor_[0], dtype=np.float64)
        ood_meta = {
            "config": ood_detector.config.to_dict(),
            "cho_lower": bool(ood_detector.cho_factor_[1]),
            "mahalanobis_threshold": float(ood_detector.mahalanobis_threshold_),
            "knn_threshold": float(ood_detector.knn_threshold_),
            "domain_envelope": (
                ood_detector.domain_envelope_.to_dict()
                if ood_detector.domain_envelope_ is not None
                else None
            ),
        }

    if background_features is not None:
        bg_x = np.asarray(background_features, dtype=np.float64)
        if bg_x.ndim == 2 and bg_x.shape[0] > 0:
            bg_x_slice = bg_x[:128]
            npz_arrays["xai_background_X"] = bg_x_slice
            if background_targets is not None:
                bg_y = np.asarray(background_targets, dtype=np.float64)
                if bg_y.ndim <= 1:
                    bg_y = bg_y.reshape(-1)
                if bg_y.shape[0] >= bg_x_slice.shape[0]:
                    npz_arrays["xai_background_y"] = bg_y[: bg_x_slice.shape[0]]

    np.savez(npz_path, **npz_arrays)

    # 3. Write model_card.json (Section 26, OD-006)
    schema = target_schema or TargetLabelSchema()
    dataset_source = str(model._training_dataset_source or "unknown")
    is_synth = dataset_source == "synthetic"
    card_path = out_dir / "model_card.json"
    card_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": metadata.artifact_id,
        "semantic_version": metadata.semantic_version,
        "model_name": model.model_name,
        "model_family": model.model_family,
        "target_names": list(model.target_names),
        "target_schema": schema.to_dict(),
        "instrument_profile_id": instrument_profile.profile_id,
        "calibration_id": metadata.calibration_id,
        "dataset_source": dataset_source,
        "is_synthetic_smoke_only": is_synth,
        "validation_metrics": dict(validation_metrics or {}),
        "subgroup_metrics": dict(subgroup_metrics or {}),
        "uncertainty_coverage": dict(uncertainty_coverage or {}),
        "domain_envelope": ood_meta["domain_envelope"] if ood_meta is not None else None,
        "reference_methods": dict(
            reference_methods
            or {
                "soc_pct": "dry_combustion_acid_pretreated_iso10694",
                "sic_pct": "calcimeter_iso10693",
                "tc_pct": "dry_combustion_dumas_iso10694",
            }
        ),
        "intended_use": (
            "Standalone soil carbon prediction (SOC/SIC/TC on dry-matter % basis), "
            "uncertainty quantification, and physical carbon stock estimation when valid bulk density is supplied."
        ),
        "limitations": [
            "Not a laboratory replacement or MRV certification statement without regional site calibration.",
            "Requires matched InstrumentProfile wavelength grid and dry-matter reference calibration.",
            "Carbon stock returns None whenever bulk_density is missing (ADR-004 / DEF-005).",
        ],
        "out_of_domain_policy": (
            "Measurements outside the calibrated Mahalanobis/k-NN/covariate envelope are flagged with "
            "QualityFlag.OUT_OF_CALIBRATION_DOMAIN and confidence_score < 0.5."
        ),
        "recalibration_triggers": [
            "Instrument lamp/detector replacement or reference drift exceeding calibration tolerance.",
            "Deployment in unseen soil texture classes, moisture regimes, or carbonate provinces.",
        ],
    }
    if is_synth:
        card_payload["warning"] = (
            "Model trained on synthetic records (`dataset_source='synthetic'`); pipeline verification only."
        )
    if model_card_overrides:
        card_payload.update(dict(model_card_overrides))
    card_path.write_text(json.dumps(card_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4. Compute SHA-256 digests and write signed manifest.json
    tracked_files = [weight_filename, npz_filename, "model_card.json"]
    file_entries: dict[str, dict[str, Any]] = {}
    for fname in tracked_files:
        fpath = out_dir / fname
        file_entries[fname] = {
            "sha256": compute_file_sha256(fpath),
            "size_bytes": int(fpath.stat().st_size),
        }

    meta_dict = metadata.to_dict()
    meta_dict["dependency_versions"] = dict(metadata.dependency_versions or collect_dependency_versions())
    meta_dict["artifact_sha256"] = ""

    fusion_cfg = fusion_pipeline.config
    preprocess_spec = {
        "chemometrics": chemometric_pipeline.config.to_dict(),
        "space_signature": chemometric_pipeline.space_signature,
        "wavelength_grid_nm": list(chemometric_pipeline.wavelength_grid_nm or ()),
        "epo": epo_meta,
        "glsw": glsw_meta,
        "fusion_config": {
            "covariate_names": list(fusion_cfg.covariate_names),
            "missing_strategy": fusion_cfg.missing_strategy,
            "spectral_scaler": fusion_cfg.spectral_scaler,
            "covariate_scaler": fusion_cfg.covariate_scaler,
            "include_eis": fusion_cfg.include_eis,
            "include_derived": fusion_cfg.include_derived,
            "include_visnir_bands": fusion_cfg.include_visnir_bands,
            "eis_frequencies_hz": list(fusion_cfg.eis_frequencies_hz),
            "eis_protocol_id": fusion_cfg.eis_protocol_id,
        },
        "feature_profile": _feature_profile_to_dict(fusion_pipeline.feature_profile),
        "conformal": conformal_meta,
        "ood": ood_meta,
    }

    core_manifest: dict[str, Any] = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "metadata": meta_dict,
        "instrument_profile": instrument_profile.to_dict(),
        "target_schema": schema.to_dict(),
        "model_spec": model_spec,
        "weight_filename": weight_filename,
        "preprocess_spec": preprocess_spec,
        "files": file_entries,
    }
    bundle_sha = _compute_manifest_digest(core_manifest)
    core_manifest["metadata"]["artifact_sha256"] = bundle_sha
    core_manifest["bundle_sha256"] = bundle_sha

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(core_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def verify_artifact_integrity(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify bundle structure and SHA-256 digests BEFORE opening weights or `.npz` files.

    Raises ArtifactCompatibilityError on missing manifests, tampered hashes,
    path traversal attempts, or raw untrusted checkpoint files.
    """
    p = Path(bundle_dir)
    if not p.exists():
        raise ArtifactCompatibilityError(
            f"Artifact path does not exist: '{p}'.",
            remediation="Provide a valid directory exported via st_ds.artifact.save_artifact.",
        )
    if p.is_file():
        raise ArtifactCompatibilityError(
            f"Refusing to load raw single-file checkpoint '{p.name}': signed bundle directory with "
            "verified manifest.json and SHA-256 digests is required (DEF-006).",
            details={"path": str(p), "reason": "raw_single_file_forbidden"},
            remediation="Package the model into a verified artifact bundle with save_artifact.",
        )

    manifest_path = p / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactCompatibilityError(
            f"Artifact directory '{p}' is missing required manifest.json.",
            details={"bundle_dir": str(p)},
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"Failed to parse artifact manifest.json: {exc}",
            details={"bundle_dir": str(p)},
        ) from exc

    fmt_ver = str(manifest.get("artifact_format_version", ""))
    if not fmt_ver.startswith("1."):
        raise ArtifactCompatibilityError(
            f"Incompatible artifact_format_version '{fmt_ver}'; expected '1.x'.",
            details={"artifact_format_version": fmt_ver},
        )

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactCompatibilityError("Artifact manifest 'files' entry is missing or malformed.")

    weight_filename = str(manifest.get("weight_filename", DEFAULT_WEIGHT_FILENAME))
    for req in (weight_filename, "preprocess_state.npz", "model_card.json"):
        if req not in files:
            raise ArtifactCompatibilityError(
                f"Artifact manifest is missing required tracked file '{req}'.",
                details={"missing_file": req},
            )

    root_resolved = p.resolve()
    for rel_name, info in files.items():
        if not isinstance(rel_name, str) or Path(rel_name).is_absolute() or ".." in Path(rel_name).parts:
            raise ArtifactCompatibilityError(
                f"Unsafe relative path '{rel_name}' detected in artifact manifest.",
                details={"filename": rel_name},
            )
        target = (p / rel_name).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError as exc:
            raise ArtifactCompatibilityError(
                f"Artifact path escape detected for '{rel_name}'.",
            ) from exc
        if not target.is_file():
            raise ArtifactCompatibilityError(
                f"Tracked artifact file '{rel_name}' is missing from bundle.",
                details={"missing_file": rel_name},
            )
        expected_sha = str(info.get("sha256", ""))
        actual_sha = compute_file_sha256(target)
        if not expected_sha or actual_sha != expected_sha:
            raise ArtifactCompatibilityError(
                f"SHA-256 integrity mismatch for artifact file '{rel_name}' (tampered or corrupt bundle, DEF-006).",
                details={"filename": rel_name, "expected_sha256": expected_sha, "actual_sha256": actual_sha},
                remediation="Do not load this bundle; re-export from a trusted training run.",
            )

    # Verify manifest digest itself
    stored_bundle_sha = str(manifest.get("bundle_sha256", ""))
    meta_copy = dict(manifest.get("metadata", {}))
    meta_copy["artifact_sha256"] = ""
    core_for_hash = {k: v for k, v in manifest.items() if k != "bundle_sha256"}
    core_for_hash["metadata"] = meta_copy
    recomputed_sha = _compute_manifest_digest(core_for_hash)
    if not stored_bundle_sha or recomputed_sha != stored_bundle_sha:
        raise ArtifactCompatibilityError(
            "Manifest bundle_sha256 verification failed; manifest.json may have been tampered with (DEF-006).",
            details={"expected": stored_bundle_sha, "actual": recomputed_sha},
        )

    return manifest


def load_artifact(
    bundle_dir: str | Path,
    *,
    expected_device_profile: InstrumentProfile | str | None = None,
    expected_calibration_id: str | None = None,
    strict_compatibility: bool = True,
) -> LoadedArtifact:
    """Verify SHA-256 integrity and safely load a complete STDS artifact bundle."""
    manifest = verify_artifact_integrity(bundle_dir)
    p = Path(bundle_dir)

    meta_d = dict(manifest["metadata"])
    metadata = ArtifactMetadata(
        artifact_id=str(meta_d["artifact_id"]),
        semantic_version=str(meta_d["semantic_version"]),
        schema_version=str(meta_d["schema_version"]),
        instrument_profile_id=str(meta_d["instrument_profile_id"]),
        calibration_id=str(meta_d["calibration_id"]),
        pipeline_version=str(meta_d["pipeline_version"]),
        dataset_hash=str(meta_d["dataset_hash"]),
        dependency_versions=dict(meta_d.get("dependency_versions", {})),
        artifact_sha256=str(manifest["bundle_sha256"]),
        code_commit=str(meta_d.get("code_commit", "uncommitted")),
        seed=int(meta_d.get("seed", 42)),
        created_utc=str(meta_d.get("created_utc", "")),
    )

    profile = _profile_from_dict(manifest["instrument_profile"])
    target_schema = _target_schema_from_dict(manifest["target_schema"])
    model_card = json.loads((p / "model_card.json").read_text(encoding="utf-8"))

    # Compatibility checks (Section 26)
    if expected_device_profile is not None:
        if isinstance(expected_device_profile, InstrumentProfile):
            same_id = expected_device_profile.profile_id == profile.profile_id
            same_grid = (
                len(expected_device_profile.wavelength_grid_nm) == len(profile.wavelength_grid_nm)
                and np.allclose(
                    expected_device_profile.wavelength_grid_nm,
                    profile.wavelength_grid_nm,
                    atol=1e-3,
                    rtol=0.0,
                )
            )
            if not (same_id and same_grid):
                msg = (
                    f"Device profile mismatch: artifact expects '{profile.profile_id}' "
                    f"({profile.n_channels} ch), received '{expected_device_profile.profile_id}' "
                    f"({expected_device_profile.n_channels} ch)."
                )
                if strict_compatibility:
                    raise ArtifactCompatibilityError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
        elif isinstance(expected_device_profile, str):
            if expected_device_profile != profile.profile_id:
                msg = (
                    f"Device profile mismatch: artifact expects '{profile.profile_id}', "
                    f"received '{expected_device_profile}'."
                )
                if strict_compatibility:
                    raise ArtifactCompatibilityError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)

    if expected_calibration_id is not None and expected_calibration_id != metadata.calibration_id:
        msg = (
            f"Calibration ID mismatch: artifact requires '{metadata.calibration_id}', "
            f"received '{expected_calibration_id}'."
        )
        if strict_compatibility:
            raise ArtifactCompatibilityError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # Load `.npz` strictly with allow_pickle=False
    npz_path = p / "preprocess_state.npz"
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            arrays = {k: np.asarray(npz[k]) for k in npz.files}
    except ValueError as exc:
        raise ArtifactCompatibilityError(
            f"Rejected preprocess_state.npz containing object/pickle data: {exc}",
        ) from exc

    pre_spec = manifest["preprocess_spec"]
    chemo_cfg_d = pre_spec["chemometrics"]
    chemo_cfg = PreprocessConfig(
        pipeline_version=str(chemo_cfg_d.get("pipeline_version", "chem_v1")),
        use_absorbance=bool(chemo_cfg_d.get("use_absorbance", True)),
        use_snv=bool(chemo_cfg_d.get("use_snv", True)),
        use_savgol=bool(chemo_cfg_d.get("use_savgol", True)),
        sg_window=int(chemo_cfg_d.get("sg_window", 7)),
        sg_polyorder=int(chemo_cfg_d.get("sg_polyorder", 2)),
        sg_deriv=int(chemo_cfg_d.get("sg_deriv", 1)),
        suppression_mode=chemo_cfg_d.get("suppression_mode", "none"),
        epo_q=chemo_cfg_d.get("epo_q"),
        epo_max_q=int(chemo_cfg_d.get("epo_max_q", 6)),
        glsw_alpha=float(chemo_cfg_d.get("glsw_alpha", 0.1)),
    )
    chemo = ChemometricPipeline(chemo_cfg)
    wl_arr = np.asarray(arrays["wavelength_grid_nm"], dtype=np.float64)
    chemo.wavelength_grid_nm = tuple(float(w) for w in wl_arr)
    chemo.space_signature = str(pre_spec["space_signature"])

    if pre_spec.get("epo") is not None:
        epo_m = pre_spec["epo"]
        epo = EPO(
            EPOConfig(
                q=epo_m.get("q"),
                max_q=int(epo_m.get("max_q", 6)),
                variance_capture_threshold=float(epo_m.get("variance_capture_threshold", 0.90)),
            )
        )
        epo.n_bands = int(epo_m["n_bands"])
        epo.q = int(epo_m["q"])
        epo.V_s = arrays["epo_V_s"]
        epo.Q = arrays["epo_Q"]
        epo.P = arrays["epo_P"]
        epo.singular_values = arrays.get("epo_singular_values")
        epo.explained_diff_variance_ratio = arrays.get("epo_explained_diff_variance_ratio")
        epo.space_signature = str(epo_m["space_signature"])
        epo.calibration_dataset_hash = str(epo_m.get("calibration_dataset_hash", "untracked"))
        chemo.epo = epo

    if pre_spec.get("glsw") is not None:
        glsw_m = pre_spec["glsw"]
        glsw = GLSW(alpha=float(glsw_m["alpha"]))
        glsw.n_bands = int(glsw_m["n_bands"])
        glsw.P = arrays["glsw_P"]
        glsw.space_signature = str(glsw_m["space_signature"])
        chemo.glsw = glsw

    chemo._is_fitted = True

    fcfg_d = pre_spec["fusion_config"]
    f_cfg = FeatureFusionConfig(
        covariate_names=tuple(fcfg_d["covariate_names"]),
        missing_strategy=fcfg_d["missing_strategy"],
        spectral_scaler=fcfg_d["spectral_scaler"],
        covariate_scaler=fcfg_d["covariate_scaler"],
        include_eis=bool(fcfg_d.get("include_eis", False)),
        include_derived=bool(fcfg_d.get("include_derived", False)),
        include_visnir_bands=bool(fcfg_d.get("include_visnir_bands", False)),
        eis_frequencies_hz=tuple(float(f) for f in fcfg_d.get("eis_frequencies_hz", [])),
        eis_protocol_id=fcfg_d.get("eis_protocol_id"),
    )
    feat_profile = _feature_profile_from_dict(pre_spec["feature_profile"])
    fusion = FeatureFusionPipeline(f_cfg, registry=FeatureRegistry())
    fusion.instrument_profile = profile
    fusion.feature_profile = feat_profile

    spec_scaler = _BranchScaler(f_cfg.spectral_scaler)
    spec_scaler.center = arrays["fusion_spectral_center"]
    spec_scaler.scale = arrays["fusion_spectral_scale"]
    cov_scaler = _BranchScaler(f_cfg.covariate_scaler)
    cov_scaler.center = arrays["fusion_covariate_center"]
    cov_scaler.scale = arrays["fusion_covariate_scale"]

    fusion._spectral_scaler = spec_scaler
    fusion._covariate_scaler = cov_scaler
    fusion._covariate_medians = arrays["fusion_covariate_medians"]
    fusion._is_fitted = True

    conformal: ConformalUncertaintyCalibrator | None = None
    if pre_spec.get("conformal") is not None:
        c_meta = pre_spec["conformal"]
        c_cfg_d = c_meta["config"]
        c_cfg = ConformalConfig(
            method=c_cfg_d["method"],
            default_alpha=float(c_cfg_d["default_alpha"]),
            min_scale=float(c_cfg_d["min_scale"]),
            non_negative=bool(c_cfg_d["non_negative"]),
        )
        conformal = ConformalUncertaintyCalibrator(c_cfg)
        conformal.nonconformity_scores_ = arrays["conformal_nonconformity_scores"]
        conformal.n_calibration_samples_ = int(c_meta["n_calibration_samples"])
        conformal._is_fitted = True

    ood: OutOfDomainDetector | None = None
    if pre_spec.get("ood") is not None:
        o_meta = pre_spec["ood"]
        o_cfg_d = o_meta["config"]
        o_cfg = OODConfig(
            mahalanobis_quantile=float(o_cfg_d["mahalanobis_quantile"]),
            knn_k=int(o_cfg_d["knn_k"]),
            knn_quantile=float(o_cfg_d["knn_quantile"]),
            shrinkage=float(o_cfg_d["shrinkage"]),
            envelope_lower_quantile=float(o_cfg_d["envelope_lower_quantile"]),
            envelope_upper_quantile=float(o_cfg_d["envelope_upper_quantile"]),
            envelope_margin_fraction=float(o_cfg_d["envelope_margin_fraction"]),
            fail_on_ood=bool(o_cfg_d["fail_on_ood"]),
        )
        ood = OutOfDomainDetector(o_cfg)
        ood.feature_mean_ = arrays["ood_feature_mean"]
        ood.feature_std_ = arrays["ood_feature_std"]
        ood.z_train_ = arrays["ood_z_train"]
        ood.cho_factor_ = (arrays["ood_cho_L"], bool(o_meta.get("cho_lower", True)))
        ood.mahalanobis_threshold_ = float(o_meta["mahalanobis_threshold"])
        ood.knn_threshold_ = float(o_meta["knn_threshold"])
        env_d = o_meta.get("domain_envelope")
        if env_d is not None:
            ood.domain_envelope_ = DomainEnvelope(
                covariate_bounds={k: (float(v[0]), float(v[1])) for k, v in env_d["covariate_bounds"].items()},
                seen_instrument_profiles=tuple(env_d["seen_instrument_profiles"]),
                seen_moisture_scenarios=tuple(env_d["seen_moisture_scenarios"]),
                seen_texture_classes=tuple(env_d["seen_texture_classes"]),
            )
        ood._is_fitted = True

    weight_path = p / str(manifest["weight_filename"])
    model = _deserialize_model_from_bundle(manifest["model_spec"], weight_path)

    return LoadedArtifact(
        bundle_dir=p,
        manifest=manifest,
        model_card=model_card,
        metadata=metadata,
        instrument_profile=profile,
        target_schema=target_schema,
        chemometric_pipeline=chemo,
        fusion_pipeline=fusion,
        model=model,
        conformal_calibrator=conformal,
        ood_detector=ood,
        feature_names=feat_profile.feature_names,
        background_features=arrays.get("xai_background_X"),
        background_targets=arrays.get("xai_background_y"),
        bundle_sha256=str(manifest["bundle_sha256"]),
    )


def inspect_artifact(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify artifact integrity and return structured manifest & model-card inspection report."""
    manifest = verify_artifact_integrity(bundle_dir)
    p = Path(bundle_dir)
    card = json.loads((p / "model_card.json").read_text(encoding="utf-8"))
    pre = manifest["preprocess_spec"]
    return {
        "integrity_verified": True,
        "bundle_dir": str(p),
        "bundle_sha256": manifest["bundle_sha256"],
        "artifact_format_version": manifest["artifact_format_version"],
        "schema_version": manifest["schema_version"],
        "metadata": manifest["metadata"],
        "instrument_profile_id": manifest["instrument_profile"]["profile_id"],
        "n_spectral_channels": manifest["instrument_profile"]["n_channels"],
        "wavelength_range_nm": manifest["instrument_profile"]["spectral_range_nm"],
        "model_name": manifest["model_spec"]["model_name"],
        "model_family": manifest["model_spec"]["model_family"],
        "weight_format": manifest["model_spec"]["weight_format"],
        "n_fused_features": len(pre["feature_profile"]["descriptors"]),
        "suppression_mode": pre["chemometrics"]["suppression_mode"],
        "has_conformal_calibrator": pre.get("conformal") is not None,
        "has_ood_detector": pre.get("ood") is not None,
        "tracked_files": manifest["files"],
        "model_card": card,
    }
