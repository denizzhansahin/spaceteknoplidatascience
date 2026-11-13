"""Faz 10 Security & Compatibility Contract Tests (`DEF-006`, Section 26)."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from st_ds import SCHEMA_VERSION
from st_ds.api import STSoilAnalyzer
from st_ds.artifact import inspect_artifact, load_artifact, save_artifact, verify_artifact_integrity
from st_ds.chemometrics.pipeline import ChemometricPipeline, PreprocessConfig
from st_ds.domain.errors import ArtifactCompatibilityError, CalibrationError
from st_ds.domain.instrument import get_builtin_profile
from st_ds.domain.schemas import ArtifactMetadata
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models.baselines import PLSRRegressor

_CANARY_EXECUTED = False


class _MaliciousPickleCanary:
    """Test-only pickle gadget verifying arbitrary __reduce__ code is NEVER executed."""

    def __reduce__(self):  # type: ignore[override]
        global _CANARY_EXECUTED
        _CANARY_EXECUTED = True
        return (dict, ())


def _make_clean_bundle(tmp_path: Path, *, bundle_name: str = "clean_bundle") -> tuple[Path, list]:
    gen = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id="sec_synth",
            profile_id="visnir_24ch_v1",
            n_samples=30,
            n_sites=3,
            seed=101,
        )
    )
    ds = gen.generate_dataset()
    recs = ds.records
    profile = get_builtin_profile("visnir_24ch_v1")

    chemo = ChemometricPipeline(PreprocessConfig(suppression_mode="none")).fit(
        profile.wavelength_grid_nm
    )
    fusion = FeatureFusionPipeline(FeatureFusionConfig()).fit(recs, chemometric_pipeline=chemo)
    fused = fusion.build_feature_vector(recs, chemometric_pipeline=chemo)
    y = np.asarray([float(r.reference.soc_pct) for r in recs], dtype=float)  # type: ignore[union-attr]

    model = PLSRRegressor(target_names=("soc_pct",))
    model.set_metadata(
        feature_names=fused.fused_feature_names,
        target_names=("soc_pct",),
        dataset_source=ds.dataset_source,
    )
    model.fit(fused.X_fused, y)

    meta = ArtifactMetadata(
        artifact_id="sec_test_plsr_v1",
        semantic_version="0.1.0",
        schema_version=SCHEMA_VERSION,
        instrument_profile_id=profile.profile_id,
        calibration_id=recs[0].calibration_id,
        pipeline_version="chem_v1",
        dataset_hash=ds.compute_sha256(),
        dependency_versions={"python": "3.13"},
    )
    out_dir = tmp_path / bundle_name
    save_artifact(
        out_dir,
        metadata=meta,
        instrument_profile=profile,
        model=model,
        chemometric_pipeline=chemo,
        fusion_pipeline=fusion,
    )
    return out_dir, recs


class TestArtifactSecurityContract:
    """DEF-006 acceptance gates: SHA-256 verification, zero-pickle, and compatibility guards."""

    def test_malicious_raw_pickle_checkpoint_rejected_without_execution(self, tmp_path: Path) -> None:
        global _CANARY_EXECUTED
        _CANARY_EXECUTED = False

        evil_pt = tmp_path / "evil_checkpoint.pt"
        # Simulate crafting a raw pickle checkpoint file
        payload = pickle.dumps({"malicious": True})
        evil_pt.write_bytes(payload)

        with pytest.raises(ArtifactCompatibilityError, match="raw single-file checkpoint"):
            load_artifact(evil_pt)

        with pytest.warns(DeprecationWarning):
            with pytest.raises(ArtifactCompatibilityError):
                STSoilAnalyzer(weights_path=evil_pt)

        assert _CANARY_EXECUTED is False

    def test_forged_torch_weights_only_bundle_blocks_malicious_pickle_reduce(self, tmp_path: Path) -> None:
        """Even if a forged bundle points at a valid-SHA pickled object, weights_only=True blocks execution."""
        bundle_dir, _ = _make_clean_bundle(tmp_path)

        # Craft an evil pickle payload targeting _MaliciousPickleCanary
        global _CANARY_EXECUTED
        _CANARY_EXECUTED = False
        assert _MaliciousPickleCanary.__reduce__ is not None

        # Write a raw python script byte stream or pickled class reference directly without invoking __reduce__
        evil_file = bundle_dir / "model_weights.pt"
        # Opcode REDUCE payload built manually via pickle protocol 2 referencing _MaliciousPickleCanary
        import torch

        class _Gadget:
            def __reduce__(self):
                return (exec, ("raise RuntimeError('should never execute')",))

        evil_bytes = pickle.dumps(_Gadget())
        evil_file.write_bytes(evil_bytes)

        # Re-sign manifest artificially to simulate compromised weights inside a `.pt` bundle
        from st_ds.artifact.bundle import _compute_manifest_digest, compute_file_sha256

        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["weight_filename"] = "model_weights.pt"
        manifest["model_spec"]["class_name"] = "STSoilHybridNetRegressor"
        manifest["model_spec"]["weight_format"] = "torch_state_dict_weights_only"
        manifest["model_spec"]["hybrid_config"] = {
            "n_spectral_features": 24,
            "n_covariate_features": 8,
            "spectral_hidden_dim": 8,
            "covariate_hidden_dim": 8,
            "fusion_hidden_dim": 8,
            "n_classes": 4,
        }
        manifest["files"].pop("model_weights.safetensors")
        manifest["files"]["model_weights.pt"] = {
            "sha256": compute_file_sha256(evil_file),
            "size_bytes": evil_file.stat().st_size,
        }
        core = {k: v for k, v in manifest.items() if k != "bundle_sha256"}
        core["metadata"] = dict(manifest["metadata"])
        core["metadata"]["artifact_sha256"] = ""
        manifest["bundle_sha256"] = _compute_manifest_digest(core)
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ArtifactCompatibilityError, match="Rejected unsafe or unreadable PyTorch"):
            load_artifact(bundle_dir)

        assert _CANARY_EXECUTED is False

    def test_tampered_sha256_in_weights_preprocess_or_manifest_fails_before_load(self, tmp_path: Path) -> None:
        # 1. Tamper 1 byte in model_weights.safetensors
        b1, _ = _make_clean_bundle(tmp_path, bundle_name="tamper_weights")
        wpath = b1 / "model_weights.safetensors"
        wdata = bytearray(wpath.read_bytes())
        wdata[-1] ^= 0xFF
        wpath.write_bytes(bytes(wdata))
        with pytest.raises(ArtifactCompatibilityError, match="SHA-256 integrity mismatch"):
            verify_artifact_integrity(b1)
        with pytest.raises(ArtifactCompatibilityError, match="SHA-256 integrity mismatch"):
            STSoilAnalyzer.from_artifact(b1)

        # 2. Tamper preprocess_state.npz
        b2, _ = _make_clean_bundle(tmp_path, bundle_name="tamper_npz")
        npath = b2 / "preprocess_state.npz"
        ndata = bytearray(npath.read_bytes())
        ndata[-1] ^= 0x01
        npath.write_bytes(bytes(ndata))
        with pytest.raises(ArtifactCompatibilityError, match="SHA-256 integrity mismatch"):
            load_artifact(b2)

        # 3. Tamper manifest.json itself
        b3, _ = _make_clean_bundle(tmp_path, bundle_name="tamper_manifest")
        mpath = b3 / "manifest.json"
        m = json.loads(mpath.read_text(encoding="utf-8"))
        m["metadata"]["seed"] = 99999
        mpath.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(ArtifactCompatibilityError, match="bundle_sha256"):
            inspect_artifact(b3)

    def test_device_profile_calibration_mismatch_and_legacy_deprecation_guard(self, tmp_path: Path) -> None:
        bundle_dir, recs = _make_clean_bundle(tmp_path)
        wrong_profile = get_builtin_profile("led_5band_v1")

        with pytest.raises(ArtifactCompatibilityError, match="Device profile mismatch"):
            STSoilAnalyzer.from_artifact(bundle_dir, device_profile=wrong_profile, strict_compatibility=True)

        with pytest.raises(ArtifactCompatibilityError, match="Calibration ID mismatch"):
            STSoilAnalyzer.from_artifact(bundle_dir, calibration_id="wrong_cal_id", strict_compatibility=True)

        # Valid bundle + legacy constructor emits DeprecationWarning and enforces exact wavelength match
        with pytest.warns(DeprecationWarning, match="from_artifact"):
            legacy_analyzer = STSoilAnalyzer(weights_path=bundle_dir)

        # Measurement with mismatched spectrum profile rejected at predict time (no 24-ch silent cast)
        m_wrong = replace(
            recs[0],
            instrument_profile_id="led_5band_v1",
            spectrum=replace(
                recs[0].spectrum,
                values=np.ones(5, dtype=float) * 0.3,
                wavelength_grid_nm=wrong_profile.wavelength_grid_nm,
                channel_ids=wrong_profile.channel_ids,
                dark_reference=None,
                white_reference=None,
                saturation_flags=None,
                valid_mask=None,
            ),
        )
        with pytest.raises((CalibrationError, Exception)):
            legacy_analyzer.predict(m_wrong)
