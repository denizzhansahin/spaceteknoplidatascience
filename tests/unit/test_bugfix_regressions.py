"""Regression tests covering critical, contract, and edge-case bug fixes across st_ds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from st_ds.api.analyzer import STSoilAnalyzer
from st_ds.artifact.bundle import collect_dependency_versions, load_artifact, save_artifact
from st_ds.calibration.wavelength import align_wavelengths
from st_ds.carbon.fractions import PROVISIONAL_CLASS_LABELS, classify_soil_provisional
from st_ds.chemometrics.pipeline import ChemometricPipeline, PreprocessConfig
from st_ds.domain.errors import SchemaError
from st_ds.domain.instrument import get_builtin_profile
from st_ds.domain.protocols import AnalyzerProtocol
from st_ds.domain.schemas import ArtifactMetadata, TargetLabelSchema
from st_ds.domain.units import UnitRegistry
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io.reference_store import ReferenceLabelStore
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models.baselines import PLSRRegressor
from st_ds.models.hybrid_net import STSoilHybridNetRegressor
from st_ds.models.loss import MultiTaskLossConfig, STSoilMultiTaskLoss
from st_ds.reporting.pdf_report import generate_prediction_pdf_report
from st_ds.signal.qc import evaluate_signal_qc
from st_ds.visualization.base import VisualizationConfig
from st_ds.visualization.model_plots import ModelPlotContext, plot_09_parity_and_ci90
from st_ds.visualization.suite import extract_dataset_arrays


def _build_synthetic_bundle(
    tmp_path: Path,
    *,
    target_name: str = "soc_pct",
    model_cls: str = "plsr",
    mc_dropout_samples: int = 1,
    multi_target: bool = False,
):
    ds = SyntheticGeneratorV2(
        SyntheticConfig(
            dataset_id=f"reg_{target_name}_{model_cls}",
            n_samples=24,
            n_sites=4,
            seed=17,
            profile_id="visnir_24ch_v1",
        )
    ).generate_dataset()
    profile = ds.instrument_profile or get_builtin_profile("visnir_24ch_v1")
    chemo = ChemometricPipeline(PreprocessConfig(suppression_mode="none")).fit(profile.wavelength_grid_nm)
    fusion = FeatureFusionPipeline(FeatureFusionConfig()).fit(ds.records, chemometric_pipeline=chemo, profile=profile)
    fused = fusion.build_feature_vector(ds.records, chemometric_pipeline=chemo)

    y_soc = np.asarray([float(r.reference.soc_pct) for r in ds.records], dtype=float)
    y_sic = np.asarray([float(r.reference.sic_pct) for r in ds.records], dtype=float)
    y_tc = np.asarray([float(r.reference.tc_pct) for r in ds.records], dtype=float)

    target_map = {"soc_pct": y_soc, "sic_pct": y_sic, "tc_pct": y_tc}
    if multi_target:
        target_names = ("soc_pct", "sic_pct", "tc_pct")
        y_bg = np.column_stack([y_soc, y_sic, y_tc])
    else:
        target_names = (target_name,)
        y_bg = target_map[target_name]

    if model_cls == "hybrid":
        model = STSoilHybridNetRegressor(
            n_spectral_features=fused.X_spectral.shape[1],
            epochs=6,
            dropout=0.1,
            mc_dropout_samples=mc_dropout_samples,
            target_names=target_names,
        )
        model.set_metadata(
            feature_names=fused.fused_feature_names,
            target_names=target_names,
            dataset_source=ds.dataset_source,
        )
        if multi_target:
            model.fit_multitask(fused.X_spectral, fused.X_covariates, soc=y_soc, sic=y_sic, tc=y_tc)
        else:
            model.fit(fused.X_fused, target_map[target_name])
    else:
        model = PLSRRegressor(max_components=4, target_names=target_names)
        model.set_metadata(
            feature_names=fused.fused_feature_names,
            target_names=target_names,
            dataset_source=ds.dataset_source,
        )
        model.fit(fused.X_fused, y_bg)

    bundle_dir = tmp_path / f"bundle_{target_name}_{model_cls}_{int(multi_target)}"
    meta = ArtifactMetadata(
        artifact_id=f"reg_{target_name}_{model_cls}",
        semantic_version="0.1.0",
        schema_version="1.0.0",
        instrument_profile_id=profile.profile_id,
        calibration_id=ds.records[0].calibration_id,
        pipeline_version=chemo.config.pipeline_version,
        dataset_hash=ds.compute_sha256(),
        dependency_versions=collect_dependency_versions(),
    )
    save_artifact(
        bundle_dir,
        metadata=meta,
        instrument_profile=profile,
        model=model,
        chemometric_pipeline=chemo,
        fusion_pipeline=fusion,
        background_features=fused.X_fused,
        background_targets=y_bg,
    )
    return ds, bundle_dir


def test_single_target_tc_and_sic_do_not_leak_into_soc(tmp_path: Path) -> None:
    ds_tc, tc_dir = _build_synthetic_bundle(tmp_path, target_name="tc_pct", model_cls="plsr")
    an_tc = STSoilAnalyzer.from_artifact(tc_dir)
    rep_tc = an_tc.predict(ds_tc.records[0])
    assert rep_tc.carbon_metrics["primary_target"] == "tc_pct"
    assert rep_tc.carbon_metrics["tc_pct"] is not None
    assert rep_tc.carbon_metrics["soc_pct"] is None
    assert rep_tc.carbon_metrics["soc_std_pct"] is None

    ds_sic, sic_dir = _build_synthetic_bundle(tmp_path, target_name="sic_pct", model_cls="plsr")
    an_sic = STSoilAnalyzer.from_artifact(sic_dir)
    rep_sic = an_sic.predict(ds_sic.records[0])
    assert rep_sic.carbon_metrics["primary_target"] == "sic_pct"
    assert rep_sic.carbon_metrics["sic_pct"] is not None
    assert rep_sic.carbon_metrics["soc_pct"] is None
    assert rep_sic.carbon_metrics["soc_std_pct"] is None


def test_hybrid_net_masked_loss_zero_and_multitask_roundtrip(tmp_path: Path) -> None:
    criterion = STSoilMultiTaskLoss(MultiTaskLossConfig())
    outputs = {
        "soc_mean": torch.tensor([1.0, 2.0]),
        "soc_var": torch.tensor([0.1, 0.1]),
        "sic_mean": torch.tensor([0.5, 0.5]),
        "sic_var": torch.tensor([0.1, 0.1]),
        "tc_aux_mean": torch.tensor([1.5, 2.5]),
        "class_logits": torch.zeros((2, 4)),
    }
    _, bd = criterion(
        outputs,
        soc_target=torch.tensor([1.0, 2.0]),
        soc_mask=torch.tensor([True, True]),
        sic_target=torch.tensor([9999.0, 9999.0]),
        sic_mask=torch.tensor([False, False]),
        tc_target=torch.tensor([9999.0, 9999.0]),
        tc_mask=torch.tensor([False, False]),
    )
    assert bd["loss_sic_nll"] == 0.0
    assert bd["loss_sic_huber"] == 0.0
    assert bd["loss_tc"] == 0.0

    ds, bundle_dir = _build_synthetic_bundle(
        tmp_path,
        target_name="soc_pct",
        model_cls="hybrid",
        mc_dropout_samples=5,
        multi_target=True,
    )
    loaded = load_artifact(bundle_dir)
    assert isinstance(loaded.model, STSoilHybridNetRegressor)
    assert loaded.model.mc_dropout_samples == 5
    assert loaded.background_targets is not None
    assert loaded.background_targets.ndim == 2
    assert loaded.background_targets.shape[1] == 3

    analyzer = STSoilAnalyzer(loaded_artifact=loaded)
    xai_rep = analyzer.explain(ds.records[0])
    assert xai_rep.dataset_dependence_hsic.n_samples == loaded.background_features.shape[0]


def test_batch_pdf_handles_errors_key_and_none_records(tmp_path: Path) -> None:
    ds, bundle_dir = _build_synthetic_bundle(tmp_path, target_name="soc_pct", model_cls="plsr")
    analyzer = STSoilAnalyzer.from_artifact(bundle_dir)

    led_prof = get_builtin_profile("led_5band_v1")
    bad_rec = replace(
        ds.records[1],
        sample_id="BAD_SAMPLE_01",
        instrument_profile_id=led_prof.profile_id,
        spectrum=replace(
            ds.records[1].spectrum,
            values=ds.records[1].spectrum.values[: led_prof.n_channels],
            wavelength_grid_nm=led_prof.wavelength_grid_nm,
            channel_ids=led_prof.channel_ids,
            dark_reference=None,
            white_reference=None,
            valid_mask=None,
            saturation_flags=None,
        ),
    )
    batch = analyzer.predict_batch([ds.records[0], bad_rec, ds.records[2]], fail_fast=False)
    assert batch.n_succeeded == 2
    assert batch.n_failed == 1

    payload = batch.to_dict()
    assert None in payload["results"]
    assert len(payload["errors"]) == 1

    pdf_out = tmp_path / "batch_partial.pdf"
    generate_prediction_pdf_report(payload, pdf_out, lang="en")
    assert pdf_out.is_file() and pdf_out.stat().st_size > 1000


def test_provenance_tracks_input_and_training_sources_separately(tmp_path: Path) -> None:
    ds, bundle_dir = _build_synthetic_bundle(tmp_path, target_name="soc_pct", model_cls="plsr")
    analyzer = STSoilAnalyzer.from_artifact(bundle_dir)

    real_meas = replace(ds.records[0], dataset_source="real_lab")
    rep = analyzer.predict(real_meas)
    assert rep.identity["dataset_source"] == "real_lab"
    assert rep.provenance["training_dataset_source"] == "synthetic"
    assert rep.provenance["measurement_dataset_source"] == "real_lab"

    pdf_out = tmp_path / "mixed_provenance.pdf"
    generate_prediction_pdf_report(rep.to_dict(), pdf_out, lang="en")
    assert pdf_out.is_file() and pdf_out.stat().st_size > 1000


def test_provisional_taxonomy_and_analyzer_protocol(tmp_path: Path) -> None:
    schema = TargetLabelSchema()
    assert set(schema.class_labels) == set(PROVISIONAL_CLASS_LABELS)
    for soc, sic in ((0.5, 0.5), (0.5, 3.0), (5.0, 0.5), (15.0, 1.0)):
        cls = classify_soil_provisional(soc, sic)
        assert cls.label in schema.class_labels

    ds, bundle_dir = _build_synthetic_bundle(tmp_path, target_name="soc_pct", model_cls="plsr")
    analyzer = STSoilAnalyzer.from_artifact(bundle_dir)
    assert isinstance(analyzer, AnalyzerProtocol)

    batch = analyzer.predict_batch(ds.records[:6], include_xai=True)
    assert batch.batch_xai_hsic is not None
    assert batch.batch_xai_hsic["n_samples"] == 6


def test_reference_store_csv_strict_boolean_parsing(tmp_path: Path) -> None:
    bad_dm_csv = tmp_path / "bad_dm.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "soc_pct": 1.5,
                "sic_pct": 0.4,
                "tc_pct": 1.9,
                "dry_matter_basis": "false",
            }
        ]
    ).to_csv(bad_dm_csv, index=False)
    with pytest.raises(SchemaError):
        ReferenceLabelStore.load_csv(bad_dm_csv)

    no_derive_csv = tmp_path / "no_derive.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "S2",
                "sic_pct": 1.0,
                "tc_pct": 2.5,
                "is_soc_derived_from_difference": "false",
                "dry_matter_basis": "true",
            }
        ]
    ).to_csv(no_derive_csv, index=False)
    store = ReferenceLabelStore.load_csv(no_derive_csv)
    assert store.labels["S2"].soc_pct is None


def test_edge_cases_units_qc_wavelength_baselines_and_plots(tmp_path: Path) -> None:
    assert UnitRegistry.to_canonical("impedance", 500.0, "mΩ") == pytest.approx(0.5)
    assert UnitRegistry.to_canonical("impedance", 2.0, "MΩ") == pytest.approx(2e6)

    prof = get_builtin_profile("led_5band_v1")
    from st_ds.domain.schemas import SpectralSignalMode, Spectrum

    spec_2d = Spectrum(
        values=np.full((3, prof.n_channels), 0.4),
        wavelength_grid_nm=prof.wavelength_grid_nm,
        channel_ids=prof.channel_ids,
        mode=SpectralSignalMode.REFLECTANCE,
        valid_mask=tuple(False for _ in range(prof.n_channels)),
    )
    qc_rep = evaluate_signal_qc(spec_2d)
    assert np.isfinite(qc_rep.snr_estimate)
    assert qc_rep.snr_estimate == 0.0

    # Wavelength alignment preserves valid_mask on exact pos - 1 tolerance match
    shifted_wl = tuple(w + 1e-6 for w in prof.wavelength_grid_nm)
    mask_in = (True, False, True, True, True)
    spec_shift = Spectrum(
        values=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
        wavelength_grid_nm=shifted_wl,
        channel_ids=prof.channel_ids,
        mode=SpectralSignalMode.REFLECTANCE,
        valid_mask=mask_in,
    )
    aligned = align_wavelengths(spec_shift, prof)
    assert aligned.spectrum.valid_mask == mask_in

    # Small-sample PLSR fit with 3 samples doesn't crash
    reg = PLSRRegressor(max_components=3, inner_cv_folds=3)
    reg.fit(np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]]), np.array([1.0, 2.0, 3.0]))
    assert reg.predict(np.array([[0.2, 0.4]])).shape == (1,)

    # Constant y_true parity plot doesn't crash with LinAlgError
    ds = SyntheticGeneratorV2(SyntheticConfig(dataset_id="const_plot", n_samples=6, n_sites=2, seed=5)).generate_dataset()
    arrays = extract_dataset_arrays(ds)
    ctx = ModelPlotContext(
        target_name="soc_pct",
        y_true=np.ones(6, dtype=float),
        y_pred=np.linspace(0.9, 1.1, 6),
        pred_std=np.full(6, 0.1),
        ci_lower=np.full(6, 0.8),
        ci_upper=np.full(6, 1.2),
        benchmark=None,
        best_entry=None,
        reports=tuple([None] * 6),
        batch_result=None,
        fused_features=None,
        ood_batch=None,
        conformal_calibrator=None,
    )
    rec = plot_09_parity_and_ci90(
        ctx,
        arrays,
        plots_dir=tmp_path,
        config=VisualizationConfig(formats=("png",)),
        dataset_source="synthetic",
    )
    assert len(rec.files) >= 1 and Path(rec.files[0]).is_file()
