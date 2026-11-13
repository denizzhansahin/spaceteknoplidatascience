"""Integration tests for flexible targets, explicit CLI model/covariate selection, and custom CSV/profile loading."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from st_ds.api.analyzer import STSoilAnalyzer
from st_ds.domain.errors import SchemaError
from st_ds.domain.instrument import InstrumentProfile, SpectralModality, resolve_instrument_profile
from st_ds.domain.quality import QualityFlag
from st_ds.domain.schemas import (
    CovariateSet,
    Measurement,
    SoilReference,
    SpatialCoordinates,
    SpectralSignalMode,
    Spectrum,
)
from st_ds.features.fusion import FeatureFusionConfig, FeatureFusionPipeline
from st_ds.io.dataset import (
    SoilDataset,
    load_dataset_csv,
    load_dataset_json,
    save_dataset_csv,
    save_dataset_json,
)
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.models.baselines import PLSRRegressor
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import STSoilTrainer, TrainerConfig
from st_ds.visualization import VisualizationConfig, generate_full_visualization_suite

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_custom_food_dataset(tmp_path: Path) -> tuple[Path, Path, SoilDataset, InstrumentProfile]:
    wl_grid = tuple(float(w) for w in range(700, 1021, 40))
    ch_ids = tuple(f"BAND_{int(w)}" for w in wl_grid)
    profile = InstrumentProfile(
        profile_id="test_food_nir_9ch_v1",
        sensor_model="TestFoodNIR9",
        modality=SpectralModality.VIS_NIR_SWIR,
        wavelength_grid_nm=wl_grid,
        channel_ids=ch_ids,
        fwhm_nm=tuple(15.0 for _ in wl_grid),
        metadata={"domain": "food_protein"},
    )
    profile_json = tmp_path / "custom_food_profile.json"
    profile_json.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    rng = np.random.default_rng(314)
    wl_arr = np.asarray(wl_grid)
    rows: list[dict[str, object]] = []
    for i in range(24):
        site = f"silo_{i // 6 + 1:02d}"
        protein = 9.0 + 0.25 * i + float(rng.normal(0.0, 0.1))
        moisture = 10.0 + 0.4 * (i % 4)
        absorb = (
            0.20
            + 0.025 * protein * np.exp(-0.5 * ((wl_arr - 900.0) / 35.0) ** 2)
            + 0.010 * moisture * np.exp(-0.5 * ((wl_arr - 980.0) / 25.0) ** 2)
        )
        refl = np.clip(10.0 ** (-absorb) + rng.normal(0.0, 0.001, size=wl_arr.size), 0.02, 0.98)
        row: dict[str, object] = {
            "sample_id": f"GRAIN_{i + 1:03d}",
            "site_id": site,
            "timestamp_utc": "2026-11-12T10:00:00Z",
            "instrument_id": "food_nir_01",
            "instrument_profile_id": profile.profile_id,
            "calibration_id": "food_cal_v1",
            "dataset_source": "real_lab",
            "temperature_c": round(21.0 + 0.3 * (i % 5), 2),
            "moisture_volumetric_pct": round(moisture, 2),
            "lat": 37.8 + 0.004 * (i // 6) + 0.0005 * (i % 6),
            "lon": 32.4 + 0.005 * (i % 6),
            "protein_pct": round(protein, 3),
        }
        for w, val in zip(wl_grid, refl):
            row[f"wl_{w:.2f}nm"] = round(float(val), 6)
        rows.append(row)

    csv_path = tmp_path / "food_protein.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    ds = load_dataset_csv(csv_path, profile=profile, target_columns=["protein_pct"])
    return csv_path, profile_json, ds, profile


def test_custom_target_and_profile_roundtrip(tmp_path: Path) -> None:
    csv_path, profile_json, ds, profile = _make_custom_food_dataset(tmp_path)
    resolved = resolve_instrument_profile(str(profile_json))
    assert resolved.profile_id == profile.profile_id
    assert resolved.wavelength_grid_nm == profile.wavelength_grid_nm

    with pytest.raises(SchemaError):
        resolve_instrument_profile("nonexistent_profile_xyz")

    assert ds.records[0].reference is not None
    assert ds.records[0].reference.soc_pct is None
    assert ds.records[0].reference.get_target("protein_pct") is not None

    # CSV round-trip with companion .meta.json
    saved_csv = tmp_path / "roundtrip.csv"
    save_dataset_csv(ds, saved_csv)
    reloaded_csv = load_dataset_csv(saved_csv)
    assert reloaded_csv.records[0].reference is not None
    assert reloaded_csv.records[0].reference.get_target("protein_pct") == pytest.approx(
        ds.records[0].reference.get_target("protein_pct")
    )

    # JSON round-trip
    saved_json = tmp_path / "roundtrip.json"
    save_dataset_json(ds, saved_json)
    reloaded_json = load_dataset_json(saved_json)
    assert reloaded_json.records[5].reference is not None
    assert reloaded_json.records[5].reference.get_target("protein_pct") == pytest.approx(
        ds.records[5].reference.get_target("protein_pct")
    )
    assert csv_path.exists()


def test_custom_target_training_and_prediction_no_soil_warnings(tmp_path: Path) -> None:
    _, profile_json, ds, _ = _make_custom_food_dataset(tmp_path)
    fusion_cfg = FeatureFusionConfig(covariate_names=("moisture_volumetric_pct", "temperature_c"))
    trainer = STSoilTrainer(
        TrainerConfig(target_name="protein_pct", fusion_config=fusion_cfg, seed=42)
    )
    folds = GroupedSplitBuilder.group_kfold_split(ds, n_splits=4, group_by="site_id")
    bench = trainer.run_benchmark(ds, split_folds=folds, include_optional_models=False)
    assert bench.target_name == "protein_pct"
    assert "plsr" in bench.entries
    assert bench.entries["plsr"].oof_metrics.r2 > 0.7

    # Train via CLI with custom profile JSON, --covariates, --model plsr
    bundle_dir = tmp_path / "food_bundle"
    csv_path = tmp_path / "food_protein.csv"
    train_pdf = tmp_path / "food_train.pdf"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "st_ds.cli",
            "train",
            "--input",
            str(csv_path),
            "--instrument-profile",
            str(profile_json),
            "--target",
            "protein_pct",
            "--covariates",
            "moisture_volumetric_pct,temperature_c",
            "--model",
            "plsr",
            "--folds",
            "4",
            "--pdf",
            str(train_pdf),
            "--lang",
            "tr",
            "--output",
            str(bundle_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    train_payload = json.loads(res.stdout)
    assert train_payload["selected_model"] == "plsr"
    assert train_payload["target"] == "protein_pct"
    assert train_pdf.exists() and train_pdf.stat().st_size > 1000

    # Verify single prediction suppresses soil-only carbonate / bulk_density flags
    analyzer = STSoilAnalyzer.from_artifact(bundle_dir)
    pred_report = analyzer.predict(ds.records[0], include_xai=False, include_stock=True)
    assert "protein_pct" in pred_report.carbon_metrics
    assert QualityFlag.CARBONATE_BAND_UNAVAILABLE not in pred_report.quality.flags
    assert QualityFlag.MISSING_BULK_DENSITY not in pred_report.quality.flags


def test_cli_tc_plsr_target_and_model_flexibility(tmp_path: Path) -> None:
    gen = SyntheticGeneratorV2(SyntheticConfig(dataset_id="tc_plsr_test_v1", n_samples=32, n_sites=4, seed=19))
    ds = gen.generate_dataset()
    ds_json = tmp_path / "soil_ds.json"
    save_dataset_json(ds, ds_json)

    viz_dir = tmp_path / "viz_tc_plsr"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "st_ds.cli",
            "visualize",
            "--input",
            str(ds_json),
            "--target",
            "tc_pct",
            "--model",
            "plsr",
            "--folds",
            "4",
            "--dpi",
            "85",
            "--no-interactive-3d",
            "--lang",
            "en",
            "--output",
            str(viz_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(res.stdout)
    assert payload["command"] == "visualize"
    pdf_path = Path(payload["pdf_report_path"])
    assert pdf_path.exists() and pdf_path.stat().st_size > 100_000

    manifest = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["target_name"] == "tc_pct"
    html_text = Path(payload["dashboard_path"]).read_text(encoding="utf-8")
    assert "plsr" in html_text
    assert "tc_pct" in html_text


def test_new_examples_syntax_valid() -> None:
    bash_script = REPO_ROOT / "examples" / "01_all_cli_commands_and_params.sh"
    res = subprocess.run(["bash", "-n", str(bash_script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    for py_name in ("02_all_python_api_parameters.py", "03_custom_csv_and_non_soil_domains.py"):
        src = (REPO_ROOT / "examples" / py_name).read_text(encoding="utf-8")
        ast.parse(src, filename=py_name)
