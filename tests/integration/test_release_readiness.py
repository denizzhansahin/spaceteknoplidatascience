"""Faz 11 release readiness, packaging, documentation, replay, and profiling tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from st_ds.validation.model_card import build_model_card_markdown, validate_model_card_sections
from st_ds.validation.release_checks import (
    SYNTHETIC_DISCLAIMER,
    collect_environment_manifest,
    profile_inference_and_load,
    run_scientific_holdout_check,
    verify_deterministic_replay,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_environment_manifest_includes_pinned_dependencies() -> None:
    env = collect_environment_manifest()
    deps = env["dependencies"]
    for pkg in ("numpy", "scipy", "torch", "sklearn", "pandas", "pydantic", "safetensors"):
        assert pkg in deps, f"Missing dependency '{pkg}' in environment manifest."


def test_scientific_holdout_disjoint_sites_and_model_card(tmp_path: Path) -> None:
    env = collect_environment_manifest()
    holdout = run_scientific_holdout_check(
        seed=42,
        n_samples=60,
        n_sites=6,
        holdout_site_count=2,
        hybrid_epochs=5,
        work_dir=tmp_path,
    )
    assert holdout["disclaimer"] == SYNTHETIC_DISCLAIMER
    assert holdout["is_synthetic"] is True
    assert set(holdout["train_sites"]).isdisjoint(set(holdout["holdout_sites"]))
    assert len(holdout["holdout_sites"]) == 2
    assert len(holdout["oof_ranking"]) >= 2

    metrics = holdout["holdout_metrics"]
    for key in (
        "n_samples",
        "mae",
        "rmse",
        "bias",
        "bland_altman_loa_lower",
        "bland_altman_loa_upper",
    ):
        assert metrics[key] is not None

    md = build_model_card_markdown(
        artifact_id="test_holdout_card",
        semantic_version=env["stds_version"],
        model_name=holdout["selected_model"],
        model_family=holdout["selected_model"],
        instrument_profile_id="visnir_24ch_v1",
        dataset_source="synthetic",
        n_train_samples=holdout["n_train"],
        train_sites=holdout["train_sites"],
        holdout_sites=holdout["holdout_sites"],
        holdout_metrics=metrics,
        environment=env,
        bundle_sha256=holdout["bundle_sha256"],
    )
    assert validate_model_card_sections(md) == []


def test_deterministic_replay_matches_weights_and_predictions() -> None:
    replay = verify_deterministic_replay(seed=42, n_samples=36, hybrid_epochs=4)
    assert replay["passed"] is True
    assert replay["weights_bytes_identical"] is True
    assert replay["max_prediction_diff"] <= 1e-12


def test_performance_profiling_outputs_positive_metrics() -> None:
    prof = profile_inference_and_load(seed=42, n_samples=36, n_warmup=1, n_repeats=5)
    assert prof["cold_load_ms"] > 0.0
    assert prof["single_predict_p50_ms"] > 0.0
    assert prof["single_predict_p95_ms"] >= prof["single_predict_p50_ms"]
    assert prof["batch_ms_per_sample"] > 0.0
    assert prof["peak_tracemalloc_mb"] > 0.0
    assert prof["process_max_rss_mb"] > 0.0


def test_cli_entrypoints_emit_zero_runtime_warnings() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "st_ds.cli", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RuntimeWarning" not in proc.stderr


def test_documentation_and_quickstart_example_exist_and_run() -> None:
    for rel in (
        ".env",
        ".env.example",
        ".gitignore",
        "README.md",
        "docs/TOPRAK_ANALIZ.md",
        "docs/ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md",
        "docs/FIRST_TIME_ROADMAP_GUIDE.md",
        "docs/RELEASE_NOTES.md",
        "docs/MODEL_CARD_TEMPLATE.md",
        "docs/SYNTHETIC_TESTING_GUIDE.md",
        "docs/SYSTEM_AND_USAGE_GUIDE.md",
        "docs/assets/01_spectral_3d_landscape.png",
        "docs/assets/02_chemometrics_epo_comparison.png",
        "docs/assets/03_model_parity_ci90.png",
        "docs/assets/04_xai_ig_and_bands.png",
        "docs/assets/05_spatial_kriging_3d.png",
        "docs/assets/06_covariance_and_clusters_3d.png",
        "scripts/01_generate_synthetic_data.sh",
        "scripts/02_train_and_evaluate.sh",
        "scripts/03_run_full_pipeline.sh",
        "examples/cli_workflow.sh",
        "benchmarks/run_release_checks.py",
    ):
        assert (REPO_ROOT / rel).is_file(), f"Expected file '{rel}' does not exist."

    template_text = (REPO_ROOT / "docs/MODEL_CARD_TEMPLATE.md").read_text(encoding="utf-8")
    assert validate_model_card_sections(template_text) == []

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples/synthetic_python_quickstart.py")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Quickstart failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "Batch Inference: succeeded=2, failed=1" in proc.stdout
