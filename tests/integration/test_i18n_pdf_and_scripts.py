"""Integration tests for optional .env TR/EN i18n, PDF report export, bash pipelines & .gitignore."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from st_ds.config import get_runtime_config, load_dotenv_dict, resolve_lang
from st_ds.io.dataset import save_dataset_json
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2
from st_ds.visualization import VisualizationConfig, generate_full_visualization_suite

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_optional_dotenv_precedence_and_no_dependency(tmp_path: Path, monkeypatch) -> None:
    # 1) When .env file does NOT exist and STDS_LANG is unset -> safe fallback to 'tr'
    monkeypatch.delenv("STDS_LANG", raising=False)
    missing_env = tmp_path / "nonexistent.env"
    parsed_missing, found_missing = load_dotenv_dict(missing_env)
    assert parsed_missing == {}
    assert found_missing is None
    assert resolve_lang(None, env_file=missing_env) == "tr"

    # 2) When .env has STDS_LANG=en -> resolves to 'en'
    custom_env = tmp_path / ".env"
    custom_env.write_text("STDS_LANG=en\nSTDS_DPI=180\nSTDS_PDF_EXPORT=true\n", encoding="utf-8")
    assert resolve_lang(None, env_file=custom_env) == "en"
    cfg = get_runtime_config(env_file=custom_env)
    assert cfg.lang == "en"
    assert cfg.dpi == 180
    assert cfg.pdf_export is True

    # 3) OS environment beats .env, explicit param beats both
    monkeypatch.setenv("STDS_LANG", "tr")
    assert resolve_lang(None, env_file=custom_env) == "tr"
    assert resolve_lang("en", env_file=custom_env) == "en"


def test_bilingual_visualization_and_multipage_pdf_generation(tmp_path: Path) -> None:
    gen = SyntheticGeneratorV2(SyntheticConfig(dataset_id="pdf_i18n_v1", n_samples=30, n_sites=5, seed=42))
    ds = gen.generate_dataset()
    wet_ds, dry_ds = gen.generate_paired_wet_dry_dataset(n_pairs=10)

    # English run
    out_en = tmp_path / "viz_en"
    rep_en = generate_full_visualization_suite(
        ds,
        out_en,
        wet_dataset=wet_ds,
        dry_dataset=dry_ds,
        config=VisualizationConfig(
            n_folds=3,
            dpi=90,
            include_interactive_3d=False,
            include_pdf_report=True,
            lang="en",
        ),
    )
    assert rep_en.lang == "en"
    assert rep_en.n_plots == 26
    assert rep_en.pdf_report_path is not None
    pdf_en = Path(rep_en.pdf_report_path)
    assert pdf_en.is_file()
    raw_bytes_en = pdf_en.read_bytes()
    assert raw_bytes_en.startswith(b"%PDF-")
    assert len(raw_bytes_en) > 100_000  # Multi-page PDF embedding 26 PNGs
    dash_en = Path(rep_en.dashboard_path).read_text(encoding="utf-8")
    assert "Open Complete PDF Report" in dash_en
    assert "SYNTHETIC DATA VERIFICATION OUTPUT" in dash_en

    # Turkish run
    out_tr = tmp_path / "viz_tr"
    rep_tr = generate_full_visualization_suite(
        ds,
        out_tr,
        wet_dataset=wet_ds,
        dry_dataset=dry_ds,
        config=VisualizationConfig(
            n_folds=3,
            dpi=90,
            include_interactive_3d=False,
            include_pdf_report=True,
            lang="tr",
        ),
    )
    assert rep_tr.lang == "tr"
    assert rep_tr.pdf_report_path is not None
    assert Path(rep_tr.pdf_report_path).read_bytes().startswith(b"%PDF-")
    dash_tr = Path(rep_tr.dashboard_path).read_text(encoding="utf-8")
    assert "Tam Kapsamlı PDF Raporunu Aç" in dash_tr


def test_cli_train_predict_report_export_pdf_and_lang(tmp_path: Path) -> None:
    gen = SyntheticGeneratorV2(SyntheticConfig(dataset_id="cli_pdf_v1", n_samples=30, n_sites=5, seed=42))
    ds = gen.generate_dataset()
    ds_path = tmp_path / "ds.json"
    save_dataset_json(ds, ds_path)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    bundle_dir = tmp_path / "bundle"
    train_pdf = tmp_path / "train.pdf"
    p_train = subprocess.run(
        [
            sys.executable,
            "-m",
            "st_ds.cli",
            "train",
            "--input",
            str(ds_path),
            "--output",
            str(bundle_dir),
            "--folds",
            "3",
            "--hybrid-epochs",
            "4",
            "--lang",
            "en",
            "--pdf",
            str(train_pdf),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_train.returncode == 0, p_train.stderr
    assert train_pdf.is_file() and train_pdf.read_bytes().startswith(b"%PDF-")

    pred_json = tmp_path / "pred.json"
    pred_pdf = tmp_path / "pred.pdf"
    p_pred = subprocess.run(
        [
            sys.executable,
            "-m",
            "st_ds.cli",
            "predict",
            "--artifact",
            str(bundle_dir),
            "--input",
            str(ds_path),
            "--output",
            str(pred_json),
            "--pdf",
            str(pred_pdf),
            "--lang",
            "en",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_pred.returncode == 0, p_pred.stderr
    assert pred_pdf.is_file() and pred_pdf.read_bytes().startswith(b"%PDF-")

    exported_json = tmp_path / "exported.json"
    exported_pdf = tmp_path / "exported.pdf"
    p_exp = subprocess.run(
        [
            sys.executable,
            "-m",
            "st_ds.cli",
            "report",
            "export",
            "--input",
            str(pred_json),
            "--output",
            str(exported_json),
            "--format",
            "both",
            "--pdf-output",
            str(exported_pdf),
            "--lang",
            "tr",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_exp.returncode == 0, p_exp.stderr
    exp_summary = json.loads(p_exp.stdout)
    assert exp_summary["lang"] == "tr"
    assert exp_summary["format"] == "both"
    assert exported_json.is_file()
    assert exported_pdf.is_file() and exported_pdf.read_bytes().startswith(b"%PDF-")


def test_scripts_syntax_and_gitignore_rules() -> None:
    scripts = sorted((REPO_ROOT / "scripts").glob("*.sh"))
    assert len(scripts) == 3
    for sc in scripts:
        assert os.access(sc, os.X_OK), f"Script must be executable: {sc}"
        chk = subprocess.run(["bash", "-n", str(sc)], capture_output=True, text=True, check=False)
        assert chk.returncode == 0, f"bash -n failed for {sc}: {chk.stderr}"

    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!.env\n" in gi
    assert "!.env.example\n" in gi
    assert "runs/\n" in gi
    assert "!runs_examples/\n" in gi


def test_cli_multitarget_epo_train_predict_and_audit(tmp_path: Path) -> None:
    gen = SyntheticGeneratorV2(SyntheticConfig(dataset_id="mt_epo_v1", n_samples=24, n_sites=4, seed=42))
    ds = gen.generate_dataset()
    wet_ds, dry_ds = gen.generate_paired_wet_dry_dataset(n_pairs=10)

    ds_path = tmp_path / "ds.json"
    wet_path = tmp_path / "wet.json"
    dry_path = tmp_path / "dry.json"
    save_dataset_json(ds, ds_path)
    save_dataset_json(wet_ds, wet_path)
    save_dataset_json(dry_ds, dry_path)

    bundle_dir = tmp_path / "bundle_mt_epo"
    pred_path = tmp_path / "preds.json"
    viz_dir = tmp_path / "viz_mt_epo"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    p_train = subprocess.run(
        [
            sys.executable,
            "-m",
            "st_ds.cli",
            "train",
            "--input",
            str(ds_path),
            "--output",
            str(bundle_dir),
            "--model",
            "plsr",
            "--target",
            "soc_pct",
            "--multi-target",
            "--suppression-mode",
            "epo",
            "--epo-q",
            "2",
            "--wet",
            str(wet_path),
            "--dry",
            str(dry_path),
            "--folds",
            "2",
            "--lang",
            "en",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_train.returncode == 0, p_train.stderr

    p_pred = subprocess.run(
        [
            sys.executable,
            "-m",
            "st_ds.cli",
            "predict",
            "--artifact",
            str(bundle_dir),
            "--input",
            str(ds_path),
            "--output",
            str(pred_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_pred.returncode == 0, p_pred.stderr
    pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
    first_rep = pred_payload["results"][0]
    cm = first_rep["carbon_metrics"]
    assert cm["soc_pct"] is not None
    assert cm["sic_pct"] is not None
    assert cm["tc_pct"] is not None
    assert cm["closure_residual_pct"] is not None

    p_viz = subprocess.run(
        [
            sys.executable,
            "-m",
            "st_ds.cli",
            "visualize",
            "--input",
            str(ds_path),
            "--artifact",
            str(bundle_dir),
            "--wet",
            str(wet_path),
            "--dry",
            str(dry_path),
            "--output",
            str(viz_dir),
            "--dpi",
            "85",
            "--no-interactive-3d",
            "--lang",
            "en",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_viz.returncode == 0, p_viz.stderr
    from st_ds.artifact.bundle import inspect_artifact

    inspected = inspect_artifact(bundle_dir)
    assert inspected["suppression_mode"] == "epo"
    audit_csv = (viz_dir / "tables" / "05_tahmin_ve_stok_denetim_raporu.csv").read_text(encoding="utf-8")
    assert "warning [synthetic]" in audit_csv
    assert "- / -" not in audit_csv

