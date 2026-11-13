"""Faz 10 Unit Tests: Unified `st-ds` CLI Subcommands & Structured Error Handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from st_ds.cli import main
from st_ds.io.dataset import save_dataset_json
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2


class TestCLIWorkflows:
    """Exercises all 7 PDF Section 27 CLI commands using the shared st_ds engine."""

    def test_cli_full_workflow_validate_epo_train_evaluate_inspect_predict_export(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        gen = SyntheticGeneratorV2(
            SyntheticConfig(
                dataset_id="cli_synth_ds",
                profile_id="visnir_24ch_v1",
                n_samples=36,
                n_sites=3,
                seed=19,
            )
        )
        ds = gen.generate_dataset()
        dry_ds, wet_ds = gen.generate_paired_wet_dry_dataset(n_pairs=10)

        ds_path = tmp_path / "dataset.json"
        wet_path = tmp_path / "wet.json"
        dry_path = tmp_path / "dry.json"
        save_dataset_json(ds, ds_path)
        save_dataset_json(wet_ds, wet_path)
        save_dataset_json(dry_ds, dry_path)

        # 1. st-ds dataset validate
        rc = main(["dataset", "validate", "--input", str(ds_path), "--for-training"])
        assert rc == 0
        val_out = json.loads(capsys.readouterr().out)
        assert val_out["is_valid"] is True
        assert val_out["n_records"] == 36

        # 2. st-ds calibration fit-epo
        epo_out_dir = tmp_path / "epo_out"
        rc = main(["calibration", "fit-epo", "--wet", str(wet_path), "--dry", str(dry_path), "--output", str(epo_out_dir)])
        assert rc == 0
        epo_res = json.loads(capsys.readouterr().out)
        assert epo_res["invariants"]["symmetry_frobenius_error"] < 1e-10
        assert (epo_out_dir / "epo_state.npz").is_file()

        # 3. st-ds evaluate
        eval_json = tmp_path / "benchmark.json"
        rc = main(["evaluate", "--input", str(ds_path), "--folds", "3", "--output", str(eval_json)])
        assert rc == 0
        eval_res = json.loads(capsys.readouterr().out)
        assert len(eval_res["evidence_ranking"]) >= 2
        assert eval_json.is_file()

        # 4. st-ds train -> exports signed artifact bundle
        artifact_dir = tmp_path / "trained_artifact"
        rc = main([
            "train",
            "--input",
            str(ds_path),
            "--output",
            str(artifact_dir),
            "--model",
            "plsr",
            "--folds",
            "3",
        ])
        assert rc == 0
        train_res = json.loads(capsys.readouterr().out)
        assert train_res["selected_model"] == "plsr"
        assert (artifact_dir / "manifest.json").is_file()
        assert (artifact_dir / "model_card.json").is_file()
        assert (artifact_dir / "preprocess_state.npz").is_file()
        assert (artifact_dir / "model_weights.safetensors").is_file()

        # 5. st-ds artifact inspect
        rc = main(["artifact", "inspect", "--artifact", str(artifact_dir)])
        assert rc == 0
        insp_res = json.loads(capsys.readouterr().out)
        assert insp_res["integrity_verified"] is True
        assert insp_res["bundle_sha256"] == train_res["bundle_sha256"]

        # 6. st-ds predict (on single Measurement JSON & SoilDataset JSON)
        single_meas_path = tmp_path / "single_measurement.json"
        single_meas_path.write_text(json.dumps(ds.records[0].to_dict(), indent=2), encoding="utf-8")
        pred_report_path = tmp_path / "single_report.json"

        rc = main([
            "predict",
            "--artifact",
            str(artifact_dir),
            "--input",
            str(single_meas_path),
            "--output",
            str(pred_report_path),
        ])
        assert rc == 0
        pred_out = json.loads(capsys.readouterr().out)
        assert pred_out["identity"]["sample_id"] == ds.records[0].sample_id
        assert pred_report_path.is_file()

        # 7. st-ds report export
        exported_path = tmp_path / "exported_report.json"
        rc = main(["report", "export", "--input", str(pred_report_path), "--output", str(exported_path)])
        assert rc == 0
        exp_out = json.loads(capsys.readouterr().out)
        assert exp_out["report_type"] == "prediction_report"
        assert exported_path.is_file()

        # 8. st-ds visualize
        viz_dir = tmp_path / "viz_cli"
        rc = main([
            "visualize",
            "--input",
            str(ds_path),
            "--artifact",
            str(artifact_dir),
            "--output",
            str(viz_dir),
            "--dpi",
            "85",
            "--no-interactive-3d",
        ])
        assert rc == 0
        viz_out = json.loads(capsys.readouterr().out)
        assert viz_out["command"] == "visualize"
        assert viz_out["n_plots"] >= 26
        assert Path(viz_out["dashboard_path"]).is_file()

    def test_cli_structured_error_hides_raw_traceback_by_default(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing_bundle = tmp_path / "does_not_exist"
        rc = main(["artifact", "inspect", "--artifact", str(missing_bundle)])
        assert rc == 1
        captured = capsys.readouterr()
        err_json = json.loads(captured.err)
        assert "error_code" in err_json
        assert "traceback" not in err_json
