"""Executable release verification & profiling runner for STDS Faz 11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from st_ds.validation.model_card import build_model_card_markdown, validate_model_card_sections
from st_ds.validation.release_checks import (
    collect_environment_manifest,
    profile_inference_and_load,
    run_scientific_holdout_check,
    verify_deterministic_replay,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run STDS Faz 11 release checks and performance profiling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None, help="Optional output JSON report path.")
    parser.add_argument("--output-model-card", default=None, help="Optional generated Model Card Markdown path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    env = collect_environment_manifest()
    holdout = run_scientific_holdout_check(seed=args.seed)
    replay = verify_deterministic_replay(seed=args.seed)
    perf = profile_inference_and_load(seed=args.seed)

    model_card_md = build_model_card_markdown(
        artifact_id=f"stds_{holdout['selected_model']}_holdout",
        semantic_version=env["stds_version"],
        model_name=holdout["selected_model"],
        model_family=holdout["selected_model"],
        instrument_profile_id="visnir_24ch_v1",
        dataset_source="synthetic",
        n_train_samples=holdout["n_train"],
        train_sites=holdout["train_sites"],
        holdout_sites=holdout["holdout_sites"],
        holdout_metrics=holdout["holdout_metrics"],
        environment=env,
        bundle_sha256=holdout["bundle_sha256"],
    )
    missing_sections = validate_model_card_sections(model_card_md)

    report = {
        "environment": env,
        "holdout": holdout,
        "deterministic_replay": replay,
        "performance_profile": perf,
        "model_card_missing_sections": missing_sections,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if args.output_model_card:
        mc_out = Path(args.output_model_card)
        mc_out.parent.mkdir(parents=True, exist_ok=True)
        mc_out.write_text(model_card_md, encoding="utf-8")

    print(text)
    return 0 if (replay["passed"] and not missing_sections) else 1


if __name__ == "__main__":
    raise SystemExit(main())
