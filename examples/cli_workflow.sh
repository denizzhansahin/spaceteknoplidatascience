#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  RESOLVED_PYTHON="$PYTHON_BIN"
elif [[ -x "/opt/anaconda3/bin/python" ]]; then
  RESOLVED_PYTHON="/opt/anaconda3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  RESOLVED_PYTHON="$(command -v python3)"
else
  RESOLVED_PYTHON="$(command -v python)"
fi

WORK_DIR="$(mktemp -d -t stds_cli_workflow_XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "[1/8] Sentetik JSON ve eslesmis wet/dry veri setleri olusturuluyor ($WORK_DIR)..."
PYTHONPATH=. "$RESOLVED_PYTHON" - <<PY
from pathlib import Path
from st_ds.io.dataset import save_dataset_json
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2

out = Path("$WORK_DIR")
gen = SyntheticGeneratorV2(SyntheticConfig(dataset_id="cli_demo_v1", n_samples=60, n_sites=6, seed=42))
ds = gen.generate_dataset()
save_dataset_json(ds, out / "dataset.json")
wet_ds, dry_ds = gen.generate_paired_wet_dry_dataset(n_pairs=18)
save_dataset_json(wet_ds, out / "wet.json")
save_dataset_json(dry_ds, out / "dry.json")
PY

echo "[2/8] st-ds dataset validate"
PYTHONPATH=. "$RESOLVED_PYTHON" -m st_ds.cli dataset validate \
  --input "$WORK_DIR/dataset.json" \
  --for-training \
  --output "$WORK_DIR/validate.json" >/dev/null

echo "[3/8] st-ds calibration fit-epo"
PYTHONPATH=. "$RESOLVED_PYTHON" -m st_ds.cli calibration fit-epo \
  --wet "$WORK_DIR/wet.json" \
  --dry "$WORK_DIR/dry.json" \
  --output "$WORK_DIR/epo_matrix.json" \
  --q 2 >/dev/null

echo "[4/8] st-ds evaluate"
PYTHONPATH=. "$RESOLVED_PYTHON" -m st_ds.cli evaluate \
  --input "$WORK_DIR/dataset.json" \
  --target soc_pct \
  --folds 3 \
  --output "$WORK_DIR/benchmark.json" >/dev/null

echo "[5/8] st-ds train"
PYTHONPATH=. "$RESOLVED_PYTHON" -m st_ds.cli train \
  --input "$WORK_DIR/dataset.json" \
  --output "$WORK_DIR/artifact_bundle" \
  --model auto \
  --target soc_pct \
  --folds 3 \
  --hybrid-epochs 8 >/dev/null

echo "[6/8] st-ds artifact inspect & predict"
PYTHONPATH=. "$RESOLVED_PYTHON" -m st_ds.cli artifact inspect \
  --artifact "$WORK_DIR/artifact_bundle" \
  --output "$WORK_DIR/inspect.json" >/dev/null

PYTHONPATH=. "$RESOLVED_PYTHON" -m st_ds.cli predict \
  --artifact "$WORK_DIR/artifact_bundle" \
  --input "$WORK_DIR/dataset.json" \
  --output "$WORK_DIR/predictions.json" >/dev/null

echo "[7/8] st-ds report export"
PYTHONPATH=. "$RESOLVED_PYTHON" -m st_ds.cli report export \
  --input "$WORK_DIR/predictions.json" \
  --output "$WORK_DIR/exported_report.json"

echo "[8/8] st-ds visualize (26+ 2D/3D grafik, 5 CSV/MD tablo ve HTML pano)"
PYTHONPATH=. "$RESOLVED_PYTHON" -m st_ds.cli visualize \
  --input "$WORK_DIR/dataset.json" \
  --artifact "$WORK_DIR/artifact_bundle" \
  --wet "$WORK_DIR/wet.json" \
  --dry "$WORK_DIR/dry.json" \
  --folds 3 \
  --output "$WORK_DIR/viz_report" >/dev/null

echo "CLI uctan uca sentetik is akisi ve gorsellestirme paketi basariyla tamamlandi."
