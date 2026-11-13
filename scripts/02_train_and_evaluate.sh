#!/usr/bin/env bash
# ==============================================================================
# SpaceTeknopoliDataScience (STDS) — 02: Eğitim & Değerlendirme Betiği
#                                       02: Training & Evaluation Pipeline
# ==============================================================================
# AMAÇ (TR):
#   Veri doğrulama -> EPO nem bastırma uyumu -> Çoklu model kıyaslama (CV) ->
#   Model eğitimi -> İmzalı SHA-256 artefakt paketi + PDF Eğitim Özeti üretir.
#   Girdi verilmezse otomatik olarak sentetik veri seti üretir.
#
# PURPOSE (EN):
#   Runs data validation -> EPO fitting -> grouped CV benchmark -> model training
#   -> SHA-256 artifact bundle inspection -> PDF training summary inside
#   timestamped `runs/<YYYYMMDD_HHMMSS>_train_eval/`.
#
# KULLANIM / USAGE:
#   bash scripts/02_train_and_evaluate.sh
#   bash scripts/02_train_and_evaluate.sh /path/to/dataset.json
#   STDS_LANG=en bash scripts/02_train_and_evaluate.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  RESOLVED_PYTHON="$PYTHON_BIN"
elif [[ -x "/opt/anaconda3/bin/python" ]]; then
  RESOLVED_PYTHON="/opt/anaconda3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  RESOLVED_PYTHON="$(command -v python3)"
else
  RESOLVED_PYTHON="$(command -v python)"
fi

SETTINGS_JSON="$(PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -c '
import json
from st_ds.config import get_runtime_config
cfg = get_runtime_config()
print(json.dumps({"lang": cfg.lang, "runs_dir": cfg.runs_dir}))
')"
ACTIVE_LANG="$("$RESOLVED_PYTHON" -c "import json; print(json.loads('''$SETTINGS_JSON''')['lang'])")"
RUNS_BASE="${STDS_RUNS_DIR:-$("$RESOLVED_PYTHON" -c "import json; print(json.loads('''$SETTINGS_JSON''')['runs_dir'])")}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$REPO_ROOT/$RUNS_BASE/${TIMESTAMP}_train_eval"
TARGET="${TARGET:-soc_pct}"
MODEL="${MODEL:-auto}"
FOLDS="${FOLDS:-3}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
INPUT_PATH="${1:-}"

mkdir -p "$RUN_DIR"

echo "========================================================================"
echo " [STDS 02] Model Eğitimi & Çapraz Doğrulama / Training & Evaluation"
echo "------------------------------------------------------------------------"
echo "  Dil / Language   : $ACTIVE_LANG"
echo "  Hedef / Target   : $TARGET | Model: $MODEL | Folds: $FOLDS"
echo "  Çıktı / Output   : $RUN_DIR"
echo "========================================================================"

# 1. Girdi yolu verilmediyse bu çalıştırmaya özel sentetik veri hazırla
if [[ -z "$INPUT_PATH" ]]; then
  echo "[1/6] Girdi verilmedi; sentetik veri üretiliyor..."
  bash "$SCRIPT_DIR/01_generate_synthetic_data.sh" "$RUN_DIR/data" >/dev/null
  INPUT_PATH="$RUN_DIR/data/dataset.json"
  WET_PATH="$RUN_DIR/data/wet_pairs.json"
  DRY_PATH="$RUN_DIR/data/dry_pairs.json"
else
  echo "[1/6] Kullanıcı veri seti doğrulamaya alınıyor: $INPUT_PATH"
  WET_PATH="${WET_PATH:-}"
  DRY_PATH="${DRY_PATH:-}"
fi

# 2. Şema & Fiziksel Kısıt Doğrulaması
echo "[2/6] Veri seti doğrulanıyor (st-ds dataset validate)..."
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli dataset validate \
  --input "$INPUT_PATH" \
  --for-training \
  --output "$RUN_DIR/01_validation.json" >/dev/null

# 3. İsteğe bağlı EPO Nem Baskılama Matrisi Uydurma
if [[ -n "${WET_PATH:-}" && -f "$WET_PATH" && -n "${DRY_PATH:-}" && -f "$DRY_PATH" ]]; then
  echo "[3/6] EPO nem baskılama projeksiyonu hesaplanıyor (calibration fit-epo)..."
  PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli calibration fit-epo \
    --wet "$WET_PATH" \
    --dry "$DRY_PATH" \
    --output "$RUN_DIR/02_epo_projection.json" \
    --q 2 >/dev/null
else
  echo "[3/6] Islak/Kuru çifti verilmedi; fit-epo adımı atlandı."
fi

# 4. Gruplanmış K-Katlı Çapraz Doğrulama (Leakage-Free Model Benchmark)
echo "[4/6] Çoklu model kıyaslaması çalıştırılıyor (st-ds evaluate)..."
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli evaluate \
  --input "$INPUT_PATH" \
  --target "$TARGET" \
  --folds "$FOLDS" \
  --seed "$SEED" \
  --include-optional \
  --output "$RUN_DIR/03_cv_benchmark.json" >/dev/null

# 5. Nihai Eğitim + SHA-256 Artefakt Paketi + Eğitim PDF Raporu
echo "[5/6] Nihai model eğitiliyor ve imzalı artefakt dışa aktarılıyor (st-ds train)..."
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli train \
  --input "$INPUT_PATH" \
  --output "$RUN_DIR/artifact_bundle" \
  --model "$MODEL" \
  --target "$TARGET" \
  --folds "$FOLDS" \
  --hybrid-epochs "$EPOCHS" \
  --seed "$SEED" \
  --lang "$ACTIVE_LANG" \
  --pdf "$RUN_DIR/04_training_summary.pdf" \
  > "$RUN_DIR/04_train_result.json"

# 6. Artefakt SHA-256 Bütünlük Denetimi
echo "[6/6] Artefakt SHA-256 bütünlüğü doğrulanıyor (artifact inspect)..."
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli artifact inspect \
  --artifact "$RUN_DIR/artifact_bundle" \
  --output "$RUN_DIR/05_artifact_inspect.json" >/dev/null

echo ""
echo "Eğitim & Değerlendirme Tamamlandı / Training Complete -> $RUN_DIR"
ls -lh "$RUN_DIR"
