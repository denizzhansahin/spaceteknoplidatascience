#!/usr/bin/env bash
# ==============================================================================
# SpaceTeknopoliDataScience (STDS) — 03: Uçtan Uca Tam Sistem Çalıştırma Betiği
#                                       03: End-to-End Full System Pipeline
# ==============================================================================
# AMAÇ (TR):
#   Tek komutla veriyi alır, doğrular, kemometri + EPO uygular, modeli eğitir,
#   tahmin ve XAI çıkarımını çalıştırır ve zaman damgalı `runs/` altında TÜM
#   çıktı türlerini üretir:
#     • JSON (tahmin raporu, benchmark, doğrulama, viz_manifest)
#     • CSV & Markdown tablolar (5 analitik tablo)
#     • PNG (26+ bilimsel 2D & 3D grafik)
#     • İnteraktif 3D HTML & Çevrimdışı Web Panosu (index.html)
#     • Çok Sayfalı Kapsamlı PDF Raporları (giren/çıkan veri + ilişkiler + TÜM PNG'ler)
#
# PURPOSE (EN):
#   End-to-end orchestration from raw input -> preprocessing -> training/inference
#   -> XAI -> CSV/MD/JSON -> 26+ PNGs -> Interactive 3D HTML -> Comprehensive
#   Multi-Page PDF Reports inside `runs/<YYYYMMDD_HHMMSS>_full_run/`.
#
# KULLANIM / USAGE:
#   bash scripts/03_run_full_pipeline.sh                     # Sentetik demo verisiyle
#   bash scripts/03_run_full_pipeline.sh veri.csv            # Kendi CSV/JSON verinizle
#   STDS_LANG=en bash scripts/03_run_full_pipeline.sh        # İngilizce çıktı ile
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
print(json.dumps({"lang": cfg.lang, "runs_dir": cfg.runs_dir, "dpi": cfg.dpi}))
')"
ACTIVE_LANG="$("$RESOLVED_PYTHON" -c "import json; print(json.loads('''$SETTINGS_JSON''')['lang'])")"
RUNS_BASE="${STDS_RUNS_DIR:-$("$RESOLVED_PYTHON" -c "import json; print(json.loads('''$SETTINGS_JSON''')['runs_dir'])")}"
ACTIVE_DPI="${STDS_DPI:-$("$RESOLVED_PYTHON" -c "import json; print(json.loads('''$SETTINGS_JSON''')['dpi'])")}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$REPO_ROOT/$RUNS_BASE/${TIMESTAMP}_full_run"
TARGET="${TARGET:-soc_pct}"
MODEL="${MODEL:-auto}"
FOLDS="${FOLDS:-3}"
EPOCHS="${EPOCHS:-16}"
SEED="${SEED:-42}"
INPUT_ARG="${1:-}"
PROFILE_ARG="${INSTRUMENT_PROFILE:-}"

mkdir -p "$RUN_DIR"

echo "========================================================================"
echo " [STDS 03] Tam Sistem Çalıştırma Hattı / Full Pipeline Execution"
echo "------------------------------------------------------------------------"
echo "  Aktif Dil / Language : $ACTIVE_LANG (.env veya STDS_LANG üzerinden)"
echo "  Zaman Damgalı Klasör : $RUN_DIR"
echo "  DPI / Target / Model : $ACTIVE_DPI DPI | $TARGET | $MODEL | $FOLDS-fold"
echo "========================================================================"

# AŞAMA 1 — Veri Hazırlama (Kullanıcı Girdisi veya Sentetik Üretim)
WET_ARG=""
DRY_ARG=""
if [[ -z "$INPUT_ARG" ]]; then
  echo "[1/6] Örnek girdi verilmedi -> Sentetik veri ve Islak/Kuru çiftler üretiliyor..."
  bash "$SCRIPT_DIR/01_generate_synthetic_data.sh" "$RUN_DIR/01_input_data" >/dev/null
  INPUT_PATH="$RUN_DIR/01_input_data/dataset.json"
  WET_ARG="$RUN_DIR/01_input_data/wet_pairs.json"
  DRY_ARG="$RUN_DIR/01_input_data/dry_pairs.json"
else
  INPUT_PATH="$INPUT_ARG"
  WET_ARG="${WET_PATH:-}"
  DRY_ARG="${DRY_PATH:-}"
  mkdir -p "$RUN_DIR/01_input_data"
  cp "$INPUT_PATH" "$RUN_DIR/01_input_data/" 2>/dev/null || true
  echo "[1/6] Kullanıcı verisi kullanılıyor: $INPUT_PATH"
fi

PROFILE_FLAGS=()
if [[ -n "$PROFILE_ARG" ]]; then
  PROFILE_FLAGS=(--instrument-profile "$PROFILE_ARG")
fi

EPO_TRAIN_FLAGS=()
if [[ -n "$WET_ARG" && -f "$WET_ARG" && -n "$DRY_ARG" && -f "$DRY_ARG" ]]; then
  EPO_TRAIN_FLAGS=(--suppression-mode epo --epo-q 2 --wet "$WET_ARG" --dry "$DRY_ARG")
fi

MULTI_TARGET_FLAG=()
VALIDATE_SIC_FLAG=()
if [[ "$TARGET" == "soc_pct" || "$TARGET" == "sic_pct" || "$TARGET" == "tc_pct" ]]; then
  if [[ "${MULTI_TARGET:-1}" != "0" ]]; then
    MULTI_TARGET_FLAG=(--multi-target)
  fi
else
  VALIDATE_SIC_FLAG=(--no-require-sic)
fi

# AŞAMA 2 — Veri Şeması ve Fiziksel Kısıt Doğrulaması
echo "[2/6] Şema & fiziksel kısıt doğrulaması (st-ds dataset validate)..."
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli dataset validate \
  --input "$INPUT_PATH" \
  ${PROFILE_FLAGS[@]+"${PROFILE_FLAGS[@]}"} \
  --for-training \
  --target "$TARGET" \
  ${VALIDATE_SIC_FLAG[@]+"${VALIDATE_SIC_FLAG[@]}"} \
  --output "$RUN_DIR/02_validation_report.json" >/dev/null

# AŞAMA 3 — Gruplanmış CV Benchmark & İmzalı Model Eğitimi
echo "[3/6] Gruplanmış çapraz doğrulama & SHA-256 model eğitimi (st-ds train)..."
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli train \
  --input "$INPUT_PATH" \
  --output "$RUN_DIR/03_artifact_bundle" \
  --model "$MODEL" \
  --target "$TARGET" \
  ${PROFILE_FLAGS[@]+"${PROFILE_FLAGS[@]}"} \
  --folds "$FOLDS" \
  ${EPO_TRAIN_FLAGS[@]+"${EPO_TRAIN_FLAGS[@]}"} \
  ${MULTI_TARGET_FLAG[@]+"${MULTI_TARGET_FLAG[@]}"} \
  --hybrid-epochs "$EPOCHS" \
  --seed "$SEED" \
  --lang "$ACTIVE_LANG" \
  --pdf "$RUN_DIR/03_training_summary.pdf" \
  > "$RUN_DIR/03_train_summary.json"

# AŞAMA 4 — Çıkarım + XAI + Karbon Stoku + Appendix B JSON & PDF Rapor Dışa Aktarımı
echo "[4/6] Tahmin, XAI ve karbon stoku çıkarımı + PDF dışa aktarım (predict + report export)..."
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli predict \
  --artifact "$RUN_DIR/03_artifact_bundle" \
  --input "$INPUT_PATH" \
  ${PROFILE_FLAGS[@]+"${PROFILE_FLAGS[@]}"} \
  --include-xai \
  --lang "$ACTIVE_LANG" \
  --output "$RUN_DIR/04_predictions.json" >/dev/null

PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli report export \
  --input "$RUN_DIR/04_predictions.json" \
  --output "$RUN_DIR/04_validated_report.json" \
  --format both \
  --pdf-output "$RUN_DIR/04_predictions_report.pdf" \
  --lang "$ACTIVE_LANG" >/dev/null

# AŞAMA 5 — Tam Bilimsel Görselleştirme Paketi (26+ PNG + 5 CSV/MD + 3D HTML + Dashboard + Ana PDF)
echo "[5/6] 26+ grafik (2D/3D PNG), 5 CSV/MD tablo, HTML pano ve tam PDF raporu üretiliyor..."
EPO_VIZ_FLAGS=()
if [[ -n "$WET_ARG" && -f "$WET_ARG" && -n "$DRY_ARG" && -f "$DRY_ARG" ]]; then
  EPO_VIZ_FLAGS=(--wet "$WET_ARG" --dry "$DRY_ARG")
fi

PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli visualize \
  --input "$INPUT_PATH" \
  --artifact "$RUN_DIR/03_artifact_bundle" \
  ${PROFILE_FLAGS[@]+"${PROFILE_FLAGS[@]}"} \
  ${EPO_VIZ_FLAGS[@]+"${EPO_VIZ_FLAGS[@]}"} \
  --target "$TARGET" \
  --model "$MODEL" \
  --folds "$FOLDS" \
  --dpi "$ACTIVE_DPI" \
  --format png \
  --include-optional \
  --seed "$SEED" \
  --lang "$ACTIVE_LANG" \
  --output "$RUN_DIR/05_visualization_suite" \
  > "$RUN_DIR/05_visualize_summary.json"

# AŞAMA 6 — Ana PDF'i kolay erişim için çalıştırma köküne de kopyala & Özet Yazdır
echo "[6/6] Çalıştırma çıktıları düzenleniyor..."
MAIN_PDF="$RUN_DIR/05_visualization_suite/reports/stds_analysis_report.pdf"
if [[ -f "$MAIN_PDF" ]]; then
  cp "$MAIN_PDF" "$RUN_DIR/00_STDS_FULL_ANALYSIS_REPORT.pdf"
fi

N_PNGS="$(find "$RUN_DIR/05_visualization_suite/plots" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')"
N_CSVS="$(find "$RUN_DIR/05_visualization_suite/tables" -maxdepth 1 -name '*.csv' | wc -l | tr -d ' ')"
N_MDS="$(find "$RUN_DIR/05_visualization_suite/tables" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')"

echo ""
echo "========================================================================"
echo " [STDS] ÇALIŞTIRMA BAŞARIYLA TAMAMLANDI / FULL RUN SUCCEEDED"
echo "------------------------------------------------------------------------"
echo "  Kök Klasör / Run Dir  : $RUN_DIR"
echo "  Ana PDF Raporu (TÜM PNG): $RUN_DIR/00_STDS_FULL_ANALYSIS_REPORT.pdf"
echo "  Tahmin PDF Raporu     : $RUN_DIR/04_predictions_report.pdf"
echo "  Eğitim PDF Özeti      : $RUN_DIR/03_training_summary.pdf"
echo "  Web Pano / Dashboard  : $RUN_DIR/05_visualization_suite/index.html"
echo "  Grafikler (PNG)       : $N_PNGS adet ($RUN_DIR/05_visualization_suite/plots/)"
echo "  Tablolar (CSV / MD)   : $N_CSVS CSV, $N_MDS MD ($RUN_DIR/05_visualization_suite/tables/)"
echo "  İnteraktif 3D HTML    : $RUN_DIR/05_visualization_suite/interactive/"
echo "  Model Artefaktı       : $RUN_DIR/03_artifact_bundle/"
echo "========================================================================"
