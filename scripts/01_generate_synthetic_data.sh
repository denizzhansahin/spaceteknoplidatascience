#!/usr/bin/env bash
# ==============================================================================
# SpaceTeknopoliDataScience (STDS) — 01: Sentetik Veri Oluşturma Betiği
#                                       01: Synthetic Data Generation Script
# ==============================================================================
# AMAÇ (TR):
#   Model eğitimi ve uçtan uca testler için gerçekçi sentetik toprak spektrum
#   veri setlerini (JSON + CSV + eşleşmiş Islak/Kuru EPO çiftleri) üretir ve
#   zaman damgalı `runs/<YYYYMMDD_HHMMSS>_synthetic_data/` klasörüne yazar.
#
# PURPOSE (EN):
#   Generates realistic synthetic soil spectral datasets (JSON + CSV + paired
#   Wet/Dry EPO pairs) inside timestamped `runs/<YYYYMMDD_HHMMSS>_synthetic_data/`.
#
# KULLANIM / USAGE:
#   bash scripts/01_generate_synthetic_data.sh
#   N_SAMPLES=120 N_SITES=8 bash scripts/01_generate_synthetic_data.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 1. Python yorumlayıcısını otomatik bul (varsa Anaconda, yoksa python3)
if [[ -n "${PYTHON_BIN:-}" ]]; then
  RESOLVED_PYTHON="$PYTHON_BIN"
elif [[ -x "/opt/anaconda3/bin/python" ]]; then
  RESOLVED_PYTHON="/opt/anaconda3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  RESOLVED_PYTHON="$(command -v python3)"
else
  RESOLVED_PYTHON="$(command -v python)"
fi

# 2. İsteğe bağlı .env ayarlarını (sistem bağımlı olmadan) ve parametreleri çöz
SETTINGS_JSON="$(PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -c '
import json
from st_ds.config import get_runtime_config
cfg = get_runtime_config()
print(json.dumps({"lang": cfg.lang, "runs_dir": cfg.runs_dir}))
')"
ACTIVE_LANG="$("$RESOLVED_PYTHON" -c "import json; print(json.loads('''$SETTINGS_JSON''')['lang'])")"
RUNS_BASE="${STDS_RUNS_DIR:-$("$RESOLVED_PYTHON" -c "import json; print(json.loads('''$SETTINGS_JSON''')['runs_dir'])")}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${1:-$REPO_ROOT/$RUNS_BASE/${TIMESTAMP}_synthetic_data}"
N_SAMPLES="${N_SAMPLES:-80}"
N_SITES="${N_SITES:-8}"
N_PAIRS="${N_PAIRS:-20}"
SEED="${SEED:-42}"

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo " [STDS 01] Sentetik Veri Üretimi Başlatılıyor / Generating Synthetic Data"
echo "------------------------------------------------------------------------"
echo "  Dil / Language   : $ACTIVE_LANG"
echo "  Çıktı / Output   : $OUT_DIR"
echo "  Örnek / Samples  : $N_SAMPLES (Sites: $N_SITES, EPO pairs: $N_PAIRS)"
echo "========================================================================"

# 3. Ana veri setini (JSON + CSV) ve Islak/Kuru EPO kalibrasyon çiftlerini üret
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" - <<PY
import json
from pathlib import Path
from st_ds.io.dataset import save_dataset_csv, save_dataset_json, build_dataset_card, generate_data_quality_report
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2

out = Path("$OUT_DIR")
gen = SyntheticGeneratorV2(
    SyntheticConfig(
        dataset_id="stds_synth_${TIMESTAMP}",
        n_samples=int("$N_SAMPLES"),
        n_sites=int("$N_SITES"),
        seed=int("$SEED"),
    )
)

# 3a. Ana eğitim/analiz veri seti (Vis-NIR + Toprak Kovaryatları + Karbon Referansları + GPS)
ds = gen.generate_dataset()
json_path = save_dataset_json(ds, out / "dataset.json")
csv_path = save_dataset_csv(ds, out / "dataset.csv")

# 3b. Dış Parametre Ortogonalizasyonu (EPO) için eşleşmiş Islak / Kuru spektrumlar
wet_ds, dry_ds = gen.generate_paired_wet_dry_dataset(n_pairs=int("$N_PAIRS"))
wet_path = save_dataset_json(wet_ds, out / "wet_pairs.json")
dry_path = save_dataset_json(dry_ds, out / "dry_pairs.json")

# 3c. Veri Kalitesi & Kimlik Kartı özeti (data_profile.json)
profile = {
    "dataset_card": build_dataset_card(ds),
    "quality_report": generate_data_quality_report(ds),
    "files": {
        "dataset_json": str(json_path),
        "dataset_csv": str(csv_path),
        "dataset_meta_json": str(out / "dataset.meta.json"),
        "wet_pairs_json": str(wet_path),
        "dry_pairs_json": str(dry_path),
    },
}
(out / "data_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
print("Generated:", json_path.name, csv_path.name, wet_path.name, dry_path.name)
PY

# 4. Şema ve birim bütünlüğünü CLI ile doğrula
echo "[Doğrulama / Validate] st-ds dataset validate..."
PYTHONPATH="$REPO_ROOT" "$RESOLVED_PYTHON" -m st_ds.cli dataset validate \
  --input "$OUT_DIR/dataset.json" \
  --for-training \
  --output "$OUT_DIR/validation_report.json"

echo ""
echo "Tamamlandı / Completed -> $OUT_DIR"
ls -lh "$OUT_DIR"
