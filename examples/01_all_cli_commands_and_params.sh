#!/usr/bin/env bash
# ==============================================================================
# STDS (`st_ds`) — 01: TÜM CLI KOMUTLARI VE PARAMETRE VARYANTLARI ÖRNEĞİ
# ==============================================================================
# AMAÇ:
#   `scripts/` altındaki yardımcı bash hatlarına bağlı kalmadan, doğrudan terminal
#   üzerinden `python -m st_ds.cli` (veya `pip install -e .` sonrası `st-ds`)
#   komutlarının TÜM parametrelerinin ne işe yaradığını göstererek çalıştırır.
#
# NOT:
#   Etkilenen geçici dosyalar iş bitince otomatik temizlenir (`trap ... EXIT`).
# ==============================================================================
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "/opt/anaconda3/bin/python" ]]; then
  PY="/opt/anaconda3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY="$(command -v python)"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK_DIR="$(mktemp -d -t stds_all_cli_params_XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo ">>> Çalışma alanı (otomatik silinecek): $WORK_DIR"

# ------------------------------------------------------------------------------
# ADIM 0: Örnek JSON, CSV (hem sidecar'lı hem ham dış CSV) ve Wet/Dry EPO verisi
# ------------------------------------------------------------------------------
PYTHONPATH=. "$PY" - <<PY
from pathlib import Path
from st_ds.io.dataset import save_dataset_csv, save_dataset_json
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2

out = Path("$WORK_DIR")
gen = SyntheticGeneratorV2(
    SyntheticConfig(
        dataset_id="cli_params_demo_v1",
        profile_id="visnir_24ch_v1",   # Yerleşik 24 bantlı Vis-NIR profili
        n_samples=48,
        n_sites=6,                     # GroupKFold sızıntısız bölme için 6 saha
        seed=42,
    )
)
ds = gen.generate_dataset()
save_dataset_json(ds, out / "soil_train.json")
save_dataset_csv(ds, out / "soil_train.csv")

# Ham dış CSV örneği: .meta.json olmadan tek başına --instrument-profile test etmek için
(out / "soil_raw.csv").write_text((out / "soil_train.csv").read_text(encoding="utf-8"), encoding="utf-8")

# ÖNEMLİ: generate_paired_wet_dry_dataset() dönüş sırası -> (dry_ds, wet_ds)
dry_ds, wet_ds = gen.generate_paired_wet_dry_dataset(n_pairs=16)
save_dataset_json(dry_ds, out / "dry.json")
save_dataset_json(wet_ds, out / "wet.json")
PY

# ------------------------------------------------------------------------------
# ADIM 1: `dataset validate` — Şema, Birim ve Etiket Doğrulama
#   --for-training     : Her satırda site_id (!= unknown_site) ve hedef etiket arar.
#   --target           : Kontrol edilecek hedef sütun/alan adı (soc_pct/sic_pct/tc_pct/özel).
#   --no-require-sic   : SIC eksikse eğitim doğrulamasının geçmesine izin verir.
#   --instrument-profile: Sidecar (.meta.json) olmayan ham CSV için profil ID veya JSON yolu.
# ------------------------------------------------------------------------------
echo "[1/8] st-ds dataset validate (ham CSV + --instrument-profile + --target tc_pct)..."
PYTHONPATH=. "$PY" -m st_ds.cli dataset validate \
  --input "$WORK_DIR/soil_raw.csv" \
  --instrument-profile visnir_24ch_v1 \
  --for-training \
  --target tc_pct \
  --no-require-sic \
  --output "$WORK_DIR/01_validate.json" >/dev/null

# ------------------------------------------------------------------------------
# ADIM 2: `calibration fit-epo` — Dış Parametre Ortogonalizasyonu (Nem Bastırma)
#   --q                : Ortogonal silinecek nem tekil bileşen sayısı (verilmezse otomatik).
#   --no-absorbance    : Veriniz zaten absorbans ise ekler; reflectance'ta eklemeyin.
#   --no-snv / --no-savgol : İsterseniz SNV veya Savitzky-Golay türevini kapatabilirsiniz.
# ------------------------------------------------------------------------------
echo "[2/8] st-ds calibration fit-epo (--q 2)..."
PYTHONPATH=. "$PY" -m st_ds.cli calibration fit-epo \
  --wet "$WORK_DIR/wet.json" \
  --dry "$WORK_DIR/dry.json" \
  --q 2 \
  --output "$WORK_DIR/02_epo_state.json" >/dev/null

# ------------------------------------------------------------------------------
# ADIM 3: `evaluate` — Saha Bazlı GroupKFold Kıyaslama
#   --target tc_pct         : Hedefi soc_pct yerine Toplam Karbon (tc_pct) yapıyoruz.
#   --folds 3               : 3 katlı site_id grubu çapraz doğrulaması.
#   --suppression-mode glsw : Alternatif bozucu etki bastırma modu (none | epo | glsw).
#   --include-optional      : RandomForest, LocalPLSR ve HybridNet'i de kıyaslamaya ekler.
# ------------------------------------------------------------------------------
echo "[3/8] st-ds evaluate (--target tc_pct --suppression-mode glsw --include-optional)..."
PYTHONPATH=. "$PY" -m st_ds.cli evaluate \
  --input "$WORK_DIR/soil_train.json" \
  --target tc_pct \
  --folds 3 \
  --suppression-mode glsw \
  --wet "$WORK_DIR/wet.json" \
  --dry "$WORK_DIR/dry.json" \
  --include-optional \
  --seed 42 \
  --output "$WORK_DIR/03_eval_glsw_tc.json" >/dev/null

# ------------------------------------------------------------------------------
# ADIM 4: `train` — Sabit Model veya `auto` + Çoklu Hedef + EPO + PDF Özeti
#   --model plsr            : `auto` yerine açıkça PLSR seçer (diğerleri: kernel_ridge_rbf,
#                             random_forest, local_plsr_mbl, st_soil_hybrid_net).
#   --target tc_pct         : Ana hedef tc_pct olarak eğitilir.
#   --multi-target          : Varsa soc_pct, sic_pct, tc_pct hedeflerini ortak eğitir.
#   --suppression-mode epo  : EPO nem bastırma projeksiyonunu modele gömer.
#   --epo-q 2               : EPO rank q=2.
#   --hybrid-epochs 8       : Hibrit YSA için PyTorch epok sayısı.
#   --lang en --pdf ...     : Eğitim özetini İngilizce PDF olarak da üretir.
# ------------------------------------------------------------------------------
echo "[4/8] st-ds train (--model plsr --target tc_pct --suppression-mode epo --multi-target)..."
PYTHONPATH=. "$PY" -m st_ds.cli train \
  --input "$WORK_DIR/soil_train.json" \
  --output "$WORK_DIR/04_bundle_tc_plsr" \
  --model plsr \
  --target tc_pct \
  --folds 3 \
  --suppression-mode epo \
  --epo-q 2 \
  --wet "$WORK_DIR/wet.json" \
  --dry "$WORK_DIR/dry.json" \
  --multi-target \
  --hybrid-epochs 8 \
  --artifact-id demo_tc_plsr_v1 \
  --semantic-version 0.1.0 \
  --seed 42 \
  --lang en \
  --pdf "$WORK_DIR/04_train_summary_en.pdf" >/dev/null

# ------------------------------------------------------------------------------
# ADIM 5: `artifact inspect` — SHA-256 Bütünlük ve Model Kartı İnceleme
# ------------------------------------------------------------------------------
echo "[5/8] st-ds artifact inspect..."
PYTHONPATH=. "$PY" -m st_ds.cli artifact inspect \
  --artifact "$WORK_DIR/04_bundle_tc_plsr" \
  --output "$WORK_DIR/05_inspect.json" >/dev/null

# ------------------------------------------------------------------------------
# ADIM 6: `predict` — Çıkarım Parametreleri (--include-xai, --no-stock, --fail-fast, --pdf)
#   --include-xai : IG + Duyarlılık + toplu HSIC XAI açıklamalarını üretir.
#   --no-stock    : bulk_density olsa bile karbon stoku hesaplamasını devre dışı bırakır.
#   --lang tr     : Çıkarım PDF'ini Türkçe üretir (.env ayarından bağımsız).
# ------------------------------------------------------------------------------
echo "[6/8] st-ds predict (--include-xai --no-stock --lang tr --pdf)..."
PYTHONPATH=. "$PY" -m st_ds.cli predict \
  --artifact "$WORK_DIR/04_bundle_tc_plsr" \
  --input "$WORK_DIR/soil_raw.csv" \
  --instrument-profile visnir_24ch_v1 \
  --include-xai \
  --no-stock \
  --lang tr \
  --output "$WORK_DIR/06_predictions.json" \
  --pdf "$WORK_DIR/06_predictions_tr.pdf" >/dev/null

# ------------------------------------------------------------------------------
# ADIM 7: `report export` — Appendix B Doğrulama & JSON/PDF Dönüştürme
#   --format both : Hem doğrulanmış JSON hem PDF üretir (diğer seçenekler: json | pdf).
# ------------------------------------------------------------------------------
echo "[7/8] st-ds report export (--format both)..."
PYTHONPATH=. "$PY" -m st_ds.cli report export \
  --input "$WORK_DIR/06_predictions.json" \
  --output "$WORK_DIR/07_validated_predictions.json" \
  --format both \
  --pdf-output "$WORK_DIR/07_exported_predictions_en.pdf" \
  --lang en >/dev/null

# ------------------------------------------------------------------------------
# ADIM 8: `visualize` — 26+ Grafik, Tablolar, HTML Pano & Ana Rapor PDF'i
#   --target sic_pct          : İnorganik Karbon (sic_pct) hedefli görselleştirme!
#   --model random_forest     : Artifact verilmezse öne çıkarılacak model familyası.
#   --include-optional        : Grafik 08'de 5 modelin tümünü gösterir.
#   --dpi 100 --format png    : Çözünürlük ve çıktı formatı ('png,svg' de verilebilir).
#   --no-interactive-3d       : Plotly HTML üretmeden hızlandırılmış çalıştırma örneği.
# ------------------------------------------------------------------------------
echo "[8/8] st-ds visualize (--target sic_pct --model random_forest --dpi 100 --no-interactive-3d)..."
PYTHONPATH=. "$PY" -m st_ds.cli visualize \
  --input "$WORK_DIR/soil_train.json" \
  --wet "$WORK_DIR/wet.json" \
  --dry "$WORK_DIR/dry.json" \
  --target sic_pct \
  --model random_forest \
  --folds 3 \
  --include-optional \
  --dpi 100 \
  --format png \
  --no-interactive-3d \
  --lang tr \
  --seed 42 \
  --output "$WORK_DIR/08_viz_sic_rf" >/dev/null

echo "TÜM CLI parametre varyantları başarıyla doğrulandı:"
ls -lh "$WORK_DIR"/04_train_summary_en.pdf "$WORK_DIR"/06_predictions_tr.pdf "$WORK_DIR/08_viz_sic_rf/reports/stds_analysis_report.pdf"
