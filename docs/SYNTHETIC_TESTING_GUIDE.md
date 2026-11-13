# STDS — Sentetik Veriler ile İlk Kurulum ve Uçtan Uca Test Rehberi

Bu rehber, gerçek laboratuvar/saha verileri henüz hazır değilken **`st_ds` sisteminin tüm veri bilimi, kalibrasyon, eğitim, çıkarım, XAI, karbon stoku ve güvenlik kapılarını ilk sentetik verilerle adım adım nasıl test edeceğinizi** anlatır.

> [!IMPORTANT]
> **Bilimsel Uyarı (`Risk 9` / `ADR-005`):**
> `SyntheticGeneratorV2` fiziksel esinli (Beer–Lambert, nem, kil, karbonat bantları ve `TC = SOC + SIC`) gerçekçi sentetik veriler üretir; ancak sentetik skorlar **yalnızca yazılım boru hattı ve matematiksel tutarlılık doğrulaması içindir**. Gerçek toprak/saha doğruluğu olarak raporlanamaz.

---

## 1. İlk Çalıştırma ve Ortam Hazırlığı

Proje kök dizininde Python `>=3.11` (bu ortamda `/opt/anaconda3/bin/python`) kullanılır:

```bash
cd /Users/denizhan/Desktop/SpaceTeknopoliDataScience

# [Seçenek A — Önerilen] Editable kurulum yaparsanız 'st-ds' komutu doğrudan çalışır:
# /opt/anaconda3/bin/pip install --no-build-isolation --no-deps -e .

# [Seçenek B — Kurulumsuz] Repo kökünden doğrudan test etmek için pytest çalıştırın:
/opt/anaconda3/bin/pytest tests/ -v
```

- **Seçenek A** ile kurduysanız aşağıdaki CLI örneklerindeki `PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli` yerine doğrudan **`st-ds`** yazabilirsiniz.
- **Seçenek B** ile hiç kurulum yapmadan repo içinden çalıştırmak için komutların başına **`PYTHONPATH=.`** eklemeniz yeterlidir.
- **Hazır Bash Otomasyonu:** Tek komutla zaman damgalı `runs/` klasörüne sentetik veri veya tüm PDF/PNG çıktıları üretmek için `bash scripts/01_generate_synthetic_data.sh` ya da `bash scripts/03_run_full_pipeline.sh` betiklerini çalıştırabilirsiniz (`.env` / `STDS_LANG=en` veya `--lang en` ile İngilizce çıktı alabilirsiniz).

---

## 2. Sentetik Veri Üretimi (`SyntheticGeneratorV2`)

`st_ds.io.synthetic` modülündeki `SyntheticGeneratorV2`, seçilen enstrüman profiline (`visnir_24ch_v1`, `led_5band_v1`, `multispectral_18ch_v1`, `ftnir_1300_2550nm_v1`, `libs_ci_hi_v1`) uygun deterministik veriler üretir:

- `generate_dataset()`: Farklı sahalar (`site_01..site_06`), toprak sınıfları, çevresel eş değişkenler (`ph`, `moisture_volumetric_pct`, `temperature_c`, `ec_ds_m`, `bulk_density`) ve `SoilReference` etiketleri (`soc_pct`, `sic_pct`, `tc_pct`).
- `generate_paired_wet_dry_dataset(n_pairs=...)`: **EPO** (External Parameter Orthogonalization) veya **GLSW** nem etkisi baskılama testi için aynı örneğin kuru (`dry`) ve nemli (`wet`) taramalarını üretir.
- `generate_multi_replicate_dataset(n_replicates=3)`: Aynı numune için çoklu tekrar (`rep_1`, `rep_2`, `rep_3`) ve tekrar kalitesi kontrolü üretir.

### Hızlı Sentetik Veri Üretme Betiği

Aşağıdaki Python kodu geçici bir çalışma klasörüne (`/tmp/stds_synth_demo`) ilk test dosyalarınızı yazar:

```bash
PYTHONPATH=. /opt/anaconda3/bin/python - <<'PY'
from pathlib import Path
from st_ds.io.dataset import save_dataset_csv, save_dataset_json
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2

work = Path("/tmp/stds_synth_demo")
work.mkdir(parents=True, exist_ok=True)

# 1) Ana sentetik veri seti (96 örnek, 6 farklı saha)
gen = SyntheticGeneratorV2(
    SyntheticConfig(
        dataset_id="ilk_sentetik_deney_v1",
        profile_id="visnir_24ch_v1",
        n_samples=96,
        n_sites=6,
        seed=42,
        missing_bulk_density_ratio=0.05,  # %5 eksik bulk density ile DEF-005 testi
    )
)
full_ds = gen.generate_dataset()

# Geliştirme (site_01..04) ve hiç görülmemiş Holdout (site_05..06) ayrımı
from st_ds.io.dataset import SoilDataset

train_recs = [r for r in full_ds.records if r.site_id in {"site_01", "site_02", "site_03", "site_04"}]
test_recs = [r for r in full_ds.records if r.site_id in {"site_05", "site_06"}]

train_ds = SoilDataset(dataset_id="synth_train_v1", records=train_recs, dataset_source="synthetic")
test_ds = SoilDataset(dataset_id="synth_holdout_v1", records=test_recs, dataset_source="synthetic")

save_dataset_json(train_ds, work / "train_dataset.json")
save_dataset_json(test_ds, work / "holdout_dataset.json")
save_dataset_csv(test_ds, work / "holdout_dataset.csv")

# 2) Tekil Prediction testi için referans etiketsiz örnek kaydı
single_payload = test_recs[0].to_dict()
import json
(work / "single_measurement.json").write_text(json.dumps(single_payload, indent=2), encoding="utf-8")

# 3) EPO nem transferi için paired wet/dry veri seti
wet_ds, dry_ds = gen.generate_paired_wet_dry_dataset(n_pairs=24)
save_dataset_json(wet_ds, work / "wet_pairs.json")
save_dataset_json(dry_ds, work / "dry_pairs.json")

print(f"Hazirlanan dosyalar -> {work}:")
for p in sorted(work.iterdir()):
    print(" -", p.name)
PY
```

---

## 3. Adım Adım `st-ds` CLI ile Sentetik Test Akışı

Yukarıda `/tmp/stds_synth_demo` içine yazılan dosyalarla tüm CLI zincirini sırayla deneyin:

### Adım 1 — Şema, Birim ve Fiziksel Tutarlılık Doğrulaması

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli dataset validate \
  --input /tmp/stds_synth_demo/train_dataset.json \
  --for-training
```
Çıktıda `"is_valid": true` ve `"n_errors": 0` görmelisiniz (`TC = SOC + SIC` ve saha lineage kontrolleri dahil).

### Adım 2 — Paired Wet/Dry Verisiyle EPO Matrisi Çıkarma

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli calibration fit-epo \
  --wet /tmp/stds_synth_demo/wet_pairs.json \
  --dry /tmp/stds_synth_demo/dry_pairs.json \
  --output /tmp/stds_synth_demo/epo_projection.json \
  --q 2
```

### Adım 3 — Baseline ve Hibrit Modelleri Sızıntısız Kıyaslama (`GroupKFold`)

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli evaluate \
  --input /tmp/stds_synth_demo/train_dataset.json \
  --target soc_pct \
  --folds 3 \
  --include-optional \
  --output /tmp/stds_synth_demo/cv_benchmark.json
```
Çıktıdaki `summary_table` bölümünde `plsr`, `kernel_ridge_rbf`, `random_forest`, `local_plsr_mbl` ve `st_soil_hybrid_net` OOF RMSE/MAE/R²/RPIQ sıralamasını görürsünüz.

### Adım 4 — Model Eğitimi ve İmzalı Artifact Bundle Oluşturma

```bash
PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli train \
  --input /tmp/stds_synth_demo/train_dataset.json \
  --output /tmp/stds_synth_demo/artifact_bundle \
  --model auto \
  --target soc_pct \
  --folds 3 \
  --hybrid-epochs 12 \
  --artifact-id synth_visnir_demo_v1
```
Bu komut `artifact_bundle/` içinde `.safetensors` ağırlıkları, JSON preprocessing/fusion/conformal/OOD durumları ve SHA-256 imzalı `manifest.json` üretir (**sıfır pickle**).

### Adım 5 — Artifact Denetimi ve Hiç Görülmemiş Holdout Verisiyle Çıkarım

```bash
# SHA-256 bütünlük doğrulaması + Model Card özeti
PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli artifact inspect \
  --artifact /tmp/stds_synth_demo/artifact_bundle

# Tekil ölçüm tahmini (+ XAI duyarlılık/IG açıklaması)
PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli predict \
  --artifact /tmp/stds_synth_demo/artifact_bundle \
  --input /tmp/stds_synth_demo/single_measurement.json \
  --include-xai \
  --output /tmp/stds_synth_demo/single_prediction.json

# Holdout veri seti üzerinde toplu tahmin + kanonik rapor aktarımı
PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli predict \
  --artifact /tmp/stds_synth_demo/artifact_bundle \
  --input /tmp/stds_synth_demo/holdout_dataset.json \
  --output /tmp/stds_synth_demo/holdout_predictions.json

PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli report export \
  --input /tmp/stds_synth_demo/holdout_predictions.json \
  --output /tmp/stds_synth_demo/exported_appendix_b_report.json
```

---

## 4. Python ile Doğrudan Uçtan Uca Sentetik Test (Tek Komut)

Hazır örnek script ve Faz 11 release benchmark aracı ile aynı boru hattını tek seferde çalıştırabilirsiniz:

```bash
# Python quickstart (üretim -> benchmark -> bundle -> predict_batch -> DEF-005 -> Kriging)
PYTHONPATH=. /opt/anaconda3/bin/python examples/synthetic_python_quickstart.py

# Tam CLI iş akışı scripti (geçici dizinde 7 adımı çalıştırıp temizler)
bash examples/cli_workflow.sh

# Bilimsel holdout + deterministik replay + gecikme/RAM profilleme raporu
PYTHONPATH=. /opt/anaconda3/bin/python benchmarks/run_release_checks.py \
  --output-json /tmp/stds_synth_demo/release_report.json
```

---

## 5. Güvenlik ve Fail-Closed Doğrulama Deneyleri

Aşağıdaki kısa Python kontrolü ile STDS'nin üç kritik bilimsel/güvenlik kapısını sentetik ortamda canlı test edebilirsiniz:

```bash
PYTHONPATH=. /opt/anaconda3/bin/python - <<'PY'
from dataclasses import replace
import shutil
from pathlib import Path
import numpy as np

from st_ds import STSoilAnalyzer
from st_ds.carbon.stock import compute_layer_stock
from st_ds.domain.errors import ArtifactCompatibilityError
from st_ds.domain.quality import QualityFlag, QualityStatus
from st_ds.io.dataset import load_dataset_json

bundle = Path("/tmp/stds_synth_demo/artifact_bundle")
holdout = load_dataset_json("/tmp/stds_synth_demo/holdout_dataset.json")
analyzer = STSoilAnalyzer.from_artifact(bundle)

# 1) DEF-005: Eksik bulk_density asla 1.35 varsayimina dusmez
stock_res = compute_layer_stock(carbon_value=2.0, bulk_density=None, depth_top=0.0, depth_bottom=30.0)
assert stock_res.stock_mg_ha is None
assert stock_res.quality_status == QualityStatus.INSUFFICIENT_DATA
assert QualityFlag.MISSING_BULK_DENSITY in stock_res.quality_flags
print("[OK] DEF-005 fail-closed missing bulk density dogrulandi.")

# 2) OOD kapisi: Kalibrasyon alani disi spektral sekilde OUT_OF_DOMAIN bayragi
rec = holdout.records[0]
extreme_spec = replace(rec.spectrum, values=np.linspace(0.05, 0.95, len(rec.spectrum.values)))
extreme_rec = replace(rec, sample_id="ood_probe", spectrum=extreme_spec)
ood_rep = analyzer.predict(extreme_rec)
assert ood_rep.diagnostics["is_ood"] is True
assert ood_rep.status == QualityStatus.OUT_OF_DOMAIN
print("[OK] OOD tespit kapisi dogrulandi:", ood_rep.status.value)

# 3) DEF-006: Bundle byte degisikliginde ArtifactCompatibilityError (fail-closed)
tampered_dir = Path("/tmp/stds_synth_demo/tampered_bundle")
if tampered_dir.exists():
    shutil.rmtree(tampered_dir)
shutil.copytree(bundle, tampered_dir)
weights_file = tampered_dir / "model_weights.safetensors"
raw = bytearray(weights_file.read_bytes())
raw[-1] ^= 0xFF
weights_file.write_bytes(bytes(raw))

try:
    STSoilAnalyzer.from_artifact(tampered_dir)
    raise AssertionError("Tampered bundle yuklenmemeliydi!")
except ArtifactCompatibilityError as exc:
    print("[OK] SHA-256 butunluk kapisi dogrulandi:", exc.error_code)
PY
```

İhtiyacınız bittiğinde geçici klasörü silebilirsiniz: `rm -rf /tmp/stds_synth_demo`.
