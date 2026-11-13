# STDS `v0.1.0` — Örnek Çalıştırma Çıktıları / Example Run Outputs (`runs_examples/`)

> **Bilgi (TR):** Normal çalışma sırasında `scripts/*.sh` çıktıları `.gitignore` kapsamındaki yerel `runs/<YYYYMMDD_HHMMSS>_*/` klasörlerine kaydedilir. Depoyu GitHub üzerinden inceleyen kullanıcıların sistemin **giren veriden çıkan PDF, PNG, CSV, Markdown, HTML ve SHA-256 imzalı model artefaktlarına** kadar tüm yaşam döngüsünü doğrudan görebilmesi için iki adet tam vitrin çalıştırması (`TR` ve `EN`) bu klasörde paylaşılmıştır.
>
> **Note (EN):** During normal runs, `scripts/*.sh` writes into the `.gitignore`d `runs/<YYYYMMDD_HHMMSS>_*/` folder. Two complete showcase runs (`TR` and `EN`) are committed here so GitHub visitors can immediately inspect every artifact produced end-to-end by STDS without executing the pipeline first.

---

## Klasör İçeriği / Directory Layout

| Vitrin Klasörü / Showcase Folder | Dil / Language | İçerik Özeti / Highlights | Ana Rapor / Primary PDF |
| :--- | :--- | :--- | :--- |
| `example_full_run_tr/` | Türkçe (`tr`) | Tam sentetik veri + doğrulama + EPO + CV + eğitim + tahmin/XAI + 26 PNG + 5 CSV/MD tablo + interaktif 3D HTML + web pano + PDF raporlar | [00_STDS_FULL_ANALYSIS_REPORT.pdf](example_full_run_tr/00_STDS_FULL_ANALYSIS_REPORT.pdf) |
| `example_full_run_en/` | English (`en`) | Same complete end-to-end run localized in English | [00_STDS_FULL_ANALYSIS_REPORT.pdf](example_full_run_en/00_STDS_FULL_ANALYSIS_REPORT.pdf) |

---

## Bir Tam Çalıştırma (`*_full_run`) İçindeki Tüm Dosyalar / Inside a Full Run

1. `00_STDS_FULL_ANALYSIS_REPORT.pdf` — Veri girişi → ön işleme → model çıktısı → XAI/karbon stoku ilişkileri + sayısal tablolar + **üretilen tüm 26 PNG grafiğini** içeren kapsamlı çok sayfalı PDF raporu.
2. `01_input_data/` — Girdi spektral/toprak verisi (`dataset.json`, `dataset.csv`, `dataset.csv.meta.json`, `wet_pairs.json`, `dry_pairs.json`, `data_profile.json`, `validation_report.json`).
3. `02_validation_report.json` — Fiziksel sınır ve birim sözlüğü (`DatasetValidator`) doğrulama raporu.
4. `03_artifact_bundle/`, `03_train_summary.json`, `03_training_summary.pdf` — Gruplanmış K-fold CV sonucu, `SHA-256` imzalı taşınabilir model paketi (`manifest.json`, `model.safetensors`, `preprocess.json`) ve eğitim özet PDF'i.
5. `04_predictions.json`, `04_validated_report.json`, `04_predictions_report.pdf` — Tahminler, %90 Conformal aralıklar, OOD durumları, HSIC/IG XAI raporları, karbon stoku (`Mg C ha⁻¹`) ve Appendix B tahmin PDF raporu.
6. `05_visualization_suite/`
   - `index.html` — Tek tıklamalı çevrimdışı görselleştirme panosu
   - `plots/` — 26 adet yüksek çözünürlüklü 2D & 3D PNG bilimsel grafik (`01`..`26`)
   - `tables/` — 5 adet senkronize `.csv` ve `.md` analitik tablo
   - `interactive/` — Etkileşimli 3D Plotly HTML dosyaları (mevcutsa)
   - `reports/stds_analysis_report.pdf` ve `viz_manifest.json`

---

## Ayrı Aşamalar ile Tam Çalıştırma Farkı / Full Run vs. Individual Bash Scripts

- **`scripts/03_run_full_pipeline.sh` (`*_full_run`)**: Üretimdeki tam boru hattıdır; yukarıdaki 6 aşamanın tamamını tek komutta çalıştırır.
- **`scripts/01_generate_synthetic_data.sh` (`*_synthetic_data`)**: Yalnızca sentetik veri + `wet/dry` EPO çifti + doğrulama üretmek istediğinizde kullanılır (`03` bunu Stage 1'de otomatik çağırır).
- **`scripts/02_train_and_evaluate.sh` (`*_train_eval`)**: 26 görsel veya tam tahmin panosu üretmeden; bağımsız `st-ds calibration fit-epo` (`02_epo_projection.npz`), `st-ds evaluate` (`03_cv_benchmark.json`), `st-ds train` ve `st-ds artifact inspect` (`05_artifact_inspect.json`) dosyalarını incelemek istediğinizde kullanılır.
