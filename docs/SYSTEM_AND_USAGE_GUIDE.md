# STDS — Tam Sistem Mimarisi, Modül Kataloğu ve Python / CLI Kullanım Dokümantasyonu

**SpaceTeknopoliDataScience (`st_ds`)**, çok sensörlü toprak spektrumu (Vis-NIR / Multispektral / LED / FT-NIR / LIBS / EIS) ve çevresel eş değişkenler kullanarak Toprak Organik Karbonu (**SOC**), İnorganik Karbonu (**SIC**), Toplam Karbonu (**TC**) ve fiziksel **Karbon Stokunu** ($Mg\;C\;ha^{-1}$) belirsizlik aralığı, OOD kontrolü ve XAI açıklamalarıyla birlikte üreten **bağımsız bilimsel analitik kütüphanesi ve CLI aracıdır**.

---

## 1. Sistem Mimarisi ve Temel Tasarım İlkeleri (`ADR-001`..`ADR-007`)

| Karar | İlke | Sistemdeki Karşılığı |
| --- | --- | --- |
| **`ADR-001`** | Bağımsız MVP sınırı | Donanım sürücüsü, FastAPI/NestJS sunucusu, PostgreSQL ve GIS arayüzü içermez; yalnızca Python API + `st-ds` CLI sunar. |
| **`ADR-002`** | Fail-closed alan doğrulaması | Kanonik birimler (`UnitRegistry`), dalga boyu uyumu ve kimyasal kapatma (`TC = SOC + SIC`) ihlalinde sessiz düzeltme yerine yapılandırılmış hata (`STDSError`) döner. |
| **`ADR-003`** | Veri sızıntısı yasağı | Tüm CV bölmeleri `GroupedSplitBuilder` (`site_id`/`sample_id`) ile yapılır; ölçekleme, PCA/EPO ve kalibrasyon yalnızca `train_idx` üzerinde eğitilir. |
| **`ADR-004` / `DEF-005`** | Eksik bulk density ve OOD kapıları | `bulk_density` yoksa `stock_mg_ha = None` (`MISSING_BULK_DENSITY`) döner; `1.35` gibi gizli varsayım asla atanmaz. |
| **`ADR-005`** | Kanıta dayalı kıyaslama | `STSoilTrainer` tüm baseline'ları (`PLSR`, `KernelRidge`, `RandomForest`, `LocalPLSR`) ve `STSoilHybridNetRegressor` modelini aynı katlarda kıyaslar. |
| **`ADR-006` / `DEF-006`** | Sıfır-pickle güvenli paket | Modeller `.safetensors` + JSON + SHA-256 `manifest.json` ile saklanır; XAI modülü yerel duyarlılık, IG, HSIC ve 2. derece etkileşimi ayrı tutar. |
| **`ADR-007`** | Tekrarlanabilir ortam ve determinizm | `pyproject.toml` sabit sürümler, tohum (`seed`) kontrolü ve `verify_deterministic_replay` ile aynı ağırlık/tahmin garantisi. |

### Veri Akış Mimarisi

```text
Measurement (Spektrum + Eş Değişkenler + Konum + Çevre Meta Verisi)
   │
   ├─► Sinyal Kalite Kontrolü & Kalibrasyon (dark/white radyometri, ISS, SNR, doygunluk, LIBS QC)
   │
   ├─► Kemometri (Soğurma -> SNV/MSC -> Savitzky-Golay -> EPO/GLSW nem bastırma)
   │
   ├─► Özellik Füzyonu (spektral blok + eş değişkenler + eksiklik işaretçileri, isimli profil)
   │
   ├─► Model Çıkarımı (STSoilHybridNet / PLSR / KRR / RF / Local PLSR)
   │       ├─► Conformal Prediction (CI90/CI95 aralığı + aleatorik/epistemik belirsizlik)
   │       ├─► OOD Algılayıcı (spektral Mahalanobis + eş değişken + artık -> OOD_HIGH_RISK)
   │       └─► XAI Motoru (Yerel Duyarlılık, Integrated Gradients, B>=4 için HSIC, 2D etkileşim)
   │
   ├─► Karbon Fraksiyonlama & Stok (TC = SOC + SIC kapatması, S = C * rho_b * d * (1 - f_c))
   │
   └─► Appendix B PredictionReport (durum, kalite bayrakları, aşama süreleri, köken, artifact SHA-256)
```

---

## 2. Paket ve Modül Kataloğu (`st_ds.*`)

| Paket | Ana Sorumluluk | Doğrulanmış Temel Semboller |
| --- | --- | --- |
| `st_ds.domain` | Şemalar, birimler, enstrüman profilleri, kalite ve hata sözleşmesi | `Measurement`, `Spectrum`, `SoilReference`, `PredictionReport`, `UnitRegistry`, `QualityStatus`, `QualityFlag`, `get_builtin_profile`, `STDSError` |
| `st_ds.io` | JSON/CSV/Parquet I/O, doğrulayıcı ve sentetik üretici | `SoilDataset`, `load_dataset_json`, `save_dataset_json`, `load_dataset_csv`, `DataValidator`, `SyntheticGeneratorV2`, `SyntheticConfig` |
| `st_ds.calibration` | Radyometri (`(raw-dark)/(white-dark)`), ISS aktarım ve dalga boyu uyumu | `calibrate_reflectance`, `CalibratedSpectrum`, `ISSModel`, `fit_iss`, `apply_iss`, `align_wavelengths`, `AlignedSpectrum` |
| `st_ds.signal` | Spektral kalite kontrolü, arPLS baseline, tepe uydurma, DLIA ve LIBS | `evaluate_signal_qc`, `estimate_arpls_baseline`, `correct_baseline`, `filter_libs_shots`, `fit_pseudovoigt_peaks`, `summarize_dlia_outputs` |
| `st_ds.chemometrics` | Soğurma, SNV, dalga boyu-duyarlı Savitzky–Golay, EPO/GLSW ve boru hattı | `ChemometricPipeline`, `PreprocessConfig`, `EPO`, `EPOConfig`, `GLSW`, `savgol_derivative`, `snv_transform`, `to_absorbance` |
| `st_ds.features` | İsimlendirilmiş spektral + çevresel özellik birleştirme | `FeatureFusionPipeline`, `FeatureFusionConfig`, `FusedFeatures`, `FeatureProfileSpec`, `FeatureRegistry` |
| `st_ds.models` | Baseline regresörler, `STSoilHybridNetRegressor` ve fizik-bilinçli kayıp | `PLSRRegressor`, `KernelRidgeRegressor`, `RandomForestSoilRegressor`, `LocalPLSRRegressor`, `STSoilHybridNetRegressor`, `STSoilMultiTaskLoss` |
| `st_ds.training` | Grup bölmeleri, metrikler (RMSE/MAE/R²/RPIQ/Bland-Altman) ve trainer | `GroupedSplitBuilder`, `STSoilTrainer`, `TrainerConfig`, `BenchmarkReport`, `compute_regression_metrics` |
| `st_ds.uncertainty` | Conformal kalibrasyon, ensemble belirsizlik ve OOD algılama | `ConformalUncertaintyCalibrator`, `ConformalConfig`, `GroupedBootstrapEnsemble`, `OutOfDomainDetector`, `OODConfig` |
| `st_ds.xai` | Açıklanabilirlik motoru (IG, HSIC, etkileşim ve grup sıralama) | `SoilXAIEngine`, `XAIExplanationReport`, `local_sensitivity`, `integrated_gradients`, `compute_feature_hsic_ranking` |
| `st_ds.carbon` | Karbon fraksiyonlama ve ilk-derece kovaryans yayılımlı stok | `partition_carbon`, `classify_soil_provisional`, `compute_layer_stock`, `compute_profile_stock` |
| `st_ds.spatial` | Ampirik variogram uydurma, Ordinary Kriging ve GeoJSON | `fit_variogram`, `ordinary_kriging`, `kriging_to_geojson`, `KrigingPrediction` |
| `st_ds.artifact` | Sıfır-pickle `.safetensors` + SHA-256 artifact paketi | `save_artifact`, `load_artifact`, `inspect_artifact`, `verify_artifact_integrity`, `LoadedArtifact` |
| `st_ds.api` | Dışa açık sabit tüketici cephesi | `STSoilAnalyzer`, `BatchPredictionResult`, `BatchFailureRecord`, `PUBLIC_API_VERSION` |
| `st_ds.config` / `st_ds.i18n` | İsteğe bağlı `.env` okuyucusu ve çift dilli (`TR`/`EN`) metin motoru | `resolve_lang`, `get_runtime_config`, `STDSRuntimeConfig`, `pick_text`, `t` |
| `st_ds.reporting` | Çok sayfalı A4 yatay kapsamlı bilimsel PDF rapor üreteci | `generate_analysis_pdf_report`, `generate_prediction_pdf_report` |
| `st_ds.visualization` | 26+ bilimsel 2D/3D grafik, 5 CSV/MD tablosu, etkileşimli 3D HTML ve çevrimdışı HTML panosu | `generate_full_visualization_suite`, `VisualizationConfig`, `VisualizationReport` |
| `st_ds.cli` | `st-ds` komut satırı arayüzü | `build_parser`, `main` (`python -m st_ds.cli` / `st-ds`) |
| `st_ds.validation` | Faz 11 bilimsel holdout, replay, profilleme ve Model Kartı | `run_scientific_holdout_check`, `verify_deterministic_replay`, `profile_inference_and_load`, `build_model_card_markdown` |

---

## 3. Python API Kullanım Rehberi

### 3.1 Veri Yükleme, Doğrulama ve Eğitim (`STSoilTrainer`)

```python
from st_ds.io.dataset import load_dataset_json
from st_ds.io.validator import DataValidator
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import STSoilTrainer, TrainerConfig

dataset = load_dataset_json("train_dataset.json")
summary = DataValidator.validate_dataset(dataset, for_training=True)
assert summary.is_valid, summary.errors

trainer = STSoilTrainer(
    TrainerConfig(target_name="soc_pct", hybrid_epochs=30, seed=42)
)
folds = GroupedSplitBuilder.group_kfold_split(dataset, n_splits=4, group_by="site_id")
report = trainer.run_benchmark(dataset, split_folds=folds, include_optional_models=True)

print("Model Sıralaması (OOF RMSE):", report.summary_table())
```

### 3.2 İmzalı Artifact Yükleme ve Çıkarım (`STSoilAnalyzer`)

```python
from st_ds import STSoilAnalyzer, inspect_artifact
from st_ds.io.dataset import load_dataset_json

# 1) Önceden eğitilmiş paketin metadata & SHA-256 özetini incele
info = inspect_artifact("artifact_bundle/")
print("Artifact ID:", info["metadata"]["artifact_id"])

# 2) Doğrulanmış analizörü yükle (profil veya SHA-256 uyuşmazlığında fail-closed)
analyzer = STSoilAnalyzer.from_artifact("artifact_bundle/")

# 3) Tekil ölçüm tahmini (+ XAI + Karbon stoku)
test_ds = load_dataset_json("holdout_dataset.json")
rep = analyzer.predict(test_ds.records[0], include_xai=True, include_stock=True)
print(rep.to_json())

# 4) Kısmi hata yalıtımlı toplu tahmin (fail_fast=False)
batch = analyzer.predict_batch(test_ds.records, fail_fast=False, include_stock=True)
print(f"Başarılı: {batch.n_succeeded}/{batch.n_total}, Hatalı: {batch.n_failed}")
for err in batch.errors:
    print("Hata kaydı:", err.to_dict())

# 5) Bağımsız XAI açıklaması (B >= 4 örnekle HSIC aktif olur)
xai_report = analyzer.explain(test_ds.records[:8])
print(xai_report.to_dict())
```

### 3.3 Bağımsız Karbon Stoku ve Belirsizlik Yayılımı

`st_ds.carbon.stock` farklı fiziksel birimleri otomatik olarak kanonik birimlere çevirir ve analitik kovaryans yayılımı uygular:

```python
from st_ds.carbon.stock import compute_layer_stock, compute_profile_stock

# Tek katman karbon stoku: S = C * rho_b * d * (1 - f_c)
layer = compute_layer_stock(
    carbon_value=18.5,
    carbon_unit="g/kg",      # Otomatik 1.85 % değerine çevrilir
    bulk_density=1320.0,
    bulk_density_unit="kg/m3",  # Otomatik 1.32 g/cm3 değerine çevrilir
    depth_top=0.0,
    depth_bottom=30.0,
    depth_unit="cm",
    coarse_fraction=10.0,
    coarse_unit="pct",       # Otomatik 0.10 değerine çevrilir
    carbon_std=1.2,
    bulk_density_std=60.0,
)
print(f"Stok: {layer.stock_mg_ha:.2f} ± {layer.stock_std_mg_ha:.2f} Mg C/ha")

# Çok katmanlı profil toplama
profile = compute_profile_stock([
    {"label": "0-15cm", "carbon_value": 2.4, "bulk_density": 1.22, "depth_top": 0, "depth_bottom": 15},
    {"label": "15-30cm", "carbon_value": 1.3, "bulk_density": 1.38, "depth_top": 15, "depth_bottom": 30},
])
print("Toplam Profil Stoku:", profile.total_stock_mg_ha)
```

### 3.4 Opsiyonel 2D Ordinary Kriging Haritalama

```python
import numpy as np
from st_ds.spatial.kriging import kriging_to_geojson, ordinary_kriging

sample_lon_lat = np.array([
    [32.850, 39.920], [32.852, 39.921], [32.854, 39.919], [32.856, 39.922],
    [32.851, 39.924], [32.853, 39.925], [32.855, 39.924], [32.857, 39.926],
])
soc_values = np.array([1.8, 2.0, 1.7, 2.2, 2.1, 2.3, 1.9, 2.4])
queries = np.array([[32.853, 39.922], [32.880, 39.950]])  # 2. nokta convex hull dışı

res = ordinary_kriging(sample_lon_lat, soc_values, queries, crs_epsg="EPSG:4326", coord_order="lon_lat")
print("Tahminler:", res.predicted_values, "Ekstrapolasyon:", res.is_extrapolated_mask)
geojson_fc = kriging_to_geojson(res)
```

---

## 4. `st-ds` CLI Tam Başvuru Kılavuzu

**İlk kez çalıştırırken iki seçenekten birini kullanabilirsiniz:**
- **Seçenek A (Editable kurulum):** `pip install --no-build-isolation --no-deps -e .` yaptıktan sonra doğrudan `st-ds ...` yazabilirsiniz.
- **Seçenek B (Kurulumsuz repo içinden):** `PYTHONPATH=. /opt/anaconda3/bin/python -m st_ds.cli ...` şeklinde çağırabilirsiniz.

Hatalarda ham traceback bastırılmaz; yapılandırılmış JSON hata çıktısı üretir (ayrıntılı traceback için `--debug` ekleyin).

| Komut | Amaç | Temel Bayraklar |
| --- | --- | --- |
| `st-ds dataset validate` | Şema, birim ve opsiyonel eğitim etiketlerini doğrular | `--input`, `--instrument-profile`, `--for-training`, `--no-require-sic`, `--output` |
| `st-ds calibration fit-epo` | Eşleştirilmiş wet/dry veri setlerinden EPO projeksiyonu çıkarır | `--wet`, `--dry`, `--output`, `--q`, `--no-absorbance`, `--no-snv`, `--no-savgol` |
| `st-ds evaluate` | Grup-bazlı CV üzerinde candidate modelleri kıyaslar | `--input`, `--target {soc_pct,sic_pct,tc_pct}`, `--folds`, `--seed`, `--include-optional`, `--output` |
| `st-ds train` | Sızıntısız eğitim + kalibrasyon yapıp `.safetensors` artifact paketi kaydeder | `--input`, `--output`, `--model {auto,plsr,kernel_ridge,random_forest,local_plsr,hybrid_net}`, `--target`, `--folds`, `--seed`, `--suppression-mode {none,epo,glsw}`, `--epo-q`, `--hybrid-epochs`, `--artifact-id`, `--semantic-version`, `--pdf`, `--lang` |
| `st-ds artifact inspect` | Paket SHA-256 imzasını doğrular, manifest ve Model Card'ı döner | `--artifact`, `--output` |
| `st-ds predict` | Tekil `Measurement` veya `SoilDataset` (JSON/CSV) üzerinde çıkarım yapar | `--artifact`, `--input`, `--output`, `--include-xai`, `--no-stock`, `--fail-fast`, `--pdf`, `--lang` |
| `st-ds report export` | Çıktıyı Appendix B şemasına karşı doğrulayıp JSON ve/veya PDF aktarır | `--input`, `--output`, `--format {json,pdf,both}`, `--pdf-output`, `--lang` |
| `st-ds visualize` | 26+ bilimsel 2D/3D grafik, 5 CSV/MD tablo, çevrimdışı `index.html` ve çok sayfalı PDF üretir | `--input`, `--output`, `--artifact`, `--instrument-profile`, `--wet`, `--dry`, `--target`, `--folds`, `--format`, `--dpi`, `--no-html`, `--no-interactive-3d`, `--no-pdf`, `--include-optional`, `--seed`, `--lang` |

### 4.1 İsteğe Bağlı `.env` Dil Seçimi (`TR` / `EN`) ve Hazır Bash Betikleri (`scripts/`)
- Sistem `.env` dosyasına bağımlı değildir; kök dizindeki **`.env`** / **`.env.example`** yoksa varsayılan olarak Türkçe (`tr`) çalışır (`Öncelik: --lang > OS Env STDS_LANG > .env STDS_LANG > 'tr'`).
- Tek komutla zaman damgalı `runs/` çıktısı üretmek için: `bash scripts/01_generate_synthetic_data.sh`, `bash scripts/02_train_and_evaluate.sh`, `bash scripts/03_run_full_pipeline.sh`.

### Çıkış Kodları
- `0`: Başarılı (veya `dataset validate` için `is_valid=true`).
- `1`: Doğrulama / STDS domain hatası (`STDSError`, geçersiz şema, integrity hatası, profil uyuşmazlığı).
- `2`: Beklenmeyen çalışma zamanı hatası (`STDS-CLI-ERROR`).

---

## 5. `PredictionReport` (Appendix B) ve Kalite Durumları

Her tekil tahmin aşağıdaki kök anahtarlara sahip standart Appendix B JSON yapısını üretir:

- `identity`: `correlation_id`, `sample_id`, `site_id`, `timestamp_utc`, `instrument_profile_id`, `calibration_id`
- `status`: `QualityStatus` değeri:
  - `ok`: Tüm kalite, OOD ve domain kapıları geçti.
  - `warning`: Tahmin üretildi ancak örneğin eksik yardımcı eş değişken veya kapatma uyarısı gibi ikincil uyarı bayrakları var.
  - `insufficient_data`: Kritik stok verisi (ör. `bulk_density`) eksik; eksik değer tahmini yapılmaz (`DEF-005`).
  - `out_of_domain`: Örnek spektrumu veya özellikleri kalibrasyon alanı dışında (`OOD_HIGH_RISK`).
  - `not_valid_for_reporting` / `rejected`: Doymuş/düşük SNR veya hatalı sinyal nedeniyle raporlanamaz.
- `carbon_metrics`: `primary_target`, `soc_pct`, `soc_std_pct`, `sic_pct`, `tc_pct`, `closure_residual_pct`, `confidence_interval` (`lower_pct`, `upper_pct`, `half_width_pct`, `nominal_coverage`, `calibration_method`)
- `carbon_stock`: `stock_mg_ha`, `stock_std_mg_ha`, `dominant_uncertainty_source`, `quality_flags`
- `classification`: Geçici spektral/karbon sınıf olasılıkları
- `quality`: `QualityMetadata` (`quality_score`, `flags`, `missing_covariates`, `notes`)
- `xai`: `include_xai=True` ise yerel duyarlılık, IG, HSIC (batch `B >= 4`) ve grup sıralamaları
- `provenance`: `stds_package_version`, `api_version`, `schema_version`, `artifact_id`, `artifact_sha256`, `model_name`, `dataset_source`, `training_dataset_source`, `measurement_dataset_source`
- `diagnostics`: `ood_score`, `is_ood`, `snr_estimate`, `stage_timings_ms`, `warning_codes`

---

## 6. Wheel Paketleme ve Çevrimdışı Kurulum

Hedef sahada/edge cihazda internet erişimi olmadan kurulum yapmak için önce wheel paketi üretilir ve `pip install --no-index` ile kurulur:

```bash
# 1) Bağımsız wheel derlemesi (ağ olmadan)
/opt/anaconda3/bin/pip wheel --no-build-isolation --no-deps . -w dist/wheelhouse

# 2) Hedef ortamda çevrimdışı kurulum örneği
/opt/anaconda3/bin/pip install --no-index --no-deps --force-reinstall dist/wheelhouse/st_ds-0.1.0-py3-none-any.whl
```

> [!NOTE]
> Edge (ör. Raspberry Pi 5 ARM64) ortamına dağıtırken performans metriklerini gerçek hedef donanımda `benchmarks/run_release_checks.py` ile yeniden ölçün; farklı CPU/SIMD donanımları arasında gecikme varsayımı yapmayın.

---

## 7. İlgili Dokümantasyon

- **Veri Seti Hazırlama ve Şema Rehberi (TR & EN):** [DATA_PREPARATION_AND_SCHEMA_GUIDE.md](DATA_PREPARATION_AND_SCHEMA_GUIDE.md)
- **Tüm CLI & Python Parametreleri Referansı (TR & EN):** [CLI_AND_PYTHON_PARAMETERS_GUIDE.md](CLI_AND_PYTHON_PARAMETERS_GUIDE.md)
- **Toprak Analizi Çift Dilli (TR & EN) Bilimsel & Uygulama Rehberi:** [TOPRAK_ANALIZ.md](TOPRAK_ANALIZ.md)
- **Toprak Dışı & Genel Spektral / Çok-Sensörlü Veri Analizi Rehberi (TR & EN):** [GENERAL_DATA_ANALYSIS_GUIDE.md](GENERAL_DATA_ANALYSIS_GUIDE.md)
- **Akademik Metodoloji ve Matematiksel Formülasyon (%90 Bilimsel / %10 Yazılım İlişkisi):** [ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md](ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md)
- **Sıfırdan Kurulum ve Adım Adım Kullanım Yol Haritası:** [FIRST_TIME_ROADMAP_GUIDE.md](FIRST_TIME_ROADMAP_GUIDE.md)
- **İlk Sentetik Test Rehberi:** [SYNTHETIC_TESTING_GUIDE.md](SYNTHETIC_TESTING_GUIDE.md)
- **Sürüm Notları (`v0.1.0`):** [RELEASE_NOTES.md](RELEASE_NOTES.md)
