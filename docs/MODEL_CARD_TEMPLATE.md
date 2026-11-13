# STDS Model Kartı Şablonu (Model Card)

> Her dışa aktarılan artifact paketinin (`save_artifact`) yanında doldurun veya `st_ds.validation.build_model_card_markdown(...)` ile otomatik oluşturun.

## Model Kimliği
- **Artifact Kimliği:** `<artifact_id>`
- **Semantik Sürüm:** `<0.1.0>`
- **Model Adı / Ailesi:** `<plsr | kernel_ridge_rbf | random_forest | local_plsr_mbl | st_soil_hybrid_net>`
- **Enstrüman Profili:** `<visnir_24ch_v1 | led_5band_v1 | multispectral_18ch_v1 | ftnir_1300_2550nm_v1 | libs_ci_hi_v1>`
- **Bundle SHA-256 Özeti:** `<manifest.bundle_sha256>`

## Veri Seti ve Saha Kapsamı
- **Veri Kaynağı:** `<synthetic | real_lab>`
- **Eğitim Örnek Sayısı:** `<N>`
- **Kalibrasyon Sahaları (Görülen):** `<site_id listesi>`
- **Holdout Sahaları (Hiç Görülmemiş):** `<site_id listesi>`

## Değerlendirme Metrikleri
| Metrik | Değer | Notlar |
| --- | --- | --- |
| `n_samples` | `<N_eval>` | Hiç görülmemiş `site_id` holdout seti |
| `rmse` | `<değer>` | Kanonik `%` kuru madde bazında |
| `mae` | `<değer>` | Kanonik `%` kuru madde bazında |
| `bias` | `<değer>` | Ortalama `(y_pred - y_true)` |
| `r2` | `<değer>` | Belirleme katsayısı |
| `rpiq` | `<değer>` | Çeyrekler arası açıklık / RMSE |
| `bland_altman_95_loa` | `[<alt>, <üst>]` | `bias ± 1.96 * SD(diff)` |

## Sentetik ve Gerçek Saha Ayrımı
- Sentetik veri setleri (`SyntheticGeneratorV2`, `dataset_source="synthetic"`) **yalnızca** boru hattı sözleşmelerini, sayısal kararlılığı, conformal belirsizlik altyapısını, XAI katmanını ve güvenlik kapılarını doğrular.
- Sentetik doğrulama metrikleri hiçbir koşulda gerçek laboratuvar veya tarla toprağı doğruluğu yerine sunulamaz (`Risk 9`).

## Kullanım Amacı ve Fail-Closed Kapılar
- Operasyonel agronomik/MRV raporlaması için çıktıda `status` alanı `"ok"` (veya uzman incelemesinden geçmiş `"warning"`) olmalıdır.
- Eksik hacim ağırlığında (`bulk_density`) `carbon_stock.stock_mg_ha` değeri her zaman `None` ve bayrak `QualityFlag.MISSING_BULK_DENSITY` olarak kalır (`DEF-005`).
- Kalibrasyon alanı dışındaki girdiler `QualityStatus.OUT_OF_DOMAIN` (`OOD_HIGH_RISK`) tetikler.
- `model_weights.safetensors`, `preprocess_state.npz` veya `manifest.json` üzerindeki herhangi bir byte bozulması `ArtifactCompatibilityError` fırlatarak yüklemeyi durdurur (`DEF-006`).

## Çalışma Ortamı
- **Platform:** `<İşletim Sistemi / CPU Mimarisi>`
- **Python:** `<3.11+>`
- **Bağımlılıklar:** `<collect_dependency_versions() çıktısındaki sabit sürümler>`
