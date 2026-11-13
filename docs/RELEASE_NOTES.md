# STDS (`st-ds`) — Sürüm ve Geliştirme Notları (MVP `v0.1.0`)

**Sürüm:** `v0.1.0` (MVP İlk Kararlı Sürüm)  
**Tarih:** 13 Kasım 2026  
**Şema Sürümü (`SCHEMA_VERSION`):** `1.0.0`  
**Genel API Sürümü (`PUBLIC_API_VERSION`):** `1.0.0`  
**Artifact Biçimi (`ARTIFACT_FORMAT_VERSION`):** `1.0.0`  
**Gereksinimler:** Python `>=3.11` (`numpy`, `scipy`, `torch`, `scikit-learn`, `pandas`, `pydantic`, `safetensors`, `matplotlib`; opsiyonel `.[viz]` ile `plotly` ve `seaborn`)

---

## 1. Sürüm Özeti

`st-ds v0.1.0`, **SpaceTeknopoliDataScience (STDS)** projesinin donanımdan ve web sunucularından tamamen bağımsız (**`ADR-001`**), uçtan uca doğrulanmış **ilk kararlı MVP sürümüdür**. Toprak organik karbonu (**SOC**), inorganik karbonu (**SIC**), toplam karbonu (**TC**) ve fiziksel toprak karbon stokunu ($Mg\;C\;ha^{-1}$) tek bir deterministik, sıfır-pickle ve fail-closed bilimsel boru hattında birleştirir.

---

## 2. Faz Bazlı Gelişim Geçmişi (Faz 0 → Faz 11)

| Faz | Kazanımlar | Doğrulanan Temel Sorumluluk |
| --- | --- | --- |
| **Faz 0** | Mimari Denetim ve ADR Dondurma | `ADR-001..007` kararları ve `DEF-001..009` kritik kusur çözüm sözleşmeleri ([docs/faz0/](faz0/STDS_ARCHITECTURE_DECISIONS.md)). |
| **Faz 1** | Kanonik Alan Sözleşmesi ve Birim Motoru | Sabit 24 kanal varsayımının kaldırılması (`InstrumentProfile`), `UnitRegistry`, `Measurement`, `PredictionReport`, `STDSError` hiyerarşisi. |
| **Faz 2** | Veri Seti I/O, Sızıntısız Bölme ve Sentetik Üretici v2 | `SoilDataset` (real vs synthetic izolasyonu), `GroupedSplitBuilder` (`site_id`/`sample_id`), `SyntheticGeneratorV2` (`TC = SOC + SIC`, paired wet/dry). |
| **Faz 3** | Kalibrasyon ve Sinyal Kalite Kontrolü (QC) | Dark/White radyometrik düzeltme (`calibrate_reflectance`), ISS aktarımı (`fit_iss`/`apply_iss`), `align_wavelengths`, SNR/doygunluk QC, arPLS baseline, Pseudo-Voigt LIBS ve DLIA. |
| **Faz 4** | Fiziksel Dalga Boyu Kemometrisi ve EPO/GLSW | Dalga boyu grid-duyarlı Savitzky–Golay türevi (`DEF-002`), kalibrasyon-çıkarım uzay imzalı `EPO` ve `GLSW` nem bastırma (`DEF-001`). |
| **Faz 5** | Çoklu Sensör Özellik Kayıt Defteri ve Füzyon | `FeatureRegistry`, Vis-NIR kimyasal bantları, EIS özellikleri, eksik eş değişken işaretçileri ve isimlendirilmiş `FeatureFusionPipeline`. |
| **Faz 6** | Baseline Modeller, `STSoilHybridNet` ve Eğitim Kıyaslaması | Aynı `GroupKFold` katlarında `PLSR`, `KernelRidge`, `RandomForest`, `LocalPLSR` ve çift dallı `STSoilHybridNetRegressor` kıyası (`ADR-005`). |
| **Faz 7** | Belirsizlik Kalibrasyonu ve OOD Kapısı | OOF Conformal Prediction (`CI90` kapsamı), aleatorik/epistemik belirsizlik ayrımı ve spektral/kovaryat `OutOfDomainDetector` (`OOD_HIGH_RISK`). |
| **Faz 8** | Bilimsel Açıklanabilirlik (XAI) | Ölçek-normalize yerel duyarlılık, Integrated Gradients, toplu RBF-HSIC (`B < 4` ise fail-closed `insufficient_sample_count`), 2. derece etkileşim (`ADR-006`). |
| **Faz 9** | Karbon Fraksiyonlama, Stok Yayılımı ve Kriging | `TC = SOC + SIC` kapatma denetimi, 1. derece kovaryans yayılımlı stok (`DEF-005` eksik `bulk_density` asla varsayılmaz), opsiyonel 2D `ordinary_kriging` + GeoJSON. |
| **Faz 10** | Sıfır-Pickle Güvenli Paket, `STSoilAnalyzer` ve `st-ds` CLI | `.safetensors` + `preprocess_state.npz` + SHA-256 `manifest.json` (`DEF-006`), `predict` / `predict_batch` kısmi hata yalıtımı, 8 alt komutlu `st-ds` CLI (`visualize` dahil). |
| **Faz 11 & Görselleştirme** | Sürüm Sertleştirme, 26+ 2D/3D Bilimsel Görselleştirme ve Türkçe Yol Haritası | `st_ds.visualization` motoru (26+ grafik, 4 adet 3D yüzey/küme, 5 CSV/MD tablo, çevrimdışı `index.html` panosu), `st-ds visualize`, `st_ds.validation` holdout/replay/profilleme, [FIRST_TIME_ROADMAP_GUIDE.md](FIRST_TIME_ROADMAP_GUIDE.md). |
| **v0.1.0 Nihai Tamamlama** | Çift Dilli (`TR/EN` `.env` & `--lang`), Çok Sayfalı PDF Raporu & Bash Hatları | `st_ds.config` (`.env` opsiyonel), `st_ds.i18n`, `st_ds.reporting` (tüm PNG'leri ve girdi/çıktı yaşam döngüsünü içeren çok sayfalı PDF), hazır `scripts/*.sh` (`runs/` zaman damgalı klasörler), `.gitignore` (`.env` takip edilir), çift dilli [TOPRAK_ANALIZ.md](TOPRAK_ANALIZ.md), toprak-dışı genel rehber [GENERAL_DATA_ANALYSIS_GUIDE.md](GENERAL_DATA_ANALYSIS_GUIDE.md) ve aktif EPO + çoklu hedef (`SOC+SIC+TC`) + tam 2D Kriging desteği. |

---

## 3. Kapatılan Kritik Güvenlik ve Bilimsel Kusurlar (`DEF-001`..`DEF-009`)

1. **`DEF-001` (EPO Uzay Uyuşmazlığı):** `EPO.fit` ve `EPO.transform` arasındaki farklı ön-işleme hatası kaldırıldı; uzay imzası (`space_signature`) uyuşmazlığı `CalibrationError` ile durdurulur.
2. **`DEF-002` (Fiziksel Savitzky–Golay Türevi):** İndeks bazlı türev yerine gerçek $\Delta\lambda$ (nm) ölçekli ve düzensiz grid destekli polinom türevi getirildi.
3. **`DEF-003` / `DEF-009` (Veri Sızıntısı):** Tekrar taramalar (`replicate`), eşleşmiş wet/dry ölçümler ve aynı `site_id` örnekleri asla eğitim/doğrulama arasında bölünmez; tüm ölçekleme/EPO adımları kat içinde sıfırdan eğitilir.
4. **`DEF-004` / `ADR-002` (Birim ve Sabit Kanal Kısıtı):** Birimsiz girdi kabul edilmez; sabit 24 kanal varsayımı yerine dinamik `InstrumentProfile` uygulanır.
5. **`DEF-005` (Sessiz Bulk Density Varsayımı):** `bulk_density` eksik olduğunda asla `1.35 g/cm³` varsayımı yapılmaz; `stock_mg_ha = None` ve `QualityFlag.MISSING_BULK_DENSITY` üretilir.
6. **`DEF-006` (Güvensiz Pickle Çöplüğü):** `.pkl` / `torch.load(weights_only=False)` kullanımı tamamen yasaklandı; yalnızca `.safetensors` + saf sayısal `.npz` + SHA-256 doğrulamalı `manifest.json` yüklenir.

---

## 4. Bilinen Sınırlar ve Kapsam Dışı Öğeler (MVP v0.1.0)

- **Donanım & Sunucu İzolasyonu (`ADR-001`):** USB/UART/I2C sürücüleri, FastAPI/NestJS servisleri, PostgreSQL bağlantısı ve GIS arayüzleri bu sürümde yoktur; yalnızca Python API ve `st-ds` CLI sunar.
- **Sentetik vs Gerçek Saha Verisi (`Risk 9`):** `SyntheticGeneratorV2` tarafından üretilen metrikler yalnızca yazılım ve matematiksel boru hattı doğrulamasıdır; gerçek saha performansı gerçek laboratuvar/arazi eşleşmiş kalibrasyonu gerektirir.
- **Donanım Profillemesi:** Geliştirme aşamasında `macOS arm64` üzerinde ölçülen milisaniye ve bellek metrikleri, Raspberry Pi 5 veya hedef edge donanımına geçildiğinde `benchmarks/run_release_checks.py` ile yerinde yeniden ölçülmelidir.

---

## 5. İlk Çalıştırma Kontrol Listesi (First-Run Checklist)

1. Kurulum yapın (editable): `pip install --no-build-isolation --no-deps -e .` (veya opsiyonel etkileşimli 3D için `pip install -e ".[viz,dev]"`).
2. İlk kez kurulum ve baştan sona yol haritası için [FIRST_TIME_ROADMAP_GUIDE.md](FIRST_TIME_ROADMAP_GUIDE.md) rehberini izleyin.
3. Test paketini doğrulayın: `/opt/anaconda3/bin/pytest tests/ -v`.
4. İlk sentetik uçtan uca deneyi çalıştırın: [SYNTHETIC_TESTING_GUIDE.md](SYNTHETIC_TESTING_GUIDE.md) ve `PYTHONPATH=. python examples/synthetic_python_quickstart.py`.
5. Detaylı Python API, 26+ bilimsel grafik, `.env` dil seçimi (`TR`/`EN`), çok sayfalı PDF üretimi ve `st-ds` CLI komutları için [SYSTEM_AND_USAGE_GUIDE.md](SYSTEM_AND_USAGE_GUIDE.md) belgesine bakın.
6. Matematiksel formüller, fiziksel/istatistiksel problem-çözüm kuramı ve veri dönüşüm yaşam döngüsü için [ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md](ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md) ve toprak analizi özelinde çift dilli [TOPRAK_ANALIZ.md](TOPRAK_ANALIZ.md) kılavuzuna bakın.
7. Tek komutla zaman damgalı `runs/` klasörüne sentetik veri, eğitim veya tam PDF+PNG paketi üretmek için `bash scripts/03_run_full_pipeline.sh` betiğini çalıştırın (`STDS_LANG=en` veya `.env` ile İngilizce seçebilirsiniz).
