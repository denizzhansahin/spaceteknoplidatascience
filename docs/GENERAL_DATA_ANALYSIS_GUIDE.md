# STDS — Toprak Dışı & Genel Spektral / Çok-Sensörlü Veri Analizi Rehberi (`TR & EN`)

> **Dil Seçimi / Language:** **Bölüm A (Türkçe)** · **Part B (English)**  
> **Sürüm / Version:** `st-ds v0.1.0`  
> **Kapsam / Scope:** STDS (**SpaceTeknopoliDataScience**) motoru bugün doğrudan **toprak karbonu analizi (SOC / SIC / TC / Karbon Stoku)** için uzmanlaşmıştır (~%20 domain katmanı); ancak çekirdek mimarisinin **~%80'i tamamen alan-bağımsızdır**. Gelecekte **hava kalitesi, gıda spektroskopisi, donanım/sensör telemetrisi, madencilik veya su kalitesi** analizleri için de `--target <ozel_hedef>`, özel `InstrumentProfile` JSON ve Python API ile nasıl kullanılacağını açıklar.

---

## İçindekiler / Table of Contents

- **Bölüm A — Türkçe Kapsamlı Rehber**
  - [A.1. Büyük Ayrım: %80 Alan-Bağımsız Motor vs. %20 Toprak Karbonu Katmanı](#a1-büyük-ayrım-80-alan-bağımsız-motor-vs-20-toprak-karbonu-katmanı)
  - [A.2. Toprak Dışı Somut Uygulama Alanları](#a2-toprak-dışı-somut-uygulama-alanları)
  - [A.3. CLI ve Şema Sınırları vs. Python API Esnekliği](#a3-cli-ve-şema-sınırları-vs-python-api-esnekliği)
  - [A.4. Örnek Kod 1 — Yaprak Azotu / Cevher Analizi (Custom InstrumentProfile + Bilimsel Paket)](#a4-örnek-kod-1--yaprak-azotu--cevher-analizi-custom-instrumentprofile--bilimsel-paket)
  - [A.5. Örnek Kod 2 — Genel Tabular / Çok-Sensörlü Füzyon, Conformal Belirsizlik, OOD & Kriging](#a5-örnek-kod-2--genel-tabular--çok-sensörlü-füzyon-conformal-belirsizlik-ood--kriging)
- **Part B — Complete English Guide**
  - [B.1. Architectural Split: ~80% Domain-Agnostic Engine vs. ~20% Soil Carbon Layer](#b1-architectural-split-80-domain-agnostic-engine-vs-20-soil-carbon-layer)
  - [B.2. Concrete Non-Soil Application Domains](#b2-concrete-non-soil-application-domains)
  - [B.3. CLI Schema Boundaries vs. Python API Flexibility](#b3-cli-schema-boundaries-vs-python-api-flexibility)
  - [B.4. End-to-End Python Recipe (Custom Wavelengths, EPO Suppression & Full PDF Suite)](#b4-end-to-end-python-recipe-custom-wavelengths-epo-suppression--full-pdf-suite)

---

# Bölüm A — Türkçe Kapsamlı Rehber

## A.1. Büyük Ayrım: %80 Alan-Bağımsız Motor vs. %20 Toprak Karbonu Katmanı

STDS (`st_ds`) mimari olarak modüler kurulmuştur. Çekirdek makine öğrenmesi, sinyal işleme, belirsizlik kalibrasyonu ve raporlama altyapısının **yaklaşık %80'i fiziksel olarak herhangi bir sürekli spektral veya çok-sensörlü regresyon verisine doğrudan uygulanabilir**. Toprağa özgü mantık yalnızca veri şeması etiketlerinde ve karbon kapatma/stok formüllerinde yer alır:

| Katman | Alan-Bağımsız (~%80) — Doğrudan Yeniden Kullanılabilir | Toprağa Özgü (~%20) — Karbon Domain Katmanı |
| --- | --- | --- |
| **Girdi Vektörleri (`domain`)** | `InstrumentProfile` ile **keyfi uzunluk ve birimde** herhangi bir eksen (`nm`, `cm^-1`, `Hz`, kanal indeksi), `SpectrumBlock`, sınırsız sayısal eş değişken (`covariates`), GPS (`lat`, `lon`). | `TargetLabelSchema` alan adları (`soc_pct`, `sic_pct`, `tc_pct`), katman derinliği (`layer_depth_cm`) ve hacim ağırlığı (`bulk_density_g_cm3`). |
| **Ön-İşleme & Kemometri (`calibration`)** | Karanlık/beyaz radyometrik kalibrasyon, ISS sensörler arası aktarım, SNR/doygunluk QC, **SNV**, **MSC**, grid-duyarlı **Savitzky-Golay (`d/dλ`)**, eşleşmiş bozucu etki bastırma (**EPO / GLSW** — nem, sıcaklık veya matris etkisi fark etmeksizin). | Vis-NIR spesifik spektral indeks isimleri (`clay_2200nm`, `carbonate_2340nm`, `organic_slope`) — istenirse özel `FeatureRegistry` ile değiştirilebilir. |
| **Modeller & Eğitim (`models`, `training`)** | `PLSRRegressor`, `KernelRidgeRegressor`, `RandomForestSoilRegressor`, `LocalPLSRRegressor`, `STSoilHybridNetRegressor` (hem 1D skaler hem 2D çoklu hedef $Y \in \mathbb{R}^{n \times k}$), saha/grup bazlı `GroupKFold`. | Çift dallı ağın fiziksel karbon kapatma ağırlığı (`tc_pct ≈ soc_pct + sic_pct`) yalnızca `sic` ve `tc` hedefleri verildiğinde aktive olur. |
| **Belirsizlik, OOD & XAI (`uq`, `xai`)** | **Conformal Prediction** (modelden ve alandan bağımsız dağılım-bağımsız kapsam garantisi), **Mahalanobis + kNN OOD** dedektörü, **Integrated Gradients**, **RBF-HSIC**, 2. derece kısmi türev etkileşimi. | Yok — tamamen matematiksel ve alan-bağımsızdır. |
| **Jeoistatistik & Raporlama (`visualization`, `reporting`)** | Deneysel semivariogram, **2D/3D Ordinary Kriging** (herhangi bir $(lon, lat, z)$ alanı için), 26 PNG grafik, 5 CSV/MD tablo, çevrimdışı `index.html`, çok sayfalı **TR/EN PDF raporu**, SHA-256 `.safetensors` paketi. | Yalnızca **Plot 20** (karbon fraksiyon kapatma) ve **Plot 21** (`0-30 cm` stok & yoğunluk tornadosu) karbon birimlerini bekler; stok `bulk_density` yoksa kendini güvenle kapatır (`None`). |

---

## A.2. Toprak Dışı Somut Uygulama Alanları

Aşağıdaki alanların tamamı STDS'nin **sızıntısız CV, EPO bozucu etki bastırma, Conformal belirsizlik aralığı, OOD kilidi, XAI bant önem sıralaması ve Kriging haritalama** bileşenlerini doğrudan kullanabilir:

1. **Bitki / Yaprak Dokusu & Tarımsal Kalite Spektroskopisi (Agronomi & Gıda)**
   - **Girdi:** Taze veya kuru yaprak Vis-NIR / SWIR spektrumu (`400–2500 nm`), eş değişkenler (`LAI`, `SPAD`, yaprak sıcaklığı, bitki yaşı), parsel GPS koordinatı.
   - **Bozucu Etki Bastırma (EPO):** Taze yapraklardaki **su/nem farkını** (`wet_pairs` = taze sulu yaprak, `dry_pairs` = etüvde kurutulmuş yaprak) EPO matrisi ile ortogonal olarak bastırın; böylece model doğrudan azot, klorofil, protein veya şeker (`Brix`) sinyaline odaklanır.
   - **Çıktı:** Tahmin + %90 Conformal güven aralığı + hangi dalga boylarının belirleyici olduğu (`XAI IG/Sensitivity`).
2. **Mineraloji, Cevher Tenörü & Kayaç Jeokimyası (Madencilik & Jeoloji)**
   - **Girdi:** Karot veya kayaç örneği SWIR / LIBS / FT-IR spektrumu (`1300–2500 nm` kil/karbonat/hidroksil soğurma bantları), yoğunluk/manyetik duyarlılık kovaryatları, sondaj koordinatları (`lon`, `lat`).
   - **Fayda:** Galeri veya maden sahasında **sondaj grubu (`site_id`) bazlı `GroupKFold`** ile farklı damar/bölgeye genelleme başarısını ölçün; saha dışına çıkıldığında (`OOD_HIGH_RISK`) yanlış tenör tahminini engelleyin; `23_ordinary_kriging_2d_ve_3d` ile cevher dağılım haritası üretin.
3. **Su Kalitesi, Atıksu & Çevresel İzleme (Hidroloji & Çevre Mühendisliği)**
   - **Girdi:** UV-Vis absorbans spektrumu (`200–750 nm`, çözünmüş organik madde **DOC** ve nitrat pikleri) + elektrot sensör eş değişkenleri (`EC`, `pH`, `DO`, bulanıklık `NTU`, sıcaklık) + göl/nehir istasyon koordinatları.
   - **EPO Kullanımı:** Bulanıklık veya mevsimsel sıcaklık etkisini eşleşmiş kalibrasyon çiftleriyle bastırın.
   - [x] **Gıda & Kalite Spektroskopisi (Tahıl Proteini / Şeker Brix / Nem) + PAT:** NIR spektrumu (`700–1050 nm`) + örnek sıcaklığı → `protein_pct` veya `brix_deg`.
   - [x] **Hava Kalitesi / Çevresel Gaz & Partikül Analizi:** UV-Vis / optik bantlar + sıcaklık/nem/EC sensörleri → `pm25_ugm3` veya `no2_ppb` + istasyon GPS Kriging haritası.
   - [x] **Donanım & Sensör Telemetrisi / EIS Empedans Analizi:** Çok-frekanslı EIS (`z_real_ohm`, `z_imag_ohm`) + sıcaklık → donanım sürüklenmesi / aşınma indeksi (`wear_index`).
   - [x] **Endüstriyel Proses / Polimer Kalite Kontrolü & Tabular Çok-Sensörlü Füzyon**
   - Yakın kızılötesi (PAT / NIR) hat içi tablet etken madde tayini, yem/tahıl nem-protein analizi veya **yalnızca sayısal sensör dizilerinden** oluşan çok değişkenli kestirim problemleri.

---

## A.3. CLI ve Şema Sınırları vs. Python API Esnekliği

Mevcut `v0.1.0` sürümünde toprak dışındaki verileri analiz ederken bilmeniz gereken iki net yol vardır:

| Yöntem | Ne Zaman Tercih Edilir? | Önemli Detay & Sınır |
| --- | --- | --- |
| **Yol 1: `st-ds` CLI + PDF / 26 PNG Paketi (`--target <özel>` destekli)** | Sıfır kod yazarak **26 grafik, 5 tablo, HTML pano, Kriging haritası ve Tam PDF Raporu** almak istediğinizde. | Kendi sensör dalga boylarınızı içeren özel bir `InstrumentProfile` JSON'unu **`--instrument-profile sensor.json`**, kendi sayısal hedef sütununuzu (ör. `protein_pct`, `pm25_ugm3`, `wear_index`) **`--target protein_pct`** (ve karbona özgü olmayan hedeflerde `--no-require-sic`) olarak geçin; CLI `SoilReference.custom_targets` üzerine okuyup tam eğitim, OOD, Conformal UQ, XAI ve görselleştirmeyi çalıştırır. |
| **Yol 2: Doğrudan Python API (`st_ds.*`)** | Özel kovaryat isimleri (`covariate_names`), saf tabular mühendislik veya programatik döngüler kurmak istediğinizde. | `ChemometricPipeline`, `FeatureFusionPipeline`, `PLSRRegressor` / `KernelRidgeRegressor` / `RandomForestSoilRegressor` / `LocalPLSRRegressor`, `ConformalUncertaintyCalibrator`, `OutOfDomainDetector`, `compute_spatial_kriging_bundle` ve `.safetensors` paketi tamamen hedef isminden bağımsız çalışır. |

---

## A.4. Örnek Kod 1 — Yaprak Azotu / Cevher Analizi (Custom InstrumentProfile + Bilimsel Paket)

Aşağıdaki betik, toprak dışındaki bir **UV-Vis / NIR cihazına** (`700–1100 nm`, 41 kanal) ait örnekleri STDS kemometri boru hattından geçirir, `PLSR` / `KernelRidge` / `RandomForest` modellerini sızıntısız gruplanmış çapraz doğrulama ile eğitir ve sonuçları raporlar:

```python
import numpy as np
from st_ds.domain.instrument import InstrumentProfile, SpectralModality
from st_ds.domain.schemas import (
    CovariateSet,
    Measurement,
    SoilReference,
    SpatialCoordinates,
    Spectrum,
    SpectralSignalMode,
)
from st_ds.chemometrics.pipeline import PreprocessConfig
from st_ds.io.dataset import SoilDataset
from st_ds.training.splits import GroupedSplitBuilder
from st_ds.training.trainer import STSoilTrainer, TrainerConfig

# 1) Toprak dışı cihaz tanımı (ör. Yaprak NIR veya Su UV-Vis spektrometresi)
wl = tuple(float(x) for x in np.linspace(700.0, 1100.0, 41))
ch_ids = tuple(f"ch_{int(round(w))}" for w in wl)
leaf_profile = InstrumentProfile(
    profile_id="leaf_nir_41ch",
    sensor_model="custom_foliar_nir",
    modality=SpectralModality.VIS_NIR_SWIR,
    wavelength_grid_nm=wl,
    channel_ids=ch_ids,
    metadata={"domain": "foliar_agronomy"},
)

# 2) Ölçüm nesneleri (istediğiniz sayısal sensör kovaryatlarını ekleyin)
rng = np.random.default_rng(42)
records = []
wl_arr = np.asarray(wl)
for i in range(60):
    site = f"parsel_{i // 10}"
    nitrogen_pct = 1.8 + 0.06 * i + rng.normal(0.0, 0.08)
    # Klorofil kırmızı-kenar (700-760 nm) ve su bandı (970 nm) yansıma simülasyonu
    spectrum_vals = np.clip(
        0.48
        - 0.07 * nitrogen_pct * np.exp(-0.5 * ((wl_arr - 740.0) / 35.0) ** 2)
        - 0.05 * np.exp(-0.5 * ((wl_arr - 970.0) / 25.0) ** 2)
        + rng.normal(0.0, 0.002, size=wl_arr.size),
        0.01,
        0.99,
    )
    records.append(
        Measurement(
            sample_id=f"LEAF_{i:03d}",
            timestamp_utc="2026-11-13T00:00:00Z",
            instrument_id="foliar_unit_01",
            instrument_profile_id=leaf_profile.profile_id,
            calibration_id="cal_foliar_v1",
            site_id=site,
            spectrum=Spectrum(
                values=spectrum_vals,
                wavelength_grid_nm=wl,
                channel_ids=ch_ids,
                mode=SpectralSignalMode.REFLECTANCE,
            ),
            covariates=CovariateSet(
                temperature_c=float(24.0 + rng.normal(0, 1.5)),
                moisture_gravimetric_pct=float(65.0 + rng.normal(0, 4.0)),
            ),
            coordinates=SpatialCoordinates(
                lat=39.90 + 0.01 * (i // 10) + 0.0005 * (i % 10),
                lon=32.85 + 0.01 * (i % 10),
            ),
            reference=SoilReference(soc_pct=float(nitrogen_pct)),  # Ana hedef yuvası (ör. Azot %)
            dataset_source="real_lab",
        )
    )

dataset = SoilDataset(dataset_id="leaf_n_demo", records=tuple(records), dataset_source="real_lab")

# 3) Sızıntısız gruplanmış çapraz doğrulama kıyası (Parsel bazlı GroupKFold)
cfg = TrainerConfig(
    target_name="soc_pct",
    preprocess_config=PreprocessConfig(use_absorbance=True, use_snv=True, use_savgol=True),
    seed=42,
)
folds = GroupedSplitBuilder.group_kfold_split(dataset, n_splits=3, group_by="site_id")
report = STSoilTrainer(cfg).run_benchmark(dataset, split_folds=folds)
best_key = report.evidence_ranking[0]
print("Kazanan model:", best_key, "OOF RMSE:", report.entries[best_key].oof_metrics.rmse)
```

---

## A.5. Örnek Kod 2 — Genel Tabular / Çok-Sensörlü Füzyon, Conformal Belirsizlik, OOD & Kriging

Eğer spektral bantlarınız hiç yoksa veya tamamen genel `(X, y, lon, lat)` matrisleriyle çalışıyorsanız çekirdek istatistik/makine öğrenmesi modüllerini doğrudan çağırabilirsiniz:

```python
import numpy as np
from st_ds.models.baselines import RandomForestSoilRegressor
from st_ds.uncertainty.conformal import ConformalConfig, ConformalUncertaintyCalibrator
from st_ds.uncertainty.ood import OODConfig, OutOfDomainDetector
from st_ds.spatial.kriging import ordinary_kriging

rng = np.random.default_rng(7)
X = rng.normal(size=(80, 10))
y = 2.5 * X[:, 0] - 1.3 * X[:, 1] + 0.4 * (X[:, 2] ** 2) + rng.normal(0, 0.15, size=80)

# 1) Model eğitimi + epistemik belirsizlik
model = RandomForestSoilRegressor(n_estimators=120, seed=7)
model.set_metadata(feature_names=tuple(f"f_{j}" for j in range(10)), target_names=("y_custom",), dataset_source="real_lab")
model.fit(X[:60], y[:60])

y_cal_pred, cal_std = model.predict_distribution(X[40:60])
conformal = ConformalUncertaintyCalibrator(ConformalConfig(alpha=0.10)).fit(
    y[40:60], y_cal_pred, pred_std_cal=cal_std
)

# 2) Yeni gözlem tahmini + Conformal %90 aralık + OOD kontrolü
ood = OutOfDomainDetector(OODConfig()).fit(X[:60], X[:60])
y_test_pred, test_std = model.predict_distribution(X[60:])
intervals = conformal.predict_interval(y_test_pred, pred_std=test_std)
ood_batch = ood.score_batch(X[60:], X[60:])
print("OOD oranı:", ood_batch.ood_rate, "Ortalama %90 aralık genişliği:", np.mean(intervals.upper_bound - intervals.lower_bound))

# 3) Mekansal 2D Kriging yüzeyi (herhangi bir hedef z için)
lon = 32.8 + rng.uniform(-0.02, 0.02, size=20)
lat = 39.9 + rng.uniform(-0.02, 0.02, size=20)
z_val = 10.0 + 30.0 * (lon - 32.8) + rng.normal(0, 0.1, size=20)
krig = ordinary_kriging(lon, lat, z_val, grid_size=24)
print("Kriging variogram modeli:", krig.variogram.model_name, "Range:", krig.variogram.range_m)
```

---
---

# Part B — Complete English Guide

## B.1. Architectural Split: ~80% Domain-Agnostic Engine vs. ~20% Soil Carbon Layer

`SpaceTeknopoliDataScience (STDS)` is engineered as a modular scientific pipeline. Roughly **80% of the analytical, signal-processing, machine-learning, uncertainty, XAI, geospatial, and reporting stack is domain-agnostic** and applies directly to any spectral, multi-sensor, or spatially indexed continuous target problem. Only ~20% encodes soil-carbon domain physics:

| Subsystem | Domain-Agnostic (~80%) — Directly Reusable | Soil-Carbon-Specific (~20%) |
| --- | --- | --- |
| **Inputs & Units (`domain`)** | Arbitrary wavelength/axis grid via `InstrumentProfile`, arbitrary numeric `covariates`, sample/site lineage, `(lat, lon)` coordinates, `synthetic` vs `real_field` isolation. | Target slot names (`soc_pct`, `sic_pct`, `tc_pct`), soil depth (`layer_depth_cm`), bulk density (`bulk_density_g_cm3`). |
| **Preprocessing & EPO (`calibration`)** | Dark/white calibration, ISS transfer, SNR/saturation QC, **SNV**, **MSC**, physical grid-aware **Savitzky-Golay (`d/dλ`)**, paired nuisance suppression (**EPO / GLSW** against moisture, temperature, or matrix drift). | Named Vis-NIR soil spectral indices (`clay_2200nm`, `carbonate_2340nm`, `organic_slope`) — swappable via custom `FeatureRegistry`. |
| **Models & Leak-Free CV (`models`, `training`)** | `PLSRRegressor`, `KernelRidgeRegressor`, `RandomForestSoilRegressor`, `LocalPLSRRegressor`, `STSoilHybridNetRegressor` (1D scalar or 2D multi-target $Y \in \mathbb{R}^{n \times k}$), `GroupKFold` (`site_id`/`sample_id`). | Hybrid net's physical mass-balance closure term (`tc_pct ≈ soc_pct + sic_pct`) activates only when both `sic` and `tc` targets are supplied. |
| **UQ, OOD & XAI (`uq`, `xai`)** | Distribution-free **Conformal Prediction**, **Mahalanobis + kNN OOD** gating, **Integrated Gradients**, **RBF-HSIC** dependence test, 2nd-order mixed-derivative interaction heatmap. | None — purely mathematical and domain-agnostic. |
| **Geostatistics & Reporting (`visualization`, `reporting`)** | Experimental Matheron semivariogram, **2D/3D Ordinary Kriging** for any scalar field `z(lon, lat)`, 26+ 2D/3D PNGs, 5 CSV/MD tables, offline `index.html`, bilingual **TR/EN multi-page PDF report**, SHA-256 zero-pickle `.safetensors` bundle. | **Plot 20** (SOC+SIC vs TC mass-balance closure) and **Plot 21** (layer carbon stock & density variance tornado) assume carbon targets; missing bulk density cleanly disables stock computation (`None`) without guessing defaults. |

---

## B.2. Concrete Non-Soil Application Domains

1. **Agronomy & Plant Tissue Spectroscopy:** Predict foliar nitrogen (`N%`), chlorophyll, protein, or fruit `Brix` from Vis-NIR leaf spectra; use paired fresh vs. oven-dried leaves in **EPO** (`--suppression-mode epo`) to project out leaf water absorption.
2. **Mining, Mineralogy & Ore Grade Spectroscopy:** Process SWIR / FT-IR / LIBS drill-core spectra alongside density/magnetic susceptibility covariates; enforce **drill-hole grouped CV (`GroupKFold`)**, catch new ore facies via **OOD gating**, and map grades with **2D/3D Kriging**.
3. **Water Quality & Environmental Monitoring:** Fuse UV-Vis absorbance spectra (`200–750 nm`) with probe covariates (`EC`, `pH`, `DO`, turbidity `NTU`) to estimate **DOC**, nitrates, or contaminants across monitoring stations.
4. **Industrial PAT / Polymer / Pharmaceutical Tabular & Multi-Sensor Regression:** Calibrate inline NIR composition models or arbitrary multi-sensor tables with full Conformal uncertainty intervals and SHA-256 verifiable `.safetensors` bundles.

---

## B.3. CLI Schema Boundaries vs. Python API Flexibility

- **Option 1 — Zero-Code CLI (`st-ds`) + Full 26 PNG & PDF Suite:** `TargetLabelSchema` currently validates scalar target keys named `soc_pct`, `sic_pct`, `tc_pct`. For non-soil projects that want the turnkey bash pipeline (`scripts/03_run_full_pipeline.sh`), map your primary scalar target into `soc_pct`, leave `sic_pct`/`tc_pct` and `bulk_density_g_cm3` unset, supply your custom `InstrumentProfile`, and run with `MULTI_TARGET=0 TARGET=soc_pct` (single-target mode). Stock computation cleanly stays `None` (`no hidden 1.35 fallback`).
- **Option 2 — Direct Python API (`st_ds.*`):** Call `STSoilTrainer`, `PLSRRegressor` / `KernelRidgeRegressor` / `RandomForestSoilRegressor` / `LocalPLSRRegressor`, `ConformalUncertaintyCalibrator`, `OutOfDomainDetector`, `XAI`, and `ordinary_kriging` directly with arbitrary target names (`target_names=("nitrate_mg_l",)` or multi-output arrays) and arbitrary covariate matrices.

---

## B.4. End-to-End Python Recipe (Custom Wavelengths, EPO Suppression & Full PDF Suite)

```python
import numpy as np
from st_ds.domain.instrument import InstrumentProfile, SpectralModality
from st_ds.domain.schemas import (
    CovariateSet,
    Measurement,
    SoilReference,
    SpatialCoordinates,
    Spectrum,
    SpectralSignalMode,
)
from st_ds.io.dataset import SoilDataset
from st_ds.visualization.suite import VisualizationConfig, generate_full_visualization_suite

# Example: Mineral / Hydrothermal Clay SWIR reflectance (1300-2500 nm, 61 bands)
wl = tuple(float(x) for x in np.linspace(1300.0, 2500.0, 61))
ch_ids = tuple(f"swir_{int(round(w))}" for w in wl)
swir_profile = InstrumentProfile(
    profile_id="mineral_swir_61ch",
    sensor_model="custom_swir_core_scanner",
    modality=SpectralModality.VIS_NIR_SWIR,
    wavelength_grid_nm=wl,
    channel_ids=ch_ids,
)

rng = np.random.default_rng(123)
wl_arr = np.asarray(wl)
samples = []
for idx in range(48):
    kaolinite_pct = 5.0 + 0.6 * idx + rng.normal(0, 1.2)
    spec = np.clip(
        0.65
        - 0.008 * kaolinite_pct * np.exp(-0.5 * ((wl_arr - 2205.0) / 18.0) ** 2)
        - 0.004 * kaolinite_pct * np.exp(-0.5 * ((wl_arr - 2165.0) / 14.0) ** 2)
        + rng.normal(0, 0.003, size=wl_arr.size),
        0.02,
        0.98,
    )
    samples.append(
        Measurement(
            sample_id=f"CORE_{idx:03d}",
            timestamp_utc="2026-11-13T00:00:00Z",
            instrument_id="core_scanner_01",
            instrument_profile_id=swir_profile.profile_id,
            calibration_id="cal_swir_v1",
            site_id=f"BH_{idx // 8:02d}",
            spectrum=Spectrum(
                values=spec,
                wavelength_grid_nm=wl,
                channel_ids=ch_ids,
                mode=SpectralSignalMode.REFLECTANCE,
            ),
            covariates=CovariateSet(
                temperature_c=float(21.0 + rng.normal(0, 1.0)),
                ec_ds_m=float(0.3 + 0.02 * idx),
            ),
            coordinates=SpatialCoordinates(
                lat=38.40 + 0.004 * (idx // 8) + 0.0004 * (idx % 8),
                lon=27.10 + 0.004 * (idx % 8),
            ),
            reference=SoilReference(soc_pct=float(kaolinite_pct)),
            dataset_source="real_lab",
        )
    )

swir_ds = SoilDataset(dataset_id="mineral_swir_demo", records=tuple(samples), dataset_source="real_lab")

# Generates all 2D/3D plots, 5 CSV/MD tables, interactive 3D HTML, dashboard, and PDF report:
report = generate_full_visualization_suite(
    swir_ds,
    "runs/non_soil_mineral_demo",
    config=VisualizationConfig(target_name="soc_pct", lang="en", include_pdf_report=True),
)
print("Generated plots:", report.n_plots, "PDF:", report.pdf_report_path)
```

---

## İlgili Dokümantasyon / Related Documentation

- **Toprak Analizi Çift Dilli Rehberi (TR & EN):** [TOPRAK_ANALIZ.md](TOPRAK_ANALIZ.md)
- **Akademik Metodoloji & Algoritmalar:** [ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md](ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md)
- **Tam Sistem Mimarisi, Python API & CLI:** [SYSTEM_AND_USAGE_GUIDE.md](SYSTEM_AND_USAGE_GUIDE.md)
- **İlk Kurulum Yol Haritası:** [FIRST_TIME_ROADMAP_GUIDE.md](FIRST_TIME_ROADMAP_GUIDE.md)
