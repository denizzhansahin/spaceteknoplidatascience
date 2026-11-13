# STDS — Veri Seti Hazırlama, Satır/Sütun Kuralları ve Şema Rehberi (`TR & EN`)

> **Dil Seçimi / Language:** **Bölüm A (Türkçe)** · **Part B (English)**  
> **Sürüm / Version:** `st-ds v0.1.0` | **Şema / Schema:** `1.0.0`  
> **Kapsam / Scope:** STDS'ye dışarıdan gerçek veri (toprak, gıda, hava kalitesi, mineraloji veya sensör telemetrisi) hazırlayan kullanıcının bilmesi gereken **tüm dosya formatları, satır/sütun tanımları, veri tipleri, fiziksel birimler, sınır kuralları, spektral bant adlandırması, eğitim vs. tahmin ayrımı ve özel `InstrumentProfile` JSON yapısı**.

---

## İçindekiler / Table of Contents

- **Bölüm A — Türkçe Tam Veri Hazırlama Rehberi**
  - [A.1. Bir Satır Ne Temsil Eder? (Satır Mantığı & Tekillik Anahtarı)](#a1-bir-satır-ne-temsil-eder-satır-mantığı--tekillik-anahtarı)
  - [A.2. Desteklenen 4 Girdi Formatı ve Hangisini Seçmeli?](#a2-desteklenen-4-girdi-formatı-ve-hangisini-seçmeli)
  - [A.3. Eksiksiz Sütun Sözlüğü (Ad, Veri Tipi, Zorunluluk, Birim, Sınır)](#a3-eksiksiz-sütun-sözlüğü-ad-veri-tipi-zorunluluk-birim-sınır)
  - [A.4. Spektral Bant Sütunları (`wl_{nm:.2f}nm`) ve Yerleşik/Özel Cihaz Profilleri](#a4-spektral-bant-sütunları-wl_nm2fnm-ve-yerleşiközel-cihaz-profilleri)
  - [A.5. Eğitim Verisi vs. Tahmin (Inference) Verisi vs. EPO Islak/Kuru Çiftleri](#a5-eğitim-verisi-vs-tahmin-inference-verisi-vs-epo-islakkuru-çiftleri)
  - [A.6. Toprak Dışı Veri Seti Hazırlama (Gıda, Hava, Donanım/Sensör)](#a6-toprak-dışı-veri-seti-hazırlama-gıda-hava-donanımsensör)
  - [A.7. Katı Doğrulama Kuralları (Fail-Closed) ve Sık Yapılan Hatalar](#a7-katı-doğrulama-kuralları-fail-closed-ve-sık-yapılan-hatalar)
- **Part B — Complete English Data Preparation & Schema Guide**
  - [B.1. Row Semantics & Supported Formats](#b1-row-semantics--supported-formats)
  - [B.2. Complete Column & Data Type Specification](#b2-complete-column--data-type-specification)
  - [B.3. Spectral Columns, Custom InstrumentProfile JSON & Custom Targets](#b3-spectral-columns-custom-instrumentprofile-json--custom-targets)

---

# Bölüm A — Türkçe Tam Veri Hazırlama Rehberi

## A.1. Bir Satır Ne Temsil Eder? (Satır Mantığı & Tekillik Anahtarı)

Tablosal bir dosyada (`CSV` / `Parquet`) **her satır (record)** tek bir fiziksel sensör taramasını (`Measurement`) temsil eder. Sistem aynı fiziksel örneğin birden fazla tekrar taramasını (`replicate_id`) ve farklı nem koşullarında alınan ölçümlerini (`moisture_scenario`) destekler, ancak her satır için aşağıdaki **3 parçalı birleşik anahtar tekil olmak zorundadır**:

```text
Benzersiz Satır Anahtarı = (sample_id, replicate_id, moisture_scenario)
```

- Aynı `sample_id` değerini hem `"dry"` hem `"wet"` koşulunda taradıysanız geçerlidir (`("ORN_001", "rep_1", "dry")` ve `("ORN_001", "rep_1", "wet")`).
- Aynı `sample_id` için aynı `moisture_scenario` altında 3 tekrar aldıysanız `replicate_id` değerlerini `"rep_1"`, `"rep_2"`, `"rep_3"` şeklinde ayırmalısınız; aksi halde `SchemaError` (duplicate key) fırlatılır.
- **Veri Sızıntısı Önleme Kuralı (`ADR-003`):** Çapraz doğrulama bölünmeleri (`GroupKFold`) varsayılan olarak **`site_id`** sütununa göre gruplanır. Farklı saha/tarla/parti kimlikleri verilmezse (`n_unique_sites < 2`), grup bazlı çapraz doğrulama çalıştırılamaz.

---

## A.2. Desteklenen 4 Girdi Formatı ve Hangisini Seçmeli?

| Format | Uzantı | Ne Zaman Kullanılır? | Komutta Nasıl Verilir? |
| --- | --- | --- | --- |
| **1. Kanonik JSON (`SoilDataset`)** | `.json` | En eksiksiz format: 2D multi-shot LIBS matrisleri, karanlık/beyaz referans dizileri, EIS frekans süpürmeleri (`EISMeasurement`), `missing_reasons` ve SHA-256 manifest bütünlüğü korunur. | `--input verim.json` |
| **2. Yan Dosyalı CSV** | `.csv` + `.csv.meta.json` | `save_dataset_csv()` ile kaydedilmiş 1D spektral veri; Excel/pandas uyumludur, dalga boyu ızgarası yan `.meta.json` dosyasından otomatik okunur. | `--input verim.csv` |
| **3. Ham Dış CSV (Sidecar Olmadan)** | `.csv` | Kendi laboratuvar veya cihazınızdan doğrudan aktardığınız ham CSV tablosu. **Tek şart:** Spektral kolonların `wl_{nm:.2f}nm` formatında olması ve komuta `--instrument-profile` verilmesidir. | `--input ham_veri.csv --instrument-profile visnir_24ch_v1` (veya `profil.json`) |
| **4. Parquet + Sidecar** | `.parquet` + `.parquet.meta.json` | Çok büyük (`10.000+` satır) veri setlerinde hızlı disk okuma/yazma için (Python `load_dataset_parquet` / `save_dataset_parquet`). | Python API üzerinden yüklenir |

---

## A.3. Eksiksiz Sütun Sözlüğü (Ad, Veri Tipi, Zorunluluk, Birim, Sınır)

### 1. Kimlik, Köken ve Grup Sütunları

| Sütun Adı | Veri Tipi | Eğitimde Zorunlu? | Tahminde Zorunlu? | Varsayılan / Açıklama |
| --- | --- | --- | --- | --- |
| `sample_id` | `str` | **Evet** | **Evet** | Boş olamaz (`"TR_ANK_001"`). Referans laboratuvar ve wet/dry EPO eşleştirmeleri bu sütunla yapılır. |
| `site_id` | `str` | **Evet** (≠ `"unknown_site"`) | Önerilir | Saha / tarla / istasyon / parti kimliği. `GroupKFold` sızıntısız bölmeleri için eğitimde en az `2` farklı `site_id` şarttır. |
| `timestamp_utc` | `str` (ISO-8601) | **Evet** | **Evet** | Örn: `"2026-11-12T10:30:00Z"` veya `"+00:00"`. Spektral ve yardımcı sensör zaman hizalamasında (`max_time_delta_s`) kullanılır. |
| `instrument_id` | `str` | **Evet** | **Evet** | Fiziksel cihaz seri/birim kimliği (ör. `"field_probe_01"`). |
| `instrument_profile_id` | `str` | **Evet** | **Evet** | Cihaz dalga boyu şeması kimliği (ör. `"visnir_24ch_v1"` veya özel `"gida_nir_12ch"`). Kayıt ile profil kimliği birebir eşleşmelidir. |
| `calibration_id` | `str` | **Evet** | **Evet** | Radyometrik kalibrasyon sürümü (ör. `"cal_2026_11_v1"`). |
| `dataset_source` | `str` | **Evet** | **Evet** | Sadece **`"real_lab"`** veya **`"synthetic"`** olabilir. Aynı dosya/veri setinde bu iki değer asla karıştırılamaz. |
| `session_id` | `str` | Opsiyonel | Opsiyonel | Varsayılan: `"session_1"`. Oturum bazlı hata dilimlemesi için. |
| `replicate_id` | `str` | Opsiyonel | Opsiyonel | Varsayılan: `"rep_1"`. Tekrar taramalarını ayırmak için (`rep_1`, `rep_2`). |
| `moisture_scenario` | `str` | Opsiyonel | Opsiyonel | Yalnızca `"dry"`, `"wet"` veya `"field_moist"` (varsayılan `"dry"`). EPO içindeki wet/dry eşlemesi bunu kullanır. |
| `sample_preparation` | `str` | Opsiyonel | Opsiyonel | Varsayılan: `"air_dried_sieved_2mm"`. |
| `soil_texture_class` | `str / null` | Opsiyonel | Opsiyonel | Örn. `"clay_loam"`, `"sandy_loam"`. Alt grup performans tablolarında kullanılır. |

### 2. Derinlik & Konum Sütunları

| Sütun Adı | Veri Tipi | Kanonik Birim | Fiziksel Sınır | Zorunluluk ve Davranış |
| --- | --- | --- | --- | --- |
| `depth_top_cm` | `float` | `cm` | `[0.0, 500.0]` | Opsiyonel (varsayılan `0.0`). `depth_bottom_cm > depth_top_cm` zorunludur. |
| `depth_bottom_cm` | `float` | `cm` | `[0.0, 500.0]` | Opsiyonel (varsayılan `30.0`). Karbon stoku kalınlığı $d = \text{bottom} - \text{top}$ için kullanılır. |
| `lat` | `float / null` | Derece (`EPSG:4326`) | `[-90.0, 90.0]` | **Plot 22–23 (Variogram & Kriging)** için gerekli (`lon` ile birlikte, en az 8 farklı `(lon, lat)` noktası). |
| `lon` | `float / null` | Derece (`EPSG:4326`) | `[-180.0, 180.0]` | **Plot 22–23** için gerekli. Eksikse Kriging haritası kendini güvenle kilitler (Fail-Closed). |
| `crs_epsg` | `str` | — | — | Opsiyonel (varsayılan `"EPSG:4326"`). |

### 3. Çevresel & Sensör Eş Değişkenleri (`covariates`)

Eksik bırakıldıklarında (`NaN` / `None` / boş hücre) eğitim katında öğrenilen medyan + eksiklik göstergesi (`median_with_indicator`) devreye girer; yalnızca `bulk_density_g_cm3` eksikse **stok tahmini sessiz varsayım yerine kapatılır (`None`)**:

| Sütun Adı | Veri Tipi | Kanonik Birim | `UnitRegistry` İzinli Birim Alternatifleri | Fiziksel Sınır | Rol |
| --- | --- | --- | --- | --- | --- |
| `ph` | `float / null` | boyutsuz | `dimensionless`, `ph`, `unitless` | `[2.0, 12.0]` | Model özelliği (kovaryat dalı) |
| `moisture_volumetric_pct` | `float / null` | `%` (`pct`) | `pct`, `%`, `m3/m3`, `cm3/cm3`, `v/v`, `fraction` | `[0.0, 100.0]` | Model özelliği & alt grup nem analizi |
| `moisture_gravimetric_pct` | `float / null` | `%` (`pct`) | `pct`, `%`, `g/g`, `kg/kg`, `fraction` | `[0.0, 200.0]` | Opsiyonel gravimetrik nem |
| `temperature_c` | `float / null` | `°C` (`degC`) | `degc`, `c`, `celsius`, `k`, `degf`, `f` | `[-40.0, 85.0]` | Sensör / numune sıcaklık sürükleme kovaryatı |
| `ec_ds_m` | `float / null` | `dS/m` | `ds/m`, `ms/cm`, `s/m`, `us/cm`, `ms/m` | `[0.0, 200.0]` | Elektriksel iletkenlik kovaryatı |
| `bulk_density_g_cm3` | `float / null` | `g/cm³` (`g/cm3`) | `g/cm3`, `mg/m3`, `t/m3`, `kg/m3` | `[0.2, 2.65]` | **Karbon Stoku (`Mg C ha^-1`) için ZORUNLU** |
| `bulk_density_source` | `str / null` | — | `"measured_core"`, `"pedotransfer"`, vb. | — | Ölçüm provenance bilgisi |
| `coarse_fragments_fraction` | `float / null` | oran `[0, 1)` | `fraction`, `g/g`, `v/v`, `pct`, `%` | `[0.0, 0.99]` | `>2 mm` taşlık düzeltme katsayısı ($1 - f_c$) |

### 4. Referans Hedef Değişkenleri (`reference`)

| Sütun Adı | Veri Tipi | Kanonik Birim | Fiziksel Sınır | Açıklama |
| --- | --- | --- | --- | --- |
| `soc_pct` | `float / null` | `% w/w` (kuru baz) | `[0.0, 60.0]` | Toprak Organik Karbonu. `g/kg`, `ppm` vb. veriler Python'da `SoilReference.from_units(..., unit="g/kg")` ile çevrilebilir. |
| `sic_pct` | `float / null` | `% w/w` (kuru baz) | `[0.0, 60.0]` | Toprak İnorganik Karbonu. `--no-require-sic` ile opsiyonel yapılabilir; `--multi-target` ile ortak eğitilir. |
| `tc_pct` | `float / null` | `% w/w` (kuru baz) | `[0.0, 60.0]` | Toplam Karbon. Verilmezse ve `soc_pct` + `sic_pct` varsa `TC = SOC + SIC` otomatik türetilir; 3'ü de varsa `|TC - (SOC + SIC)| <= 0.05%` kapatma kuralı denetlenir. |
| `soc_method`, `sic_method`, `tc_method` | `str / null` | — | — | Laboratuvar metodu (ör. `"dry_combustion_acid_pretreated_iso10694"`). Toprak eğitiminde varsayılan atanır. |
| `soil_class_id`, `soil_class_name` | `int / str` | — | — | Opsiyonel toprak sınıf etiketi. |
| **Özel Hedef Kolonu** (ör. `protein_pct`, `pm25_ugm3`) | `float / null` | Serbest (ör. `%`, `µg/m³`) | Sonlu sayı (`np.isfinite`) | **Toprak dışı veya özel hedef!** `SoilReference.custom_targets` içine ve ham CSV sütunu olarak eklenip `--target <kolon_adi>` ile eğitilebilir. |

---

## A.4. Spektral Bant Sütunları (`wl_{nm:.2f}nm`) ve Yerleşik/Özel Cihaz Profilleri

CSV/Parquet dosyalarında spektral ölçümler **her dalga boyu bir sütun olacak şekilde** (`1D` vektör) yer alır. Kolon adları kesinlikle iki ondalıklı nanometre formatında olmalıdır:

```text
wl_{dalga_boyu:.2f}nm   -->   Örnek: wl_450.00nm, wl_550.00nm, wl_1415.00nm, wl_2520.00nm
```

### Yerleşik 5 Cihaz Profili (`st_ds.domain.instrument.BUILTIN_PROFILES`)

| `instrument_profile_id` | Kanal Sayısı | Dalga Boyu Aralığı (nm) | Modalite (`SpectralModality`) | Kullanım Senaryosu |
| --- | --- | --- | --- | --- |
| `led_5band_v1` | `5` | `460, 660, 850, 940, 1450` | `led_discrete` | Düşük maliyetli 5-LED saha probu |
| `multispectral_18ch_v1` | `18` | `410 – 940` | `multispectral_array` | AS7265x / kompakt Vis-NIR dizi sensör |
| `visnir_24ch_v1` | `24` | `450 – 2520` | `vis_nir_swir` | Varsayılan tam aralıklı toprak + su + karbonat sensörü |
| `ftnir_1300_2550nm_v1` | `126` | `1300 – 2550` (`10 nm` adım) | `ft_nir` | Yüksek çözünürlüklü MEMS FT-NIR laboratuvar spektrometresi |
| `libs_ci_hi_v1` | `84` | `192.1–194.1`, `228.7–230.7`, `246.8–248.8`, `655.3–657.3` | `libs` | Lazer indüklü plazma spektroskopisi (C I ve H I pencereleri) |

### Kendi Cihaz Profilinizi (`InstrumentProfile` JSON) Oluşturma

Farklı bir dalga boyu sayınız (ör. `650–1050 nm` arası `9` bantlı gıda NIR veya `220–400 nm` UV hava sensörü) varsa şu JSON dosyasını (`custom_profile.json`) oluşturup CLI'da `--instrument-profile custom_profile.json` olarak verebilirsiniz:

```json
{
  "profile_id": "gida_nir_9ch_v1",
  "sensor_model": "FoodSpec-Mini-9",
  "modality": "vis_nir_swir",
  "wavelength_grid_nm": [700.0, 740.0, 780.0, 820.0, 860.0, 900.0, 940.0, 980.0, 1020.0],
  "channel_ids": ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8", "CH9"],
  "fwhm_nm": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
  "calibration_file_id": "gida_cal_v1"
}
```
> ⚠️ **Kurallar:** `wavelength_grid_nm` mutlaka **kesin artan** sırada olmalı, `150.0–25000.0 nm` aralığında bulunmalı ve CSV'deki kolon adları (`wl_700.00nm`, `wl_740.00nm`, ...) ile birebir örtüşmelidir.

---

## A.5. Eğitim Verisi vs. Tahmin (Inference) Verisi vs. EPO Islak/Kuru Çiftleri

1. **Eğitim & Çapraz Doğrulama Verisi (`st-ds train` / `evaluate`):**
   - Her satırda hedef sütunu (`soc_pct` veya özel `--target` adı) dolu olmalıdır.
   - `site_id` farklı grup/saha isimleri taşımalıdır (en az `2` benzersiz saha; 3-katlı CV için en az `3`).
   - Karbon hedeflerinde (`soc_pct`, `sic_pct`, `tc_pct`) `sic_pct` kolonu yoksa `--no-require-sic` (veya tek hedef) kullanın.
2. **Yeni Örnek Tahmin / Çıkarım Verisi (`st-ds predict`):**
   - Hedef sütunları (`soc_pct`, `sic_pct`, `tc_pct` vb.) **olmayabilir veya boş (`NaN`) bırakılabilir**!
   - `sample_id`, `timestamp_utc`, `instrument_id`, `instrument_profile_id`, `calibration_id`, `dataset_source` ve spektral kolonlar yeterlidir (`site_id` verilmezse `"unknown_site"` alabilir).
3. **Islak / Kuru EPO Kalibrasyon Çiftleri (`--wet` ve `--dry`):**
   - `wet` ve `dry` dosyaları **aynı `sample_id` değerlerine** ve aynı dalga boyu ızgarasına sahip fiziksel eşleri içermelidir (`wet.records[i]` ile `dry.records[j]`, `sample_id` üzerinden otomatik eşleşir).

---

## A.6. Toprak Dışı Veri Seti Hazırlama (Gıda, Hava, Donanım/Sensör)

Örneğin bir **tahıl/gıda protein spektrometresi** veya **hava kalitesi UV sensörü** için 6 satırlık minimalist ham CSV şablonu şöyle görünür (`wl_700.00nm ...` ve özel hedef `protein_pct`):

```csv
sample_id,site_id,timestamp_utc,instrument_id,instrument_profile_id,calibration_id,dataset_source,temperature_c,moisture_volumetric_pct,protein_pct,wl_700.00nm,wl_740.00nm,wl_780.00nm,wl_820.00nm,wl_860.00nm,wl_900.00nm,wl_940.00nm,wl_980.00nm,wl_1020.00nm
GIDA_01,depo_A,2026-11-12T08:00:00Z,nir_01,gida_nir_9ch_v1,cal_v1,real_lab,22.1,11.8,13.4,0.451,0.432,0.419,0.405,0.391,0.376,0.342,0.311,0.298
GIDA_02,depo_B,2026-11-12T08:05:00Z,nir_01,gida_nir_9ch_v1,cal_v1,real_lab,23.0,12.1,14.8,0.438,0.419,0.404,0.390,0.374,0.358,0.328,0.299,0.285
```

Çalıştırma komutu:
```bash
st-ds train --input gida.csv --instrument-profile custom_profile.json --target protein_pct --model plsr --output gida_model_bundle
```
(Ayrıntılı Python & CLI örnekleri için [examples/03_custom_csv_and_non_soil_domains.py](../examples/03_custom_csv_and_non_soil_domains.py) dosyasına bakın.)

---

## A.7. Katı Doğrulama Kuralları (Fail-Closed) ve Sık Yapılan Hatalar

| Hata / Belirti | Gerçek Nedeni | Kesin Çözüm |
| --- | --- | --- |
| `SchemaError: Dataset source mismatch` | Aynı veri setinde `"real_lab"` ve `"synthetic"` satırları karıştırılmış (`Section 31`). | Sentetik ve gerçek laboratuvar verilerini ayrı dosyalarda tutun. |
| `SchemaError: Duplicate record key` | Aynı `(sample_id, replicate_id, moisture_scenario)` iki kez geçiyor. | Tekrar ölçümlere `rep_1`, `rep_2` gibi farklı `replicate_id` verin. |
| `SchemaError: Loading CSV requires either a '.meta.json' sidecar or an explicit InstrumentProfile` | Dış ham CSV verilirken `.meta.json` yoksa `--instrument-profile` geçilmemiş. | Komuta `--instrument-profile visnir_24ch_v1` veya `--instrument-profile profil.json` ekleyin. |
| `UnitError` / `Value X for bulk_density is out of physical bounds` | Birim dönüştürülmeden girilmiş (ör. `g/cm3` yerine `1320 kg/m3` veya `%` yerine `18.5 g/kg` yazılmış). | Tablo A.3'teki kanonik birime çevirin veya Python'da `CovariateSet.from_units` / `SoilReference.from_units` kullanın. |
| `SchemaError: closure residual ... exceeds tolerance` | Aynı satırda `|tc_pct - (soc_pct + sic_pct)| > 0.05%`. | Laboratuvar ölçüm tutarsızlığını düzeltin veya yalnızca bağımsız ölçülen sütunları girin (`tc_pct` boşsa otomatik türetilir). |
| `carbon_stock.stock_mg_ha = None` (`MISSING_BULK_DENSITY`) | `bulk_density_g_cm3` boş bırakılmış. | Toprak stoku istiyorsanız gerçek hacim ağırlığı ölçümü (`0.2..2.65 g/cm³`) girin; toprak dışı analizde normaldir. |
| Kriging haritasında uyarı (< 8 nokta) | `lat`/`lon` eksik ya da 8 farklı `(lon, lat)` koordinat çifti yok. | Her saha/örnek için gerçek ondalık GPS `(lat, lon)` ekleyin. |

---

# Part B — Complete English Data Preparation & Schema Guide

## B.1. Row Semantics & Supported Formats

Every row in a tabular CSV/Parquet file represents a single scan (`Measurement`). Each row MUST have a unique composite key:

```text
Unique Row Key = (sample_id, replicate_id, moisture_scenario)
```

Supported formats:
1. **Canonical JSON (`.json`):** Full lossless representation via `save_dataset_json()` / `load_dataset_json()` with SHA-256 integrity, dark/white references, EIS frequency sweeps, and `custom_targets`.
2. **CSV + `.meta.json` Sidecar:** Created by `save_dataset_csv()`; auto-detects wavelength grid, spectral mode, and custom target columns.
3. **Standalone External CSV (`.csv` only):** Exported directly from Excel/spectrometers using spectral columns named **`wl_{wavelength_nm:.2f}nm`** (e.g., `wl_450.00nm`). Load via `--instrument-profile <builtin_id_or_json_path>`.
4. **Parquet + `.meta.json` Sidecar:** Efficient columnar storage via `save_dataset_parquet()` / `load_dataset_parquet()`.

---

## B.2. Complete Column & Data Type Specification

- **Mandatory Identifiers (`str`):** `sample_id`, `site_id` (unique groups required for grouped CV training), `timestamp_utc` (ISO-8601 UTC), `instrument_id`, `instrument_profile_id`, `calibration_id`, `dataset_source` (`"real_lab"` or `"synthetic"` — never mix inside one dataset).
- **Optional Lineage (`str`):** `session_id` (`"session_1"`), `replicate_id` (`"rep_1"`), `moisture_scenario` (`"dry" | "wet" | "field_moist"`), `sample_preparation`, `soil_texture_class`.
- **Depth & GPS (`float`):** `depth_top_cm` (`0..500`, `bottom > top`), `depth_bottom_cm`, `lat` (`-90..90`), `lon` (`-180..180`), `crs_epsg` (`"EPSG:4326"`). Spatial Kriging (Plots 22–23) requires at least `8` distinct `(lon, lat)` pairs.
- **Covariates (`float`, nullable):** `ph` (`2..12`), `moisture_volumetric_pct` (`0..100`), `moisture_gravimetric_pct` (`0..200`), `temperature_c` (`-40..85`), `ec_ds_m` (`0..200`), `bulk_density_g_cm3` (`0.2..2.65`, mandatory for carbon stock; never silently imputed), `coarse_fragments_fraction` (`0..0.99`).
- **Targets (`float`, nullable during inference):**
  - Canonical soil carbon targets (`0..60%` dry-matter): `soc_pct`, `sic_pct`, `tc_pct` (`|tc_pct - (soc_pct + sic_pct)| <= 0.05%`).
  - **Custom domain targets** (e.g. `protein_pct`, `pm25_ugm3`, `wear_index`): any finite numeric column passed via `--target <col_name>` (CLI) or `SoilReference(custom_targets={...})` (Python).

---

## B.3. Spectral Columns, Custom InstrumentProfile JSON & Custom Targets

Spectral columns in CSV MUST use two-decimal nanometer headers: `wl_{nm:.2f}nm`, ordered strictly ascending to match the active `InstrumentProfile` (`led_5band_v1`, `multispectral_18ch_v1`, `visnir_24ch_v1`, `ftnir_1300_2550nm_v1`, `libs_ci_hi_v1`, or your custom JSON file passed to `--instrument-profile my_sensor.json`).

**Inference files (`st-ds predict`)** may omit all target columns; **paired EPO files (`--wet`, `--dry`)** must share identical `sample_id` values across both files so spectral difference vectors $X_{\text{wet}} - X_{\text{dry}}$ match one-to-one.

---

> **İlgili Kılavuzlar / Related Guides:**  
> • [CLI ve Python Parametreleri Tam Referansı (TR & EN)](CLI_AND_PYTHON_PARAMETERS_GUIDE.md)  
> • [Toprak Dışı & Genel Veri Analizi Rehberi (TR & EN)](GENERAL_DATA_ANALYSIS_GUIDE.md)  
> • [Örnek Kodlar ve Veri Hazırlama Betiği (`examples/`)](../examples/README.md)
