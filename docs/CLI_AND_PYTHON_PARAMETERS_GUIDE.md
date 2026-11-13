# STDS — Tüm CLI & Python Parametreleri Referans Rehberi (`TR & EN`)

> **Dil / Language:** **Bölüm A (Türkçe)** · **Part B (English)**  
> **Sürüm / Version:** `st-ds v0.1.0`  
> **Amaç / Purpose:** Kullanıcıların *"Hedef değişken ve model neden aynıydı? Parametreleri değiştirirsem ne olur? CLI ve Python sınıfları hangi parametreleri ve değer aralıklarını kabul eder? Bash betikleri olmadan doğrudan nasıl çalıştırırım?"* sorularının **eksiksiz, kanıtlı ve kopyala-çalıştır karşılığıdır**.

---

## İçindekiler / Table of Contents

- **Bölüm A — Türkçe Kapsamlı Parametre Sözlüğü**
  - [A.1. Önemli Aydınlatma: `soc_pct` ve `kernel_ridge_rbf` Neden Hep Aynıydı, Nasıl Değiştirilir?](#a1-önemli-aydınlatma-soc_pct-ve-kernel_ridge_rbf-neden-hep-aynıydı-nasıl-değiştirilir)
  - [A.2. Ortam & İsteğe Bağlı `.env` Parametreleri ve Öncelik Kuralı](#a2-ortam--isteğe-bağlı-env-parametreleri-ve-öncelik-kuralı)
  - [A.3. Tüm `st-ds` CLI Alt Komutları ve Bayrakları (`Ne Verirsen Ne Olur?`)](#a3-tüm-st-ds-cli-alt-komutları-ve-bayrakları-ne-verirsen-ne-olur)
  - [A.4. Bash Olmadan Doğrudan CLI Çalıştırma Tarifleri](#a4-bash-olmadan-doğrudan-cli-çalıştırma-tarifleri)
  - [A.5. Python API Yapılandırma Sınıfları & Tüm Parametreler](#a5-python-api-yapılandırma-sınıfları--tüm-parametreler)
- **Part B — Complete English CLI & Python Parameter Reference**
  - [B.1. Parameter Hierarchy, Target/Model Control & Runtime Flags](#b1-parameter-hierarchy-targetmodel-control--runtime-flags)
  - [B.2. CLI Subcommand Matrix & Python Config Defaults](#b2-cli-subcommand-matrix--python-config-defaults)

---

# Bölüm A — Türkçe Kapsamlı Parametre Sözlüğü

## A.1. Önemli Aydınlatma: `soc_pct` ve `kernel_ridge_rbf` Neden Hep Aynıydı, Nasıl Değiştirilir?

1. **Hedef Değişken (`--target`, varsayılan: `soc_pct`):**
   - `soc_pct`, veri setindeki (`SoilReference` veya CSV tablosundaki) **sütun adıdır**.
   - Değiştirmek için **`--target`** bayrağını verirsiniz:
     - Toprak karbonu hedefleri: `soc_pct` (Organik Karbon), `sic_pct` (İnorganik Karbon), `tc_pct` (Toplam Karbon).
     - Özel / toprak dışı hedefler: Veri setinizde (veya `SoilReference.custom_targets` içinde) `protein_pct`, `pm25_ugm3`, `wear_index` gibi hangi sayısal sütun varsa doğrudan `--target protein_pct` verebilirsiniz.
     - Ayrıca `st-ds train --multi-target` kullanıldığında `soc_pct`, `sic_pct` ve `tc_pct` **üçü birden aynı anda** eğitilir (ancak Sayfa 1'deki ana KPI birincil `--target` adını gösterir).
2. **Seçilen Model (`--model`, varsayılan: `auto`):**
   - `--model auto` verildiğinde sistem tüm aday modelleri (`plsr`, `kernel_ridge_rbf`, `random_forest`, `local_plsr_mbl`, `st_soil_hybrid_net`) aynı CV katlarında yarıştırır ve **OOF RMSE'si en düşük olanı otomatik seçer**.
   - Sentetik veri üreteci pürüzsüz Beer-Lambert Gauss eğrileri ürettiği için RBF Kernel Ridge (`kernel_ridge_rbf`) bu 80 örnekte en düşük RMSE'yi alır ve `auto` hep onu seçer.
   - Belli bir modeli kendiniz seçmek için hem `st-ds train` hem `st-ds visualize` komutlarında **`--model plsr`**, **`--model random_forest`**, **`--model local_plsr`**, **`--model hybrid_net`** veya **`--model kernel_ridge_rbf`** verebilirsiniz!

---

## A.2. Ortam & İsteğe Bağlı `.env` Parametreleri ve Öncelik Kuralı

Sistem `.env` dosyasına bağımlı değildir; hiç `.env` olmasa bile yerleşik varsayılanlarla çalışır. Varsa öncelik sırası:

```text
Komut/Python Açık Parametresi (--lang en)  >  OS Ortam Değişkeni (STDS_LANG=en)  >  .env Dosyası  >  Varsayılan ('tr')
```

| Değişken | Alabileceği Değerler | Varsayılan | Ne Verirsen Ne Olur? |
| --- | --- | --- | --- |
| `STDS_LANG` | `tr`, `en` (alias: `turkish`, `eng`, `tr-tr`, `en-us`) | `tr` | Tüm grafik başlıkları, PDF sayfaları, tablolar ve `index.html` dili değişir. |
| `STDS_RUNS_DIR` | Göreli veya mutlak klasör yolu | `runs` | `scripts/*.sh` betiklerinin çıktı klasörü (`runs` veya `runs_examples`). |
| `STDS_DPI` | Tamsayı `>= 72` (ör. `100`, `160`, `220`, `300`) | `160` | PNG grafik çözünürlüğü. Yüksek DPI baskı kalitesi verir ama PDF/disk boyutunu büyütür. |
| `STDS_PDF_EXPORT` | `true`/`1`/`yes` veya `false`/`0`/`no` | `true` | `false` verilirse `visualize` yalnızca PNG/CSV/HTML üretir, PDF oluşturmaz. |
| `STDS_DEFAULT_PROFILE` | Yerleşik profil ID'si | `visnir_24ch_v1` | Çalışma zamanı varsayılan spektral sensör profili kimliği. |
| `STDS_ENV_FILE` | `.env` dosya yolu | *bulunan ilk `.env`* | Farklı bir ayar dosyasını zorla okutmak için kullanılır. |

---

## A.3. Tüm `st-ds` CLI Alt Komutları ve Bayrakları (`Ne Verirsen Ne Olur?`)

> 💡 **Çağrı Şekli:** Paket kuruluysa **`st-ds ...`**, kurulu değilse repo kökünden **`PYTHONPATH=. python -m st_ds.cli ...`** yazın.

### Genel Kök Bayraklar

| Bayrak | Alabileceği Değer | Varsayılan | Etkisi |
| --- | --- | --- | --- |
| `--version` | *(bayrak)* | — | Sürümü (`st-ds 0.1.0`) basar ve çıkar. |
| `--debug` | *(bayrak)* | `False` | Normalde temiz JSON hata veren CLI'da tam Python hata izini (traceback) basar. |
| `--lang` | `tr`, `en` | `None` → `tr` | Aktif çıktı/rapor dilini `.env` üstüne yazar. |

---

### 1) `st-ds dataset validate` — Şema, Birim ve Etiket Doğrulama

| Parametre | Tip / Seçenekler | Varsayılan | Ne Verirsen Ne Olur? |
| --- | --- | --- | --- |
| `--input` *(Zorunlu)* | `.json` veya `.csv` yolu | — | Yüklenecek veri seti. CSV'nin `.meta.json` yanında yoksa `--instrument-profile` ister. |
| `--instrument-profile` | Profil ID'si (`visnir_24ch_v1`, `led_5band_v1`, `multispectral_18ch_v1`, `ftnir_1300_2550nm_v1`, `libs_ci_hi_v1`) veya `.json` dosya yolu | `None` | Yanında `.meta.json` olmayan ham CSV'lerin `wl_*.nm` bantlarını bu profile göre okur ve doğrular. |
| `--for-training` | *(bayrak)* | `False` | Açılırsa her satırda geçerli `site_id` (`≠ unknown_site`) ve hedef etiketi arar; tahmin dosyası kontrolü için kapalı tutun. |
| `--target` | Herhangi geçerli hedef adı (`soc_pct`, `sic_pct`, `tc_pct`, `protein_pct`, ...) | `soc_pct` | `--for-training` ile birlikte hangi referans kolonunun/hedefin varlığının denetleneceğini belirler. |
| `--no-require-sic` | *(bayrak)* | `False` | Açılırsa karbon eğitim verisinde `sic_pct` boş olan satırlara izin verilir. |
| `--output` | `.json` yolu | `None` | Doğrulama sonucunu (`is_valid`, `errors`, `warnings`) bu JSON dosyasına da kaydeder. |

---

### 2) `st-ds calibration fit-epo` — Nem / Bozucu Etki Bastırma Matrisi Uydurma

| Parametre | Tip / Seçenekler | Varsayılan | Ne Verirsen Ne Olur? |
| --- | --- | --- | --- |
| `--wet`, `--dry` *(Zorunlu)* | `.json` veya `.csv` yolu | — | Aynı `sample_id` değerlerine sahip ıslak ve kuru kalibrasyon çiftleri. Fark matrisi $D = Z_{\text{wet}} - Z_{\text{dry}}$ SVD'si alınır. |
| `--instrument-profile` | Profil ID'si veya `.json` yolu | `None` | Ham CSV çiftleri verildiyse dalga boyu şemasını çözer. |
| `--output` *(Zorunlu)* | Klasör veya `.json` | — | `epo_state.json` (SHA-256 `space_signature`, rank `q`, ortogonallik/projeksiyon invariantları) + `epo_state.npz` ($P, V_s, Q$) yazar. |
| `--q` | Tamsayı (`1 .. n_bands-1`) | `None` (otomatik) | Verilmezse rank $q$ scree/enerji eşiğiyle otomatik seçilir; ör. `--q 2` ilk 2 nem tekil bileşenini ortogonal olarak siler (`P = I - V_s V_s^T`). |
| `--no-absorbance` | *(bayrak)* | `False` | Reflectance → $-\log_{10}(R)$ dönüşümünü kapatır (veriniz zaten absorbans ise kullanın). |
| `--no-snv` | *(bayrak)* | `False` | Satır içi Standard Normal Variate (ortalama sıfır, std bir) ölçeklemeyi kapatır. |
| `--no-savgol` | *(bayrak)* | `False` | Savitzky–Golay 1. türev yumuşatmasını kapatır. |

---

### 3) `st-ds evaluate` — Sızıntısız Çoklu Model Çapraz Doğrulama Kıyaslaması

| Parametre | Tip / Seçenekler | Varsayılan | Ne Verirsen Ne Olur? |
| --- | --- | --- | --- |
| `--input` *(Zorunlu)* | `.json` / `.csv` | — | Eğitim/kıyaslama veri seti. |
| `--target` | Herhangi hedef adı (`soc_pct`, `sic_pct`, `tc_pct`, özel hedef) | `soc_pct` | Tüm modeller bu hedef vektörü üzerinde eğitilip RMSE/MAE/R²/RPIQ hesaplanır. |
| `--instrument-profile` | Profil ID veya `.json` yolu | `None` | Ham CSV girişinde spektral şemayı sağlar. |
| `--covariates` | Virgüllü kolon adları veya `none` | `ph,moisture_volumetric_pct,temperature_c,ec_ds_m` | Toprak dışı/özel CSV'lerde yalnız mevcut yardımcı sütunları seçer veya `none` ile yalnızca spektral çalışır. |
| `--folds` | Tamsayı `>= 2` | `3` | `GroupKFold(site_id)` kat sayısı (veri setindeki saha sayısından büyükse otomatik `min(folds, n_sites)` alınır). |
| `--suppression-mode` | `none`, `epo`, `glsw` | `none` | `epo`: dik nem bastırma uygular; `glsw`: istatistiksel grup içi kovaryans sönümleme uygular (`--wet` + `--dry` veya veri içi `wet/dry` çiftleri gerekir). |
| `--epo-q` | Tamsayı `>= 1` | `None` | EPO projeksiyon rankı $q$. |
| `--wet`, `--dry` | `.json` / `.csv` | `None` | Harici eşleşmiş kalibrasyon çiftleri (her CV eğitim katında yalnızca `train_idx` veya harici kalibrasyon uzayı kullanılarak sızıntı engellenir). |
| `--include-optional` | *(bayrak)* | `False` | Kapalıysa yalnız `plsr` + `kernel_ridge_rbf`; açılırsa `random_forest`, `local_plsr_mbl` ve `st_soil_hybrid_net` da kıyaslanır. |
| `--seed` | Tamsayı | `42` | Tekrarlanabilir tohum (RF, PyTorch ağırlık ilklendirme ve shuffle). |
| `--output` | `.json` yolu | `None` | Kıyaslama JSON raporunu diske yazar. |

---

### 4) `st-ds train` — Model Eğitimi ve Sıfır-Pickle `.safetensors` Artefakt Üretimi

| Parametre | Tip / Seçenekler | Varsayılan | Ne Verirsen Ne Olur? |
| --- | --- | --- | --- |
| `--input` *(Zorunlu)* | `.json` / `.csv` | — | Eğitim veri seti. |
| `--output` *(Zorunlu)* | Klasör yolu | — | `.safetensors` ağırlıkları + `manifest.json` (SHA-256) + `MODEL_CARD.md` + ön-işleme/Conformal/OOD durumunu yazar. |
| `--model` | `auto`, `plsr`, `kernel_ridge` (`kernel_ridge_rbf`), `random_forest`, `local_plsr` (`local_plsr_mbl`), `hybrid_net` (`st_soil_hybrid_net`) | `auto` | `auto`: Tüm 5 modeli kıyaslayıp en düşük OOF RMSE'li modeli seçer. Sabit bir model verirseniz doğrudan o aile eğitilir. |
| `--target` | `soc_pct`, `sic_pct`, `tc_pct` veya özel hedef adı | `soc_pct` | Ana tahmin/kalibrasyon hedefi (ör. `--target tc_pct` veya `--target protein_pct`). |
| `--multi-target` | *(bayrak)* | `False` | Açılırsa veri setinde eksiksiz olan `soc_pct`, `sic_pct`, `tc_pct` hedeflerini ortak eğitir (çıktıda kütle kapatma ve tam SIC/TC üretir). |
| `--suppression-mode` | `none`, `epo`, `glsw` | `none` | Kemometrik ön-işleme bastırma katmanı. |
| `--epo-q`, `--wet`, `--dry` | Tamsayı / Dosya yolu | `None` | `--suppression-mode epo/glsw` için rank ve harici nem çiftleri. |
| `--folds` | Tamsayı `>= 2` | `3` | OOF kalibrasyon ve Conformal belirsizlik aralığı için kat sayısı. |
| `--hybrid-epochs` | Tamsayı `>= 1` | `35` | `STSoilHybridNetRegressor` PyTorch eğitim epok sayısı (hızlı duman testi için `8`, tam eğitim için `35–80`). |
| `--artifact-id` | Metin | `stds_<model>_<dataset>` | Artefakt kimliği etiketi. |
| `--semantic-version` | Metin | `0.1.0` | Artefakt semantik sürüm numarası. |
| `--instrument-profile` | Profil ID / `.json` | `None` | Ham CSV girdisinde dalga boyu profili. |
| `--covariates` | Virgüllü kolon adları veya `none` | Standart 4 toprak kovaryatı | Gıda/hava/özel analizde mevcut olmayan toprak sütunlarını (ör. `ph`) hariç tutmak veya `none` yapmak için kullanılır. |
| `--pdf` | `.pdf` yolu | `None` | Verilirse eğitim özeti PDF'i (`--lang` dilinde) ayrıca üretilir. |
| `--lang` | `tr`, `en` | `.env` / `tr` | PDF özet rapor dili. |
| `--seed` | Tamsayı | `42` | Deterministik eğitim tohumu. |

---

### 5) `st-ds artifact inspect` — Paket Bütünlük (SHA-256) ve Metadata Kontrolü

- `--artifact <klasör>` *(Zorunlu)*: Tüm dosyaların SHA-256 karma değerini `manifest.json` ile doğrular (dosyaya elle müdahale edildiyse hata fırlatır).
- `--output <dosya.json>`: Doğrulama özetini JSON olarak yazar.

---

### 6) `st-ds predict` — Eğitilmiş Artefakt ile Tekil veya Toplu Çıkarım

| Parametre | Tip / Seçenekler | Varsayılan | Ne Verirsen Ne Olur? |
| --- | --- | --- | --- |
| `--artifact` *(Zorunlu)* | Artefakt klasörü | — | Önce SHA-256 + cihaz profili uyumluluğu doğrulanır, uyuşmazsa `ArtifactCompatibilityError` verilir. |
| `--input` *(Zorunlu)* | Tekil `Measurement` `.json` veya `SoilDataset` `.json`/`.csv` | — | Tekil JSON gelirse `PredictionReport`, liste/dataset gelirse `BatchPredictionResult` döner. Hedef kolonlar gerekmez. |
| `--instrument-profile` | Profil ID / `.json` | Artefakt profili | Ham CSV tahmin dosyası için profil belirtir; verilmezse artefaktın kendi profilini kullanır. |
| `--include-xai` | *(bayrak)* | `False` | Açılırsa her örnek için yerel duyarlılık + Integrated Gradients (`B >= 4` toplu veride ek olarak HSIC) hesaplar. |
| `--no-stock` | *(bayrak)* | `False` | Açılırsa `bulk_density_g_cm3` olsa bile karbon stoku hesaplanmaz. |
| `--fail-fast` | *(bayrak)* | `False` | Kapalıyken toplu tahminde hatalı satır izole edilir (`errors` listesine konur); açılırsa ilk bozuk örnekte durur. |
| `--output`, `--pdf`, `--lang` | Yol / `tr|en` | `None` | JSON çıktısı ve isteğe bağlı tekil/toplu PDF raporu üretir. |

---

### 7) `st-ds report export` — Appendix B Doğrulama ve PDF Dışa Aktarım

- `--input` *(Zorunlu)*: `PredictionReport`, `BatchPredictionResult` veya `train` özet JSON'u.
- `--format`: **`json`** (sadece doğrulanmış JSON), **`pdf`** (sadece PDF), **`both`** (hem doğrulanmış JSON hem PDF).
- `--output` *(Zorunlu)*: Çıktı yolu; `--format both` ile PDF ayrı yere yazılacaksa `--pdf-output rapor.pdf` verilir.
- `--lang`: `tr` veya `en`.

---

### 8) `st-ds visualize` — 26+ 2D/3D Grafik, 5 CSV/MD Tablo, HTML Pano & Tam Kapsamlı PDF

| Parametre | Tip / Seçenekler | Varsayılan | Ne Verirsen Ne Olur? |
| --- | --- | --- | --- |
| `--input` *(Zorunlu)* | `.json` / `.csv` | — | Görselleştirilecek veri seti. |
| `--output` *(Zorunlu)* | Klasör yolu | — | `plots/`, `tables/`, `interactive/`, `reports/`, `index.html`, `viz_manifest.json` burada oluşur. |
| `--artifact` | Artefakt klasörü | `None` | Verilirse kayıtlı modeli kullanır; verilmezse bu veri seti üzerinde geçici in-memory model eğitir. |
| `--instrument-profile` | Profil ID / `.json` | `None` | Ham CSV girdisinde dalga boyu tanımı. |
| `--wet`, `--dry` | `.json` / `.csv` | `None` | Verilirse in-memory EPO eğitilir ve Grafik 03 (`EPO/GLSW Nem Etkisi Bastırma`) iki eğriyle tam çizilir. |
| `--target` | Herhangi hedef (`soc_pct`, `sic_pct`, `tc_pct`, özel) | `soc_pct` | Tüm parite, Bland-Altman, kalıntı, XAI, Semivariogram ve 2D/3D Kriging grafikleri bu hedef için üretilir. |
| `--model` | `auto`, `plsr`, `kernel_ridge_rbf`, `random_forest`, `local_plsr_mbl`, `st_soil_hybrid_net` | `auto` | `--artifact` verilmediğinde grafik/raporda vitrine çıkarılacak modeli belirler (`auto` = en düşük OOF RMSE). |
| `--covariates` | Virgüllü kolon adları veya `none` | Artefakt ayarı veya standart 4 kovaryat | Özel/toprak dışı veri setlerinde grafik + belirsizlik boru hattının kullanacağı kovaryatları seçer. |
| `--folds` | Tamsayı `>= 2` | `4` | Grafik 08 (Model Kıyaslama) kat sayısı. |
| `--format` | Virgüllü metin (`png`, `svg`, `png,svg`) | `png` | Vektörel SVG kopyalarını da üretmek için `png,svg` verin. |
| `--dpi` | Tamsayı `>= 72` | `.env` / `160` | PNG kalitesi. |
| `--no-html`, `--no-interactive-3d`, `--no-pdf` | *(bayraklar)* | `False` | İlgili çıktıları atlayarak hız kazandırır. |
| `--include-optional` | *(bayrak)* | `False` | Grafik 08 ve Tablo 02'ye 5 modelin tamamını (`RandomForest`, `LocalPLSR`, `HybridNet` dahil) sokar. |
| `--lang` | `tr`, `en` | `.env` / `tr` | Tüm görsel/tablo/PDF metinlerinin dili. |
| `--seed` | Tamsayı | `42` | Tekrarlanabilir grafik düzeni ve model tohumu. |

---

## A.4. Bash Olmadan Doğrudan CLI Çalıştırma Tarifleri

Herhangi bir `.sh` dosyasına bağlı kalmadan kendi terminalinizde tek seferde kullanabileceğiniz 4 doğrudan örnek:

```bash
# 1) Ham dış CSV dosyasını özel hedef (tc_pct) + sabit PLSR modeliyle eğitip PDF özet al:
st-ds train \
  --input toprağım.csv \
  --instrument-profile visnir_24ch_v1 \
  --target tc_pct \
  --model plsr \
  --folds 4 \
  --suppression-mode epo --epo-q 2 --wet islak.csv --dry kuru.csv \
  --lang tr \
  --pdf egitim_ozeti.pdf \
  --output modeller/tc_plsr_bundle

# 2) Tüm modelleri yarıştırıp yalnızca doğrulama raporu çıkar (JSON):
st-ds evaluate \
  --input toprağım.csv \
  --instrument-profile visnir_24ch_v1 \
  --target soc_pct \
  --folds 5 \
  --include-optional \
  --output benchmark_raporu.json

# 3) Hedef sütunu OLMAYAN yeni saha örneklerini tahmin et + XAI açıklaması + PDF üret:
st-ds predict \
  --artifact modeller/tc_plsr_bundle \
  --input yeni_sahalar.csv \
  --include-xai \
  --lang en \
  --output tahminler.json \
  --pdf tahmin_raporu_en.pdf

# 4) Özel gıda/hava sensörü profili + özel hedef sütunu ile 26 grafik & tam PDF üret:
st-ds visualize \
  --input gida_protein.csv \
  --instrument-profile gida_profil.json \
  --target protein_pct \
  --model random_forest \
  --include-optional \
  --dpi 180 \
  --lang tr \
  --output gida_ciktilari
```

---

## A.5. Python API Yapılandırma Sınıfları & Tüm Parametreler

Python içinde doğrudan kullanabileceğiniz ana sınıflar ve değiştirebileceğiniz parametreler (çalıştırılabilir karşılıkları için [examples/02_all_python_api_parameters.py](../examples/02_all_python_api_parameters.py) dosyasına bakın):

### 1. Sentetik Veri Üretimi (`SyntheticConfig` & `SyntheticGeneratorV2`)
- `dataset_id: str = "stds_synthetic_v2"` — Veri seti manifest kimliği.
- `profile_id: str = "visnir_24ch_v1"` — `led_5band_v1`, `multispectral_18ch_v1`, `visnir_24ch_v1`, `ftnir_1300_2550nm_v1`, `libs_ci_hi_v1`.
- `n_samples: int = 120` — Üretilecek örnek satır sayısı.
- `n_sites: int = 6` — Benzersiz saha (`site_id`) grubu sayısı (`>= 2`).
- `seed: int = 42` — Deterministik rastgelelik tohumu.
- `noise_std_reflectance: float = 0.0025` — Spektral Gauss ölçüm gürültüsü std sapması.
- `include_eis: bool = True` — 12-frekanslı elektrokimyasal empedans (`EISMeasurement`) eklensin mi?
- `missing_bulk_density_ratio: float = 0.0` — `0.0..1.0`; ör. `0.25` verilirse satırların %25'inde hacim ağırlığı eksik bırakılır (stok fail-closed testi için).

### 2. Kemometri Ön-İşleme (`PreprocessConfig` & `ChemometricPipeline`)
- `use_absorbance: bool = True` — $-\log_{10}(R)$ Beer-Lambert lineerleştirme.
- `use_snv: bool = True` — Satır içi saçılma düzeltmesi (`(x - mean)/std`).
- `use_savgol: bool = True`, `sg_window: int = 7` (tek sayı `>= 3`), `sg_polyorder: int = 2` (`< sg_window`), `sg_deriv: int = 1` (`0`, `1` veya `2`).
- `suppression_mode: "none" | "epo" | "glsw"` — `PreprocessConfig()` doğrudan oluşturulduğunda varsayılan `"epo"`, `TrainerConfig()` içinde varsayılan `"none"`'dur.
- `epo_q: int | None = None` — Sabit bastırma bileşen sayısı (`None` = otomatik rank, `1..epo_max_q`); `epo_max_q: int = 6`; `glsw_alpha: float = 0.1` (küçük alpha daha sert sönümleme yapar).

### 3. Çoklu Sensör Özellik Füzyonu (`FeatureFusionConfig`)
- `covariate_names: tuple[str, ...]` — Varsayılan: `("ph", "moisture_volumetric_pct", "temperature_c", "ec_ds_m")`. `()` verilirse model **yalnız spektral** çalışır!
- `missing_strategy: "median_with_indicator" | "fail_closed"` — Eksik kovaryatlarda eğitim medyanı + ikili gösterge sütunu üretir veya anında hata verir.
- `spectral_scaler: "none" | "standard"`, `covariate_scaler: "standard" | "robust"`.
- `include_eis: bool = False`, `include_derived: bool = False`, `include_visnir_bands: bool = False` — Ek fiziksel öznitelikler (karbonat/kil/organik bant oranları, EIS Nyquist özeti).

### 4. Regresyon Modelleri ve Hiperparametreleri
| Sınıf (`st_ds.models.*`) | Başlıca Parametreler & Varsayılanlar | Etkisi |
| --- | --- | --- |
| `PLSRRegressor` | `max_components=10`, `inner_cv_folds=5`, `scale=True`, `clip_negative=True` | Küçük veri ve doğrusal eşdoğrusallıkta güçlü; iç CV ile optimum bileşen $k^*$ seçer. |
| `KernelRidgeRegressor` | `alpha_grid=(1e-3..10.0)`, `sigma_multipliers=(0.5, 1, 2, 4)`, `inner_cv_folds=5` | Doğrusal olmayan spektral eğrilerde çok yüksek doğruluk + RBF sonsal belirsizliği verir. |
| `RandomForestSoilRegressor` | `n_estimators=150`, `max_depth=12`, `min_samples_leaf=2`, `random_state=42` | Ağaç tabanlı doğrusal olmayan etkileşimleri yakalar; ağaçlar arası std hesaplar. |
| `LocalPLSRRegressor` | `n_neighbors=20`, `n_components=3`, `bandwidth=None`, `min_library_size=8` | Her test örneği için spektral en yakın $k$ komşuyla yerel ağırlıklı PLSR kurar (MBL). |
| `STSoilHybridNetRegressor` | `n_spectral_features`, `spectral_hidden_dim=64`, `covariate_hidden_dim=32`, `fusion_hidden_dim=64`, `dropout=0.1`, `epochs=50`, `learning_rate=2e-3`, `batch_size=32` | Spektral & kovaryat dallarını ayırır; `fit_multitask` ile `SOC + SIC + TC` ve fiziksel kapatma cezasını birlikte optimize eder. |

### 5. Belirsizlik & Alan Dışı Dedektörü (`ConformalConfig`, `OODConfig`) & Çıkarım (`STSoilAnalyzer`)
- `ConformalConfig(method="normalized"|"absolute", default_alpha=0.10, min_scale=1e-3, non_negative=True)` — `default_alpha=0.10` → %90 Conformal aralık; tahmin anında `alpha=0.05` → daha geniş %95 aralık.
- `OODConfig(mahalanobis_quantile=0.99, knn_k=5, knn_quantile=0.99, shrinkage=0.15, fail_on_ood=False)` — Eğitim bulutu dışındaki yeni numuneleri (Mahalanobis + kNN + fiziksel zarf) yakalar.
- `STSoilAnalyzer.predict(m, include_xai=False, include_stock=True, alpha=None)` — Tahmin zamanında `alpha` değiştirerek aynı modelden hem %90 (`alpha=0.10`) hem %95 (`alpha=0.05`) aralık alabilirsiniz!

---

# Part B — Complete English CLI & Python Parameter Reference

## B.1. Parameter Hierarchy, Target/Model Control & Runtime Flags

- **Why were `soc_pct` and `kernel_ridge_rbf` shown by default?**
  1. `soc_pct` is simply the default column/field name for Soil Organic Carbon (`--target soc_pct`). You can select **`--target sic_pct`**, **`--target tc_pct`**, or **any custom finite numeric target** (e.g., `--target protein_pct`, `--target pm25_ugm3`) defined in your CSV or `SoilReference.custom_targets`.
  2. `auto` evaluates all candidate models via `GroupKFold` and picks the lowest OOF RMSE model (which on smooth synthetic spectra is `kernel_ridge_rbf`). Pass **`--model plsr`**, **`--model random_forest`**, **`--model local_plsr`**, **`--model hybrid_net`**, or **`--model kernel_ridge_rbf`** to `st-ds train` or `st-ds visualize` to choose any model explicitly.
- **Configuration Precedence:**
  `CLI / Python explicit argument > OS env (STDS_LANG, STDS_DPI, STDS_PDF_EXPORT, STDS_RUNS_DIR) > optional .env file > built-in defaults ('tr', 160 DPI, PDF enabled)`.

## B.2. CLI Subcommand Matrix & Python Config Defaults

All 8 CLI subcommands work directly without bash wrappers (`st-ds <subcommand>` or `PYTHONPATH=. python -m st_ds.cli <subcommand>`):
- `dataset validate`: `--input`, `--instrument-profile <id_or_json>`, `--for-training`, `--target <col>`, `--no-require-sic`, `--output`
- `calibration fit-epo`: `--wet`, `--dry`, `--instrument-profile`, `--output`, `--q`, `--no-absorbance`, `--no-snv`, `--no-savgol`
- `evaluate`: `--input`, `--target`, `--instrument-profile`, `--covariates`, `--folds`, `--suppression-mode {none,epo,glsw}`, `--epo-q`, `--wet`, `--dry`, `--include-optional`, `--seed`, `--output`
- `train`: `--input`, `--output`, `--model`, `--target`, `--instrument-profile`, `--covariates`, `--folds`, `--suppression-mode`, `--epo-q`, `--wet`, `--dry`, `--multi-target`, `--hybrid-epochs`, `--artifact-id`, `--semantic-version`, `--pdf`, `--lang`, `--seed`
- `artifact inspect`: `--artifact`, `--output`
- `predict`: `--artifact`, `--input`, `--instrument-profile`, `--include-xai`, `--no-stock`, `--fail-fast`, `--output`, `--pdf`, `--lang`
- `report export`: `--input`, `--output`, `--format {json,pdf,both}`, `--pdf-output`, `--lang`
- `visualize`: `--input`, `--output`, `--artifact`, `--instrument-profile`, `--covariates`, `--wet`, `--dry`, `--target`, `--model`, `--folds`, `--format`, `--dpi`, `--no-html`, `--no-interactive-3d`, `--no-pdf`, `--include-optional`, `--lang`, `--seed`

For hands-on runnable Python and CLI scripts covering every option, see [examples/README.md](../examples/README.md).
