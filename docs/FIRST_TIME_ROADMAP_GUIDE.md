# STDS v0.1.0 — İlk Kez Kuran Kullanıcı İçin Sıfırdan Adım Adım Yol Haritası

> **Sürüm:** `0.1.0` (MVP İlk Kararlı Sürüm)  
> **Hedef Kitle:** Sistemi bilgisayarına ilk kez kuran araştırmacı, veri bilimcisi veya mühendis.  
> **Önemi:** Eğer projede başka dokümanlara bakmadan *"veriyi nasıl hazırlayacağım, komutları hangi sırayla çalıştıracağım, hangi yöntemi seçeceğim ve klasörde oluşan onlarca dosya ne anlama geliyor?"* diyorsanız **başlangıç noktanız burasıdır**.

> ⚠️ **Sentetik ve Gerçek Veri Ayrımı:** Bu pakette gelen sentetik veri üreteci yalnızca yazılımın çalıştığını doğrulamak içindir. Tarımsal veya çevresel karar almak için mutlaka kendi gerçek laboratuvar/spektrometre ölçümlerinizi yükleyin; çıktıların tamamı üzerinde `VERİ KAYNAĞI: SENTETİK` veya `GERÇEK LABORATUVAR / SAHA` etiketi yer alır.

---

## 1. Büyük Resim: Sistem Baştan Sona Nasıl Çalışır?

STDS (Soil Carbon Spectroscopy & Sensor Fusion Toolkit), toprak spektrumlarını (`Vis-NIR` / `SWIR` / `LIBS` / `Raman`) ve yardımcı sensör ölçümlerini (`pH`, `nem`, `sıcaklık`, `EC`, `hacim ağırlığı`, `koordinat`) birleştirip:

1. Verinin fiziksel kurallara ve cihaz profiline uygunluğunu denetler (`dataset validate`),
2. Farklı modelleri saha bazlı çapraz doğrulama ile sızıntısız yarıştırır (`evaluate`),
3. En iyi modeli **pickle içermeyen güvenli artefakt paketi** olarak diske kaydeder (`train`),
4. Yeni örnekler için **%90 Conformal güven aralığı**, **OOD (alan dışı uyarısı)**, **karbon fraksiyon kapatma kontrolü** ve **katman karbon stoku (`Mg C ha⁻¹`)** hesaplar (`predict`),
5. Tüm veri, model, XAI ve harita analizlerini **26+ bilimsel 2D/3D grafik**, **5 analitik CSV/Markdown tablosu** ve **çevrimdışı HTML panosu** olarak üretir (`visualize`).

> 💡 **Toprak Dışı Veri Analizi & Tam Parametreler:** STDS motorunun ~%80'i gıda, hava kalitesi, donanım sensörü, yaprak azotu ve genel tabular veriler için de kullanılabilir; bkz. **[Toprak Dışı & Genel Veri Analizi (TR & EN)](GENERAL_DATA_ANALYSIS_GUIDE.md)**, **[Veri Seti Hazırlama & Şema Kuralları](DATA_PREPARATION_AND_SCHEMA_GUIDE.md)** ve **[Tüm CLI/Python Parametreleri](CLI_AND_PYTHON_PARAMETERS_GUIDE.md)**.

---

## 2. Kurulum ve İlk Sağlık Kontrolü

Önce proje kök klasöründe Python ortamınızı hazırlayın (Python 3.10+ önerilir):

```bash
cd /Users/denizhan/Desktop/SpaceTeknopoliDataScience

# Temel paket + 2D/3D grafik ve test araçları ile kurulum:
pip install -e ".[viz,dev]"
```

Kurulumun başarılı olduğunu doğrulamak için hızlı testleri çalıştırın:

```bash
pytest tests/ -v
st-ds --help
```

Eğer `pytest` tüm testleri yeşil geçiriyor ve `st-ds --help` altında `dataset`, `calibration`, `train`, `evaluate`, `artifact`, `predict`, `report`, `visualize` komutları listeleniyorsa kurulum tamamdır.

### 2.1. Dil Seçimi (`TR` / `EN`), İsteğe Bağlı `.env` ve Hazır Bash Betikleri (`scripts/`)

- **Varsayılan Dil Türkçe (`tr`):** Sistem `.env` dosyasına bağımlı olmadan varsayılan olarak Türkçe grafik, tablo, HTML pano ve PDF üretir.
- **İngilizce (`en`) Çalıştırma:** İsterseniz proje kökündeki **`.env`** dosyasında `STDS_LANG=en` yapabilir (bkz. [`.env.example`](../.env.example)), ortam değişkeni (`STDS_LANG=en`) verebilir veya herhangi bir komuta **`--lang en`** parametresini ekleyebilirsiniz (`Öncelik: --lang > OS Env > .env > 'tr'`).
- **Tek Komutla Otomasyon (`scripts/` → Zaman Damgalı `runs/`):**
  ```bash
  bash scripts/01_generate_synthetic_data.sh   # Sentetik JSON/CSV + Islak/Kuru EPO verisi
  bash scripts/02_train_and_evaluate.sh        # Doğrulama + EPO + CV + Eğitim + PDF özeti
  bash scripts/03_run_full_pipeline.sh         # Tüm PNG + CSV/MD + HTML + Çok Sayfalı Tam PDF
  ```
  Tüm çıktılar zaman damgalı `runs/<YYYYMMDD_HHMMSS>_*/` klasörlerine yazılır (`runs/` klasörü `.gitignore` ile hariç tutulur; `.env` ise git deposunda takip edilir).
- **Toprak Analizi Çift Dilli Rehberi:** Toprak kimyası ve analiz iş akışı için [TOPRAK_ANALIZ.md](TOPRAK_ANALIZ.md) dosyasına bakın.

---

## 3. Verilerinizi Hazırlama (JSON, CSV ve Parquet Kuralları)

Kendi toprak örneklerinizi sisteme sokmadan önce aşağıdaki **fiziksel birimleri ve alan adlarını** hazırlamanız gerekir. Yanlış birim (örneğin `%` yerine `g/kg` verilmesi veya derinliğin metre yazılması) sistemin `SchemaError` ile işlemi güvenli şekilde durdurmasına yol açar.

### 3.1. Zorunlu ve Opsiyonel Alanlar Sözlüğü

| Alan / Kolon Adı | Zorunlu mu? | Beklenen Birim ve Format | Açıklama |
| --- | --- | --- | --- |
| `sample_id` | Evet | Metin (ör. `"TR_ANK_001"`) | Her ölçüm için benzersiz kimlik |
| `site_id` | Eğitim için çok önerilir | Metin (ör. `"ankara_polatli"`) | Saha/tarla grubu; çapraz doğrulamada veri sızıntısını önler |
| `instrument_profile_id` | Evet | Metin (ör. `"visnir_24ch_v1"`) | Spektrometre dalga boyu ızgarası kimliği |
| **Spektral Bantlar** | Evet | Yansıma/şiddet (sayısal dizi) | JSON'da `spectrum.values`; CSV'de `wl_350.00nm`, `wl_360.00nm` ... |
| `soc_pct` | Eğitimde Evet, Tahminde Hayır | `%` (kuru madde bazında `>= 0`) | Toprak Organik Karbonu |
| `sic_pct` | Opsiyonel | `%` (kuru madde bazında `>= 0`) | Toprak İnorganik Karbonu |
| `tc_pct` | Opsiyonel | `%` (`TC ≈ SOC + SIC`) | Toplam Karbon |
| `ph` | Opsiyonel | `0 – 14` pH birimi | Toprak pH değeri |
| `moisture_volumetric_pct` | Opsiyonel (EPO için önemli) | `%` (`0 – 100`) | Hacimsel toprak nemi |
| `temperature_c` | Opsiyonel | `°C` | Sensör/toprak sıcaklığı |
| `ec_ds_m` | Opsiyonel | `dS/m` | Elektriksel iletkenlik |
| `bulk_density_g_cm3` | Stok için **Zorunlu** | `g/cm³` (`0.1 – 2.65`) | Eksikse stok hesabı yapılmaz (`INSUFFICIENT_DATA`) |
| `coarse_fragments_fraction` | Opsiyonel | `0.0 – 0.99` oran | Taş/çakıl oranı (varsayılan `0.0`) |
| `depth_top_cm`, `depth_bottom_cm` | Stok için Evet | `cm` (varsayılan `0 - 30 cm`) | Örneklenen toprak derinlik aralığı |
| `lat`, `lon` | Kriging haritası için Evet | Ondalık derece (`EPSG:4326`) | Enlem (`[-90, 90]`) ve Boylam (`[-180, 180]`) |
| `moisture_scenario` | Opsiyonel | `"dry"`, `"wet"`, `"field"` vb. | Alt grup hata analizi için nem sınıfı |

### 3.2. CSV Dosyası Hazırlamanın İki Yolu

1. **STDS Yan Dosyalı CSV (En Güvenli Yol):** `save_dataset_csv()` kullandığınızda `verim.csv` yanında otomatik olarak `verim.csv.meta.json` oluşur. Bu sayede `st-ds` dalga boylarını otomatik tanır.
2. **Ham Dış CSV Dosyası (Excel / Laboratuvar Çıktısı):** Eğer elinizde yalnızca tek bir `.csv` dosyası varsa:
   - Spektral kolon adlarını yerleşik veya özel cihaz profilinizin dalga boylarına tam olarak **`wl_{dalga_boyu:.2f}nm`** biçiminde isimlendirin (örnek: `wl_400.00nm`, `wl_420.00nm`, ...).
   - Komutları çalıştırırken `--instrument-profile <profil_id>` parametresini verin (veya Python'da `load_dataset_csv("dosyam.csv", profile=profiliniz)` kullanın).
   - Yerleşik profil ID'lerini görmek için Python'da `from st_ds.domain.instrument import BUILTIN_PROFILES; print(list(BUILTIN_PROFILES.keys()))` çalıştırabilirsiniz (`led_5band_v1`, `multispectral_18ch_v1`, `visnir_24ch_v1`, `ftnir_1300_2550nm_v1`, `libs_ci_hi_v1`).

### 3.3. Büyük Veri İçin Parquet Kullanımı

Binlerce spektrumunuz varsa Python içinde `save_dataset_parquet(dataset, "toprak.parquet")` ve `load_dataset_parquet("toprak.parquet")` kullanabilirsiniz. Hem dosya boyutunu ciddi ölçüde küçültür hem de `.meta.json` şema bilgisini yanında saklar.

---

## 4. Hangi Veri Analizi Yöntemini Seçmeliyim? (Karar Rehberi)

Verinizin doğasına göre STDS içindeki modülleri aşağıdaki rehbere göre seçebilirsiniz:

| Durumunuz | Önerilen Yöntem / Ayar | Neden? |
| --- | --- | --- |
| **Kuru ve ıslak saha ölçümleri karışık** | `calibration fit-epo` + `--suppression-mode epo` + `visualize --wet ... --dry ...` | `EPO` (External Parameter Orthogonalization), su bantlarındaki paraziti matematiksel olarak dik izdüşümle bastırır. |
| **Hafif nem/sıcaklık sürüklenmesi var, eşli ıslak-kuru seti yok** | `--suppression-mode glsw` | `GLSW` istatistiksel grup içi farklardan ağırlıklandırma yaparak sürüklenmeyi azaltır. |
| **Az örnek var (`N < 150`) ve hızlı temel çizgi istiyorsunuz** | `--model plsr` veya `--model auto` | `PLSR` küçük spektral veri setlerinde yüksek eşdoğrusallığa çok dayanıklıdır. |
| **Doğrusal olmayan ilişkiler + belirsizlik + çoklu hedef (`SOC`, `SIC`, `TC`)** | `--model st_soil_hybrid_net` | Hem spektral 1D-CNN dalı hem kovaryat dalı taşır; aleatorik/epistemik belirsizlik ve kütle dengesi cezası üretir. |
| **Tahmin güvenilir mi, ne kadar hata payı var?** | Conformal + OOD kapısı | Artefakt içine otomatik gömülür; eğitim dağılımı dışındaki örnekleri `OUT_OF_DOMAIN` olarak işaretler. |
| **Hangi dalga boyu veya sensör tahmini sürüklüyor?** | `--include-xai` | `Integrated Gradients`, `HSIC` (istatistiksel bağımlılık) ve 2. derece etkileşim puanlarını hesaplar. |
| **Sahada noktadan noktaya harita çıkarmak istiyorum** | `ordinary_kriging` / `visualize` (Grafik 22-23) | En az 8 farklı koordinat noktası varsa Matheron semivariogramı ve 2D/3D Kriging belirsizlik haritası üretir. |

---

## 5. Adım Adım Çalıştırma Sırası (İlk Başlangıç Senaryosu)

Aşağıdaki komutları **tam bu sırayla** kopyalayıp çalıştırabilirsiniz. Örnek veri olarak önce paket içindeki sentetik üreteçle hızlı bir çalışma alanı hazırlıyoruz; kendi veriniz hazır olduğunda sadece `--input` yolunu kendi `.json`/`.csv` dosyanıza çevirmeniz yeterlidir.

### Adım 1 — Örnek Veri Üret (veya Kendi Verini Koy)

```bash
python -c '
from pathlib import Path
from st_ds.io.dataset import save_dataset_csv, save_dataset_json
from st_ds.io.synthetic import SyntheticConfig, SyntheticGeneratorV2

out = Path("calisma_alani/veri")
out.mkdir(parents=True, exist_ok=True)
gen = SyntheticGeneratorV2(SyntheticConfig(dataset_id="ilk_deneme_v1", n_samples=48, n_sites=8, seed=42))
ds = gen.generate_dataset()
wet, dry = gen.generate_paired_wet_dry_dataset(n_pairs=16)

save_dataset_json(ds, out / "toprak_verisi.json")
save_dataset_csv(ds, out / "toprak_verisi.csv")
save_dataset_json(wet, out / "nem_islak.json")
save_dataset_json(dry, out / "nem_kuru.json")
print("Hazırlanan ölçüm sayısı:", len(ds.records))
'
```

### Adım 2 — Veri Şeması ve Kalite Doğrulaması (`dataset validate`)

```bash
st-ds dataset validate --input calisma_alani/veri/toprak_verisi.json --for-training --output calisma_alani/01_dogrulama.json
```

*Ne elde edersiniz?* Tüm örneklerin dalga boyu uyumu, fiziksel sınır ihlalleri ve `OK / WARNING / UNRELIABLE` sayıları `01_dogrulama.json` içine yazılır. (Not: Etilmiş paket kurulumu yerine doğrudan kaynak klasörden çalıştırıyorsanız `st-ds` yerine `PYTHONPATH=. python -m st_ds.cli ...` yazabilirsiniz.)

### Adım 3 — Modelleri Saha Bazlı Çapraz Doğrulamayla Yarıştır (`evaluate`)

```bash
st-ds evaluate \
  --input calisma_alani/veri/toprak_verisi.json \
  --target soc_pct \
  --folds 4 \
  --include-optional \
  --output calisma_alani/02_model_kiyaslama.json
```

*Ne elde edersiniz?* Tüm modellerin (`plsr`, `kernel_ridge_rbf`, `local_plsr_mbl`, `random_forest`, `st_soil_hybrid_net`) aynı `site_id` katlamalarındaki `RMSE`, `MAE`, `Bias`, `R²`, `RPIQ` ve `Bland-Altman LoA` skorları.

### Adım 4 — Nihai Modeli Eğit ve Güvenli Artefakt Paketine Kaydet (`train`)

```bash
st-ds calibration fit-epo \
  --wet calisma_alani/veri/nem_islak.json \
  --dry calisma_alani/veri/nem_kuru.json \
  --output calisma_alani/epo_matrisi.json

st-ds train \
  --input calisma_alani/veri/toprak_verisi.json \
  --output calisma_alani/03_model_artefakti \
  --model auto \
  --target soc_pct \
  --suppression-mode epo \
  --semantic-version 0.1.0
```

*Önemli kontrol — Artefakt Bütünlüğünü Denetle:*

```bash
st-ds artifact inspect --artifact calisma_alani/03_model_artefakti
```

### Adım 5 — Yeni Örnekler Üzerinde Tahmin, Stok ve XAI Çalıştır (`predict` + `report export`)

```bash
st-ds predict \
  --artifact calisma_alani/03_model_artefakti \
  --input calisma_alani/veri/toprak_verisi.json \
  --include-xai \
  --output calisma_alani/04_tahmin_sonuclari.json

st-ds report export \
  --input calisma_alani/04_tahmin_sonuclari.json \
  --output calisma_alani/05_onayli_rapor.json
```

### Adım 6 — 26+ Bilimsel 2D/3D Grafik, Tablo ve HTML Panosunu Üret (`visualize`)

```bash
st-ds visualize \
  --input calisma_alani/veri/toprak_verisi.json \
  --artifact calisma_alani/03_model_artefakti \
  --wet calisma_alani/veri/nem_islak.json \
  --dry calisma_alani/veri/nem_kuru.json \
  --include-optional \
  --output calisma_alani/06_gorsel_rapor
```

İşlem bitince tarayıcınızda **`calisma_alani/06_gorsel_rapor/index.html`** dosyasını açmanız yeterlidir (hiçbir sunucu kurulumu veya internet bağlantısı gerektirmez).

### Python Üzerinden Tek Betikle Aynı İş Akışı

```python
from st_ds import STSoilAnalyzer, generate_full_visualization_suite, load_dataset_json

dataset = load_dataset_json("calisma_alani/veri/toprak_verisi.json")
analyzer = STSoilAnalyzer.from_artifact("calisma_alani/03_model_artefakti")

# Toplu tahmin + XAI + stok hesabı
batch = analyzer.predict_batch(dataset.records, include_xai=True, include_stock=True)
print("Başarılı örnek:", batch.n_succeeded, "Hatalı örnek:", batch.n_failed)

# 26+ grafik (4 tanesi 3D) + 5 CSV/MD tablo + index.html
viz = generate_full_visualization_suite(
    dataset,
    "calisma_alani/06_gorsel_rapor",
    analyzer=analyzer,
)
print("Üretilen grafik sayısı:", viz.n_plots, "| HTML Pano:", viz.dashboard_path)
```

---

## 6. Klasörde Oluşan Tüm Çıktı Dosya Türleri ve Anlamları

STDS çalıştığında diskinizde oluşan **her dosya uzantısının** kesin görevi aşağıdadır:

| Dosya / Uzantı | Nerede Oluşur? | İçeriği ve Nasıl Açılır? |
| --- | --- | --- |
| **`.json`** (`01_dogrulama.json`, `04_tahmin_sonuclari.json`, `viz_manifest.json`) | Çalışma ve görselleştirme klasörü | Yapılandırılmış standart JSON çıktıları (API / dashboard / arşiv entegrasyonu için). |
| **`manifest.json`** | `03_model_artefakti/` | Model sürümü, hedef şeması, cihaz profili ve dosyaların SHA-256 güvenlik özetleri. |
| **`model_card.json`** | `03_model_artefakti/` | Doğrulama metrikleri, eğitim kapsamı, OOD ve yeniden kalibrasyon kısıtları. |
| **`.safetensors`** (`model_weights.safetensors`) | `03_model_artefakti/` | Kod çalıştırma riski olmayan **zero-pickle** model ağırlıkları ve parametre matrisleri. |
| **`.npz`** (`preprocess_state.npz`) | `03_model_artefakti/` | Dalga boyu ızgarası, EPO matrisi, ölçekleyici ortalamaları, Conformal puanlar ve OOD eşikleri. |
| **`.csv`** ve **`.csv.meta.json`** | `veri/` ve `06_gorsel_rapor/tables/` | Excel/Python ile doğrudan açılabilen tablolar; `.meta.json` spektral kolon birimi ve profil bilgisini tutar. |
| **`.parquet`** | Veri dışa aktarma | Sütun bazlı sıkıştırılmış yüksek hızlı veri formatı. |
| **`.md`** | `tables/` ve `docs/` | Raporlara veya Git depolarına doğrudan kopyalanabilen Markdown tabloları ve Model Kartı şablonları. |
| **`.png`** (ve opsiyonel **`.svg`**) | `06_gorsel_rapor/plots/` | 26 adet yüksek çözünürlüklü statik bilimsel 2D/3D grafik (`--format png,svg` ile vektörel alınabilir). |
| **`.html`** (`index.html` & `interactive/*.html`) | `06_gorsel_rapor/` | `index.html`: Tüm analizlerin çevrimdışı özet panosu. `interactive/*.html`: Fare ile döndürülebilir 3D Plotly grafikleri. |
| **`.pdf`** (`stds_analysis_report.pdf`, `*_report.pdf`) | `06_gorsel_rapor/reports/` ve `runs/<TS>_*/` | Girdi profilini, kemometri/EPO akışını, XAI ilişkilerini, tüm tabloları ve **üretilen 26+ PNG grafiklerin tamamını** içeren çok sayfalı A4 yatay kapsamlı rapor. |
| **`.geojson`** | Mekansal Kriging dışa aktarımı | QGIS veya GIS yazılımlarına aktarılabilen mekansal karbon ve belirsizlik katmanları. |

---

## 7. Üretilen 26+ Grafik ve 5 Tabloyu Nasıl Okumalısınız?

Görselleştirme çıktıları dört ana kategoriye ayrılmıştır:

### A. Spektral, Kemometri & QC (`plots/01_*` – `07_*`)
1. **`01_ham_spektrumlar_ve_kimyasal_bantlar`**: Nem/toprak gruplarının ortalama ± 1 standart sapma spektrumları ve O-H (`1450`, `1940 nm`), kil (`2200 nm`), karbonat (`2340 nm`) bantları.
2. **`02_kemometri_donusum_asamalari`**: Ham spektrum → Görünür Absorbans → SNV → Savitzky-Golay 1. Türev ile saçılma temizliğinin görsel kanıtı.
3. **`03_epo_nem_baskilama_karsilastirma`**: Islak ve kuru ölçümler arasındaki fark spektrumunun EPO sonrası ne kadar (% kaç) küçüldüğü.
4. **`04_spektral_3d_yuzey` *(3D #1)***: Artan `SOC (%)` sırasına göre dizilmiş tüm spektrumların 3 boyutlu topografyası.
5. **`05_dalga_boyu_korelasyon_matrisi`**: Bantlar arası eşdoğrusallık ısı haritası ve hangi dalga boyunun SOC ile en güçlü korelasyon verdiğini gösteren profil.
6. **`06_sinyal_qc_teshis`**: Örnek bazında QC skoru dağılımı ve SNR vs SOC grafiği (`OK / WARNING / UNRELIABLE`).
7. **`07_eis_veya_kovaryat_matrisi`**: EIS verisi varsa Nyquist diyagramı, yoksa Nem-pH-SOC saçılımı + tüm kovaryat/karbon korelasyon matrisi.

### B. Model Performansı, Belirsizlik & OOD (`plots/08_*` – `16_*`)
8. **`08_model_kiyaslama_metrikleri`**: Aday modellerin `RMSE`, `MAE`, `R²` ve `RPIQ` kıyaslama çubukları.
9. **`09_gercek_vs_tahmin_ve_ci90`**: 1:1 parite doğrusu ve her noktanın **%90 Conformal güven aralığı** hata çubukları.
10. **`10_bland_altman_uyum`**: Ortalama sapma (Bias) ve `±1.96 SD` uyum sınırları (sistematik kayma var mı?).
11. **`11_artik_teshis_4panel`**: Eşvaryanslılık, hata histogramı, Normal Q-Q grafiği ve nem senaryosuna göre hata kutuları.
12. **`12_alt_grup_hata_isi_haritasi`**: Hangi saha (`site_id`) veya nem koşulunda hatanın yükseldiğini gösteren ısı haritası.
13. **`13_conformal_kalibrasyon_kapsami`**: Nominal güven düzeyleri (`%50–%95`) ile gerçekleşen kapsama oranlarının uyumu.
14. **`14_belirsizlik_aleatoric_epistemic`**: Model içsel standart sapması + conformal genişleme payı ayrıştırması.
15. **`15_ood_etki_alani_haritasi`**: Mahalanobis vs kNN oran haritası; `1.0` eşiğini aşan alan dışı (`OOD`) örnekler.
16. **`16_pca_3d_gizli_uzay` *(3D #2)***: Birleşik spektral+kovaryat uzayının ilk 3 temel bileşeninde `SOC (%)` gradyanı.

### C. Açıklanabilirlik (XAI), Karbon Dengesi & Mekansal Kriging (`plots/17_*` – `23_*`)
17. **`17_xai_duyarlilik_ve_ig_spektral`**: Modelin tahmin sırasında en çok tepki verdiği dalga boyları (`IG` + `|∂Ŷ/∂x|`).
18. **`18_xai_hsic_ve_kovaryat_onemi`**: `nHSIC` küresel bağımlılık sıralaması (nedensellik değil istatistiksel bağımlılık) ve birim-ayrıştırılmış grup önemleri.
19. **`19_xai_ikili_etkilesim_isi_haritasi`**: İkinci derece karma kısmi türev etkileşim matrisi.
20. **`20_karbon_fraksiyonlari_ve_kapatma`**: `SOC + SIC` yığılmış çubukları ve `TC − (SOC + SIC)` kütle dengesi artıkları (`±0.05%` tolerans).
21. **`21_karbon_stoku_ve_varyans_tornado`**: Katman karbon stoku (`Mg C ha⁻¹`) ve Taylor hata yayılımı tornado grafiği (belirsizliğin ne kadarı karbondan, ne kadarı hacim ağırlığından geliyor?).
22. **`22_mekansal_variogram_modeli`**: Deneysel Matheron semivariogramı ve `nugget`, `sill`, `range` parametreleri.
23. **`23_ordinary_kriging_2d_ve_3d` *(3D #3)***: 2D belirsizlik konturlu saha haritası ve 3D arazi yüzeyi.

### D. Görsel Tablolar, 3D Kovaryat Yüzeyi & 5 CSV/MD Tablosu (`plots/24_*` – `26_*` + `tables/*`)
24. **`24_veri_seti_ozet_tablosu`**: Min, Medyan, Ortalama ± SS, Maksimum özet tablosu görseli.
25. **`25_tahmin_denetim_tablosu`**: Örnek bazlı tahmin, `%90 CI`, stok, baskın belirsizlik ve kalite statüsü tablosu görseli.
26. **`26_kovaryat_3d_iliski_yuzeyi` *(3D #4)***: En bilgilendirici iki toprak kovaryatı üzerinde ikinci derece `SOC` tepki yüzeyi.
- **Klasördeki 5 CSV + 5 MD Tablosu (`tables/01_*` – `05_*`):** Veri özeti, model kıyaslama, alt grup hata matrisi, XAI sıralaması ve tam tahmin denetim listesi.

---

## 8. Sık Karşılaşılan Hatalar ve Çözümleri

| Hata / Kalite Durumu | Ne Anlama Gelir? | Ne Yapmalısınız? |
| --- | --- | --- |
| `SchemaError: Bulk density missing...` / Stok `None` (`INSUFFICIENT_DATA`) | `bulk_density_g_cm3` eksik. STDS gizlice varsayılan `1.35` değeri atamaz. | Ölçümlerinize gerçek hacim ağırlığı (`g/cm³`) ekleyin veya stoku kapatmak için `--no-stock` kullanın. |
| `QualityStatus.OUT_OF_DOMAIN` (`is_ood=True`) | Yeni ölçüm eğitim verisinin spektral/kovaryat zarfının dışında. | Sonucu yüksek riskli kabul edin; bu sahadan referans lab örneği alıp kalibrasyona ekleyin. |
| `QualityStatus.NOT_VALID_FOR_REPORTING` | Sinyal doygunluğu veya kritik düşük SNR (`qc_score` eşik altında). | Ölçümü spektrometre ile tekrarlayın, beyaz/karanlık kalibrasyonu yenileyin. |
| `ArtifactCompatibilityError` (`SHA-256 mismatch` / dalga boyu hatası) | Artefakt dosyaları değiştirilmiş veya yeni verinin spektrometre profili farklı. | Veriyi doğru cihaz profiline yeniden örnekleyin ve artefaktı yeniden eğitin. |
| `Kriging` veya `23_ordinary_kriging_2d_ve_3d` uyarı kutusu gösteriyor | 8'den az benzersiz saha koordinatı var veya tüm noktalar aynı koordinatta. | En az 8 ayrı `(lon, lat)` koordinatlı örnek ekleyin. |
