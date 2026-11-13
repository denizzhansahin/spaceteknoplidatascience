# Toprak Analizi İçin STDS Kullanım ve Bilimsel Metodoloji Kılavuzu (TR / EN)
# STDS Soil Analysis Scientific Methodology & Operational Guide (TR / EN)

> **Proje / Project:** SpaceTeknopoliDataScience (`st-ds`) — **MVP `v0.1.0`**  
> **Dil Politikası / Language Policy:** Varsayılan **Türkçe (`tr`)** • İsteğe bağlı `.env` (`STDS_LANG=en`), ortam değişkeni (`STDS_LANG=en`) veya `--lang en` parametresi ile **İngilizce (`en`)** tam destek.  
> **Akademik Referans / Companion Paper:** [ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md](ACADEMIC_METHODOLOGY_AND_ALGORITHMS.md)  
> **Kanıt Statüsü / Evidence Boundary (`Risk 9`):** Sayısal kalibrasyon örnekleri sentetik boru hattını (`SyntheticGeneratorV2`) yansıtır; gerçek tarla raporlaması için laboratuvar referanslı toprak spektrumları kullanılmalıdır.

---

## İçindekiler / Table of Contents

1. **Toprak Analizinde Bilimsel Problem ve Çözüm Çerçevesi** / *Scientific Problem & Solution Framework in Soil Analysis*
2. **Toprak Örneğinin Girdiden Bilimsel Karar Çıktısına Dönüşümü** / *Lifecycle of a Soil Sample from Sensor Signal to Decision*
3. **Toprağa Özgü Spektral Kimya, Matematiksel Formüller ve Yazılım Eşleşmesi** / *Soil-Specific Spectral Chemistry, Mathematics & Software Mapping*
4. **Toprak Karbon Havuzları (SOC / SIC / TC) ve Fiziksel Stok Muhasebesi** / *Soil Carbon Pools & Physical Stock Accounting ($\mathrm{Mg\;C\;ha^{-1}}$)*
5. **Tarla Ölçeğinde Mekânsal Toprak Haritalama (Variogram & Ordinary Kriging)** / *Field-Scale Spatial Soil Mapping*
6. **Toprak Analizi İçin Çift Dilli Kullanım (`.env`, CLI `--lang`, Hazır `scripts/` & Çok Sayfalı PDF)** / *Bilingual Soil Operations, Bash Pipelines & PDF Reporting*

---

## 1. Toprak Analizinde Bilimsel Problem ve Çözüm Çerçevesi

### 🇹🇷 Türkçe
Geleneksel toprak organik ve inorganik karbon analizleri (kuru yakma elementel analizörleri ve Walkley–Black ıslak oksidasyonu) yüksek laboratuvar maliyeti, kimyasal atık yükü ve uzun dönüş süreleri nedeniyle yüksek yoğunluklu tarla örneklemesini kısıtlar. Proksimal toprak algılayıcıları (**Vis-NIR**, **SWIR**, **FT-NIR**, **LIBS** ve **EIS**) saniyeler içinde tahribatsız ölçüm alabilir; ancak doğrudan sahadan gelen sinyaller **toprak neminin güçlü su absorpsiyon bantları ($1450\text{–}1940\text{ nm}$)**, **agregat ve tane boyutu saçılması**, **kireçli/karbonatlı topraklarda $\mathrm{SIC}$ girişimi**, **tekrarlı nokta sızıntısı** ve **eksik hacim ağırlığı ($\rho_b$) varsayımları** nedeniyle güvenilir olmaktan çıkabilir. **`st-ds`**, bu engellerin her birini deterministik kemometri, fiziksel türevler, grup-ayrık doğrulama, **Conformal** belirsizlik kalibrasyonu ve **fail-closed** karbon muhasebesi ile çözer.

### 🇬🇧 English
Conventional laboratory soil carbon assays (dry combustion elemental analyzers and Walkley–Black wet oxidation) limit high-density field monitoring due to per-sample cost, laboratory turnaround latency, and destructive preparation. Proximal soil sensors (**Vis-NIR**, **SWIR**, **FT-NIR**, **LIBS**, and **EIS**) acquire non-destructive spectra in seconds, yet raw field signals suffer from **moisture masking ($1450\text{–}1940\text{ nm}$ water bands)**, **particle-size scatter**, **inorganic carbonate ($\mathrm{SIC}$) interference in calcareous soils**, **site/replicate data leakage**, and **silent bulk-density ($\rho_b$) imputation**. **`st-ds`** resolves every bottleneck through deterministic chemometrics, wavelength-aware derivatives, grouped validation, **Conformal** uncertainty bounds, and **fail-closed** soil carbon stock accounting.

| Problem Kodu | Toprak Sahası Engeli / *Soil Field Obstacle* | Uygulanan Matematiksel Yöntem / *Mathematical Solution* | Toprak Bilimine Katkısı / *Agronomic Impact* |
| --- | --- | --- | --- |
| **P1** | Tarla nemi ($\theta_v$) organik bantları maskeler | **SNV** + **EPO** ($P_{\mathrm{EPO}} = I - V_q V_q^\top$) / **GLSW** | Islak ve kuru tarla okumalarını aynı kalibrasyon uzayına indirger (`st_ds.chemometrics.epo`) |
| **P2** | Çoklu cihaz / düzensiz $\Delta\lambda$ (LED vs spektrometre) | Fiziksel $\Delta\lambda$ kaydırılmış Vandermonde **Savitzky–Golay** + `InstrumentProfile` | Bant aralıkları farklı toprak problarında tutarlı türev sağlar (`st_ds.chemometrics.savgol`) |
| **P3** | Kalsiyum karbonat ($CaCO_3$) $\mathrm{SOC}$ tahminini şişirir | Çok görevli **SOC + SIC + TC** başlığı + $\mathrm{TC} = \mathrm{SOC} + \mathrm{SIC}$ kapatması | Kireçli Anadolu/Akdeniz topraklarında $\mathrm{SOC}$ ve $\mathrm{SIC}$ ayrımını korur (`st_ds.models`, `st_ds.carbon`) |
| **P4** | Tekrar taramalar veya aynı tarla parselinin bölünmesi | `GroupedSplitBuilder` (`site_id`/`sample_id`) + kat içi (`fold-only`) fit | Görülmemiş tarım parsellerinde gerçek genelleme skorunu verir (`st_ds.training.splits`) |
| **P5** | Hacim ağırlığı ($\rho_b$) olmadan stok uydurması | Eksik $\rho_b$'de **Fail-Closed** durdurma (`stock_mg_ha=None`) + **Taylor $\sigma_S$** | Sahte `1.35 g/cm³` varsayımını yasaklar; hata kaynağını tornado grafiğiyle açar (`st_ds.carbon.stock`) |
| **P6** | Kalibrasyon dışı yeni toprak tiplerinde aşırı güven | **Mahalanobis + k-NN OOD** kapısı & sonlu örneklem **Conformal CI90** | Kalibrasyon dışı toprağı `OUT_OF_DOMAIN` bayrağıyla işaretler (`st_ds.uncertainty`) |

---

## 2. Toprak Örneğinin Girdiden Bilimsel Karar Çıktısına Dönüşümü

```text
[Tarla / Laboratuvar Girdisi — Field / Lab Input]
  • Spektrum: Vis-NIR / SWIR / Multispektral / LIBS + Beyaz/Karanlık kalibrasyon (I_soil, I_white, I_dark)
  • Toprak Kovaryatları: Nem θ_v (% veya m³/m³), pH, Sıcaklık (°C), EC (dS/m), Hacim Ağırlığı ρ_b (g/cm³), Derinlik (cm), Kaba Parça f_c
  • Konum & Kimlik: sample_id, site_id, horizon/derinlik, lon/lat koordinatları
         │
         ▼  (1) Şema & Birim Doğrulaması — UnitRegistry & InstrumentProfile (st_ds.domain, st_ds.io)
  Tüm ölçüm birimleri kanonik iç standarda (%C, g/cm³, cm, nm, Ω, °C) çekilir.
         │
         ▼  (2) Radyometri & Kalite Kontrolü — calibrate_reflectance + evaluate_signal_qc (st_ds.calibration, st_ds.signal)
  Doymuş/düşük SNR taramalar elenir; LIBS için ArPLS (λ=10⁴) zemin düzeltmesi uygulanır.
         │
         ▼  (3) Kemometri & Nem Baskılama — Absorbans → SNV → Fiziksel SavGol → EPO/GLSW (st_ds.chemometrics)
  Parçacık saçılması ve tarla neminin spektral imzası SVD tabanlı ortogonal projeksiyonla bastırılır.
         │
         ▼  (4) İsimlendirilmiş Çoklu-Sensör Füzyonu — FeatureFusionPipeline (st_ds.features)
  X_clean + Vis-NIR tanısal bant özetleri + EIS öznitelikleri + eksik kovaryat bayrakları birleştirilir.
         │
         ▼  (5) Çıkarım, Conformal Belirsizlik & OOD — STSoilAnalyzer (st_ds.models, st_ds.uncertainty, st_ds.api)
  SOC / SIC / TC tahminleri, %90 Conformal güven aralığı (CI90) ve alan dışı (OOD) risk skoru üretilir.
         │
         ▼  (6) Açıklanabilirlik, Stok & Harita — XAI + Carbon Stock + Kriging + PDF/PNG (st_ds.xai, st_ds.carbon, st_ds.spatial, st_ds.reporting)
  Integrated Gradients dalga boyu atıfları, Mg C ha⁻¹ stok, Ordinary Kriging yüzeyleri ve tam PDF raporu çıkar.
```

---

## 3. Toprağa Özgü Spektral Kimya, Matematiksel Formüller ve Yazılım Eşleşmesi

### 3.1 Tanısal Toprak Absorpsiyon Bantları / *Diagnostic Soil Absorption Bands*

| Dalga Boyu / *Wavelength* | Kimyasal Bağ / *Chromophore* | Toprak Özelliği / *Soil Property* | Sistem Karşılığı / *Module Mapping* |
| --- | --- | --- | --- |
| **$450\text{–}580\text{ nm}$** | $\mathrm{Fe^{3+}}$ elektron geçişi & humik koyulaşma | Demir oksitler (hematit/goetit) ve organik madde rengi | `st_ds.features` (`band_iron_organic`), Plot 01 |
| **$1380\text{–}1500\text{ nm}$** ($\sim 1450\text{ nm}$) | $\mathrm{O\text{-}H}$ 1. üst ton (*overtone*) | Gözenek suyu ve kil tabakası nemi | `st_ds.chemometrics.epo` tarafından baskılanır |
| **$1880\text{–}2000\text{ nm}$** ($\sim 1940\text{ nm}$) | $\mathrm{H_2O}$ eğilme + gerilme kombinasyonu | Serbest/kapiler toprak nemi pik bandı | Plot 03 (EPO öncesi/sonrası fark genliği) |
| **$2160\text{–}2240\text{ nm}$** ($\sim 2200\text{ nm}$) | $\mathrm{Al\text{-}OH}$ ve kaolinit/illit/smektit kafesi | Kil mineralojisi ve organik-mineral kompleksi | `st_ds.visualization.spectral_plots` bant taraması |
| **$2300\text{–}2380\text{ nm}$** ($\sim 2340\text{ nm}$) | $\mathrm{CO_3^{2-}}$ karbonat titreşimi | İnorganik karbon ($\mathrm{CaCO_3}$ / $\mathrm{SIC}$) ayırıcı bandı | `SIC` tahmini ve XAI IG atıf doğrulaması |
| **LIBS ($193.1, 229.7, 247.8\text{ nm}$)** | $\mathrm{C\text{ I}}$ atomik emisyon + $\mathrm{H\text{ I}}$ ($656.3\text{ nm}$) | Doğrudan elemental karbon çizgisi | `st_ds.signal` (`ArPLS` + `Pseudo-Voigt`) |

### 3.2 Çekirdek Formüller ve Toprak Yorumu / *Core Equations & Soil Interpretation*

1. **Radyometrik Kalibrasyon ve Beer–Lambert Soğurma (*Radiometric Calibration & Absorbance*):**
   $$R(\lambda) = \frac{I_{\text{soil}}(\lambda) - I_{\text{dark}}(\lambda)}{I_{\text{white}}(\lambda) - I_{\text{dark}}(\lambda)}, \qquad A(\lambda) = -\log_{10}\!\big(\max(R(\lambda), \varepsilon)\big)$$
   - **🇹🇷** Toprak reflektansını konsantrasyonla yaklaşık doğrusal optik yoğunluğa dönüştürür (`st_ds.calibration.radiometry`, `st_ds.chemometrics.transforms`).
   - **🇬🇧** Converts raw soil detector counts into concentration-linear absorbance space while masking saturated bands.

2. **SNV Parçacık Saçılması Normalizasyonu (*Standard Normal Variate*):**
   $$\tilde{a}_i(\lambda) = \frac{a_i(\lambda) - \bar{a}_i}{s_i}$$
   - **🇹🇷** Öğütülmemiş/kaba tarla agregatlarının yol açtığı çarpımsal ve toplamsal baz çizgisi kaymasını örnek başına giderir.
   - **🇬🇧** Removes additive baseline shift and multiplicative particle-size scatter caused by coarse field soil aggregates.

3. **Dış Parametre Ortogonalizasyonu — EPO (*Moisture Suppression*):**
   Eşleşmiş ıslak–kuru fark matrisi $D = Z_{\text{wet}} - Z_{\text{dry}}$ tekil değer ayrıştırmasından ($D = U \Sigma V^\top$) ilk $q$ nem vektörü ($V_q$) seçilir:
   $$P_{\mathrm{EPO}} = I - V_q V_q^\top, \qquad Z_{\text{clean}} = Z \, P_{\mathrm{EPO}}$$
   - **🇹🇷** Toprak neminin yol açtığı spektral varyansa ortogonal alt uzaya projeksiyon yaparak tarla nemi değiştikçe $\mathrm{SOC}$ tahmininin savrulmasını önler (`st_ds.chemometrics.epo`, uzay imzası `space_signature` ile mühürlüdür).
   - **🇬🇧** Projects spectra onto the subspace orthogonal to moisture perturbations (`V_q`), stabilizing SOC predictions across wet/dry field regimes under a strict SHA-256 `space_signature` lock.

4. **Çok Görevli Kayıp ve Fiziksel Kapatma (*Multi-Task Loss & Carbon Closure*):**
   $$\mathcal{L} = \mathcal{L}_{\mathrm{SOC}} + \lambda_{\mathrm{SIC}}\mathcal{L}_{\mathrm{SIC}} + \lambda_{\mathrm{TC}}\mathcal{L}_{\mathrm{TC}} + \lambda_{\text{close}}\big(\widehat{\mathrm{TC}} - (\widehat{\mathrm{SOC}} + \widehat{\mathrm{SIC}})\big)^2, \qquad \widehat{\mathrm{SOC}}, \widehat{\mathrm{SIC}}, \widehat{\mathrm{TC}} \ge 0$$
   - **🇹🇷** `Softplus` çıktılar sayesinde negatif karbon fiziksel olarak imkansız kılınır (`STSoilHybridNetRegressor`); kapanma artığı $r_{\mathrm{TC}} = \widehat{\mathrm{TC}} - (\widehat{\mathrm{SOC}} + \widehat{\mathrm{SIC}})$ kalibrasyon uyarısı üretir.
   - **🇬🇧** Positivity is enforced via `Softplus` heads while closure regularization keeps total carbon consistent with organic + inorganic fractions.

5. **Conformal Belirsizlik ve Alan Dışı (OOD) Güvenlik Kapısı (*Uncertainty & OOD Gate*):**
   $$q_{1-\alpha} = \mathrm{Quantile}\!\left(\frac{|y_j - \hat{y}_j|}{\max(\hat{\sigma}_j, \varepsilon)},\; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right), \qquad \mathrm{CI}_{90}(x^\star) = \hat{y}(x^\star) \pm q_{0.90}\hat{\sigma}(x^\star)$$
   $$\mathrm{OOD}(x^\star) = \max\!\left(\frac{d_{\text{Mahal}}(x^\star)}{\tau_M},\; \frac{d_{\text{kNN}}(x^\star)}{\tau_K}\right) > 1.0 \implies \texttt{OUT\_OF\_DOMAIN}$$
   - **🇹🇷** Yeni toprak örneği eğitim veri setinin toprak/mineral/nem zarfı dışındaysa sistem nokta tahminiyle kullanıcıyı yanıltmaz (`st_ds.uncertainty`).
   - **🇬🇧** Distribution-free finite-sample intervals paired with dual Mahalanobis + k-NN applicability-domain gating flag novel soil matrices before erroneous agronomic decisions occur.

---

## 4. Toprak Karbon Havuzları ve Fiziksel Stok Muhasebesi

### 🇹🇷 Türkçe — Hacim Ağırlığı Olmadan Asla Stok Varsayılmaz (`DEF-005`)
Toprak organik karbon konsantrasyonu ($C$, kuru ağırlık yüzdesi $\%$) tek başına alan başına karbon stokunu vermez. $d$ ($\mathrm{cm}$) kalınlığındaki, hacim ağırlığı $\rho_b$ ($\mathrm{g\;cm^{-3}}$) ve $>2\text{ mm}$ taş/kaba parça hacim kesri $f_c \in [0, 1)$ olan bir toprak katmanının stoku:

$$S\;[\mathrm{Mg\;C\;ha^{-1}}] = C\;(\%) \times \rho_b\;(\mathrm{g\;cm^{-3}}) \times d\;(\mathrm{cm}) \times (1 - f_c)$$

Birinci derece Taylor varyans yayılımı (parametre belirsizliklerinin katkısını tornado grafiği olarak veren denklem):

$$\sigma_S^2 \approx \left(\frac{\partial S}{\partial C}\right)^2 \sigma_C^2 + \left(\frac{\partial S}{\partial \rho_b}\right)^2 \sigma_{\rho_b}^2 + \left(\frac{\partial S}{\partial d}\right)^2 \sigma_d^2 + \left(\frac{\partial S}{\partial f_c}\right)^2 \sigma_{f_c}^2$$

Örnek veride $\rho_b$ (`bulk_density_g_cm3`) eksikse `st-ds` sessizce `1.35 g/cm³` varsayımı **yapmaz**; `stock_mg_ha = None` ve `INSUFFICIENT_DATA` (`MISSING_BULK_DENSITY`) döndürür.

### 🇬🇧 English — Fail-Closed Carbon Stock Accounting (`DEF-005`)
Soil carbon concentration ($C$, wt%) alone cannot quantify carbon stocks per hectare. For layer thickness $d$ ($\mathrm{cm}$), fine-earth bulk density $\rho_b$ ($\mathrm{g\;cm^{-3}}$), and coarse fragment fraction $f_c \in [0, 1)$:

- First-order Taylor uncertainty attribution (`Plot 21`) quantifies how much of $\sigma_S^2$ stems from spectral model error ($\sigma_C^2$) versus field bulk density uncertainty ($\sigma_{\rho_b}^2$).
- Whenever `bulk_density` is absent, `st_ds.carbon.stock` enforces a strict **fail-closed** rule (`stock_mg_ha = None`, `MISSING_BULK_DENSITY`) instead of fabricating a silent `1.35 g/cm³` default.

---

## 5. Tarla Ölçeğinde Mekânsal Toprak Haritalama (Variogram & Ordinary Kriging)

### 🇹🇷 & 🇬🇧
En az 8 ayrı saha koordinatına $(\text{lon}, \text{lat})$ sahip toprak örnekleri yerel metrik düzleme $(x, y)\;[\mathrm{m}]$ projekte edilir (`st_ds.spatial`):
1. **Ampirik Matheron Semivariogramı:** $\hat{\gamma}(h) = \frac{1}{2|N(h)|}\sum_{(i,j)\in N(h)}(z_i - z_j)^2$ (`Plot 22`).
2. **Ordinary Kriging Sistemi:** Yansızlık koşulu $\sum_i w_i = 1$ altında $\hat{z}(x_0) = \sum_i w_i z_i$ ve kriging standart sapması $\sigma_K(x_0)$ hesaplanır; $\sum w_i - 1$ ağırlık toplamı hatası doğrulanır (`Plot 23`).
3. **Delaunay Dış Bükey Örtü Maskesi:** Örneklenmiş parsel sınırları dışındaki ekstrapolasyon noktaları `is_extrapolated = True` ve kırmızı kesikli sınırla işaretlenir (`kriging_to_geojson` ile GIS/QGIS uyumlu GeoJSON üretilir).

---

## 6. Toprak Analizi İçin Çift Dilli Kullanım (`.env`, CLI `--lang`, `scripts/` & Çok Sayfalı PDF)

### 6.1 İsteğe Bağlı `.env` & Parametre ile Dil Seçimi (`TR` / `EN`)
Sistem `.env` dosyasının varlığına **bağımlı değildir**. Varsayılan dil **Türkçe (`tr`)**'dir; aşağıdaki öncelik sırası uygulanır (`st_ds/config.py`):

$$\texttt{--lang / lang=} \;\succ\; \texttt{OS Env STDS\_LANG} \;\succ\; \texttt{.env STDS\_LANG} \;\succ\; \texttt{'tr' (Varsayılan)}$$

```bash
# 1) Türkçe (Varsayılan veya .env içinde STDS_LANG=tr)
bash scripts/03_run_full_pipeline.sh toprak_orneklerim.csv

# 2) İngilizce çalıştırmak için .env içini STDS_LANG=en yapın VEYA tek seferde ortam/parametre verin:
STDS_LANG=en bash scripts/03_run_full_pipeline.sh toprak_orneklerim.csv
st-ds visualize --input dataset.json --output out_en --lang en
```

### 6.2 Hazır Toprak Analizi Bash Hatları (`scripts/` → Zaman Damgalı `runs/`)

| Betik / *Script* | Toprak İş Akışı / *Soil Workflow* | Zaman Damgalı Çıktı / *Timestamped Outputs* |
| --- | --- | --- |
| `scripts/01_generate_synthetic_data.sh` | Sentetik toprak spektrumu + eşleşmiş ıslak/kuru çiftleri üretir & doğrular | `runs/<TS>_synthetic_data/` (`dataset.json`, `dataset.csv`, `wet/dry_pairs.json`, `validation_report.json`) |
| `scripts/02_train_and_evaluate.sh` | Doğrulama → EPO → GroupKFold CV kıyaslama → SHA-256 Eğitim + Eğitim PDF | `runs/<TS>_train_eval/` (`artifact_bundle/`, `03_cv_benchmark.json`, `04_training_summary.pdf`) |
| `scripts/03_run_full_pipeline.sh` | Uçtan uca girdi → işleme → tahmin/XAI → 26 PNG + 5 CSV/MD + HTML + **Tam PDF** | `runs/<TS>_full_run/` (`00_STDS_FULL_ANALYSIS_REPORT.pdf`, `04_predictions_report.pdf`, `plots/*.png`, `tables/*`, `index.html`) |

### 6.3 Her Çalıştırmada Çok Sayfalı Kapsamlı PDF Raporu (`st_ds/reporting/pdf_report.py`)
`st-ds visualize`, `st-ds train --pdf`, `st-ds predict --pdf`, `st-ds report export --format pdf|both` ve `scripts/03_run_full_pipeline.sh` her çalıştırmada çevrimdışı (`matplotlib.backends.backend_pdf.PdfPages`) çok sayfalı **A4 Yatay PDF Raporu** üretir:
- **Sayfa 1:** Yönetici Özeti, Kanıt Sınırı (`VERİ KAYNAĞI`), Ana KPI Kartları ve **4 Aşamalı Veri Yaşam Döngüsü** (Gelen Girdi Profili → Kemometri/EPO İşleme → Çıktı & Güvenlik Kapıları → Kovaryat İlişkileri & XAI).
- **Tablo Sayfaları:** Veri Özeti, Model Kıyaslama Metrikleri, Saha/Nem Alt Grup Hatası, XAI Sıralaması ve Örnek Bazlı Tahmin/Stok Denetim Tablosu.
- **Grafik Sayfaları:** Üretilen **TÜM 26+ PNG grafik** (2'li yüksek çözünürlüklü sayfa düzeni + tam 3D kriging/spektral yüzey sayfaları) başlık, kategori ve bilimsel açıklamalarıyla birlikte gömülüdür.
