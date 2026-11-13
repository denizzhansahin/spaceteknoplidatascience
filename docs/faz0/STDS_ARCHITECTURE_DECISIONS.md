# STDS (`st_ds`) — Mimari Karar Kayıtları (Architecture Decision Records - ADR)

**Tarih:** 12 Kasım 2026  
**Durum:** Faz 0 Kabul Kapısı ile Donduruldu (`ACCEPTED`)  
**Referans Spesifikasyon:** `STDS_Tam_Sistem_Mimarisi_ve_Fazli_Gelistirme_Plani.pdf` (Bölüm 30 & Bölüm 39, Sayfa 32 & 41)

---

## 1. Genel Bakış

Bu belge, **SpaceTeknopoliDataScience (`st_ds`)** projesinin çekirdek yazılım ve matematik mimarisini yöneten **6 temel Mimari Karar Kaydını (`ADR-001` .. `ADR-006`)**, çalışma ortamı standardını (`ADR-007`) ve Faz 1 öncesinde paydaş/alan uzmanı girdisiyle kapatılacak **Açık Kararları (`OD-001` .. `OD-006`)** tanımlar.

| ADR Kodu | Başlık | Durum | İlgili Fazlar |
| :--- | :--- | :--- | :--- |
| **ADR-001** | Bağımsız `st_ds` Paketi ve AgroAlert İzolasyonu | `ACCEPTED` | Faz 0–11 |
| **ADR-002** | İsimli Spektral/Kovaryat Şeması ve Canonical Birim Sistemi | `ACCEPTED` | Faz 1, Faz 5 |
| **ADR-003** | EPO Kalibrasyon ve Çıkarım Uzayı Eşleşmesi | `ACCEPTED` | Faz 4 |
| **ADR-004** | SOC, SIC ve TC Ayrımı ile Fiziksel Kapanış Kontrolü | `ACCEPTED` | Faz 6, Faz 9 |
| **ADR-005** | Kanıta Dayalı Model Seçimi (Baseline vs `STSoilHybridNet`) | `ACCEPTED` | Faz 6, Faz 11 |
| **ADR-006** | XAI Kavram Ayrımı (Duyarlılık, HSIC Bağımlılık, Etkileşim, Sıralama) | `ACCEPTED` | Faz 8 |
| **ADR-007** | Anaconda (`/opt/anaconda3`) Çalışma Ortamı ve `STSoilAnalyzer` Facade | `ACCEPTED` | Faz 0–11 |

---

## 2. Mimari Karar Kayıtları (ADR Detayları)

### ADR-001: Bağımsız `st_ds` Python Kütüphanesi ve AgroAlert İzolasyonu

- **Bağlam:** Kaynak PDF'in 7–9. ve 57+ sayfalarında AgroAlert sistemine ait donanım sürücüleri (`SoilCollectorDriver`), FastAPI sunucuları (`soil_inference_service`) ve NestJS servisleri (`SoilAnalyzerService`) yer almaktadır. Analiz motorunun donanım ve sunucu koduyla iç içe geçmesi test edilebilirliği ve bilimsel yeniden üretilebilirliği bozmaktadır.
- **Karar:**
  1. `st_ds` tamamen bağımsız bir bilimsel Python kütüphanesi olarak geliştirilecektir.
  2. Paket içinde USB/UART/I2C/SPI donanım sürücüsü, Arduino firmware'i, GPS donanım okuyucusu, WebSocket/FastAPI/NestJS sunucu katmanı veya PostgreSQL bağlantısı **bulunmayacaktır**.
  3. AgroAlert entegrasyonu, `st_ds` MVP yayınından sonra ayrı bir ikinci proje olarak yalnızca versiyonlu JSON veri sözleşmeleri (`Measurement` -> `PredictionReport`) üzerinden yapılacaktır.
- **Sonuçlar:** `st_ds` herhangi bir laboratuvar bilgisayarında, CI/CD hattında veya Raspberry Pi 5 üzerinde donanım bağlı olmadan deterministik olarak test edilebilir ve çalıştırılabilir.

---

### ADR-002: İsimli Spektral/Kovaryat Şeması ve Canonical Birim Sistemi (Sabit 24 Kanal Varsayımının Kaldırılması)

- **Bağlam:** İlk taslakta spektral girdi sabit 24 kanallı bir dizi (`Band_1..Band_24`) olarak varsayılmış ve kovaryatlar birimsiz/sıralı sayılar olarak modele verilmiştir. Bu durum 5/18 LED bantlı sensörlerin, 1300–2550 nm FT-NIR spektrometrelerin veya LIBS serilerinin kullanımını engellemekte ve birim karışıklığına (`%` vs `g/kg`, `m³/m³` vs `%`) yol açmaktadır.
- **Karar:**
  1. Sabit 24 kanal varsayımı tamamen kaldırılmıştır. Her spektral ölçüm zorunlu bir `InstrumentProfile` ve açık `wavelength_grid_nm` (veya `band_ids` + merkez dalga boyu / FWHM) meta verisi taşır.
  2. Tüm kovaryatlar (`moisture`, `ph`, `eis_real`, `eis_imag`, `temperature_c`, `bulk_density`, `depth_cm`, `coarse_fragments`) sabit dizi indeksi yerine isimli `FeatureRegistry` ve açık birim (`unit`) bilgisiyle taşınır.
  3. Giriş katmanında tüm birimler tek bir iç **Canonical Birim Sistemine** (ör. karbon: `% kuru ağırlık`, $\rho_b$: $\text{g cm}^{-3}$, derinlik: $\text{cm}$, sıcaklık: ${}^\circ\text{C}$) dönüştürülür. Birimi belirsiz veya fiziksel aralık dışı girdiler `UnitError` / `SchemaError` ile reddedilir (`Fail-Closed`).
- **Sonuçlar:** Farklı sensör profilleri ortak mimaride güvenle temsil edilir; sessiz birim hataları ve özellik sırası kaymaları önlenir.

---

### ADR-003: EPO Kalibrasyon ve Çıkarım Uzayı Eşleşmesi

- **Bağlam:** Eski taslakta `EPO.fit` işlemi ham ıslak/kuru (`wet/dry`) spektrumlar üzerinde çalışırken, `EPO.transform` işlemi `SNV + Savitzky-Golay türevi` uygulanmış spektrumlara projeksiyon yapmaktaydı (`DEF-001`).
- **Karar:**
  1. `EPO` (External Parameter Orthogonalization) ve `GLSW` yalnızca kalibrasyon/eğitim aşamasında eşleşmiş ıslak-kuru (`paired wet/dry`) toprak numuneleri üzerinde `fit` edilir; çıkarım (`inference`) sırasında asla yeniden `fit` edilmez.
  2. `EPO.fit` ve `EPO.transform` **birebir aynı spektral ön-işleme uzayını** paylaşmak zorundadır. Ön-işleme zinciri $\Phi(\cdot)$ (ör. `absorbance -> snv -> savgol_deriv1`) hem $\mathbf{X}_{\text{wet}}$ ve $\mathbf{X}_{\text{dry}}$ kalibrasyon matrislerine hem de saha spektrumuna aynı parametrelerle uygulanır.
  3. Öğrenilen projeksiyon matrisi $\mathbf{P}_{\text{epo}} = \mathbf{I}_p - \mathbf{V}_s \mathbf{V}_s^\top$ için $\mathbf{P}_{\text{epo}}^\top = \mathbf{P}_{\text{epo}}$, $\mathbf{P}_{\text{epo}}^2 \approx \mathbf{P}_{\text{epo}}$ ve $\mathbf{V}_s^\top \mathbf{V}_s \approx \mathbf{I}_q$ matematiksel değişmezleri (invariants) birim testlerle zorunlu olarak doğrulanır.
  4. EPO rankı $q$ sabit `2` olarak dondurulmaz; grup bazlı çapraz doğrulama (grouped CV) ve sinyal koruma kriteriyle seçilir.
- **Sonuçlar:** Nem bastırma projeksiyonu matematiksel olarak tutarlı hale gelir ve karbon sinyalinin yanlış uzay projeksiyonuyla bozulması engellenir.

---

### ADR-004: SOC, SIC ve TC Ayrımı ile Fiziksel Kapanış Kontrolü

- **Bağlam:** Karbonatça zengin topraklarda inorganik karbon (`SIC`) yüksek olabilir; toplam karbonu (`TC`) doğrudan organik karbon (`SOC`) gibi raporlamak tarımsal ve karbon muhasebesi açısından hatalıdır. Ayrıca eksik hacim ağırlığında (`bulk_density`) sessizce `1.35` varsayılması yanlış stok üretmektedir (`DEF-005`).
- **Karar:**
  1. `SOC` (Organik Karbon), `SIC` (İnorganik Karbon) ve `TC` (Toplam Karbon) ayrı fiziksel büyüklükler olarak modellenir ve aynı kuru madde esasında (`dry-matter basis`) raporlanır.
  2. Fiziksel kısıtlar zorunludur: $\widehat{\text{SOC}} \ge 0$, $\widehat{\text{SIC}} \ge 0$ ve $\text{TC} = \text{SOC} + \text{SIC}$.
  3. Ayrı bir `TC` tahmini veya referansı mevcut olduğunda kapanış artığı `closure_residual` ($r_{\text{TC}} = \widehat{\text{TC}} - \widehat{\text{SOC}} - \widehat{\text{SIC}}$) hesaplanır ve kalite teşhis (`diagnostics`) alanında raporlanır.
  4. Karbon stoku ($\text{Mg C ha}^{-1}$) hesabında `bulk_density` ($\rho_b$) için hiçbir varsayılan değer (`1.35` vb.) **kullanılmaz**. $\rho_b$ ölçümü yoksa stok `None` döner ve `missing_bulk_density` bayrağı işaretlenir.
- **Sonuçlar:** Fiziksel olarak imkânsız negatif karbon tahminleri ve hayali stok raporları tamamen engellenir.

---

### ADR-005: Kanıta Dayalı Model Seçimi (Baseline vs `STSoilHybridNet`)

- **Bağlam:** Kaynak PDF'te hem klasik kemometrik modeller (`PLSR`, `Kernel Ridge`, `Random Forest`, `MBL/Local PLSR`) hem de iki kollu derin ağ (`STSoilHybridNet`) tartışılmaktadır. Küçük ve orta ölçekli spektral veri setlerinde derin ağların aşırı uyum (overfitting) riski yüksektir.
- **Karar:**
  1. Tüm modeller ortak bir `RegressorProtocol` (`fit`, `predict`, `predict_distribution`, `save/load`, `model_card`) arayüzünü uygular.
  2. MVP'de en az **PLSR** ve **Kernel Ridge Regression (RBF)** zorunlu baseline modeller olarak kurulur. `Random Forest` ve `MBL/Local PLSR` opsiyonel adaylar, `STSoilHybridNet` ise deneysel çok-kollu aday olarak uygulanır.
  3. Hiçbir model önceden varsayılan (default) ilan edilmez. Model seçimi; sızıntısız grup bölmeleri (`GroupKFold` / `Leave-One-Site-Out`) üzerinde $\text{MAE}$, $\text{RMSE}$, $\text{Bias}$, $\text{RPIQ}$, Bland-Altman uyum sınırları ve tahmin aralığı kapsaması (interval coverage) kanıtlarına göre yapılır.
- **Sonuçlar:** STDS, veri büyüklüğüne ve cihaz profiline göre en güvenilir modeli bilimsel kanıtla seçer.

---

### ADR-006: XAI Kavram Ayrımı (Duyarlılık, HSIC Bağımlılık, Etkileşim ve Sıralama)

- **Bağlam:** Eski taslakta yerel gradyan duyarlılığı, veri kümesi bağımlılığı (HSIC), değişken etkileşimi (`coupling_strength`) ve bileşen sıralaması birbirine karıştırılmış; $B < 4$ durumunda sahte eşit önem dönülmüş ve farklı birimdeki değişkenler aynı sıralamaya sokulmuştur (`DEF-004`).
- **Karar:**
  1. Açıklanabilirlik (XAI) çıktıları dört ayrı kavram olarak raporlanır:
     - **Yerel Duyarlılık (`local_sensitivity` & `integrated_gradients`):** Tekil örnekte girdi değişiminin tahmine etkisi (birim/ölçek normalize).
     - **İstatistiksel Bağımlılık (`hsic_dependence`):** Veri kümesi/batch düzeyinde merkezlenmiş RBF Gram matrisleriyle normalize HSIC. $B < 4$ ise hesaplanmaz ve `insufficient_sample_count` döner.
     - **Değişken Etkileşimi (`interaction_score`):** İki maksimum gradyanın çarpımı **kullanılmaz**; ikinci mertebe çapraz kısmi türev (mixed partial derivative) veya 2D pertürbasyon etkileşimi kullanılır.
     - **Bileşen Sıralaması (`component_ranking`):** Spektral bantlar ve kovaryatlar kendi birim/grup bağlamlarında ve hedef bazında (`SOC` / `SIC` ayrı) sıralanır; nedensellik iddiası yapılmaz.
- **Sonuçlar:** XAI raporları matematiksel olarak savunulabilir ve tekrar üretilebilir hale gelir.

---

### ADR-007: Anaconda Çalışma Ortamı Standardı ve `STSoilAnalyzer` Facade Korunumu

- **Bağlam:** Geliştirme ortamı tutarlılığı ve dış API isimlendirmesi (Bölüm 25 & 30).
- **Karar:**
  1. Tüm geliştirme, paket yönetimi, eğitim ve test süreçleri **Anaconda** (`/opt/anaconda3`, Python `3.13.9`, PyTorch `2.7.1`, NumPy `2.2.6`, SciPy `1.17.0`, Scikit-Learn `1.6.1`) üzerinde yürütülür.
  2. Dış tüketici arayüzü olarak `STSoilAnalyzer` facade sınıf adı korunur (`STSoilAnalyzer.from_artifact(...)`, `predict`, `predict_batch`, `explain`, `inspect_artifact`). Eski `STSoilAnalyzer(weights_path=...)` imzası yalnızca `DeprecationWarning` üreten ve gizli varsayılanlara izin vermeyen bir geçiş sarmalayıcısı olarak desteklenir.

---

## 3. Açık Mimari Kararlar Kaydı (Open Decisions — Faz 1..2 Girişinde Kapatılacaklar)

Spesifikasyon Bölüm 39 (Sayfa 41) uyarınca, donanım ve laboratuvar protokolüne bağlı olan aşağıdaki 6 açık karar kayıt altına alınmıştır. Yazılım mimarisi (`ADR-002`) bu kararların her birini parametrik `InstrumentProfile` ve `SoilReference` şemalarıyla destekleyecek şekilde esnek tasarlanmıştır:

| Kod | Açık Karar Sorusu | Mimari Etkisi ve Varsayılan Faz 1 Yaklaşımı |
| :--- | :--- | :--- |
| **OD-001** | MVP donanım profili 24 kanal mı, 5/18 LED bandı mı, yoksa 1300–2550 nm FT-NIR mı olacak? | Faz 1'de `InstrumentProfile` her üç sınıfı da (`led_5ch`, `multispectral_18ch`, `array_24ch`, `ftnir_1300_2550nm`) şema düzeyinde tanımlayacak; gerçek kalibrasyon verisi hangi cihazdan gelirse o profil aktif artifact olacaktır. |
| **OD-002** | LIBS ilk MVP için zorunlu mu, yoksa ikinci cihaz profili mi? | Faz 3–4'te `st_ds/signal/` altında ARPLS ve Pseudo-Voigt modülleri yazılacak, ancak çekirdek `STSoilAnalyzer` boru hattında LIBS girdisi `Optional` bir sensör dalı olarak tutulacaktır. |
| **OD-003** | SIC ground-truth yöntemi ve referans laboratuvar protokolü hangisi olacak? | `SoilReference` şemasında `method_id` (ör. `calcimeter_iso10693`, `dry_combustion_iso10694`) zorunlu alan olacaktır. |
| **OD-004** | Bulk density ($\rho_b$) ölçümü penetrometre vekiliyle mi, karot (core) referansıyla mı, yoksa ikisiyle birlikte mi yapılacak? | `CovariateSet` içinde `bulk_density_source` (`core_sample`, `penetrometer_estimated`, `external_lab`) ve belirsizliği ($\sigma_{\rho_b}$) ayrı taşınacaktır. |
| **OD-005** | İlk MVP'nin toprak sınıfı taksonomisi (4 sınıf) ve eşikleri kim tarafından tanımlanacak? | Bölüm 22 uyarınca uzman onaylı ölçülebilir eşikler gelene kadar sınıflandırma başlığı (`classification`) opsiyonel/geçici taksonomi (`provisional_taxonomy_v0`) olarak işaretlenecektir. |
| **OD-006** | Bölgesel kalibrasyon için hangi saha/tekstür/nem kapsaması sağlanacak? | `model_card.json` içinde eğitim verisinin saha, tekstür ve nem sınırları otomatik özetlenecek; bu sınırların dışı Faz 7'de `out_of_domain` olarak bayraklanacaktır. |
