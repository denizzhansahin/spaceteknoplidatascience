# STDS (`st_ds`) — Faz 0: Bilinen Tasarım Hataları ve Çözüm Planı (Known Defects & Remediation Plan)

**Tarih:** 12 Kasım 2026  
**Denetçi Rolü:** STDS Faz 0 Baş Denetçisi (Architecture Auditor)  
**Referans Spesifikasyon:** `STDS_Tam_Sistem_Mimarisi_ve_Fazli_Gelistirme_Plani.pdf` (Bölüm 3, Sayfa 5; Bölüm 10–28)

---

## 1. Genel Bakış

Bu belge, kaynak PDF'teki ilk kod taslağının (`ChemometricProcessor`, `SoilXAIEngine`, `STSoilAnalyzer`) teknik denetimi sonucunda tespit edilen **6 kritik mimari ve matematiksel kusuru (`DEF-001`..`DEF-006`)** ve ek tasarım açıklarını (`DEF-007`..`DEF-009`) kayıt altına alır.

Çalışma dizininde (`/Users/denizhan/Desktop/SpaceTeknopoliDataScience`) fiziksel `.py` dosyaları henüz bulunmadığından (`ABSENT`), her kusur **Mimari Spesifikasyon PDF'indeki sayfa, bölüm ve satır numaralarıyla** referanslandırılmış; kusurlu taslak davranış ile hedef `st_ds` üretim sözleşmesi yan yana tanımlanmıştır. Hiçbir faz, bu belgedeki ilgili kabul testlerini (`pytest`) geçmeden kapatılamaz.

---

## 2. Denetlenen 6 Kritik Mimari Hata (`DEF-001` .. `DEF-006`)

### DEF-001 — EPO Fit ve Transform Uzay Uyumsuzluğu (EPO Space Mismatch)

- **Şiddet:** Kritik (Matematiksel Geçersizlik / Eğitim-Çıkarım Çökmesi)
- **Kaynak PDF Referansı:**
  - Bölüm 3, Sayfa 5, Satır 8–10: *"Kod üretim öncesi düzeltilmesi gereken açıklar: EPO fit aşaması ham wet/dry spektrumları kullanırken transform aşaması SNV + türev uygulanmış spektrumlara projeksiyon yapmaktadır; eğitim ve çıkarım uzayı eşleşmeyebilir."*
  - Bölüm 13, Sayfa 15, Satır 16–19 ve Bölüm 14, Sayfa 16, Satır 5–7.
- **Kusurlu Taslak Davranışı:**
  - `ChemometricProcessor.fit_epo(X_wet_raw, X_dry_raw)` fonksiyonu fark matrisini $\mathbf{D}_{\text{raw}} = \mathbf{X}_{\text{wet, raw}} - \mathbf{X}_{\text{dry, raw}}$ üzerinden hesaplayıp sağ tekil vektörleri ($\mathbf{V}_s$) ham reflektans/intensite uzayında öğrenmektedir.
  - Ancak çıkarım (`transform`) aşamasında girdi spektrumu önce $\tilde{\mathbf{x}} = \text{SG}(\text{SNV}(\mathbf{x}))$ dönüşümünden geçirilmekte ve ardından ham uzaya ait $\mathbf{P}_{\text{raw}} = \mathbf{I}_p - \mathbf{V}_s \mathbf{V}_s^\top$ matrisiyle çarpılmaktadır ($\tilde{\mathbf{x}} \mathbf{P}_{\text{raw}}$).
  - **Matematiksel Sonuç:** $\text{SNV}$ ve Savitzky-Golay türevi uzayın bazını, ölçeğini ve yerel türev yapısını değiştirdiğinden, ham uzaydaki nem alt-uzayı $\text{span}(\mathbf{V}_s)$ ile türev uzayındaki nem alt-uzayı birbirine ortogonal veya eşdeğer değildir. Bu işlem nem sinyalini silmediği gibi karbon (`SOC`/`SIC`) sinyalini bozar.
- **Çözüm Planı ve Mimari Sözleşme (`ADR-003`, Faz 4 — `st_ds/chemometrics/epo.py`):**
  1. `EPO` sınıfı, kendisinden önceki deterministik spektral ön-işleme zincirinin (`PreprocessSpec`: ör. `absorbance -> snv -> savgol_deriv1`) çıktısı olan uzayda hem `fit` hem `transform` yapacaktır:
     $$\mathbf{Z}_{\text{wet}} = \Phi(\mathbf{X}_{\text{wet}}), \quad \mathbf{Z}_{\text{dry}} = \Phi(\mathbf{X}_{\text{dry}})$$
     $$\mathbf{D} = \mathbf{Z}_{\text{wet}} - \mathbf{Z}_{\text{dry}}, \quad \mathbf{C}_D = \frac{1}{n-1}\mathbf{D}^\top \mathbf{D} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$
     $$\mathbf{V}_s = \mathbf{V}_{:, 1:q}, \quad \mathbf{Q} = \mathbf{V}_s \mathbf{V}_s^\top, \quad \mathbf{P}_{\text{epo}} = \mathbf{I}_p - \mathbf{Q}, \quad \mathbf{Z}_{\text{clean}} = \Phi(\mathbf{X}) \mathbf{P}_{\text{epo}}$$
  2. `EPO` nesnesi içine `input_space_signature` (ön-işleme adımlarının ve `wavelength_grid` vektörünün SHA-256 özeti) kaydedilir. `transform` çağrısında gelen spektrumun `space_signature` değeri eşleşmezse `CalibrationError` fırlatılır.
  3. Rank $q$ sabit `2` alınmaz; nested/grouped CV ve Wilks-Lambda / sinyal koruma kriteriyle seçilir.
- **Zorunlu Kabul Testleri (`tests/unit/test_epo.py`):**
  - Simetri: $\|\mathbf{P}_{\text{epo}}^\top - \mathbf{P}_{\text{epo}}\|_F < 10^{-10}$
  - İdempotentlik: $\|\mathbf{P}_{\text{epo}}^2 - \mathbf{P}_{\text{epo}}\|_F < 10^{-10}$
  - Ortonormal baz: $\|\mathbf{V}_s^\top \mathbf{V}_s - \mathbf{I}_q\|_F < 10^{-10}$
  - Uzay uyumsuzluğu testi: Ham spektrumla `EPO.transform` çağrıldığında `CalibrationError` fırlatılması.

---

### DEF-002 — Savitzky-Golay Türevinde Fiziksel Dalga Boyu Ekseni İhlali

- **Şiddet:** Yüksek (Fiziksel Birim ve Cihazlar Arası Ölçek Hatası)
- **Kaynak PDF Referansı:**
  - Bölüm 3, Sayfa 5, Satır 10–11: *"Savitzky-Golay türevi bant indeksine göre alınmaktadır; gerçek dalga boyu aralıkları eşit değilse türev fiziksel dalga boyu türevi değildir."*
  - Bölüm 11, Sayfa 13, Satır 8–11 ve Satır 20: *"Eşit aralıklı olmayan bantlarda dalga boyuna göre türev ölçeklemesi yapılmadan `axis=-1` türevi fiziksel türev olarak adlandırılmamalıdır."*
- **Kusurlu Taslak Davranışı:**
  - `scipy.signal.savgol_filter(X, window_length=..., polyorder=..., deriv=1, axis=-1)` çağrısında `delta` parametresi varsayılan `1.0` bırakılmış ve bantların dalga boyu aralıklarının ($\Delta\lambda_i = \lambda_{i+1} - \lambda_i$) sabit olduğu varsayılmıştır.
  - **Matematiksel Sonuç:** Eşit aralıklı bir grid'de bile `delta` verilmezse türev $\mathrm{d}A/\mathrm{d}\lambda$ ($\text{nm}^{-1}$) yerine $\mathrm{d}A/\mathrm{d}i$ (indeks başına değişim) üretir; çözünürlüğü $2\text{ nm}$ olan bir cihaz ile $10\text{ nm}$ olan bir cihaz arasında türev genliği $5\times$ kayar. Eşit aralıklı olmayan (non-uniform) çoklu sensör veya LED dizilerinde ise standart Savitzky-Golay konvolüsyonu matematiksel olarak geçersizdir.
- **Çözüm Planı ve Mimari Sözleşme (Faz 4 — `st_ds/chemometrics/savgol.py`):**
  1. `validate_spectral_grid(wavelengths)` fonksiyonu $\lambda$ vektörünün kesin artan ($\lambda_1 < \lambda_2 < \dots < \lambda_p$), `NaN`/`Inf` içermeyen bir dizi olduğunu doğrular.
  2. **Durum A (Uniform Grid — $|\Delta\lambda_i - \Delta\lambda| \le \epsilon$):** `savgol_filter(..., delta=float(delta_nm), axis=-1)` kullanılarak gerçek $\mathrm{d}^k A / \mathrm{d}\lambda^k$ ($\text{nm}^{-k}$) hesaplanır.
  3. **Durum B (Non-Uniform Grid):** Her kayan pencerede yerel koordinat $\xi = \lambda - \lambda_c$ üzerinde ağırlıklı/en küçük kareler polinomu $p(\xi) = \sum_{m=0}^{d} c_m \xi^m$ çözülerek merkez dalga boyunda $k! \, c_k = \left.\frac{\mathrm{d}^k p}{\mathrm{d}\lambda^k}\right|_{\lambda_c}$ analitik olarak hesaplanır (veya doğrulamalı referans grid'e interpolasyon + maske uygulanır).
  4. Kanal sayısı pencere boyutundan küçükse (ör. 5 bantlı LED profilinde `window=7`) sessiz kırpma yapılmaz; açık `SpectralQualityError` / `SchemaError` fırlatılır.
- **Zorunlu Kabul Testleri (`tests/unit/test_savgol.py`):**
  - Analitik polinom testi: $A(\lambda) = a\lambda^2 + b\lambda + c$ fonksiyonunda hem uniform hem non-uniform $\lambda$ grid üzerinde 1. türevin $2a\lambda + b$ ve 2. türevin $2a$ değerini $<10^{-9}$ hatayla vermesi.

---

### DEF-003 — Sabit 24 Bant Varsayımı (Hardcoded 24-Channel Assumption)

- **Şiddet:** Yüksek (Mimari Kısıt / Cihaz Uyumsuzluğu)
- **Kaynak PDF Referansı:**
  - Bölüm 3, Sayfa 5, Satır 11–13: *"24 bant varsayımı, 5 LED bandı veya 1300-2550 nm sürekli FT-NIR gibi farklı cihaz profilleriyle uyumlu değildir."*
  - Bölüm 9, Sayfa 11, Satır 1–19 ve Bölüm 39, Sayfa 41 (`ADR-002`).
- **Kusurlu Taslak Davranışı:**
  - Veri okuyucu, kemometrik ön-işleyici ve `STSoilHybridNet` giriş katmanı sabit `24` spektral kanala (`Band_1..Band_24`) göre kodlanmıştır.
  - Su bantları ($1410\text{–}1450\text{ nm}$, $1900\text{–}1950\text{ nm}$), kil bandı ($\sim 2200\text{ nm}$), organik fonksiyonel grup bandı ($2130\text{–}2207\text{ nm}$) ve karbonat bantları ($2310\text{–}2341\text{ nm}$, $2500\text{–}2550\text{ nm}$) fiziksel dalga boyu ($\text{nm}$) yerine sabit dizi indeksleriyle aranmıştır.
- **Çözüm Planı ve Mimari Sözleşme (`ADR-002`, Faz 1 & Faz 5 — `st_ds/domain/instrument.py`, `st_ds/features/`):**
  1. Her ölçüm zorunlu bir `InstrumentProfile` (`profile_id`, `wavelength_grid_nm`, `fwhm_nm`, `spectral_range_nm`, `channel_ids`, `modality`) ile eşleşir.
  2. Spektral indeksleme kaldırılır; tüm bant seçimleri ve kimyasal bölge öznitelikleri dalga boyu aralık sorgularıyla (`[lambda_min, lambda_max]`) çıkarılır.
  3. Bir cihazın spektral aralığı karbonat bölgesini ($>2300\text{ nm}$) kapsamıyorsa (ör. yalnızca $850\text{–}1050\text{ nm}$ silikon dedektör), eksik spektral bölge asla sıfır veya uydurma interpolasyonla doldurulmaz; modelin yetenek sınırı `model_card.json` ve `QualityFlag` içinde açıkça kısıtlanır.
- **Zorunlu Kabul Testleri (`tests/contract/test_instrument_profiles.py`):**
  - 5 bantlı LED, 18 kanallı multispektral, 24 kanallı dizi ve 256+ kanallı FT-NIR ($1300\text{–}2550\text{ nm}$) profillerinin aynı `Spectrum` ve `InstrumentProfile` şemasıyla hatasız doğrulanması; profil uyuşmazlığında `ArtifactCompatibilityError` fırlatılması.

---

### DEF-004 — XAI Eksikleri: Çağrılmayan HSIC, $B < 4$ Sahte Eşit Önem, Sahte `coupling_strength` ve Birim Karışıklığı

- **Şiddet:** Yüksek (Bilimsel Yanıltıcılık / Yanlış Açıklanabilirlik Raporu)
- **Kaynak PDF Referansı:**
  - Bölüm 3, Sayfa 5, Satır 14–20:
    1. *"XAI tarafında HSIC fonksiyonu yazılmış olsa da tekil tahmin raporlayıcısında çağrıldığı gösterilmemektedir."*
    2. *"B<4 durumunda eşit önem dönmek gerçek ölçüm değildir; bunun yerine 'yetersiz örnek' durumu dönmelidir."*
    3. *"Kodda coupling_strength, iki ayrı maksimum gradyanın çarpımıdır; gerçek değişkenler arası etkileşim/mixed partial derivative değildir."*
    4. *"Moisture yüzdesi ile SOC/SIC yüzdesinin tek bir büyüklük sıralamasında karşılaştırılması da birim ve fiziksel anlam bakımından hatalıdır."*
  - Bölüm 21, Sayfa 23, Satır 1–21 ve Bölüm 39, Sayfa 41 (`ADR-006`).
- **Kusurlu Taslak Davranışı (4 Alt Kusur):**
  - **DEF-004a (Ölü/Çağrılmayan HSIC Kodu):** `SoilXAIEngine.compute_hsic` metodu tanımlanmış ancak `STSoilAnalyzer` tekil tahmin raporunda çağrılmamıştır (ayrıca HSIC veri kümesi düzeyinde bir istatistiksel bağımlılık ölçüsüdür; tek örnek $B=1$ üzerinde tanımlı değildir).
  - **DEF-004b ($B < 4$ Sahte Eşit Önem):** Batch boyutu $B < 4$ olduğunda HSIC fonksiyonu hata veya uyarı dönmek yerine tüm özelliklere eşit önem (`1.0 / n_features`) atayarak kullanıcıya gerçek bir analiz yapılmış izlenimi vermektedir.
  - **DEF-004c (Sahte `coupling_strength`):** Spektral sinyalin maksimum mutlak gradyanı ($\max_i |\partial f / \partial s_i|$) ile kovaryatların maksimum mutlak gradyanının ($\max_j |\partial f / \partial c_j|$) skaler çarpımı `coupling_strength` olarak raporlanmıştır. Oysa toplamsal (etkileşimsiz) doğrusal bir modelde bile $f(s, c) = a s + b c \implies (\partial f / \partial s)(\partial f / \partial c) = a b \neq 0$ çıkar; gerçek etkileşim $\frac{\partial^2 f}{\partial s_i \partial c_j} = 0$'dır.
  - **DEF-004d (Birim ve Hedef Karışıklığı):** Toprak nemi ($\theta_v, \%$), pH, empedans ($\Omega$) ve karbon konsantrasyonları ($\text{SOC } \%$, $\text{SIC } \%$) birim normalizasyonu yapılmadan aynı büyüklük listesinde sıralanmaktadır.
- **Çözüm Planı ve Mimari Sözleşme (`ADR-006`, Faz 8 — `st_ds/xai/`):**
  1. **Kavram Ayrımı:** XAI raporu 4 ayrı ve bağımsız bloktan oluşur:
     - `local_sensitivity`: Birim/standart sapma ($\sigma_i$) ile normalize edilmiş yerel duyarlılık $\tilde{S}_i(x) = \sigma_i \left|\frac{\partial f(x)}{\partial x_i}\right|$ ve `Integrated Gradients`:
       $$\text{IG}_i(x) = (x_i - x_{0,i}) \int_0^1 \frac{\partial f(x_0 + \alpha(x - x_0))}{\partial x_i}\,\mathrm{d}\alpha$$
     - `dataset_dependence_hsic`: Yalnızca batch/veri kümesi ($B \ge 8$, minimum anlamlı alt sınır $B \ge 4$) analizlerinde veya eğitim referans dağılımı üzerinde merkezlenmiş Gram matrisleriyle hesaplanan normalize HSIC:
       $$\text{HSIC}(\mathbf{K}, \mathbf{L}) = \frac{\text{tr}(\mathbf{K}\mathbf{H}\mathbf{L}\mathbf{H})}{(n-1)^2}, \quad \mathbf{H} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top, \quad \text{nHSIC} = \frac{\text{HSIC}(\mathbf{K},\mathbf{L})}{\sqrt{\text{HSIC}(\mathbf{K},\mathbf{K})\,\text{HSIC}(\mathbf{L},\mathbf{L})}}$$
     - `feature_interactions`: İkinci mertebe çapraz kısmi türev (Hessian off-diagonal $\left|\frac{\partial^2 f}{\partial x_i \partial x_j}\right| \sigma_i \sigma_j$) veya kontrollü 2D pertürbasyon etkisi:
       $$I_{ij}(x) = \left| f(x_{+\Delta i, +\Delta j}) - f(x_{+\Delta i}) - f(x_{+\Delta j}) + f(x) \right|$$
     - `component_ranking`: Spektral bölgeler ve kovaryatlar kendi fiziksel grupları içinde ve hedef değişken bazında (`SOC` için ayrı, `SIC` için ayrı) sıralanır; nem yüzdesi ile karbon yüzdesi aynı listede yarıştırılmaz.
  2. **$B < 4$ Fail-Closed Kuralı:** Örnek sayısı $B < 4$ iken `compute_hsic` çağrıldığında sahte sayısal dizi dönülmez; `status="insufficient_sample_count"` ve `values=None` döner.
- **Zorunlu Kabul Testleri (`tests/unit/test_xai.py`):**
  - Doğrusal toplamsal fonksiyon $f(x_1, x_2) = 3x_1 + 5x_2$ üzerinde `interaction_score(x_1, x_2)` değerinin $0.0$ ($\le 10^{-7}$) çıkması.
  - $B \in \{1, 2, 3\}$ için `compute_hsic` çağrısının `insufficient_sample_count` bayrağı döndürmesi ve eşit ağırlık üretmemesi.

---

### DEF-005 — Fail-Closed İhlalleri: Sessiz `bulk_density = 1.35` Varsayılanı ve Eksik Kovaryat/Birim Kontrolleri

- **Şiddet:** Kritik (Yanlış Karbon Stoku Hesabı ve Sessiz Veri Bozulması)
- **Kaynak PDF Referansı:**
  - Bölüm 3, Sayfa 5, Satır 21–24: *"STSoilAnalyzer taslağında ham bant sayısı, dalga boyu vektörü, kovaryat birimleri, aralık kontrolleri, eksik sensör, kalibrasyon tarihi ve model veri sözleşmesi doğrulaması yoktur. Bulk density için sessizce 1.35 varsayılması yanlış stok üretebilir."*
  - Bölüm 5, Sayfa 7, Satır 13–16 (*"Fail closed"*), Bölüm 8, Sayfa 10 ve Bölüm 23, Sayfa 25, Satır 14–16.
- **Kusurlu Taslak Davranışı:**
  - Kullanıcının gönderdiği `covariates` sözlüğünde `bulk_density` bulunmadığında `covariates.get("bulk_density", 1.35)` ile sessizce $1.35\text{ g cm}^{-3}$ atanmakta ve $0\text{–}30\text{ cm}$ karbon stoku bu varsayımla hesaplanmaktadır.
  - Kovaryatların fiziksel birimleri ($\theta_v$: $\text{m}^3/\text{m}^3$ vs $\%$; $\text{EC}$: $\text{S/m}$ vs $\text{dS/m}$; $\text{SOC}$: oran $0.012$ vs yüzde puanı $1.2\%$) ve fiziksel geçerlilik aralıkları (ör. $\text{pH} \in [2.5, 11.0]$, $\rho_b \in [0.4, 2.2]\text{ g cm}^{-3}$, $f_{\text{coarse}} \in [0, 1)$) doğrulanmamaktadır.
- **Çözüm Planı ve Mimari Sözleşme (`ADR-002` & `ADR-004`, Faz 1 & Faz 9 — `st_ds/domain/`, `st_ds/carbon/stock.py`):**
  1. **Gizli Varsayılan Yasağı (No Hidden Defaults):** `bulk_density` ($\rho_b$) veya `depth_cm` eksikse karbon stoku **hesaplanmaz** (`stock_mg_ha = None`); raporda `QualityFlag.MISSING_BULK_DENSITY` işaretlenir.
  2. **Canonical Birim Sistemi ve Aralık Doğrulaması:** `CovariateSet` her değişkeni açık birimiyle (`value`, `unit`) alır ve iç canonical birime dönüştürür. Birimi belirtilmeyen veya fiziksel sınırlar dışında kalan (ör. negatif $\rho_b$, $\text{pH} = 18$, $\theta_v = 150\%$) girdiler `UnitError` veya `SchemaError` ile reddedilir.
  3. **Birim Tutarlılıklı Stok Formülü:**
     $$\text{Stock}_{0\text{–}d}\;(\text{Mg C ha}^{-1}) = C_{\text{pct}}\;(\%) \times \rho_b\;(\text{g cm}^{-3}) \times d\;(\text{cm}) \times (1 - f_{\text{coarse}})$$
     Eğer girdi $C$ kütle kesri ($0.012\text{ g/g}$) veya $\text{g/kg}$ ($12.0\text{ g/kg}$) olarak verilmişse önce açık birim dönüşümüyle yüzde puanına ($1.2\%$) çevrilir.
- **Zorunlu Kabul Testleri (`tests/unit/test_carbon_stock.py`, `tests/contract/test_schema_validation.py`):**
  - `bulk_density=None` verildiğinde `stock_mg_ha is None` ve `"missing_bulk_density" in report.quality.flags` doğrulaması.
  - $C = 1.2\%$ ve $C = 12.0\text{ g/kg}$ girdilerinin aynı $\rho_b = 1.30\text{ g cm}^{-3}, d = 30\text{ cm}, f_{\text{coarse}} = 0.10$ altında birebir aynı stoku ($42.12\text{ Mg C ha}^{-1}$) üretmesi.

---

### DEF-006 — Güvensiz Pickle Tabanlı Checkpoint Yükleme (`torch.load` RCE ve Provenans Eksikliği)

- **Şiddet:** Kritik (Güvenlik Açığı — CWE-502 Deserialization of Untrusted Data & Model Uyumsuzluğu)
- **Kaynak PDF Referansı:**
  - Bölüm 3, Sayfa 5, Satır 24–27: *"Güvenilmeyen checkpoint'i pickle tabanlı yükleme ile açmak güvenlik riski taşır."*
  - Bölüm 26, Sayfa 28, Satır 7–10: *"Pickle tabanlı `torch.load` güvenilmeyen kaynakta çalıştırılmaz; `weights_only` veya güvenli formatlar ve hash doğrulaması kullanılır."*
- **Kusurlu Taslak Davranışı:**
  - `STSoilAnalyzer.__init__(self, weights_path)` içinde doğrudan `torch.load(weights_path)` çağrılmaktadır. Varsayılan pickle serileştirmesi, dosya içine gömülü rastgele Python byte-kodunu (Remote Code Execution) çalıştırabilir.
  - Ayrıca yalnızca model ağırlıkları (`.pt`) yüklenmekte; modelin eğitildiği `wavelength_grid`, `EPO` matrisi, kovaryat sırası/ölçekleyicileri ve birim tablosu ağırlık dosyasından bağımsız tutulduğundan sessiz özellik kayması (silent feature drift) riski doğmaktadır.
- **Çözüm Planı ve Mimari Sözleşme (Faz 10 — `st_ds/artifact/`):**
  1. **Artifact Bundle Yapısı:** Tekil `.pt` dosyası yerine imzalı/bütünlüklü bir dizin veya arşiv yapısı (`manifest.json`, `model_card.json`, `preprocess_state.npz`, `model_weights.safetensors` veya `torch.load(..., weights_only=True)`) kullanılır.
  2. **SHA-256 Bütünlük Kontrolü:** `load_artifact` fonksiyonu, ağırlık veya durum dosyalarını belleğe almadan önce `manifest.json` içindeki SHA-256 özetlerini doğrular. Özet uyuşmazlığında veya `weights_only=False` gerektiren pickle nesnelerinde `ArtifactCompatibilityError` fırlatılır.
- **Zorunlu Kabul Testleri (`tests/contract/test_artifact_security.py`):**
  - Bozuk SHA-256 özetli veya kötücül pickle payload içeren sahte bir checkpoint dosyasının çalıştırılmadan `ArtifactCompatibilityError` ile reddedilmesi.

---

## 3. Denetimde Tespit Edilen Ek Mimari Açıklar (`DEF-007` .. `DEF-009`)

| Hata Kodu | Kaynak PDF Referansı | Açıklama | Çözüm Fazı ve Sözleşmesi |
| :--- | :--- | :--- | :--- |
| **DEF-007**<br>ISS Kalibrasyon İşaret/Yön Belirsizliği | Bölüm 10, Sayfa 12, Satır 9–12 ve Satır 21–23 | Kaynakta verilen $\text{CF}(\lambda_i) = 1 - \frac{S(\lambda_i) - M(\lambda_i)}{S(\lambda_i)} = \frac{M(\lambda_i)}{S(\lambda_i)}$ formülünde $S$ (standart) ve $M$ (usta/referans cihaz) çarpım/bölüm yönü doğrulanmamıştır. | **Faz 3 (`st_ds/calibration/iss.py`):** Bilinen referans spektrumu $\mathbf{S}_{\text{ref}}$ ve cihaz spektrumu $\mathbf{S}_{\text{inst}}$ fixture'ları üzerinde $\mathbf{S}_{\text{corrected}} \approx \mathbf{S}_{\text{ref}}$ eşitliğini sağlayan dönüşüm yönü birim testle dondurulur. |
| **DEF-008**<br>Negatif Karbon Tahmini ve $\text{TC}$ Kapanış İhlali | Bölüm 18 (Sayfa 20), Bölüm 22 (Sayfa 24) | Sınırsız doğrusal çıkış katmanları düşük karbonlu topraklarda $\widehat{\text{SOC}} < 0$ veya $\widehat{\text{TC}} \neq \widehat{\text{SOC}} + \widehat{\text{SIC}}$ üretebilir. | **Faz 6 & Faz 9 (`ADR-004`):** Çıkışlarda $\text{softplus}$ ($\ge 0$) kısıtı, eğitimde $\lambda_{\text{close}}(\widehat{\text{TC}} - \widehat{\text{SOC}} - \widehat{\text{SIC}})^2$ cezası ve çıkarımda `closure_residual` kalite alarmı uygulanır. |
| **DEF-009**<br>Satır Bazlı Rastgele Train/Test Bölme Sızıntısı | Bölüm 19, Sayfa 21, Satır 6–9 | Aynı toprak numunesinin tekrar ölçümlerinin (replicates) veya aynı parselin rastgele train/test ayrımıyla iki tarafa dağılması yapay yüksek $R^2$ üretir. | **Faz 2 & Faz 6 (`st_ds/training/splits.py`):** Rastgele `train_test_split` yasaklanır; `sample_id`, `site_id` ve `session_id` bazlı `GroupKFold` / `LeaveOneGroupOut` zorunlu kılınır. |

---

## 4. Faz Bazlı Hata Kapatma Takvimi (Defect Closure Matrix)

| Faz | Kapatılacak Hatalar | Doğrulama Kapısı (Acceptance Gate) | Durum |
| :--- | :--- | :--- | :--- |
| **Faz 1–2** | `DEF-003` (Şema kısmı), `DEF-005` (Birim/Şema kısmı), `DEF-009` (Split kısmı) | `tests/unit/test_faz1_domain_and_units.py`, `tests/unit/test_faz2_dataset_splits_synthetic.py` | **Kapatıldı** |
| **Faz 3–4** | `DEF-001` (EPO uzay), `DEF-002` (SG $\lambda$ türevi), `DEF-007` (ISS yönü) | `tests/unit/test_faz3_calibration_and_signal_qc.py`, `tests/unit/test_faz4_chemometrics_and_epo.py` | **Kapatıldı** |
| **Faz 5–6** | `DEF-003` (Model/Feature kısmı), `DEF-008` (Model kısıtları), `DEF-009` (Eğitim) | `tests/unit/test_faz5_feature_fusion.py`, `tests/unit/test_faz6_models_and_trainer.py` | **Kapatıldı** |
| **Faz 8–9** | `DEF-004` (XAI 4 alt kusur), `DEF-005` (Karbon stoku), `DEF-008` (Closure) | `tests/unit/test_faz8_xai.py`, `tests/unit/test_faz9_carbon_and_spatial.py` | **Kapatıldı** |
| **Faz 10** | `DEF-006` (Güvenli artifact yükleme ve tam API sözleşmesi) | `tests/contract/test_artifact_security.py`, `tests/unit/test_faz10_api_and_cli.py`, `tests/integration/test_end_to_end_pipeline.py` | **Kapatıldı** |

---

## 5. MVP Sonrası Kod Denetimi ve Regresyon Kapatma Kaydı (14 Kasım 2026)

`v0.1.0` kod tabanı üzerinde yapılan uçtan uca incelemede tespit edilen ek uygulama kusurları ve kenar durumlar giderilerek `tests/unit/test_bugfix_regressions.py` altında kalıcı regresyon testlerine bağlanmıştır:

1. **Tekil vs. Çoklu Hedef Yalıtımı (`st_ds/api/analyzer.py`, `st_ds/models/hybrid_net.py`, `st_ds/models/loss.py`):** `tc_pct`/`sic_pct` tekil hedeflerinin `soc_pct` alanına yazılması önlendi; `STSoilMultiTaskLoss` maskesi `False` olan hedeflerde kayıp bileşenini kesin `0.0` tutacak şekilde düzeltildi; çok görevli `STSoilHybridNetRegressor` artefakt dönüşümünde 2D `background_targets` ve `mc_dropout_samples` korundu.
2. **Toplu Çıkarım PDF & Karma Köken İzolasyonu (`st_ds/reporting/pdf_report.py`, `st_ds/api/analyzer.py`):** `BatchPredictionResult.to_dict()` çıktısındaki `"errors"` anahtarı okunacak ve `None` kısmi hata kayıtları güvenle atlanacak şekilde düzeltildi; `provenance` içinde `training_dataset_source` ile `measurement_dataset_source` ayrıştırıldı.
3. **Taksonomi, Protokol ve Kenar Durumlar (`st_ds/domain/schemas.py`, `st_ds/carbon/fractions.py`, `st_ds/io/reference_store.py`, `st_ds/domain/units.py`, `st_ds/signal/qc.py`, `st_ds/calibration/wavelength.py`, `st_ds/models/baselines.py`, `st_ds/visualization/model_plots.py`):** `TargetLabelSchema.class_labels` ile `PROVISIONAL_CLASS_LABELS` birleştirildi; `AnalyzerProtocol` çalışma zamanı uyumluluğu ve `predict_batch` `batch_xai_hsic` desteği eklendi; katı CSV boolean ayrıştırması, `mΩ`/`MΩ` birimleri, sıfır-geçerli-kanal QC, `valid_mask` hizalama korunumu, küçük örneklem `PLSRRegressor` ve sabit `y_true` parite grafiği korumaları doğrulandı.
