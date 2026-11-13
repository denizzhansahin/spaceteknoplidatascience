# Çok Sensörlü Proksimal Toprak Spektroskopisi ile Organik, İnorganik Karbon ve Karbon Stokunun Kestirimi: STDS Bilimsel Metodoloji, Matematiksel Formülasyon ve Sistem İlişkisi

**Belge Türü:** Akademik Metodoloji ve Algoritmik Tasarım Makalesi  
**Proje / Sürüm:** SpaceTeknopoliDataScience (`st_ds`) — MVP `v0.1.0`  
**Temel Mimari Referansı:** `STDS — Tam Yazılım ve Matematik Mimarisi ve Fazlı Geliştirme Planı` (`ADR-001`..`ADR-007`)  
**Kanıt Statüsü Notu:** Bu çalışmada sunulan matematiksel yöntemler doğrudan doğrulanan algoritmaları; örnek sayısal doğrulama metrikleri ise yalnızca sentetik kalibrasyon/doğrulama boru hattını (`SyntheticGeneratorV2`) ve mimari belgede anılan dış literatür bulgularının (*Risk 9* kapsamında) ayrıştırılmış sınırlarını yansıtmaktadır.

---

## Özet (Abstract)

Tarım topraklarında Organik Karbon (**SOC**), İnorganik Karbon (**SIC**) ve Toplam Karbon (**TC = SOC + SIC**) havuzlarının doğru biçimde ayrıştırılması; hem toprak verimliliği yönetimi hem de karbon muhasebesi (MRV) açısından temel bir gereksinimdir. Geleneksel kuru yakma (*dry combustion*) ve kalsimetre laboratuvar yöntemleri yüksek doğruluk sunmasına karşın yüksek maliyet, zaman gecikmesi ve yıkıcı örnekleme nedeniyle yüksek yoğunluklu mekânsal izlemeye elverişli değildir. Proksimal toprak algılama yaklaşımları (Vis-NIR / SWIR / FT-NIR reflektans spektroskopisi, Lazer İndüklü Plazma Spektroskopisi — **LIBS**, Elektrokimyasal Empedans Spektroskopisi — **EIS** ve fizikokimyasal eş değişkenler) bu darboğazı aşma potansiyeli taşır; ancak saha ölçümleri **toprak nemi maskelemesi**, **parçacık boyutu saçılması**, **düzensiz dalga boyu ızgaraları**, **lazer atış kararsızlığı**, **veri sızıntılı modelleme**, **karbonat girişimi** ve **aşırı güvenli nokta tahminleri** gibi yedi temel fiziksel-istatistiksel engelle karşı karşıyadır.

Bu makale, **STDS (`st_ds`)** bilimsel analiz motorunun söz konusu yedi probleme getirdiği matematiksel çözümleri, verinin ham foton ve empedans düzeyinden kalibre edilmiş konsantrasyon, belirsizlik aralığı, açıklanabilirlik vektörü, fiziksel karbon stoku ($Mg\;C\;ha^{-1}$) ve Ordinary Kriging yüzeyine uzanan çok aşamalı dönüşüm yaşam döngüsünü (%90 bilimsel/matematiksel kuram; %10 yazılım modül eşleştirmesi oranıyla) sistematik biçimde ortaya koymaktadır.

---

## 1. Giriş ve Bilimsel Problem Tanımı

### 1.1 Karbon Fraksiyonlarını Ayrıştırma Zorunluluğu

Bir toprak örneğinin toplam karbon içeriği iki farklı jeokimyasal ve biyolojik havuzun toplamından oluşur:

$$\mathrm{TC} = \mathrm{SOC} + \mathrm{SIC}, \qquad \mathrm{SOC} \ge 0,\; \mathrm{SIC} \ge 0$$

Karbonatça zengin (kalkerli, yüksek pH'lı ve yarı kurak/kurak) tarım topraklarında $\mathrm{SIC}$ fraksiyonu toplam karbonun baskın kısmını oluşturabilir. Bu koşullarda yalnızca toplam karbonu ($\mathrm{TC}$) ölçüp tamamını organik karbon ($\mathrm{SOC}$) olarak raporlamak, toprak sağlığı değerlendirmelerini ve karbon kredilendirme hesaplarını sistematik biçimde saptırır. Bunun yanında, $\mathrm{SOC} = \mathrm{TC} - \mathrm{SIC}$ farkından türetilen laboratuvar etiketleri ile doğrudan ölçülen etiketlerin farklı hata varyanslarına sahip olması, model eğitimi ve değerlendirmesi sırasında **etiket kökeninin** (*provenance*) korunmasını zorunlu kılar.

### 1.2 Proksimal Algılamada Yedi Temel Bilimsel Problem ve Çözüm Matrisi

Proksimal algılayıcılardan sahada tutarlı bilimsel çıktı üretebilmek için çözülen yedi temel problem ve her çözümün yöntemsel katkısı aşağıda özetlenmiştir:

| No | Bilimsel / İstatistiksel Problem | Saha / Veri Üzerindeki Etkisi | Uygulanan Çözüm Yöntemi | Bilimsel Katkısı |
| --- | --- | --- | --- | --- |
| **P1** | **Çevresel Nem Maskelemesi & Yüzey Saçılması** | Su bantları ($1410\text{–}1450$ ve $1900\text{–}1950\text{ nm}$) organik/karbonat bantlarını bastırır; tane iriliği çarpımsal kayma üretir. | **SNV** + **EPO** (*External Parameter Orthogonalization*) / **GLSW** | Nem alt-uzayına ortogonal projeksiyonla saha nemi değişkenliğine dirençli tahmin uzayı elde edilir (`DEF-001`, `Risk 3`). |
| **P2** | **Eşit Olmayan Dalga Boyu Aralığı & Cihaz Farkı** | Çoklu bant LED/multispektral sensörler düzensiz $\Delta\lambda$ adımına sahiptir; indeks bazlı türev fiziksel eğimi bozar. | Fiziksel $\Delta\lambda$ (nm) ölçekli kaydırılmış **Vandermonde Savitzky–Golay** + kanonik `InstrumentProfile` | Farklı çözünürlük ve dalga boyu aralığına sahip cihazlarda fiziksel olarak doğru türev ($\mathrm{d}^k p/\mathrm{d}\lambda^k$) ve profil uyumluluğu sağlanır (`DEF-002`, `ADR-002`). |
| **P3** | **LIBS Atış Kararsızlığı ve Sürekli Plazma Zemini** | Lazer plazması atıştan atışa dalgalanır ve sürekli (*bremsstrahlung*) zemin dar karbon çizgilerini örter. | Atış QC + **ArPLS** ($\lambda=10^4$) zemin kestirimi + **Pseudo-Voigt** pik profili | Sürekli arka plan bastırılarak $C\text{ I}$ ($193.1, 229.7, 247.8\text{ nm}$) ve $H\text{ I}$ ($656.3\text{ nm}$) çizgilerinin net alan/genlikleri ayrıştırılır. |
| **P4** | **Tekrarlı Ölçüm & Saha Veri Sızıntısı** | Aynı toprak çekirdeğinin tekrarları veya aynı tarladaki komşu noktalar train/test'e dağılırsa doğruluk yapay olarak şişer. | **Saha/Örnek Grup Bölmesi** (`site_id`/`sample_id`) + kat içi (`fold-only`) fit/transform | Bölge dışı (*out-of-site*) gerçek genelleme performansı tarafsız ölçülür (`DEF-003`). |
| **P5** | **Fiziksel Tutarsızlık & Negatif Karbon Tahminleri** | Bağımsız regresörler fiziksel sınırları ihlal edebilir ($\widehat{\mathrm{SOC}}<0$ veya $\widehat{\mathrm{TC}}\ne\widehat{\mathrm{SOC}}+\widehat{\mathrm{SIC}}$). | **Softplus** pozitif başlıklar + çok görevli **Kapatma Cezası** (*Closure Penalty*) | Negatif tahmin engellenir; kapatma artığı ($r_{\mathrm{TC}}$) otomatik etiket/ölçek kalite alarmına dönüşür (`ADR-004`). |
| **P6** | **Aşırı Güvenli Nokta Tahmini & Kara Kutu XAI** | Kalibrasyon dışı topraklarda dar/yanıltıcı güven; birimce kıyaslanamayan ham gradyan ve sahte etkileşim skorları. | **Conformal Prediction** (`CI90`), **Mahalanobis + k-NN OOD**, normalize IG, **RBF-HSIC** ($B \ge 4$), gerçek 2. derece etkileşim | Sonlu örneklem kapsam garantisi, alan dışı örnek reddi/uyarısı ve nedensellik iddiası taşımayan fiziksel açıklanabilirlik (`ADR-006`). |
| **P7** | **Konsantrasyon–Stok Yanılgısı & Mekânsal Ekstrapolasyon** | Kütle oranı (% C) hacim ağırlığı olmadan alan başına stoka ($Mg\;C\;ha^{-1}$) çevrilemez; dış bükey örtü dışı Kriging yanıltıcıdır. | Eksik $\rho_b$'de **Fail-Closed** stok yasağı + 1. derece Taylor yayılımı + **Ordinary Kriging** & **Delaunay** maskesi | `1.35 g/cm³` gibi sessiz varsayımlar önlenir; birleşik ölçüm kovaryansı ve ekstrapolasyon sınırı açıkça raporlanır (`DEF-005`). |

---

## 2. Verinin Kökeni ve Çok Aşamalı Dönüşüm Yaşam Döngüsü

Ham sensör sinyalleri ham haliyle makine öğrenmesi modellerine beslenmez; fiziksel birim doğrulamasından radyometrik normalizasyona, kemometrik projeksiyondan çoklu sensör füzyonuna kadar **10 aşamalı deterministik bir dönüşüm boru hattından** geçer.

```
[Ham Sensör ve Saha Girdileri]
   ├── Optik Ham Yoğunluk: I_soil(λ), I_white(λ), I_dark(λ) + gain, integration_time, sıcaklık
   ├── LIBS Atış Serisi  : Çoklu lazer atışı I_shot(λ), SNR, doygunluk bayrakları
   ├── EIS Empedans      : Frekans ekseni f_1..f_k üzerinde Z'(f), Z''(f) [Ω]
   └── Eş Değişkenler    : θ_v (% veya m³/m³), pH, T (°C), EC (dS/m or S/m), ρ_b (g/cm³), derinlik, kaba parça f_c
         │
         ▼  (Aşama 1: Kanonik Şema & Birim Dönüşümü)
[Kanonik Ölçüm Nesnesi]  ──► Tüm birimler sabit iç sisteme (%C, g/cm³, cm, nm, Ω, °C) çekilir.
         │
         ▼  (Aşama 2: Radyometrik Kalibrasyon & ISS)
[Kalibre Reflektans R(λ)] ──► Beyaz/karanlık akım düzeltmesi + ISS cihaz harmonizasyonu + dalga boyu ızgara hizalama
         │
         ▼  (Aşama 3: Sinyal QC, ArPLS & Pseudo-Voigt)
[Temizlenmiş Sinyal & QC] ──► Doygunluk/düşük SNR ayrıştırma; LIBS zemin çıkarımı ve C/H pik integrasyonu
         │
         ▼  (Aşama 4: Soğurma → SNV → SG → EPO/GLSW)
[Neme Dirençli X_clean]   ──► Doğrusallaştırılmış soğurma, saçılma bastırma, fiziksel türev ve nem alt-uzay projeksiyonu
         │
         ▼  (Aşama 5: İsimlendirilmiş Sensör Füzyonu)
[Birleşik Vektör X_fused] ──► X_clean + Vis-NIR kimyasal bant özetleri + EIS öznitelikleri + eksik veri işaretçileri
         │
         ▼  (Aşama 6-7: Model Çıkarımı, Conformal CI & OOD)
[Nokta + Dağılım + OOD]   ──► SOC/SIC/TC kestirimi, CI90 alt/üst sınırları, aleatorik/epistemik ayrım, OOD skoru
         │
         ▼  (Aşama 8-10: XAI, Karbon Stoku & Jeoistatistik)
[Bilimsel Karar Çıktıları]
   ├── Açıklanabilirlik  : Ölçek-normalize Duyarlılık, Integrated Gradients, RBF-HSIC, 2. derece etkileşim
   ├── Karbon Muhasebesi : S [Mg C ha⁻¹] ± σ_S (yalnızca geçerli ρ_b mevcutsa) + r_TC kapatma artığı
   └── Mekânsal Harita   : Ampirik/teorik variogram, Ordinary Kriging kestirim + varyans ızgarası & GeoJSON
```

Bu akışta her aşamanın ürettiği ara temsiller, **kalibrasyon zamanında öğrenilen nesnelerle** (ör. `ISSModel`, `EPO`, `StandardScaler`, `ConformalUncertaintyCalibrator`, `OutOfDomainDetector`) **çalışma zamanı (*inference*) verisi** arasında katı bir ayrım korur (`ADR-003`): çalışma zamanı yalnızca donmuş (*frozen*) parametrelerle dönüşüm uygular, asla test veya tekil saha örneği üzerinde yeniden öğrenme (*re-fit*) yapmaz.

---

## 3. Matematiksel Yöntemler: Fiziksel ve Kemometrik Dönüşümler

### 3.1 Radyometrik Kalibrasyon, Soğurma ve Cihaz Transferi (ISS)

Ham dedektör sayımlarında ışık kaynağı şiddetindeki dalgalanmaları ve sensör karanlık akımı (*dark current*) ofsetini gidermek için, aynı kazanç (*gain*) ve entegrasyon süresinde ölçülmüş beyaz referans ($I_{\text{white}}$) ve karanlık referans ($I_{\text{dark}}$) kullanılır:

$$R_{\text{raw}}(\lambda) = \frac{I_{\text{soil}}(\lambda) - I_{\text{dark}}(\lambda)}{I_{\text{white}}(\lambda) - I_{\text{dark}}(\lambda)}$$

Paydanın sıfıra çok yakın olması ($\le \varepsilon$) veya dedektörün doygunluk sınırına ($S_{\max}$) ulaşması durumunda ilgili dalga boyu kanalı maskelenir; beyaz referans yaşı eşiği aşıldığında `REFERENCE_AGED` kalite bayrağı tetiklenir. Beer–Lambert yasasına yakın doğrusal bir soğurma uzayına geçmek için pozitif reflektans değerleri logaritmik optik yoğunluğa (*absorbance*) dönüştürülür:

$$A(\lambda) = -\log_{10}\!\big(\max(R(\lambda), \varepsilon)\big)$$

Birden fazla spektrometrenin veya optik ön ucun tepki eğrilerini standart bir referans cihaza hizalamak amacıyla **Dahili Toprak Standardı** (*Internal Soil Standard — ISS*) düzeltme faktörü tanımlanır. Referans cihazdaki standart örnek spektrumu $S(\lambda_i)$ ve hedef cihazdaki ölçüm $M(\lambda_i)$ olmak üzere düzeltme faktörü:

$$\mathrm{CF}(\lambda_i) = 1 - \frac{S(\lambda_i) - M(\lambda_i)}{S(\lambda_i)} = \frac{M(\lambda_i)}{S(\lambda_i)}, \qquad X_{\text{corr}}(\lambda_i) = \frac{X_{\text{meas}}(\lambda_i)}{\mathrm{CF}(\lambda_i)}$$

Bu konvansiyon (`DEF-007`), $X_{\text{meas}} = M$ durumunda tam olarak $X_{\text{corr}} = S$ özdeşliğini garantiler.

### 3.2 LIBS Sinyal Temizleme: ArPLS ve Pseudo-Voigt Ayrıştırması

LIBS spektrumlarında yüksek enerjili lazer darbesinin oluşturduğu sürekli zemin (*continuum emission*), dar atomik emisyon piklerinin üzerine biner. Zemin çizgisini ($z$) piklere duyarlı olmayan biçimde kestirmek için **Asymmetrically Reweighted Penalized Least Squares (ArPLS)** amaç fonksiyonu çözülür:

$$\big(W + \lambda\, D_2^\top D_2\big)\, z = W\, y$$

Burada $D_2$, ikinci fark düzgünleştirme matrisi; $\lambda = 10^4$ düzgünlük cezasıdır. ArPLS'yi klasik taban çizgisi yöntemlerinden ayıran temel katkı, pik bölgelerini ($y_i > z_i$) sıfıra yakın ağırlıkla bastırırken pik altı gürültü bölgesini ($d_i = y_i - z_i < 0$) koruyan **lojistik ağırlık güncellemesidir**:

$$w_i^{(t+1)} = \frac{1}{1 + \exp\!\left(\frac{2\big(d_i - (-\mu_{d^-} + 2\sigma_{d^-})\big)}{\sigma_{d^-}}\right)}, \qquad d^- = \{d_i \mid d_i < 0\}$$

Zemin düzeltilmiş spektrum $y_{\text{corr}} = y - z$ üzerinde karbon ($C\text{ I}\;193.1, 229.7, 247.8\text{ nm}$; burada $247.8\text{ nm}$ çizgisinin matris girişimine açık olduğu dikkate alınır) ve hidrojen ($H\text{ I}\;656.3\text{ nm}$) pencerelerinde Lorentzian ($L$) ve Gaussian ($G$) genişlemelerini (basınç + Doppler genişlemesi) birleştiren **Pseudo-Voigt profili** uydurulur:

$$V(\lambda) = A \left[\eta\, \frac{\gamma^2}{(\lambda - \lambda_0)^2 + \gamma^2} + (1 - \eta)\, \exp\!\left(-\ln 2\, \frac{(\lambda - \lambda_0)^2}{\gamma^2}\right)\right] + b_0, \qquad \eta \in [0, 1]$$

Bu profil üzerinden pik genliği, yarı maksimumda tam genişlik ($\mathrm{FWHM} = 2\gamma$), analitik/sayısal tepe alanı ve uydurma kalıntısı (*fit residual*) öznitelik olarak çıkarılır.

### 3.3 Standart Normal Değişken (SNV) ve Fiziksel Dalga Boyu Savitzky–Golay Türevi

Yüzey pürüzlülüğü ve toprak parçacık iriliğinden kaynaklanan çarpımsal ve toplamsal temel kaymaları azaltmak için her spektrum satırı kendi ortalaması ve standart sapmasıyla satır bazında normalize edilir:

$$\mathrm{SNV}(x)_i = \frac{x_i - \bar{x}}{\sigma_x + \varepsilon}$$

Yavaş değişen spektral eğilimleri bastırıp örtüşen $\mathrm{O–H}$, $\mathrm{C–H}$ ve $\mathrm{CO_3^{2-}}$ omuzlarını ayırmak için **Savitzky–Golay türevi** uygulanır. Birçok uygulamada rastlanan kritik hata, bantların eşit aralıklı olduğu varsayılarak indeks üzerinden türev alınmasıdır; bu hata, düzensiz LED veya filtrelenmiş kanallarda fiziksel eğimi bozar (`DEF-002`). STDS yaklaşımında her pencere içindeki $j$ merkez dalga boyuna göre fiziksel kayma $\xi = \lambda_{\text{win}} - \lambda_j$ tanımlanır ve yerel polinom $p(\xi) = \sum_{r=0}^{d} c_r \xi^r$ Vandermonde matrisi $A_{mr} = (\xi_m / s)^r$ üzerinden sözde-ters (*Moore–Penrose pseudoinverse*) $A^+$ ile çözülerek doğrudan gerçek fiziksel dalga boyu türevi elde edilir:

$$\left.\frac{\mathrm{d}^k p(\lambda)}{\mathrm{d}\lambda^k}\right|_{\lambda = \lambda_j} = \frac{k!}{s^k}\, \mathbf{p}^+_{k,\cdot}\, \mathbf{y}_{\text{win}}$$

### 3.4 Nem Alt-Uzayının Ortogonal Bastırılması: EPO ve GLSW

Saha nemi ($\theta_v$), kil ve organik maddenin absorpsiyon bantlarını hem genlik hem şekil düzeyinde maskeler. Bu etkiyi hedef karbon sinyalini olabildiğince koruyarak kaldırmak amacıyla **External Parameter Orthogonalization (EPO)** ve ağırlıklı bir alternatif olarak **Generalized Least Squares Weighting (GLSW)** uygulanır.

Aynı toprak örneklerinin kuru ($X_{\text{dry}} \in \mathbb{R}^{n \times p}$) ve nemli ($X_{\text{wet}} \in \mathbb{R}^{n \times p}$) durumlarda **tamamen aynı ön-işleme uzayında** (ör. aynı soğurma + SNV + SG zinciri sonrasında; `DEF-001`) taranmasıyla fark matrisi $D$ ve kovaryans matrisi $C_D$ oluşturulur:

$$D = X_{\text{wet}} - X_{\text{dry}}, \qquad C_D = \frac{D^\top D}{n - 1} = V \Sigma V^\top$$

Tekil değer ayrışımından (*SVD*) toplam nem farkı varyansının belirlenen eşiğini (varsayılan `%90`, üst sınır `max_q = 6`) yakalayan ilk $q$ sağ tekil vektör $V_s = V_{[:,1:q]} \in \mathbb{R}^{p \times q}$ seçilir; parazit nem alt-uzayı $Q = V_s V_s^\top$ ve buna ortogonal projeksiyon matrisi $P_{\mathrm{EPO}}$ kurulur:

$$P_{\mathrm{EPO}} = I_p - V_s V_s^\top, \qquad X_{\text{clean}} = X\, P_{\mathrm{EPO}}$$

Teorik olarak $P_{\mathrm{EPO}}$, simetrik ($P^\top = P$) ve idempotent ($P^2 = P$) bir ortogonal projeksiyon operatörüdür; yani bir kez projekte edilmiş spektrum ikinci kez projekte edildiğinde değişmez ($V_s^\top V_s = I_q$). Sert bir alt-uzay kesimi yerine tekil değer büyüklüğüne göre sürekli bastırma tercih edildiğinde **GLSW** projeksiyonu kullanılır:

$$P_{\mathrm{GLSW}} = V\, R^{-1} V^\top, \qquad R = \sqrt{\frac{\Sigma^2}{\alpha} + I_p}$$

Eğitim sırasında EPO/GLSW'nin uygulandığı ön-işleme zinciri bir **SHA-256 uzay imzası (`space_signature`)** ile mühürlenir; çıkarım uzayı kalibrasyon uzayıyla eşleşmezse `CalibrationError` fırlatılarak sessiz uzay uyumsuzluğu önlenir (`ADR-003`).

### 3.5 Çoklu Sensör Öznitelik Kayıt Defteri ve Füzyon

Spektral vektör tek başına toprak matriksinin tamamını (özellikle tuzluluk, iyonik güç, sıcaklık ve gözenek suyu empedansı) temsil edemediğinden, fiziksel olarak anlamlı iki ek bilgi kanalı birleştirilir:

1. **Vis-NIR Kimyasal Pencere Özetleri:** Su ($1410\text{–}1450$, $1900\text{–}1950\text{ nm}$), organik fonksiyonel gruplar ($2130\text{–}2207\text{ nm}$), kil mineralleri ($2185\text{–}2225\text{ nm}$) ve karbonat bantları ($2310\text{–}2345$, $2500\text{–}2550\text{ nm}$) için aralık ortalamaları, eğimler ve tepe derinlikleri.
2. **EIS ve Fizikokimyasal Eş Değişkenler:** Çok frekanslı empedans özetleri ($Z'$, $Z''$, $|Z|$, faz açısı $\theta$), nem ($\theta_v$), $\mathrm{pH}$, sıcaklık ($T$) ve elektriksel iletkenlik ($\mathrm{EC}$). Eksik sensör durumları ile kalite kontrolünden kalan ölçümler **açık eksiklik göstergeleri** (*missingness indicators*) olarak ayrı kodlanır ve yalnızca kat-içi medyanlarla doldurulur.

---

## 4. Makine Öğrenmesi Modelleri ve Çok Görevli Hibrit Ağ Mimarisi

Tek bir regresyon ailesinin her toprak tekstürü, karbonat seviyesi ve örneklem rejimi için en iyi sonucu vereceği varsayılamaz (`ADR-005`). Bu nedenle STDS aynı grup katlarında (*`GroupKFold` / `Leave-One-Site-Out`*) beş farklı paradigma ailesini kıyaslar:

### 4.1 Klasik Kemometrik ve Yerel Referans Modeller

1. **Partial Least Squares Regression (PLSR):** Çok eşdoğrusal (*collinear*) dalga boyu bantlarını, $X$ ve $Y$ arasındaki kovaryansı maksimize eden düşük boyutlu gizil (*latent*) bileşenlere indirger; bileşen sayısı sızdırmasız iç çapraz doğrulamayla (`max_components = 10`) seçilir:
   $$X = T P^\top + E, \qquad Y = U Q^\top + F$$
2. **Kernel Ridge Regression (KRR):** RBF çekirdeği ile doğrusal olmayan spektral-karbon ilişkilerini örtük sonsuz boyutlu özellik uzayında çözer; kapalı form çözümü küçük-orta ölçekli toprak kalibrasyon setlerinde yüksek kararlılık sağlar:
   $$K_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right), \qquad \alpha^* = (K + \lambda I)^{-1} y, \qquad \hat{y}(x) = k(x, X)^\top \alpha^*$$
3. **Random Forest Regresörü:** Önyüklemeli (*bootstrap*) karar ağaçları ortalamasıyla ($\hat{y}(x) = \frac{1}{T}\sum_{t=1}^T f_t(x)$) keskin pH–karbonat eşiklerini yakalar; ancak kalibrasyon dışı (*OOD*) bölgede düz sabit kestirime doyduğu için tek başına ekstrapolasyon aracı olarak kullanılmaz.
4. **Memory-Based Learning (MBL / Local PLSR):** Yeni gelen her sorgu örneği için ön-işlenmiş eğitim kütüphanesinde en yakın $k$ komşuyu bulur ve Gaussian mesafe ağırlıklarıyla yerel bir regresyon kurar:
   $$w_i \propto \exp\!\left(-\frac{d(x, x_i)^2}{2h^2}\right)$$

### 4.2 Çift Kollu ve Fizik-Bilinçli `STSoilHybridNet` Mimarisi

Yüksek hacimli çok sensörlü veri mevcut olduğunda spektral sırayı ve heterojen fizikokimyasal eş değişkenleri birbirine karıştırmadan işlemek için **çift kollu hibrit sinir ağı** geliştirilmiştir:

- **Spektral Kodlayıcı (`spectral_encoder`):** `LayerNorm` + `Linear` + `SiLU` + `Dropout` + `_ResidualDenseBlock` ($\text{LN} \to \text{Linear} \to \text{SiLU} \to \text{Drop} \to \text{Linear} \to \text{Drop} + \text{skip}$) ile $64$ boyutlu spektral gizil vektör $h_{\text{spec}}$ üretir.
- **Eş Değişken Kodlayıcısı (`covariate_encoder`):** pH, nem, EIS, sıcaklık gibi farklı ölçekteki değişkenleri ayrı `LayerNorm` ve artıksal yoğun bloktan geçirerek $32$ boyutlu $h_{\text{cov}}$ temsiline dönüştürür.
- **Ortak Füzyon Omurgası (`fusion_backbone`):** $[h_{\text{spec}} \,\|\, h_{\text{cov}}]$ birleşimini $64$ boyutlu ortak temsile eşler.
- **Fizik-Kısıtlı Çoklu Başlıklar:** $\mathrm{SOC}$ ve $\mathrm{SIC}$ için ortalama ($\mu$) ve aleatorik varyans ($\sigma^2$) başlıkları, toprak sınıfı logits başlığı ($4$ geçici sınıf) ve opsiyonel yardımcı $\mathrm{TC}$ başlığı. Negatif karbon ve sıfır varyans tekilliğini önlemek için **Softplus parametrizasyonu** uygulanır:
  $$\hat{\mu}_c = \operatorname{softplus}(z_{\mu,c}) > 0, \qquad \hat{\sigma}_c^2 = \operatorname{softplus}(z_{\sigma,c}) + \varepsilon_{\min} \quad (\varepsilon_{\min}=10^{-4})$$

### 4.3 Çok Görevli Heteroskedastik Kayıp Fonksiyonu ve Kapatma Kısıtı

Model eğitimi, ölçüm gürültüsünün örneğe bağlı değiştiğini (*heteroskedastic noise*) modelleyen **Gaussian Negatif Log-Olabilirlik (NLL)** kaybı, aşırı uç değerlere karşı **Huber kaybı**, sınıflandırma için **Çapraz Entropi** ve kimyasal **kütle kapanış cezasını** tek amaç fonksiyonunda birleştirir:

$$\mathcal{L}_{\mathrm{NLL}}(y, \mu, \sigma^2) = \frac{1}{2}\left[\ln(\sigma^2) + \frac{(y - \mu)^2}{\sigma^2}\right]$$

$$\mathcal{L}_{\text{total}} = \lambda_{\mathrm{SOC}}\mathcal{L}_{\mathrm{SOC}} + \lambda_{\mathrm{SIC}}\mathcal{L}_{\mathrm{SIC}} + \lambda_{\mathrm{TC}}\mathcal{L}_{\mathrm{TC}} + \lambda_{\text{class}}\mathcal{L}_{\mathrm{CE}} + \lambda_{\text{close}}\big(\widehat{\mathrm{TC}}_{\text{aux}} - (\widehat{\mathrm{SOC}} + \widehat{\mathrm{SIC}})\big)^2 + \lambda_{\text{huber}}\mathcal{L}_{\text{Huber}}$$

Burada varsayılan fiziksel ağırlıklar ($\lambda_{\mathrm{SOC}}=1.0, \lambda_{\mathrm{SIC}}=1.0, \lambda_{\mathrm{TC}}=0.5, \lambda_{\text{class}}=0.25, \lambda_{\text{close}}=2.0, \lambda_{\text{huber}}=0.25$) özellikle kütle korunumuna yüksek öncelik vererek modelin $\mathrm{SOC}+\mathrm{SIC}$ toplamından sapmasını engeller.

---

## 5. Belirsizlik Kalibrasyonu, Alan Dışı Tespiti (OOD) ve Bilimsel XAI

### 5.1 Dağılım-Bağımsız Conformal Prediction ve Belirsizlik Ayrıştırması

Tek başına sinir ağı varyans başlığı ($\sqrt{\hat{\sigma}^2}$), saha verisinde sistematik olarak aşırı iyimser aralıklar üretebilir (`Risk 5`). Bu nedenle STDS, **iki bileşenli belirsizliği** (*aleatoric* = veri içi/sensör gürültüsü; *epistemic* = önyükleme/katman topluluğu standart sapması) raporlar ve grup dışı kalibrasyon artıkları ($r_i$) üzerinden **Normalize Split Conformal Prediction** uygular:

$$r_i = \frac{|y_i - \hat{y}_i|}{\max(\hat{\sigma}_i,\, s_{\min})}, \qquad q_{1-\alpha} = \operatorname{Quantile}\!\left(\{r_1, \dots, r_n\},\; \frac{\lceil (n+1)(1-\alpha)\rceil}{n}\right)$$

Yeni bir $x^*$ örneği için elde edilen $[\max(0, \hat{y}(x^*) - q_{1-\alpha}\hat{\sigma}(x^*)),\; \hat{y}(x^*) + q_{1-\alpha}\hat{\sigma}(x^*)]$ aralığı, sonlu örneklem altında en az $1 - \alpha$ (`CI90` için `%90`) marjinal kapsama garantisi sağlar.

### 5.2 Alan Dışı (Out-of-Domain — OOD) Kapısı

Model eğitim uzayının dışındaki bir toprak tipinde sessizce yüksek güvenli çıktı üretmesini engellemek için üç eksenli OOD detektörü çalışır:
1. **Spektral/Özellik Uzayı Mahalanobis Uzaklığı:** Eğitim kovaryansının Cholesky çarpanı ($\Sigma_{\text{reg}} = L L^\top$) üzerinden sayısal kararlılıkla çözülür:
   $$d_M(x) = \sqrt{(x - \mu_{\text{train}})^\top \Sigma_{\text{reg}}^{-1} (x - \mu_{\text{train}})}$$
2. **$k$-NN Yerel Yoğunluk Uzaklığı:** En yakın $k$ kalibrasyon komşusuna ortalama Öklid uzaklığı.
3. **Fizikokimyasal Zarf Sınırları (`DomainEnvelope`):** Eğitimdeki $\mathrm{pH}$, hacimsel nem, sıcaklık ve $\mathrm{EC}$ kantil sınırlarının aşılıp aşılmadığı. Bu eşikler aşıldığında örnek `OOD_HIGH_RISK` bayrağıyla işaretlenir.

### 5.3 Bilimsel Açıklanabilirlik (XAI) İlkeleri (`ADR-006`)

Kaynak taslak kodlarda tespit edilen iki ciddi bilimsel hata (birimce uyumsuz ham gradyan kıyası ve maksimum gradyanların çarpımını etkileşim sanmak) aşağıdaki matematiksel çerçeve ile düzeltilmiştir:

1. **Ölçek-Normalize Yerel Duyarlılık ve Integrated Gradients (IG):** Birim farkını ortadan kaldırmak için kısmi türev her özelliğin standart sapması $\sigma_i$ ile ölçeklenir; sıfır veya referans temel çizgisi $x^{(0)}$ ile giriş $x$ arasındaki Riemann yolu entegre edilir:
   $$S_i(x) = \sigma_i \left|\frac{\partial f(x)}{\partial x_i}\right|, \qquad \mathrm{IG}_i(x) = (x_i - x_i^{(0)}) \int_{0}^{1} \frac{\partial f\big(x^{(0)} + \alpha(x - x^{(0)})\big)}{\partial x_i}\,\mathrm{d}\alpha$$
2. **İstatistiksel Bağımlılık için Merkezlenmiş RBF-HSIC:** Doğrusal olmayan bağımlılık ölçüsü olan *Hilbert–Schmidt Independence Criterion* medyan sezgisel bant genişlikli RBF Gram matrisleri ($K, L$) ve merkezleme matrisi $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ ile hesaplanır:
   $$\mathrm{HSIC}(K, L) = \frac{\operatorname{tr}(K H L H)}{(n - 1)^2}, \qquad \mathrm{nHSIC}(K, L) = \frac{\mathrm{HSIC}(K, L)}{\sqrt{\mathrm{HSIC}(K, K)\,\mathrm{HSIC}(L, L)}} \in [0, 1]$$
   Yetersiz örneklemde ($n < 4$) eşit önem skoru uydurulmaz; sistem `insufficient_sample_count` durumuna geçer.
3. **Gerçek İkinci Derece Etkileşim & Birim Ayrımı:** Değişken çiftleri arasındaki etkileşim $\frac{\partial^2 f(x)}{\partial x_i \partial x_j}$ karışık ikinci kısmi türevi (sonlu farklar Hessian yaklaşımı) ile ölçülür; yüzde/$\mathrm{pH}$/empedans ve spektral kanallar birim tutarlılığı korunarak ayrı bileşen bloklarında sıralanır ve **hiçbir XAI skoru nedensellik kanıtı olarak sunulmaz**.

---

## 6. Karbon Stoku Yayılımı ve Jeoistatistiksel Haritalama

### 6.1 Fiziksel Stok Hesabı ve 1. Derece Kovaryans Yayılımı

Kütle konsantrasyonu ($C_{\text{pct}}, \%$), hacim ağırlığı ($\rho_b, \text{g cm}^{-3}$), katman kalınlığı ($d, \text{cm}$) ve taş/çakıl hacim oranı ($f_c \in [0, 0.99]$) arasındaki birim analizi yapıldığında ($1\% \text{ C} \cdot 1\text{ g cm}^{-3} \cdot 1\text{ cm}$ doğrudan $10^3\text{ kg C ha}^{-1} = 1\text{ Mg C ha}^{-1}$ değerine eşittir), tek katman ve çok katmanlı profil stoku şu biçimi alır:

$$S_{\text{layer}} = C_{\text{pct}} \cdot \rho_b \cdot d \cdot (1 - f_c) \;\;[\mathrm{Mg\;C\;ha^{-1}}], \qquad S_{\text{profile}} = \sum_{j} C_{\text{pct},j} \cdot \rho_{b,j} \cdot \Delta z_j \cdot (1 - f_{c,j})$$

`bulk_density` ($\rho_b$) eksik olduğunda `1.35 g/cm³` gibi gizli bir varsayım kesinlikle atanmaz (`DEF-005`); stok `None` döner. Tüm parametreler mevcutsa toplam stok varyansı $\theta = [C_{\text{pct}}, \rho_b, d, f_c]^\top$ parametrelerinin kovaryans matrisi üzerinden birinci derece Taylor açılımıyla yayılır:

$$\sigma_S^2 \approx \nabla_\theta S^\top\, \operatorname{Cov}(\theta)\, \nabla_\theta S, \qquad \nabla_\theta S = \begin{bmatrix} \rho_b d (1-f_c) \\ C_{\text{pct}} d (1-f_c) \\ C_{\text{pct}} \rho_b (1-f_c) \\ -C_{\text{pct}} \rho_b d \end{bmatrix}$$

### 6.2 Variogram Modelleme ve Ordinary Kriging

Saha noktalarında elde edilen karbon kestirimleri ($Z(s_i)$), metrik projeksiyon koordinatlarında ampirik **Matheron yarı-variogramı** ($\hat{\gamma}(h) = \frac{1}{2|N(h)|}\sum_{(i,j)\in N(h)}(Z(s_i)-Z(s_j))^2$) ile modellenir; en küçük kareler uyumuyla küresel (*spherical*), üstel (*exponential*) veya Gaussian teorik variogram ($c_0$ *nugget*, $c$ *partial sill*, $a$ *range*) seçilir.

Hedef $s_0$ konumunda yansız ($\sum_{i=1}^N \lambda_i = 1$) ve minimum tahmin varyanslı **Ordinary Kriging** ağırlıkları $(N+1) \times (N+1)$ doğrusal sisteminden çözülür:

$$\begin{bmatrix} C & \mathbf{1} \\ \mathbf{1}^\top & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ \mu \end{bmatrix} = \begin{bmatrix} \mathbf{c}(s_0) \\ 1 \end{bmatrix}, \qquad C_{ij} = \mathrm{sill} - \gamma(\|s_i - s_j\|)$$

Örnek noktalarının oluşturduğu **Delaunay dış bükey örtüsü** (*convex hull*) dışında kalan ızgara hücreleri `is_extrapolated_mask = True` bayrağıyla işaretlenir; böylece harita tüketicisi gerçek enterpolasyon bölgesi ile güvenilmez dış ekstrapolasyonu net biçimde ayırt eder.

---

## 7. Değerlendirme Protokolü, Bilimsel Beklentiler ve Kanıt Sınırları

### 7.1 Kabul Metrikleri ve Alt Grup Doğrulaması

Model seçimi ve sürüm kabulü tek bir küresel $R^2$ skoruna bırakılmaz; aşağıdaki altı metrik **saha, nem sınıfı, pH aralığı ve karbonat alt gruplarına** bölünerek birlikte değerlendirilir:

$$\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|, \qquad \mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}, \qquad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

$$\mathrm{RPIQ} = \frac{Q_{0.75}(y) - Q_{0.25}(y)}{\mathrm{RMSE}}, \qquad \mathrm{Bias} = \overline{\hat{y} - y}, \qquad \mathrm{LoA}_{95\%} = \mathrm{Bias} \pm 1.96\,\mathrm{SD}(\hat{y} - y)$$

### 7.2 Dış Literatür vs. MVP `v0.1.0` Sentetik Doğrulama Ayrımı (`Risk 9`)

Akademik dürüstlük ve sistem izlenebilirliği gereği iki kanıt düzeyi birbirinden kesin olarak ayrılır:
1. **Dış Literatür / Mimari Referansları (STDS Başarısı Değildir):** Mimari planlama belgesinde incelenen harici saha/cihaz çalışmaları (ör. Eylül 2026 Cense Analytics vaka incelemeleri ve 5/18 bant LED, 1300–2550 nm MEMS FT-NIR donanım önerileri), yalnızca yöntem seçimi ve donanım-bağımsız şema tasarımına ilham veren literatür girdileridir.
2. **STDS MVP `v0.1.0` Doğrulanmış Sentetik Boru Hattı Bulguları:** Kök modüllerin deterministik çalışması ve matematiksel kapanışları `SyntheticGeneratorV2` (`seed=42`, fiziksel su/organik/kil/karbonat absorpsiyon bantları ve kovaryat etkileşimleriyle üretilmiş sızıntısız saha-ayrık veri seti) üzerinde sınanmıştır:
   - Tüm doğrulama katmanında katı saha-dışı (*disjoint-site holdout*) protokolüyle **sıfır sızıntı**, sıfır `RuntimeWarning`, sıfır deterministik tekrarlama sapması ($\max|\Delta\hat{y}| = 0.0$) ve `.safetensors` + SHA-256 bütünlüğü kanıtlanmıştır.
   - Gerçek toprak örneklerinden alınan bağımsız kuru yakma / kalsimetre laboratuvar referansları sisteme yüklendiğinde, bu bilimsel yöntemlerin bölgesel ve sensör-özgü **gerçek saha kalibrasyon metrikleri** üretilecek ve sürümün `Model Card` belgesine yazılacaktır.

### 7.3 Gelecekteki Gerçek Saha Verisinden Beklenen Bilimsel Kazanımlar

STDS boru hattının gerçek çok sahalı laboratuvar/arazi kampanyalarına uygulanmasıyla şu bilimsel beklentiler hedeflenmektedir:
- **EPO/GLSW entegrasyonu** sayesinde taze/nemli saha taramaları ile kuru laboratuvar kütüphaneleri arasındaki sistematik nem bias'ının anlamlı biçimde düşürülmesi (*yakın nem bantları arasında sinyal kaybı kontrol edilerek*).
- **Çift çoklu hedef ve kapatma cezası** ile kalkerli tarım topraklarında $\mathrm{SOC}$'nin $\mathrm{SIC}$ tarafından maskelenmesinin azaltılması.
- **Conformal + Mahalanobis OOD kapısı** ile tarla sınırları dışındaki yeni toprak tekstürlerinde yanlış yüksek güven verilmesinin önüne geçilmesi ve örneklem önceliklendirme (*active sampling*) maliyetinin düşürülmesi.

---

## 8. Bilimsel Kavram $\leftrightarrow$ Yazılım Modülü Eşleştirmesi (~%10)

Aşağıdaki özet harita, yukarıdaki her matematiksel denklemin STDS kod tabanındaki doğrulanmış karşılığını gösterir:

| Bilimsel İşlem / Matematiksel Denklem | Sorumlu Paket / Dosya | Temel Fonksiyon / Sınıf Sembolü |
| --- | --- | --- |
| Kanonik Şema, Birim Dönüşümü & Profil | `st_ds/domain/` | `Measurement`, `Spectrum`, `UnitRegistry`, `InstrumentProfile`, `QualityFlag` |
| Sızıntısız Grup Bölmesi & Sentetik Fizik | `st_ds/io/`, `st_ds/training/` | `GroupedSplitBuilder`, `SoilDataset`, `SyntheticGeneratorV2` |
| Radyometrik Kalibrasyon & ISS ($R_{\text{raw}}, \mathrm{CF}$) | `st_ds/calibration/` | `calibrate_reflectance`, `fit_iss`, `apply_iss`, `align_wavelengths` |
| LIBS QC, ArPLS Zemin & Pseudo-Voigt | `st_ds/signal/` | `filter_libs_shots`, `estimate_arpls_baseline`, `fit_pseudovoigt_peaks` |
| Soğurma, SNV, Fiziksel SG & EPO/GLSW | `st_ds/chemometrics/` | `to_absorbance`, `snv_transform`, `savgol_derivative`, `EPO`, `GLSW`, `ChemometricPipeline` |
| Çoklu Sensör Özellik Kayıt Defteri | `st_ds/features/` | `FeatureRegistry`, `FeatureFusionPipeline`, `extract_eis_features` |
| Baseline Modeller & Hibrit Ağ ($\mathcal{L}_{\text{total}}$) | `st_ds/models/` | `PLSRRegressor`, `KernelRidgeRegressor`, `RandomForestSoilRegressor`, `LocalPLSRRegressor`, `STSoilHybridNetRegressor`, `STSoilMultiTaskLoss` |
| Conformal Aralık & Mahalanobis/k-NN OOD | `st_ds/uncertainty/` | `ConformalUncertaintyCalibrator`, `OutOfDomainDetector`, `GroupedBootstrapEnsemble` |
| Bilimsel XAI (IG, RBF-HSIC, 2. Türev) | `st_ds/xai/` | `SoilXAIEngine`, `local_sensitivity`, `integrated_gradients`, `compute_hsic` |
| Karbon Stoku ($S, \sigma_S^2$) & Kapatma | `st_ds/carbon/` | `partition_carbon`, `compute_layer_stock`, `compute_profile_stock` |
| Variogram, Ordinary Kriging & GeoJSON | `st_ds/spatial/` | `fit_variogram`, `ordinary_kriging`, `kriging_to_geojson` |
| Sıfır-Pickle Paket, Dış API & 26+ Görsel | `st_ds/artifact/`, `st_ds/api/`, `st_ds/visualization/` | `save_artifact`, `STSoilAnalyzer`, `generate_full_visualization_suite` |

---

## İlgili Belgeler

- **Kurulum ve Adım Adım Yol Haritası:** [FIRST_TIME_ROADMAP_GUIDE.md](FIRST_TIME_ROADMAP_GUIDE.md)
- **Sistem Mimarisi, Python API & CLI Başvurusu:** [SYSTEM_AND_USAGE_GUIDE.md](SYSTEM_AND_USAGE_GUIDE.md)
- **İlk Sentetik Deney Rehberi:** [SYNTHETIC_TESTING_GUIDE.md](SYNTHETIC_TESTING_GUIDE.md)
- **Sürüm ve Geliştirme Notları (`v0.1.0`):** [RELEASE_NOTES.md](RELEASE_NOTES.md)
- **Mimari Karar Kayıtları (`ADR-001`..`007`):** [faz0/STDS_ARCHITECTURE_DECISIONS.md](faz0/STDS_ARCHITECTURE_DECISIONS.md)
