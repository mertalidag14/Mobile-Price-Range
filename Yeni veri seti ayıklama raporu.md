# `gercek_telefonlar.csv` Veri Seti: Eksik, Hatalı ve Aykırı Değer Kontrol Raporu (v2)

Bu rapor, gerçek telefon veri seti (`gercek_telefonlar.csv`, [Kaggle – Real World Smartphone's Dataset](https://www.kaggle.com/datasets/abhijitdahatonde/real-world-smartphones-dataset)) üzerinde yapılan eksik değer taraması, tekrar kontrolü, mantıksal tutarlılık kontrolleri ve aykırı değer analizinin sonuçlarını özetliyor. v1'deki (`train.csv`) hata ayıklama raporuyla aynı adımları izliyor. Tüm sayılar `veri_inceleme_eda_gercek.ipynb` notebook'undan geliyor.

## 1. Veri Setinin Genel Yapısı

Veri seti **980 satır** (telefon) ve **22 sütundan** oluşuyor: marka, model, fiyat (₹), kullanıcı puanı ve 18 donanım özelliği. Veride **46 marka** var. En çok telefonu olan markalar Xiaomi (134), Samsung (132), Vivo (111), Realme (97) ve OPPO (88).

v1'den farklı olarak hedef değişken hazır gelmiyor. Fiyat segmenti (`price_range`) gerçek fiyattan çeyreklere bölünerek oluşturuluyor.

## 2. Eksik Değer (NaN) Kontrolü

v1'deki sentetik veride hiç eksik değer yok. Gerçek veride 8 sütunda eksik değer bulunuyor:

| Sütun | Açıklama | Eksik Sayısı | Oran | Değerlendirme |
| ----- | ----- | ----- | ----- | ----- |
| **`fast_charging`** | Hızlı şarj gücü (W) | **211** | %21,5 | 143'ünde telefonun hızlı şarjı zaten yok (`fast_charging_available = 0`), yani eksik değil, **0 W** anlamına geliyor. Kalan 68 telefonda hızlı şarj var ama watt bilgisi girilmemiş. |
| **`avg_rating`** | Kullanıcı puanı | **101** | %10,3 | Modelde kullanılmıyor, sadece sitedeki sıralamada kullanılıyor. Doldurulmuyor. |
| **`processor_speed`** | İşlemci hızı (GHz) | **42** | %4,3 | Medyanla dolduruluyor. |
| **`processor_brand`** | İşlemci markası | **20** | %2,0 | Modelde kullanılmıyor. |
| **`os`** | İşletim sistemi | **14** | %1,4 | Modelde kullanılmıyor. |
| **`battery_capacity`** | Batarya (mAh) | **11** | %1,1 | Medyanla dolduruluyor. |
| **`num_cores`** | Çekirdek sayısı | **6** | %0,6 | Medyanla dolduruluyor. |
| **`primary_camera_front`** | Ön kamera (MP) | **5** | %0,5 | Medyanla dolduruluyor. |

Temizlikten sonra modelde kullanılan sütunlarda **hiç eksik değer kalmıyor**.

## 3. Tekrarlayan Kayıt (Duplicate) Kontrolü

* Birebir tekrar eden satır: **0**
* Aynı model adıyla tekrar eden kayıt: **0**

Aynı telefonun farklı RAM/depolama varyantları ayrı satırlarda duruyor ve model adlarında bu fark yazıyor (ör. `Samsung Galaxy A14 5G (6GB RAM + 128GB)`). Bunlar tekrar değil, gerçekten farklı ürünler.

## 4. Mantıksal Hata ve Anormal Değer Tespiti

| Kontrol | Kayıt Sayısı | Değerlendirme |
| ----- | ----- | ----- |
| Hızlı şarj yok ama watt bilgisi var | **0** | Tutarlı. |
| RAM, depolamadan büyük | **0** | Tutarlı. |
| Ön kamera 0 MP | **1** (Itel A24 Pro) | Ön kamerası çok düşük çözünürlüklü giriş seviyesi bir telefon, gerçek değer olarak kalıyor. |
| Ekran genişliği (piksel) yüksekliğinden büyük | **39** | **Mantıksal hata (yer değiştirmiş değerler):** Telefon ekranında yükseklik her zaman genişlikten büyük olmalı. Bu 39 telefonda iki değer ters girilmiş, ör. Infinix Note 12 için genişlik 2400, yükseklik 1080 yazıyor (doğrusu 1080 × 2400). Farklı markalarda görülüyor, en çok itel (9), OPPO (5), Xiaomi (5), Infinix (4) ve Samsung (4). |
| Ekran 5 inçten küçük | **8** | Eski ve kompakt modeller, gerçek değer. |
| Ekran 7 inçten büyük | **18** | Katlanabilir telefonlar ve tablet benzeri cihazlar, gerçek değer. |
| Batarya 9000 mAh üzeri | **3** | Doogee V Max (22.000 mAh), Oukitel WP19 (21.000 mAh), Oukitel WP21 (9.800 mAh). Dayanıklı (rugged) telefonlar, hata değil. |
| Fiyat 165.000 ₹ üzeri | **10** | Vertu Signature Touch (650.000 ₹), Xiaomi Redmi K20 Pro Signature Edition (480.000 ₹), Porsche Design Huawei'ler gibi lüks modeller. |

v1'deki `px_height = 0` veya `sc_w = 0` türünden sıfır değerlere **rastlanmıyor**. Tek gerçek mantıksal hata, 39 telefonda ekran çözünürlüğünün genişlik ve yüksekliğinin yer değiştirmiş olması.

## 5. Aykırı Değer (Outlier) Tespiti (IQR Yöntemi)

Alt sınır $Q_1 - 1.5 \times IQR$, üst sınır $Q_3 + 1.5 \times IQR$ olarak alınıyor:

| Sütun | Alt Sınır | Üst Sınır | Aykırı Sayısı | Min – Maks | Değerlendirme |
| ----- | ----- | ----- | ----- | ----- | ----- |
| **`price`** (₹) | -20.740 | 69.230 | **107** | 3.499 – 650.000 | Fiyat sağa çarpık. Sadece dağılımı bozan en pahalı %1 (10 telefon) çıkarılıyor, diğerleri gerçek amiral gemileri. |
| **`internal_memory`** (GB) | -32 | 224 | **184** | 8 – 1024 | 256 GB, 512 GB ve 1 TB'lık üst modeller. Gerçek değer. |
| **`battery_capacity`** (mAh) | 3.750 | 5.750 | **129** | 1.821 – 22.000 | Çoğu telefon 4500–5000 mAh'ta toplandığı için sınırlar dar. Küçük iPhone'lar ve 6000 mAh'lık modeller aykırı görünüyor ama gerçek. |
| **`screen_size`** (inç) | 6,25 | 6,92 | **113** | 3,54 – 8,03 | Kompakt ve katlanabilir telefonlar. Gerçek değer. |
| **`primary_camera_front`** (MP) | -4 | 28 | **193** | 0 – 60 | 32–60 MP'lik selfie odaklı telefonlar. Gerçek değer. |
| **`fast_charging`** (W) | -54 | 138 | **13** | 10 – 240 | 150–240 W'lık çok hızlı şarjlı modeller (ör. Realme GT Neo 5). Gerçek değer. |
| **`ram_capacity`** (GB) | -2 | 14 | **11** | 1 – 18 | 16–18 GB RAM'li oyun telefonları. Gerçek değer. |
| **`primary_camera_rear`** (MP) | -36 | 124 | **18** | 2 – 200 | 200 MP kameralı modeller. Gerçek değer. |
| **`resolution_height`** (piksel) | 418 | 3.602 | **3** | 480 – 3.840 | 4K ekranlı Sony modelleri. Gerçek değer. |

## 6. Temel İstatistiksel Özet

* **Fiyat:** 3.499 ₹ ile 650.000 ₹ arasında, ortanca 19.995 ₹.
* **RAM:** 1 GB ile 18 GB arasında, ortanca 6 GB.
* **Depolama:** 8 GB ile 1 TB arasında, ortanca 128 GB.
* **Batarya:** 1.821 mAh ile 22.000 mAh arasında, ortanca 5.000 mAh.
* **İşlemci hızı:** 1,2 GHz ile 3,22 GHz arasında, ortanca 2,3 GHz.
* **Fiyat segmentleri (temizlikten sonra, 970 telefon):**

| Segment | Fiyat Aralığı (₹) | Telefon Sayısı |
| ----- | ----- | ----- |
| 0 (Düşük) | 3.499 – 12.999 | 249 |
| 1 (Orta) | 12.999 – 19.990 | 241 |
| 2 (Yüksek) | 19.990 – 34.999 | 243 |
| 3 (Çok yüksek) | 34.999 – 163.980 | 237 |

Segmentler çeyreklerle oluşturulduğu için veri seti **dengeli** kalıyor.

* **Fiyatla en güçlü korelasyonlar:** işlemci hızı (0,69), depolama (0,65), RAM (0,55), ekran çözünürlüğü (0,49), 5G (0,46). Hafıza kartı desteğinin korelasyonu negatif (-0,57): pahalı telefonların çoğunda hafıza kartı yuvası yok.

## 7. Sonuç ve Uygulanan Temizlik

1. **Lüks telefonlar:** Fiyatı 165.034 ₹'yi (%99'luk dilim) geçen **10 telefon** çıkarılıyor. Analizde **970 telefon** kalıyor.
2. **Hızlı şarj:** Watt bilgisi eksik olanlar **0 W** kabul ediliyor. Bunların çoğunda zaten hızlı şarj yok.
3. **Eksik sayısal değerler:** İşlemci hızı, çekirdek sayısı, batarya ve ön kamera **medyanla** dolduruluyor.
4. **Yer değiştirmiş çözünürlük (öneri):** Genişliği yüksekliğinden büyük görünen 39 telefonda iki değerin yer değiştirilmesi gerekiyor. Şu anki analizde bu düzeltme **uygulanmıyor**. Modelde `resolution_height` kullanıldığı için bu telefonların ekranı olduğundan düşük çözünürlüklü görünüyor. Düzeltme sonraki adım olarak öneriliyor.
5. **Aykırı değerler:** IQR'ın işaretlediği değerlerin neredeyse hepsi gerçek telefonlar (1 TB depolama, 200 MP kamera, katlanabilir ekran). Bu yüzden **silinmiyor**. KNN gibi mesafe tabanlı modeller için **StandardScaler / MinMaxScaler** uygulanıyor, ağaç tabanlı modeller ham değerlerle çalışıyor.

**v1 ile fark:** Sentetik veride hiç eksik değer yok ama `px_height = 0` gibi imkânsız değerler var. Gerçek veride ise sıfır değer yok, ama eksik bilgiler, 39 telefonda ters girilmiş çözünürlük ve gerçek uç değerler var. Bu da gerçek hayattaki veriye daha yakın bir tablo veriyor.
