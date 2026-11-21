# `train.csv` Veri Seti: Eksik, Hatalı ve Aykırı Değer Kontrol Raporu

Bu rapor, **Mobile Price Classification** (`train.csv`) veri seti üzerinde gerçekleştirilen ön veri incelemesi, eksik değer (NaN) taraması, mantıksal tutarlılık kontrolleri ve aykırı değer (outlier) analizinin sonuçlarını özetlemektedir.

## 1. Veri Seti Genel Yapısı ve Eksik Değer (NaN) Kontrolü

Veri seti toplam **2,000 satır** (gözlem) ve **21 sütundan** (20 bağımsız öznitelik + 1 hedef değişken `price_range`) oluşmaktadır.

* **Eksik Veri Durumu:** Yapılan `isnull().sum()` ve `info()` kontrolleri sonucunda veri setindeki hiçbir sütunda boş (`NaN` veya `Null`) değer **bulunmamıştır**. Tüm sütunlar %100 doluluk oranına sahiptir.

* **Tekrarlayan Kayıt (Duplicate) Durumu:** Veri setinde birebir tekrar eden satır tespit edilmemiştir.

## 2. Mantıksal Hatalı ve Anormal Değer Tespiti

Veri setinde teknik olarak `NaN` değeri bulunmamasına rağmen, alan bilgisi (domain knowledge) çerçevesinde incelendiğinde bazı sayısal sütunlarda **0 (sıfır)** olarak kodlanmış mantıksal hatalar veya eksik bilgi göstergeleri tespit edilmiştir:

| Sütun Adı | Açıklama | 0 Değerine Sahip Kayıt Sayısı | Değerlendirme / Yorum | 
| ----- | ----- | ----- | ----- | 
| **`px_height`** | Piksel Yüksekliği | **2** | **Mantıksal Hata:** Fiziksel bir ekranın piksel yüksekliğinin 0 olması imkansızdır. Hatalı veri girişi olarak değerlendirilmelidir. | 
| **`sc_w`** | Ekran Genişliği (cm) | **180** | **Eksik Bilgi Göstergesi:** Ekran genişliğinin 0 cm olması fiziksel olarak mümkün değildir; bilinmeyen değerlerin 0 ile doldurulduğunu gösterir. | 
| **`fc`** | Ön Kamera (Megapiksel) | **474** | **Donanım Özelliği:** Ön kamerası bulunmayan telefon modelleri için 0 değeri girilmiştir; normal kabul edilebilir. | 
| **`pc`** | Ana Kamera (Megapiksel) | **101** | **Donanım Özelliği:** Arka kamerası bulunmayan cihazları ifade etmektedir; yapısal bir hata değildir. | 

## 3. Aykırı Değer (Outlier) Tespiti (IQR / Kutu Grafiği Analizi)

Sayısal değişkenlerde uç değerlerin varlığını belirlemek amacıyla **Çeyrekler Arası Açıklık (IQR - Interquartile Range)** yöntemi ($Q_1 - 1.5 \times IQR$ alt sınırı ve $Q_3 + 1.5 \times IQR$ üst sınırı) uygulanmıştır:

| Sütun Adı | Alt Sınır ($Q_1 - 1.5 \times IQR$) | Üst Sınır ($Q_3 + 1.5 \times IQR$) | Tespit Edilen Aykırı Değer Sayısı | Değerlendirme / Yorum | 
| ----- | ----- | ----- | ----- | ----- | 
| **`fc` (Ön Kamera)** | $-6.0\text{ MP}$ | $16.0\text{ MP}$ | **18 adet** ($17 - 19\text{ MP}$) | Üst sınıra yakın çok az sayıda yüksek megapikselli model mevcuttur; donanımsal olarak gerçektir. | 
| **`px_height` (Piksel Yüksekliği)** | $-523.1\text{ px}$ | $1758.9\text{ px}$ | **2 adet** ($1907, 1960\text{ px}$) | Yüksek çözünürlüklü ekranlara sahip üst segment cihazları temsil etmektedir; ekstrem bir sapma değildir. | 
| **`three_g` (3G Desteği)** | $1.0$ | $1.0$ | **477 adet** (`0` değeri) | İkili (binary) bir değişken olduğu için istatistiksel olarak aykırı görünse de 3G desteklemeyen cihazları ifade eder; müdahale edilmemelidir. | 
| **Diğer Tüm Sürekli Değişkenler** (`ram`, `battery_power`, `mobile_wt`, vb.) | Değişken bazlı normal sınırlar | Değişken bazlı normal sınırlar | **0 adet** | Tüm gözlemler IQR sınırları içerisinde, oldukça düzgün ve homojen bir dağılım sergilemektedir. | 

## 4. Temel İstatistiksel Özet (`describe` Analizi)

Sayısal değişkenlerin minimum ve maksimum sınırları incelendiğinde, yukarıda belirtilen sıfır değerleri dışında uçuk veya negatif bir anomaliye rastlanmamıştır:

* **Batarya Gücü (`battery_power`):** $501\text{ mAh}$ ile $1998\text{ mAh}$ aralığındadır.

* **RAM Kapasitesi (`ram`):** $256\text{ MB}$ ile $3998\text{ MB}$ aralığındadır.

* **Cihaz Ağırlığı (`mobile_wt`):** $80\text{ gr}$ ile $200\text{ gr}$ aralığındadır.

* **Hedef Değişken (`price_range`):** `0` (Düşük Maliyet), `1` (Orta Maliyet), `2` (Yüksek Maliyet) ve `3` (Çok Yüksek Maliyet) olmak üzere 4 sınıftan oluşmaktadır. Her sınıfta **500'er adet** kayıt bulunmakta olup veri seti tamamen **dengeli (balanced)** bir dağılıma sahiptir.

## 5. Sonuç ve Ön İşleme Önerileri

1. **Veri Temizliği:** Veri seti halihazırda `NaN` değeri içermediği için herhangi bir satır silme veya eksik veri doldurma (imputation) zorunluluğu yoktur.

2. **Anomali Müdahalesi:**

   * `px_height == 0` olan **2 adet** kayıt, hatalı ölçüm olduğu için veri setinden çıkarılabilir veya medyan/ortalama değer ile güncellenebilir.

   * `sc_w == 0` olan **180 adet** kayıt için ise `sc_h` (ekran yüksekliği) ile orantılı bir tahminleme yapılabilir ya da değerler `NaN` olarak işaretlenip ortalama/medyan ile doldurulabilir.

3. **Aykırı Değer Stratejisi:** IQR yöntemiyle `fc` ve `px_height` sütunlarında tespit edilen az sayıdaki yüksek değerler gerçek cihaz özelliklerini yansıttığı için **silinmemelidir**. Bunun yerine, mesafe tabanlı algoritmalar (SVM, KNN vb.) için **ölçeklendirme (StandardScaler veya RobustScaler)** uygulanması yeterlidir.