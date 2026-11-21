# Mobile Price Range Classification (Mobil Cihaz Fiyat Segmenti Tahmini)

Bu proje, akıllı telefonların ve mobil cihazların donanım mimarilerini, performans bileşenlerini ve teknik spesifikasyonlarını (RAM kapasitesi, batarya ömrü, ekran çözünürlüğü, işlemci çekirdek sayısı, kamera özellikleri ve bağlantı teknolojileri vb.) derinlemesine analiz ederek cihazları en uygun fiyat segmentine yerleştiren uçtan uca bir Makine Öğrenmesi (Machine Learning) çözümüdür.

---

## 📌 Problem Tanımı

Günümüz mobil teknoloji pazarında hem üretici çeşitliliği hem de donanım kombinasyonları üstel bir hızla artmaktadır. Tüketiciler için bir cihazın sunduğu teknik özelliklerin fiyatını hak edip etmediğini anlamak; üreticiler, distribütörler ve perakendeciler için ise yeni piyasaya sürülecek bir cihazın donanım maliyetine ve rekabet koşullarına göre hangi fiyat bandında konumlandırılması gerektiğini belirlemek oldukça karmaşık ve hata payı yüksek bir süreçtir.

Geleneksel kural tabanlı fiyatlandırma yaklaşımları, donanım bileşenleri arasındaki doğrusal olmayan (non-linear) ilişkileri yakalamakta yetersiz kalmaktadır. Bu projede, çok boyutlu teknik parametreleri işleyerek gizli örüntüleri ortaya çıkaran mesafeye dayalı (**K-Nearest Neighbors**) ve topluluk öğrenmesi tabanlı (**Random Forest**) sınıflandırma algoritmaları kullanılmış; ek karşılaştırma olarak **XGBoost** ve **LightGBM** modelleri de denenmiştir.

---

## 🎯 Projenin Amacı

- **Yüksek Doğruluklu Segmentasyon:** Mobil cihazların 20 teknik özelliğini işleyerek onları **4 temel fiyat segmentine** (`0: Düşük / Giriş Segmenti`, `1: Orta Segment`, `2: Yüksek / Üst-Orta Segment`, `3: Çok Yüksek / Amiral Gemisi`) minimum hata payı ve maksimum genelleme yeteneği ile atamak.
- **Algoritmik Performans Karşılaştırması:** Öklid mesafesi tabanlı çalışan **KNN** algoritması ile birden fazla karar ağacının oylamasıyla (Bagging) çalışan **Random Forest** modelinin eğitim sürelerini, aşırı öğrenmeye (overfitting) karşı dirençlerini ve tahmin başarımlarını karşılaştırmalı olarak analiz etmek.
- **Özellik Etki ve Korelasyon Analizi:** Fiyat kategorisini belirlemede en yüksek ağırlığa sahip olan kritik donanım bileşenlerini (örneğin RAM miktarı, batarya kapasitesi, piksel yoğunluğu vb.) istatistiksel olarak tespit etmek ve donanım-fiyat ilişkisini şeffaf hale getirmek.

---

## 👤 Hedef Kullanıcılar

- **Mobil Cihaz Üreticileri & Ürün Yöneticileri (PM):** Ar-Ge aşamasındaki veya yeni piyasaya sürülecek bir cihazın donanım konfigürasyonuna göre hedef pazar fiyat bandını belirlemek ve maliyet-performans optimizasyonu yapmak isteyen ekipler.
- **E-Ticaret & Teknoloji Karşılaştırma Platformları:** Ürün kataloglarına eklenen binlerce cihazı manuel müdahale olmadan otomatik olarak fiyat-performans kategorilerine ayırmak ve kullanıcı arama/filtreleme algoritmalarını iyileştirmek isteyen yazılım ve veri ekipleri.
- **Veri Bilimi Öğrencileri ve Makine Öğrenmesi Araştırmacıları:** Çok sınıflı sınıflandırma (Multi-class Classification) problemlerinde özellik ölçeklendirmenin etkisini, hiperparametre optimizasyonunu ve farklı algoritma paradigmalarının kıyaslanmasını incelemek isteyen geliştiriciler.

---

## 📁 Veri Seti

| Bilgi | Değer |
| :--- | :--- |
| **Kaynak** | Kaggle – *Mobile Price Classification* |
| **Dosya** | `train.csv` |
| **Satır / Sütun** | 2000 satır, 21 sütun (20 özellik + 1 hedef) |
| **Eksik Değer** | Yok |
| **Hedef Değişken** | `price_range` (0, 1, 2, 3) |
| **Sınıf Dağılımı** | Dengeli – her sınıfta 500 örnek |

**Önemli değişkenler:** `battery_power` (mAh), `ram` (MB), `px_height` / `px_width` (piksel çözünürlüğü), `int_memory` (GB), `clock_speed` (GHz), `n_cores`, `fc` / `pc` (ön/arka kamera MP), `four_g`, `three_g`, `wifi`, `blue`, `dual_sim`, `touch_screen` (0/1).

---

## 🔍 Kapsam ve Kapsam Dışı

### ✅ Kapsam İçi
- **Veri Ön İşleme ve Temizleme:** Veri setindeki eksik değerlerin (`Null`/`NaN`) ve tekrarlı satırların tespiti, mantıksal tutarlılık kontrolleri (sıfır ekran genişliği/yüksekliği ve sıfır piksel değerleri) ve boxplot ile aykırı değer (`Outlier`) incelemesi.
- **Keşifçi Veri Analizi (EDA):** Sınıf dağılımı, temel istatistikler ve özellikler arası korelasyon (`Multicollinearity`) ısı haritası.
- **Özellik Ölçeklendirme (Feature Scaling):** KNN için ölçeklendirmesiz, `StandardScaler` ve `MinMaxScaler` dönüşümlerinin karşılaştırılması.
- **Model Eğitimi ve Optimizasyonu:** KNN ve Random Forest modellerinin `Train-Test Split` (%80/%20, stratified) ve 5-fold `Cross-Validation` ile eğitilmesi; `GridSearchCV` ile hiperparametre optimizasyonu.
- **Ek Model Karşılaştırması:** `XGBoost` ve `LightGBM`.
- **Kapsamlı Performans Değerlendirmesi:** `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`, öğrenme eğrisi, eğitim/test skoru farkı (overfitting kontrolü) ve hata analizi.
- **Özellik Önemi Analizi:** Random Forest `Feature Importance` skorları ve yalnızca en önemli 5 özellikle yeniden eğitilen modelin karşılaştırılması.

### ❌ Kapsam Dışı
- Canlı/gerçek zamanlı piyasa fiyat verilerinin (TL, USD, EUR) e-ticaret sitelerinden web scraping yöntemiyle anlık olarak çekilmesi.
- Kur dalgalanmaları, vergi oranları veya marka algısı (Apple, Samsung vb. marka değeri) gibi donanım dışı makroekonomik faktörlerin modele dahil edilmesi.
- Derin Öğrenme (Yapay Sinir Ağları / YSA) mimarilerinin veya büyük dil modellerinin projeye entegrasyonu.
- Son kullanıcıya açık bir web/mobil arayüz (UI/UX) tasarımı veya canlı sunucu üzerinde çalışan REST API dağıtımı (FastAPI / Flask Deployment).
- İkinci el veya yenilenmiş (refurbished) cihazların kozmetik hasar, pil sağlığı ve kullanım süresine dayalı fiyatlandırması.

---

## 👥 Ekip ve Roller

| Üye | Rolü ve Sorumlulukları |
| :--- | :--- |
| **Mert Ali Dağ** | **Kodlama & Model Mimarisi:** Veri ön işleme boru hatlarının (pipeline) kurulması, KNN ve Random Forest algoritmalarının kodlanması, hiperparametre ayarlamaları ve model performans metriklerinin teknik analizi. |
| **Gürkan Tural** | **Veri Analisti:** Ham veri setinin temizlenmesi, keşifçi veri analizi (EDA), donanım değişkenleri arasındaki istatistiksel korelasyonların incelenmesi ve bulguların anlaşılır grafiklerle görselleştirilmesi. |
| **Efe Erdem** | **Araştırmacı Veri Analizi:** Sınıflandırma problemine yönelik literatürdeki benzer çalışmaların ve rakip yaklaşımların araştırılması, elde edilen deneysel sonuçların raporlanması ve proje dokümantasyonunun oluşturulması. |

---

## 🛠️ Kullanılan Teknolojiler ve Algoritmalar

### 🐍 Programlama Dili & Geliştirme Ortamı
- **Python 3.x:** Zengin veri bilimi ekosistemi ve okunabilir sözdizimi nedeniyle projenin ana geliştirme dili olarak tercih edilmiştir.
- **VS Code / Jupyter Notebook:** İnteraktif veri keşfi, adım adım kod testleri ve görselleştirme çıktılarının anlık takibi için hibrit çalışma ortamı olarak kullanılmıştır.

### 📚 Temel Veri Bilimi Kütüphaneleri
- **Pandas & NumPy:** Tablosal veri manipülasyonu, eksik veri yönetimi, filtreleme ve çok boyutlu matris/vektör hesaplamaları.
- **Matplotlib & Seaborn:** Histogram, kutu grafikleri (`Boxplot`), korelasyon ısı haritası (`Heatmap`), `Confusion Matrix`, öğrenme eğrisi ve özellik önemi grafikleri.
- **XGBoost & LightGBM:** Gradient Boosting tabanlı ek karşılaştırma modelleri.
- **Joblib:** Eğitilen en iyi modelin `.pkl` dosyası olarak kaydedilmesi.

### 🤖 Makine Öğrenmesi Algoritmaları (Scikit-Learn)
- **K-Nearest Neighbors (KNN):**
  - **Çalışma Prensibi:** Yeni bir veri noktasının, özellik uzayındaki en yakın $K$ adet komşusunun çoğunluk sınıfına göre etiketlendiği parametrik olmayan, mesafe tabanlı (Öklid mesafesi) sınıflandırma algoritmasıdır.
  - **Veri Hazırlığı:** Farklı birimlere sahip donanım özelliklerinin (örn. mAh ve GHz) mesafe hesabını bozmaması için veriler `StandardScaler` / `MinMaxScaler` ile aynı ölçeğe indirgenmiştir. Ölçekleyici yalnızca eğitim verisine `fit` edilerek veri sızıntısı önlenmiştir.
  - **Hiperparametre Optimizasyonu:** 1–29 arası tek $K$ değerleri 5-fold cross-validation ile (`GridSearchCV` + `Pipeline`) denenerek en ideal komşu sayısı belirlenmiştir.
- **Random Forest (Rastgele Orman):**
  - **Çalışma Prensibi:** Veri setinin farklı alt kümeleri üzerinde eğitilen çok sayıda karar ağacının (Decision Trees) tahminlerini birleştirerek (Bagging / Ensemble Learning) nihai kararı veren güçlü bir sınıflandırma algoritmasıdır.
  - **Veri Hazırlığı:** Dallanma mantığı eşik değerlere dayandığı için değişken ölçeklendirmesine ihtiyaç duymadan ham sayısal değerlerle eğitilmiştir.
  - **Hiperparametre Optimizasyonu:** `GridSearchCV` ile `n_estimators`, `max_depth` ve `min_samples_split` kombinasyonları taranmıştır.
  - **Açıklanabilirlik (`Feature Importance`):** Her bir donanım özelliğinin ağaçlardaki safsızlığı (Gini Impurity) ne kadar azalttığı hesaplanarak, fiyat segmentini belirleyen en kritik bileşenler önem sırasına göre listelenmiştir.

---

## 📊 Sistem Mimarisi ve UML Diyagramları

Projede yazılımın yapısal tasarımını standartlaştırmak, veri işleme adımlarının mantıksal akışını netleştirmek ve kullanıcı etkileşimlerini modellemek amacıyla üç temel UML diyagramından yararlanılmıştır:

### 1. Sınıf Diyagramı (Class Diagram)
https://drive.google.com/file/d/1wXQ1T78jBfvHYGDIdV7gmmJJrdiNILgY/view

Sistemin **statik (değişmeyen) yapısal iskeletini** ortaya koyar. Kodlama mimarisinde yer alan veri yükleme, ön işleme, model eğitimi ve değerlendirme sınıflarını; bu sınıfların sahip olduğu nitelikleri (değişken/alan) ve gerçekleştirdikleri operasyonları (metot/fonksiyon) modüler kutular halinde tanımlar. Sınıflar arasındaki kalıtım, bağımlılık ve sahiplik ilişkilerini oklarla göstererek kodun sürdürülebilirliğini ve nesne yönelimli tasarım prensiplerine uygunluğunu belgeler.

### 2. Aktivite Diyagramı (Activity Diagram)
https://drive.google.com/file/d/1sY0KN6_H-XqaEXJlFyjsmkQT3QytY3K5/view

Sistemin uçtan uca **dinamik iş akışını** adım adım görselleştirir. Ham veri setinin sisteme alınmasından başlayarak veri temizleme, eksik değer kontrolü, ölçeklendirme karar noktaları (algoritma seçimine göre dallanma), model eğitimi ve performans metriklerinin raporlanmasına kadar tüm süreçleri akış şeması mantığıyla ifade eder.

### 3. Kullanım Senaryosu Diyagramı (Use Case Diagram)
https://drive.google.com/file/d/1OTLalFktwd552cqEhQhItV4Ar7u9I8-s/view

Sistemi dışarıdan kullanan **aktörleri** (veri analisti, araştırmacı, son kullanıcı vb.) ve bu aktörlerin sistem üzerinde gerçekleştirebileceği **temel işlevleri** (veri yükleme, model çalıştırma, tahmin sonucu görüntüleme, rapor inceleme vb.) tanımlar.

---

## 📂 Klasör Yapısı

```
Mobile-Price-Range/
├── train.csv                    # Veri seti
├── mobil_fiyat_projesi.py       # Tüm analiz ve modelleme kodu
├── README.md
├── sonuclar.csv                 # Model karşılaştırma tablosu (kod çalışınca oluşur)
├── cv_sonuclari.csv             # Cross-validation sonuçları (kod çalışınca oluşur)
├── en_iyi_model_rf.pkl          # Kaydedilen Random Forest modeli
├── scaler.pkl                   # Kaydedilen StandardScaler
└── grafikler/                   # Tüm grafikler (PNG)
    ├── sinif_dagilimi.png
    ├── korelasyon.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── knn_k_grafigi.png
    ├── ogrenme_egrisi.png
    └── boxplot_onemli_ozellikler.png
```

---

## ⚙️ Kurulum ve Çalıştırma

```bash
# 1. Depoyu klonlayın
git clone https://github.com/mertalidag14/Mobile-Price-Range.git
cd Mobile-Price-Range

# 2. Gerekli kütüphaneleri kurun
python -m pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm joblib

# 3. Kodu çalıştırın (train.csv aynı klasörde olmalı)
python mobil_fiyat_projesi.py
```

Jupyter Notebook kullanılıyorsa `mobil_fiyat_projesi.py` içeriği tek bir hücreye yapıştırılarak çalıştırılabilir.

---

## 📈 Sonuçlar

Test seti: 400 örnek (%20, stratified). Metrikler ağırlıklı (weighted) ortalamadır.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| KNN (k=5, ölçeklendirmesiz) | **0.9350** | **0.9346** | **0.9350** | **0.9347** |
| KNN (k=29, StandardScaler) | 0.6600 | 0.6688 | 0.6600 | 0.6623 |
| Random Forest | 0.8825 | 0.8818 | 0.8825 | 0.8819 |

### 5-Fold Cross-Validation (Accuracy)

Tek bir test bölmesine bağlı kalmamak için tüm modeller 2000 örnek üzerinde 5-fold CV ile de değerlendirildi.

| Model | CV Ortalama | CV Std |
| :--- | :---: | :---: |
| KNN (StandardScaler, CV ile seçilen k) | 0.6200 | ±0.0146 |
| Random Forest | 0.8755 | ±0.0099 |
| **XGBoost** | **0.9130** | ±0.0129 |
| LightGBM | 0.9085 | ±0.0096 |

Gradient Boosting modelleri (XGBoost, LightGBM) Random Forest'ı yaklaşık 3–4 puan geçti. Standart sapmaların düşük olması sonuçların kararlı olduğunu gösteriyor.

### Öne Çıkan Bulgular
- **En yüksek başarı ölçeklendirmesiz KNN ile elde edildi** (%93,5). Cross-validation'da XGBoost %91,3, LightGBM %90,85, Random Forest %87,55 aldı. Ölçeklendirilmiş KNN ise %66'da kaldı.
- **Ölçeklendirme KNN'i bozdu (beklenmedik bulgu):** ölçeklendirmesiz veride `ram` (250–4000 aralığı) mesafe hesabına hâkim oluyor ve RAM fiyatı en iyi açıklayan özellik olduğu için KNN çok başarılı oluyor. StandardScaler tüm özelliklere eşit ağırlık verince fiyatla ilgisiz 16 özellik mesafeyi gürültüyle dolduruyor ve başarı %50'ye düşüyor.
- **RAM açık ara en belirleyici özellik:** fiyat segmentiyle korelasyonu **0,92**, Random Forest önem skoru yaklaşık **0,48**. Ardından `battery_power` (0,20), `px_width` (0,17) ve `px_height` (0,15) geliyor. Diğer özelliklerin fiyatla korelasyonu 0,05'in altında.
- **Hatalar yalnızca komşu segmentler arasında:** Random Forest'ta 0 ile 3 hiç karışmadı. En çok karışan çift 1↔2 oldu (24 hata).
- **KNN'in düşük kalmasının nedeni:** fiyatla ilişkisiz 16 özellik mesafe hesabını gürültüyle dolduruyor. CV F1 skoru k büyüdükçe artıp k≈19'dan sonra 0,59 civarında sabitlendi.
- **Overfitting:** Random Forest eğitim skoru %100, CV skoru yaklaşık %87,5. Öğrenme eğrisi hâlâ yükseldiği için daha fazla veriyle başarının artması bekleniyor.
- **Veri seti tamamen dengeli:** her segmentte 500 örnek.

### Hipotezlerin Değerlendirilmesi
| Hipotez | Sonuç | Gerekçe |
| :--- | :---: | :--- |
| H1: Teknik özelliklerle 4 segment sınıflandırılabilir | ✅ Kabul | Random Forest %88,25 doğruluk |
| H2: KNN ve RF arasında belirgin fark var | ✅ Kabul | Ölçeklendirmeye bağlı: ölçeklendirmesiz KNN RF'den 5 puan iyi, ölçeklendirilmiş KNN 22 puan kötü |
| H3: Ölçeklendirme KNN'i olumlu etkiler | ❌ Ret | Ölçeklendirmesiz %93,5, StandardScaler %50, MinMaxScaler %42,5 (k=5) |
| H4: RAM, batarya, çözünürlük önemli | ✅ Kabul | Feature Importance ilk 4 sırası |

Bu proje, akıllı telefonların ve mobil cihazların donanım mimarilerini, performans bileşenlerini ve teknik spesifikasyonlarını (RAM kapasitesi, batarya ömrü, ekran çözünürlüğü, işlemci çekirdek sayısı, kamera özellikleri ve bağlantı teknolojileri vb.) derinlemesine analiz ederek cihazları en uygun fiyat segmentine yerleştiren uçtan uca bir Makine Öğrenmesi (Machine Learning) çözümüdür.

---

## 📌 Problem Tanımı

Günümüz mobil teknoloji pazarında hem üretici çeşitliliği hem de donanım kombinasyonları üstel bir hızla artmaktadır. Tüketiciler için bir cihazın sunduğu teknik özelliklerin fiyatını hak edip etmediğini anlamak; üreticiler, distribütörler ve perakendeciler için ise yeni piyasaya sürülecek bir cihazın donanım maliyetine ve rekabet koşullarına göre hangi fiyat bandında konumlandırılması gerektiğini belirlemek oldukça karmaşık ve hata payı yüksek bir süreçtir.

Geleneksel kural tabanlı fiyatlandırma yaklaşımları, donanım bileşenleri arasındaki doğrusal olmayan (non-linear) ilişkileri yakalamakta yetersiz kalmaktadır. Bu projede, çok boyutlu teknik parametreleri işleyerek gizli örüntüleri ortaya çıkaran mesafeye dayalı (**K-Nearest Neighbors**) ve topluluk öğrenmesi tabanlı (**Random Forest**) sınıflandırma algoritmaları kullanılmış; ek karşılaştırma olarak **XGBoost** ve **LightGBM** modelleri de denenmiştir.

---

## 🎯 Projenin Amacı

- **Yüksek Doğruluklu Segmentasyon:** Mobil cihazların 20 teknik özelliğini işleyerek onları **4 temel fiyat segmentine** (`0: Düşük / Giriş Segmenti`, `1: Orta Segment`, `2: Yüksek / Üst-Orta Segment`, `3: Çok Yüksek / Amiral Gemisi`) minimum hata payı ve maksimum genelleme yeteneği ile atamak.
- **Algoritmik Performans Karşılaştırması:** Öklid mesafesi tabanlı çalışan **KNN** algoritması ile birden fazla karar ağacının oylamasıyla (Bagging) çalışan **Random Forest** modelinin eğitim sürelerini, aşırı öğrenmeye (overfitting) karşı dirençlerini ve tahmin başarımlarını karşılaştırmalı olarak analiz etmek.
- **Özellik Etki ve Korelasyon Analizi:** Fiyat kategorisini belirlemede en yüksek ağırlığa sahip olan kritik donanım bileşenlerini (örneğin RAM miktarı, batarya kapasitesi, piksel yoğunluğu vb.) istatistiksel olarak tespit etmek ve donanım-fiyat ilişkisini şeffaf hale getirmek.

---

## 👤 Hedef Kullanıcılar

- **Mobil Cihaz Üreticileri & Ürün Yöneticileri (PM):** Ar-Ge aşamasındaki veya yeni piyasaya sürülecek bir cihazın donanım konfigürasyonuna göre hedef pazar fiyat bandını belirlemek ve maliyet-performans optimizasyonu yapmak isteyen ekipler.
- **E-Ticaret & Teknoloji Karşılaştırma Platformları:** Ürün kataloglarına eklenen binlerce cihazı manuel müdahale olmadan otomatik olarak fiyat-performans kategorilerine ayırmak ve kullanıcı arama/filtreleme algoritmalarını iyileştirmek isteyen yazılım ve veri ekipleri.
- **Veri Bilimi Öğrencileri ve Makine Öğrenmesi Araştırmacıları:** Çok sınıflı sınıflandırma (Multi-class Classification) problemlerinde özellik ölçeklendirmenin etkisini, hiperparametre optimizasyonunu ve farklı algoritma paradigmalarının kıyaslanmasını incelemek isteyen geliştiriciler.

---

## 📁 Veri Seti

| Bilgi | Değer |
| :--- | :--- |
| **Kaynak** | Kaggle – *Mobile Price Classification* |
| **Dosya** | `train.csv` |
| **Satır / Sütun** | 2000 satır, 21 sütun (20 özellik + 1 hedef) |
| **Eksik Değer** | Yok |
| **Hedef Değişken** | `price_range` (0, 1, 2, 3) |
| **Sınıf Dağılımı** | Dengeli – her sınıfta 500 örnek |

**Önemli değişkenler:** `battery_power` (mAh), `ram` (MB), `px_height` / `px_width` (piksel çözünürlüğü), `int_memory` (GB), `clock_speed` (GHz), `n_cores`, `fc` / `pc` (ön/arka kamera MP), `four_g`, `three_g`, `wifi`, `blue`, `dual_sim`, `touch_screen` (0/1).

---

## 🔍 Kapsam ve Kapsam Dışı

### ✅ Kapsam İçi
- **Veri Ön İşleme ve Temizleme:** Veri setindeki eksik değerlerin (`Null`/`NaN`) ve tekrarlı satırların tespiti, mantıksal tutarlılık kontrolleri (sıfır ekran genişliği/yüksekliği ve sıfır piksel değerleri) ve boxplot ile aykırı değer (`Outlier`) incelemesi.
- **Keşifçi Veri Analizi (EDA):** Sınıf dağılımı, temel istatistikler ve özellikler arası korelasyon (`Multicollinearity`) ısı haritası.
- **Özellik Ölçeklendirme (Feature Scaling):** KNN için ölçeklendirmesiz, `StandardScaler` ve `MinMaxScaler` dönüşümlerinin karşılaştırılması.
- **Model Eğitimi ve Optimizasyonu:** KNN ve Random Forest modellerinin `Train-Test Split` (%80/%20, stratified) ve 5-fold `Cross-Validation` ile eğitilmesi; `GridSearchCV` ile hiperparametre optimizasyonu.
- **Ek Model Karşılaştırması:** `XGBoost` ve `LightGBM`.
- **Kapsamlı Performans Değerlendirmesi:** `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`, öğrenme eğrisi, eğitim/test skoru farkı (overfitting kontrolü) ve hata analizi.
- **Özellik Önemi Analizi:** Random Forest `Feature Importance` skorları ve yalnızca en önemli 5 özellikle yeniden eğitilen modelin karşılaştırılması.

### ❌ Kapsam Dışı
- Canlı/gerçek zamanlı piyasa fiyat verilerinin (TL, USD, EUR) e-ticaret sitelerinden web scraping yöntemiyle anlık olarak çekilmesi.
- Kur dalgalanmaları, vergi oranları veya marka algısı (Apple, Samsung vb. marka değeri) gibi donanım dışı makroekonomik faktörlerin modele dahil edilmesi.
- Derin Öğrenme (Yapay Sinir Ağları / YSA) mimarilerinin veya büyük dil modellerinin projeye entegrasyonu.
- Son kullanıcıya açık bir web/mobil arayüz (UI/UX) tasarımı veya canlı sunucu üzerinde çalışan REST API dağıtımı (FastAPI / Flask Deployment).
- İkinci el veya yenilenmiş (refurbished) cihazların kozmetik hasar, pil sağlığı ve kullanım süresine dayalı fiyatlandırması.

---

## 👥 Ekip ve Roller

| Üye | Rolü ve Sorumlulukları |
| :--- | :--- |
| **Mert Ali Dağ** | **Kodlama & Model Mimarisi:** Veri ön işleme boru hatlarının (pipeline) kurulması, KNN ve Random Forest algoritmalarının kodlanması, hiperparametre ayarlamaları ve model performans metriklerinin teknik analizi. |
| **Gürkan Tural** | **Veri Analisti:** Ham veri setinin temizlenmesi, keşifçi veri analizi (EDA), donanım değişkenleri arasındaki istatistiksel korelasyonların incelenmesi ve bulguların anlaşılır grafiklerle görselleştirilmesi. |
| **Efe Erdem** | **Araştırmacı Veri Analizi:** Sınıflandırma problemine yönelik literatürdeki benzer çalışmaların ve rakip yaklaşımların araştırılması, elde edilen deneysel sonuçların raporlanması ve proje dokümantasyonunun oluşturulması. |

---

## 🛠️ Kullanılan Teknolojiler ve Algoritmalar

### 🐍 Programlama Dili & Geliştirme Ortamı
- **Python 3.x:** Zengin veri bilimi ekosistemi ve okunabilir sözdizimi nedeniyle projenin ana geliştirme dili olarak tercih edilmiştir.
- **VS Code / Jupyter Notebook:** İnteraktif veri keşfi, adım adım kod testleri ve görselleştirme çıktılarının anlık takibi için hibrit çalışma ortamı olarak kullanılmıştır.

### 📚 Temel Veri Bilimi Kütüphaneleri
- **Pandas & NumPy:** Tablosal veri manipülasyonu, eksik veri yönetimi, filtreleme ve çok boyutlu matris/vektör hesaplamaları.
- **Matplotlib & Seaborn:** Histogram, kutu grafikleri (`Boxplot`), korelasyon ısı haritası (`Heatmap`), `Confusion Matrix`, öğrenme eğrisi ve özellik önemi grafikleri.
- **XGBoost & LightGBM:** Gradient Boosting tabanlı ek karşılaştırma modelleri.
- **Joblib:** Eğitilen en iyi modelin `.pkl` dosyası olarak kaydedilmesi.

### 🤖 Makine Öğrenmesi Algoritmaları (Scikit-Learn)
- **K-Nearest Neighbors (KNN):**
  - **Çalışma Prensibi:** Yeni bir veri noktasının, özellik uzayındaki en yakın $K$ adet komşusunun çoğunluk sınıfına göre etiketlendiği parametrik olmayan, mesafe tabanlı (Öklid mesafesi) sınıflandırma algoritmasıdır.
  - **Veri Hazırlığı:** Farklı birimlere sahip donanım özelliklerinin (örn. mAh ve GHz) mesafe hesabını bozmaması için veriler `StandardScaler` / `MinMaxScaler` ile aynı ölçeğe indirgenmiştir. Ölçekleyici yalnızca eğitim verisine `fit` edilerek veri sızıntısı önlenmiştir.
  - **Hiperparametre Optimizasyonu:** 1–29 arası tek $K$ değerleri 5-fold cross-validation ile (`GridSearchCV` + `Pipeline`) denenerek en ideal komşu sayısı belirlenmiştir.
- **Random Forest (Rastgele Orman):**
  - **Çalışma Prensibi:** Veri setinin farklı alt kümeleri üzerinde eğitilen çok sayıda karar ağacının (Decision Trees) tahminlerini birleştirerek (Bagging / Ensemble Learning) nihai kararı veren güçlü bir sınıflandırma algoritmasıdır.
  - **Veri Hazırlığı:** Dallanma mantığı eşik değerlere dayandığı için değişken ölçeklendirmesine ihtiyaç duymadan ham sayısal değerlerle eğitilmiştir.
  - **Hiperparametre Optimizasyonu:** `GridSearchCV` ile `n_estimators`, `max_depth` ve `min_samples_split` kombinasyonları taranmıştır.
  - **Açıklanabilirlik (`Feature Importance`):** Her bir donanım özelliğinin ağaçlardaki safsızlığı (Gini Impurity) ne kadar azalttığı hesaplanarak, fiyat segmentini belirleyen en kritik bileşenler önem sırasına göre listelenmiştir.

---

## 📊 Sistem Mimarisi ve UML Diyagramları

Projede yazılımın yapısal tasarımını standartlaştırmak, veri işleme adımlarının mantıksal akışını netleştirmek ve kullanıcı etkileşimlerini modellemek amacıyla üç temel UML diyagramından yararlanılmıştır:

### 1. Sınıf Diyagramı (Class Diagram)
https://drive.google.com/file/d/1wXQ1T78jBfvHYGDIdV7gmmJJrdiNILgY/view

Sistemin **statik (değişmeyen) yapısal iskeletini** ortaya koyar. Kodlama mimarisinde yer alan veri yükleme, ön işleme, model eğitimi ve değerlendirme sınıflarını; bu sınıfların sahip olduğu nitelikleri (değişken/alan) ve gerçekleştirdikleri operasyonları (metot/fonksiyon) modüler kutular halinde tanımlar. Sınıflar arasındaki kalıtım, bağımlılık ve sahiplik ilişkilerini oklarla göstererek kodun sürdürülebilirliğini ve nesne yönelimli tasarım prensiplerine uygunluğunu belgeler.

### 2. Aktivite Diyagramı (Activity Diagram)
https://drive.google.com/file/d/1sY0KN6_H-XqaEXJlFyjsmkQT3QytY3K5/view

Sistemin uçtan uca **dinamik iş akışını** adım adım görselleştirir. Ham veri setinin sisteme alınmasından başlayarak veri temizleme, eksik değer kontrolü, ölçeklendirme karar noktaları (algoritma seçimine göre dallanma), model eğitimi ve performans metriklerinin raporlanmasına kadar tüm süreçleri akış şeması mantığıyla ifade eder.

### 3. Kullanım Senaryosu Diyagramı (Use Case Diagram)
https://drive.google.com/file/d/1OTLalFktwd552cqEhQhItV4Ar7u9I8-s/view

Sistemi dışarıdan kullanan **aktörleri** (veri analisti, araştırmacı, son kullanıcı vb.) ve bu aktörlerin sistem üzerinde gerçekleştirebileceği **temel işlevleri** (veri yükleme, model çalıştırma, tahmin sonucu görüntüleme, rapor inceleme vb.) tanımlar.

---

## 📂 Klasör Yapısı

```
Mobile-Price-Range/
├── train.csv                    # Veri seti
├── mobil_fiyat_projesi.py       # Tüm analiz ve modelleme kodu
├── README.md
├── sonuclar.csv                 # Model karşılaştırma tablosu (kod çalışınca oluşur)
├── cv_sonuclari.csv             # Cross-validation sonuçları (kod çalışınca oluşur)
├── en_iyi_model_rf.pkl          # Kaydedilen Random Forest modeli
├── scaler.pkl                   # Kaydedilen StandardScaler
└── grafikler/                   # Tüm grafikler (PNG)
    ├── sinif_dagilimi.png
    ├── korelasyon.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── knn_k_grafigi.png
    ├── ogrenme_egrisi.png
    └── boxplot_onemli_ozellikler.png
```

---

## ⚙️ Kurulum ve Çalıştırma

```bash
# 1. Depoyu klonlayın
git clone https://github.com/mertalidag14/Mobile-Price-Range.git
cd Mobile-Price-Range

# 2. Gerekli kütüphaneleri kurun
python -m pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm joblib

# 3. Kodu çalıştırın (train.csv aynı klasörde olmalı)
python mobil_fiyat_projesi.py
```

Jupyter Notebook kullanılıyorsa `mobil_fiyat_projesi.py` içeriği tek bir hücreye yapıştırılarak çalıştırılabilir.

---

## 📈 Sonuçlar

Test seti: 400 örnek (%20, stratified). Metrikler ağırlıklı (weighted) ortalamadır.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| KNN (k=5, ölçeklendirmesiz) | **0.9350** | **0.9346** | **0.9350** | **0.9347** |
| KNN (k=29, StandardScaler) | 0.6600 | 0.6688 | 0.6600 | 0.6623 |
| Random Forest | 0.8825 | 0.8818 | 0.8825 | 0.8819 |

### 5-Fold Cross-Validation (Accuracy)

Tek bir test bölmesine bağlı kalmamak için tüm modeller 2000 örnek üzerinde 5-fold CV ile de değerlendirildi.

| Model | CV Ortalama | CV Std |
| :--- | :---: | :---: |
| KNN (StandardScaler, CV ile seçilen k) | 0.6200 | ±0.0146 |
| Random Forest | 0.8755 | ±0.0099 |
| **XGBoost** | **0.9130** | ±0.0129 |
| LightGBM | 0.9085 | ±0.0096 |

Gradient Boosting modelleri (XGBoost, LightGBM) Random Forest'ı yaklaşık 3–4 puan geçti. Standart sapmaların düşük olması sonuçların kararlı olduğunu gösteriyor.

### Öne Çıkan Bulgular
- **En yüksek başarı ölçeklendirmesiz KNN ile elde edildi** (%93,5). Cross-validation'da XGBoost %91,3, LightGBM %90,85, Random Forest %87,55 aldı. Ölçeklendirilmiş KNN ise %66'da kaldı.
- **Ölçeklendirme KNN'i bozdu (beklenmedik bulgu):** ölçeklendirmesiz veride `ram` (250–4000 aralığı) mesafe hesabına hâkim oluyor ve RAM fiyatı en iyi açıklayan özellik olduğu için KNN çok başarılı oluyor. StandardScaler tüm özelliklere eşit ağırlık verince fiyatla ilgisiz 16 özellik mesafeyi gürültüyle dolduruyor ve başarı %50'ye düşüyor.
- **RAM açık ara en belirleyici özellik:** fiyat segmentiyle korelasyonu **0,92**, Random Forest önem skoru yaklaşık **0,48**. Ardından `battery_power` (0,20), `px_width` (0,17) ve `px_height` (0,15) geliyor. Diğer özelliklerin fiyatla korelasyonu 0,05'in altında.
- **Hatalar yalnızca komşu segmentler arasında:** Random Forest'ta 0 ile 3 hiç karışmadı. En çok karışan çift 1↔2 oldu (24 hata).
- **KNN'in düşük kalmasının nedeni:** fiyatla ilişkisiz 16 özellik mesafe hesabını gürültüyle dolduruyor. CV F1 skoru k büyüdükçe artıp k≈19'dan sonra 0,59 civarında sabitlendi.
- **Overfitting:** Random Forest eğitim skoru %100, CV skoru yaklaşık %87,5. Öğrenme eğrisi hâlâ yükseldiği için daha fazla veriyle başarının artması bekleniyor.
- **Veri seti tamamen dengeli:** her segmentte 500 örnek.

### Hipotezlerin Değerlendirilmesi
| Hipotez | Sonuç | Gerekçe |
| :--- | :---: | :--- |
| H1: Teknik özelliklerle 4 segment sınıflandırılabilir | ✅ Kabul | Random Forest %88,25 doğruluk |
| H2: KNN ve RF arasında belirgin fark var | ✅ Kabul | Ölçeklendirmeye bağlı: ölçeklendirmesiz KNN RF'den 5 puan iyi, ölçeklendirilmiş KNN 22 puan kötü |
| H3: Ölçeklendirme KNN'i olumlu etkiler | ❌ Ret | Ölçeklendirmesiz %93,5, StandardScaler %50, MinMaxScaler %42,5 (k=5) |
| H4: RAM, batarya, çözünürlük önemli | ✅ Kabul | Feature Importance ilk 4 sırası |

### Grafikler
![Sınıf Dağılımı](https://drive.google.com/file/d/1clry61bPuH7n1WfFgJVJNwtPaLngIY3k/view)
![Korelasyon Matrisi](https://drive.google.com/file/d/118hCKsLfNEXOTN-2LYjOVU6H6rRnAa5e/view))
![Önemli Özellikler](https://drive.google.com/file/d/1Lo5Yp4uIoK9J4Y6PdN1sClC0EDlU3CE5/view)
![Confusion Matrix](https://drive.google.com/file/d/1iCGDVu1-umjFcWsttzzc2H_SFGoiiorw/view)
![Feature Importance](https://drive.google.com/file/d/1IoJ6UoWNJkUXZ0r1NeQgNZlbqrNHpqOA/view)
![KNN k Grafiği](https://drive.google.com/file/d/1SLYDWlN0oin4Mffyfm2QluUx1EB-a_27/view)
![Öğrenme Eğrisi](https://drive.google.com/file/d/19-qtB8_i9pQh_IwdDf0nr6eojHsGUWaw/view)

---

## 🚀 Proje Durumu ve Gelecek Hedefleri

### ✅ Tamamlananlar
- Keşifçi veri analizi, veri kalitesi ve mantıksal tutarlılık kontrolleri
- KNN için ölçeklendirme karşılaştırması (ölçeklendirmesiz / StandardScaler / MinMaxScaler)
- KNN ve Random Forest modellerinin `GridSearchCV` ile optimize edilmesi
- `XGBoost` ve `LightGBM` ile ek model karşılaştırması
- 5-fold cross-validation, öğrenme eğrisi ve overfitting kontrolü
- Feature Importance analizi ve en önemli 5 özellikle model testi
- Hata analizi ve eğitim süresi karşılaştırması
- En iyi modelin `.pkl` olarak kaydedilmesi

### 🔭 Gelecek Hedefleri
- **Ek Algoritmalar:** `CatBoost` ve Destek Vektör Makinelerinin (`SVM`) sisteme dahil edilmesi.
- **RandomizedSearchCV:** Daha geniş parametre uzaylarında daha hızlı arama.
- **Etkileşimli Web Arayüzü (Deployment):** Kullanıcıların bir cihazın donanım özelliklerini (RAM, batarya, kamera vb.) girip anlık fiyat segmenti tahmini alabileceği, **Streamlit** veya **FastAPI** tabanlı bir web arayüzünün geliştirilmesi.
)
![Confusion Matrix](grafikler/confusion_matrix.png)
![Feature Importance](grafikler/feature_importance.png)
![KNN k Grafiği](grafikler/knn_k_grafigi.png)
![Öğrenme Eğrisi](grafikler/ogrenme_egrisi.png)

---

## 🚀 Proje Durumu ve Gelecek Hedefleri

### ✅ Tamamlananlar
- Keşifçi veri analizi, veri kalitesi ve mantıksal tutarlılık kontrolleri
- KNN için ölçeklendirme karşılaştırması (ölçeklendirmesiz / StandardScaler / MinMaxScaler)
- KNN ve Random Forest modellerinin `GridSearchCV` ile optimize edilmesi
- `XGBoost` ve `LightGBM` ile ek model karşılaştırması
- 5-fold cross-validation, öğrenme eğrisi ve overfitting kontrolü
- Feature Importance analizi ve en önemli 5 özellikle model testi
- Hata analizi ve eğitim süresi karşılaştırması
- En iyi modelin `.pkl` olarak kaydedilmesi

### 🔭 Gelecek Hedefleri
- **Ek Algoritmalar:** `CatBoost` ve Destek Vektör Makinelerinin (`SVM`) sisteme dahil edilmesi.
- **RandomizedSearchCV:** Daha geniş parametre uzaylarında daha hızlı arama.
- **Etkileşimli Web Arayüzü (Deployment):** Kullanıcıların bir cihazın donanım özelliklerini (RAM, batarya, kamera vb.) girip anlık fiyat segmenti tahmini alabileceği, **Streamlit** veya **FastAPI** tabanlı bir web arayüzünün geliştirilmesi.

