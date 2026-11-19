# Mobile Price Range Classification (Mobil Cihaz Fiyat Segmenti Tahmini)

Bu proje, akıllı telefonların ve mobil cihazların donanım mimarilerini, performans bileşenlerini ve teknik spesifikasyonlarını (RAM kapasitesi, batarya ömrü, ekran çözünürlüğü, işlemci çekirdek sayısı, kamera özellikleri ve bağlantı teknolojileri vb.) derinlemesine analiz ederek cihazları en uygun fiyat segmentine yerleştiren uçtan uca bir Makine Öğrenmesi (Machine Learning) çözümüdür.

---

## 📌 Problem Tanımı

Günümüz mobil teknoloji pazarında hem üretici çeşitliliği hem de donanım kombinasyonları üstel bir hızla artmaktadır. Tüketiciler için bir cihazın sunduğu teknik özelliklerin fiyatını hak edip etmediğini anlamak; üreticiler, distribütörler ve perakendeciler için ise yeni piyasaya sürülecek bir cihazın donanım maliyetine ve rekabet koşullarına göre hangi fiyat bandında konumlandırılması gerektiğini belirlemek oldukça karmaşık ve hata payı yüksek bir süreçtir. 

Geleneksel kural tabanlı fiyatlandırma yaklaşımları, donanım bileşenleri arasındaki doğrusal olmayan (non-linear) ilişkileri yakalamakta yetersiz kalmaktadır. Bu projede, çok boyutlu teknik parametreleri işleyerek gizli örüntüleri ortaya çıkaran mesafeye dayalı (**K-Nearest Neighbors**) ve topluluk öğrenmesi tabanlı (**Random Forest**) sınıflandırma algoritmaları kullanılmış; cihazların objektif verilere dayalı fiyat segmenti tahmini gerçekleştirilmiştir.

---

## 🎯 Projenin Amacı

- **Yüksek Doğruluklu Segmentasyon:** Mobil cihazların 20'ye yakın teknik özelliğini işleyerek onları **4 temel fiyat segmentine** (`0: Düşük / Giriş Segmenti`, `1: Orta Segment`, `2: Yüksek / Üst-Orta Segment`, `3: Çok Yüksek / Amiral Gemisi`) minimum hata payı ve maksimum genelleme yeteneği ile atamak.
- **Algoritmik Performans Karşılaştırması:** Öklid mesafesi tabanlı çalışan **KNN** algoritması ile birden fazla karar ağacının oylamasıyla (Bagging) çalışan **Random Forest** modelinin eğitim sürelerini, aşırı öğrenmeye (overfitting) karşı dirençlerini ve tahmin başarımlarını karşılaştırmalı olarak analiz etmek.
- **Özellik Etki ve Korelasyon Analizi:** Fiyat kategorisini belirlemede en yüksek ağırlığa sahip olan kritik donanım bileşenlerini (örneğin RAM miktarı, batarya kapasitesi, piksel yoğunluğu vb.) istatistiksel olarak tespit etmek ve donanım-fiyat ilişkisini şeffaf hale getirmek.

---

## 👤 Hedef Kullanıcılar

- **Mobil Cihaz Üreticileri & Ürün Yöneticileri (PM):** Ar-Ge aşamasındaki veya yeni piyasaya sürülecek bir cihazın donanım konfigürasyonuna göre hedef pazar fiyat bandını belirlemek ve maliyet-performans optimizasyonu yapmak isteyen ekipler.
- **E-Ticaret & Teknoloji Karşılaştırma Platformları:** Ürün kataloglarına eklenen binlerce cihazı manuel müdahale olmadan otomatik olarak fiyat-performans kategorilerine ayırmak ve kullanıcı arama/filtreleme algoritmalarını iyileştirmek isteyen yazılım ve veri ekipleri.
- **Veri Bilimi Öğrencileri ve Makine Öğrenmesi Araştırmacıları:** Çok sınıflı sınıflandırma (Multi-class Classification) problemlerinde özellik ölçeklendirmenin etkisini, hiperparametre optimizasyonunu ve farklı algoritma paradigmalarının kıyaslanmasını incelemek isteyen geliştiriciler.

---

## 🔍 Kapsam ve Kapsam Dışı

### ✅ Kapsam İçi
- **Veri Ön İşleme ve Temizleme:** Veri setindeki eksik değerlerin (`Null`/`NaN`) tespiti, mantıksal tutarlılık kontrolleri (örn. sıfır ekran genişliği/yüksekliği kontrolü) ve aykırı değer (`Outlier`) analizleri.
- **Keşifçi Veri Analizi (EDA):** Donanım değişkenlerinin sınıf dağılımlarının incelenmesi ve özellikler arası çoklu doğrusal bağlantı (`Multicollinearity`) / korelasyon analizleri.
- **Özellik Ölçeklendirme (Feature Scaling):** Mesafe temelli KNN algoritmasının büyük ölçekli değişkenlerden (örn. RAM veya Batarya mAh değeri) yanlı etkilenmemesi için `StandardScaler` ve `MinMaxScaler` dönüşümlerinin uygulanması.
- **Model Eğitimi ve Optimizasyonu:** KNN ve Random Forest modellerinin eğitim/test (`Train-Test Split`) ve çapraz doğrulama (`Cross-Validation`) yöntemleriyle eğitilmesi, optimum parametrelerin bulunması.
- **Kapsamlı Performans Değerlendirmesi:** Modellerin `Accuracy`, `Precision`, `Recall`, `F1-Score` metrikleri ve `Confusion Matrix` (Karmaşıklık Matrisi) üzerinden hata analizlerinin yapılması.
- **Özellik Önemi Analizi:** Random Forest modeli üzerinden elde edilen `Feature Importance` skorları ile karar mekanizmasının yorumlanması.

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
- **Pandas & NumPy:** Tablosal veri manipülasyonu, eksik veri yönetimi, filtreleme ve yüksek performanslı çok boyutlu matris/vektör hesaplamaları için kullanılmıştır.
- **Matplotlib & Seaborn:** Değişken dağılımlarını gösteren histogramlar, kutu grafikleri (`Boxplot`), ısı haritası (`Heatmap`) formatında korelasyon matrisleri ve sınıflandırma hatalarını gösteren Karmaşıklık Matrisi (`Confusion Matrix`) görselleştirmeleri için kullanılmıştır.

### 🤖 Makine Öğrenmesi Algoritmaları (Scikit-Learn)
- **K-Nearest Neighbors (KNN):**
  - **Çalışma Prensibi:** Yeni bir veri noktasının, özellik uzayındaki en yakın $K$ adet komşusunun çoğunluk sınıfına göre etiketlendiği parametrik olmayan, mesafe tabanlı (Öklid mesafesi) sınıflandırma algoritmasıdır.
  - **Veri Hazırlığı:** Farklı birimlere sahip donanım özelliklerinin (örn. mAh ve GHz) mesafe hesabını bozmaması için veriler `StandardScaler` / `MinMaxScaler` ile aynı ölçeğe indirgenmiştir.
  - **Hiperparametre Optimizasyonu:** Farklı $K$ değerleri denenerek hata oranını minimize eden ve aşırı öğrenmeyi engelleyen en ideal komşu sayısı belirlenmiştir.
- **Random Forest (Rastgele Orman):**
  - **Çalışma Prensibi:** Veri setinin farklı alt kümeleri üzerinde eğitilen çok sayıda karar ağacının (Decision Trees) tahminlerini birleştirerek (Bagging / Ensemble Learning) nihai kararı veren güçlü bir sınıflandırma algoritmasıdır.
  - **Veri Hazırlığı:** Dallanma mantığı eşik değerlere dayandığı için değişken ölçeklendirmesine ihtiyaç duymadan ham sayısal değerlerle doğrudan ve yüksek hızda eğitilmiştir.
  - **Açıklanabilirlik (`Feature Importance`):** Her bir donanım özelliğinin ağaçlardaki safsızlığı (Gini Impurity) ne kadar azalttığı hesaplanarak, fiyat segmentini belirleyen en kritik bileşenler önem sırasına göre listelenmiştir.

---

## 📊 Sistem Mimarisi ve UML Diyagramları

Projede yazılımın yapısal tasarımını standartlaştırmak, veri işleme adımlarının mantıksal akışını netleştirmek ve kullanıcı etkileşimlerini modellemek amacıyla üç temel UML diyagramından yararlanılmıştır:

### 1. Sınıf Diyagramı (Class Diagram)
https://drive.google.com/file/d/1wXQ1T78jBfvHYGDIdV7gmmJJrdiNILgY/view

Sistemin **statik (değişmeyen) yapısal iskeletini** ortaya koyar. Kodlama mimarisinde yer alan veri yükleme, ön işleme, model eğitimi ve değerlendirme sınıflarını; bu sınıfların sahip olduğu nitelikleri (değişken/alan) ve gerçekleştirdikleri operasyonları (metot/fonksiyon) modüler kutular halinde tanımlar. Sınıflar arasındaki kalıtım, bağımlılık ve sahiplik ilişkilerini oklarla göstererek kodun sürdürülebilirliğini ve nesne yönelimli tasarım prensiplerine uygunluğunu belgeler.

---

### 2. Aktivite Diyagramı (Activity Diagram)
https://drive.google.com/file/d/1sY0KN6_H-XqaEXJlFyjsmkQT3QytY3K5/view

Sistemin uçtan uca **dinamik iş akışını** adım adım görselleştirir. Ham veri setinin sisteme alınmasından başlayarak veri temizleme, eksik değer kontrolü, ölçeklendirme karar noktaları (algoritma seçimine göre dallanma), model eğitimi ve performans metriklerinin raporlanmasına kadar tüm süreçleri akış şeması mantığıyla ifade eder. Bir nevi makine öğrenmesi boru hattının (ML Pipeline) algoritmik yol haritasıdır.

---

### 3. Kullanım Senaryosu Diyagramı (Use Case Diagram)
https://drive.google.com/file/d/1OTLalFktwd552cqEhQhItV4Ar7u9I8-s/view

Sistemi dışarıdan kullanan **aktörleri** (veri analisti, araştırmacı, son kullanıcı vb.) ve bu aktörlerin sistem üzerinde gerçekleştirebileceği **temel işlevleri** (veri yükleme, model çalıştırma, tahmin sonucu görüntüleme, rapor inceleme vb.) tanımlar. *"Hangi aktör sistemde hangi işlemi başlatabilir?"* sorusuna odaklanarak, karmaşık arka plan kodlarına girmeden sistemin fonksiyonel sınırlarını ve kullanıcı etkileşim senaryolarını özetler.

---

## 🚀 Proje Durumu ve Gelecek Hedefleri

Projenin **çalışan ilk demo sürümü başarıyla tamamlanmıştır**. Mevcut aşamada veri seti üzerinde keşifçi veri analizi, ön işleme adımları, KNN ve Random Forest modellerinin kurulumu ile temel başarım kıyaslamaları gerçekleştirilmiştir.

Proje aktif olarak geliştirilmeye devam edilmekte olup, ileriki aşamalarda aşağıdaki hedefler doğrultusunda kapsamı ve yetenekleri genişletilecektir:
- **Gelişmiş Algoritmaların Entegrasyonu:** Mevcut modellere ek olarak Gradient Boosting tabanlı algoritmaların (`XGBoost`, `LightGBM`, `CatBoost`) ve Destek Vektör Makinelerinin (`SVM`) sisteme dahil edilerek doğruluk oranlarının daha da yukarı çekilmesi.
- **Otomatik Hiperparametre Optimizasyonu:** `GridSearchCV` ve `RandomizedSearchCV` teknikleri ile tüm modeller için en optimum parametre kombinasyonlarının sistematik olarak taranması.
- **Etkileşimli Web Arayüzü (Deployment):** Kullanıcıların bir cihazın donanım özelliklerini (RAM, batarya, kamera vb.) manuel olarak girip anlık fiyat segmenti tahmini alabileceği, **Streamlit** veya **FastAPI** tabanlı kullanıcı dostu bir web arayüzünün geliştirilmesi.
