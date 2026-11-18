# Mobile Price Range Classification (Mobil Cihaz Fiyat Segmenti Tahmini)

Bu proje, mobil cihazların donanım ve teknik özelliklerini (RAM, Batarya, Ekran Çözünürlüğü vb.) analiz ederek cihazları doğru fiyat segmentine sınıflandıran Makine Öğrenmesi (Machine Learning) modellerini içerir.

---

## 📌 Problem Tanımı

Mobil teknoloji pazarında cihaz çeşitliliği hızla artmaktadır. Tüketiciler ve üreticiler için bir cihazın teknik özelliklerine kıyasla hangi fiyat kategorisinde yer alması gerektiğini belirlemek zaman alıcı ve karmaşık olabilir. 

Bu projede, veri kümesindeki teknik parametreler kullanılarak mesafeye dayalı ve ağaç tabanlı sınıflandırma algoritmalarıyla (**K-Nearest Neighbors** ve **Random Forest**) fiyat segmenti tahmini gerçekleştirilmiştir.

---

## 🎯 Projenin Amacı

- **Doğru Segmentasyon:** Mobil cihazların özelliklerine göre **4 farklı fiyat segmentine** (0: Düşük, 1: Orta, 2: Yüksek, 3: Çok Yüksek) en yüksek doğrulukla atanmasını sağlamak.
- **Model Karşılaştırması:** Mesafe tabanlı algoritmalardan **KNN** ile topluluk öğrenmesi (ensemble) tabanlı **Random Forest** modelinin başarım oranlarını ve çalışma dinamiklerini kıyaslamak.
- **Özellik Etki Analizi:** Fiyat segmentini belirleyen en kritik donanım bileşenlerini (RAM, Batarya, Çözünürlük vb.) tespit etmek.

---

## 👤 Hedef Kullanıcılar

- **Mobil Cihaz Üreticileri & Tedarikçileri:** Yeni piyasaya sürülecek bir cihazın teknik donanımına göre hedef fiyat bandını belirlemek isteyen veri analistleri ve ürün yöneticileri.
- **E-Ticaret & Karşılaştırma Platformları:** Ürün kataloğundaki cihazları otomatik sınıflandırmak ve kullanıcı arama filtrelerini optimize etmek isteyen yazılım ekipleri.
- **Veri Bilimi ve Makine Öğrenmesi Araştırmacıları:** Sınıflandırma algoritmalarının kıyaslanmasını incelemek isteyen geliştiriciler.

---

## 🔍 Kapsam ve Kapsam Dışı

### ✅ Kapsam İçi
- Veri ön işleme, eksik veri kontrolü ve aykırı değer analizi.
- KNN modeli için özellik ölçeklendirme (`StandardScaler` / `MinMaxScaler`).
- KNN ve Random Forest modellerinin eğitimi ve hiperparametre optimizasyonu.
- Model başarımlarının metriklerle (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`) değerlendirilmesi ve karşılaştırılması.
- Özellik önemi (`Feature Importance`) analizi.

### ❌ Kapsam Dışı
- Canlı/Gerçek zamanlı piyasa fiyat verilerinin (TL/USD) web scraping ile çekilmesi.
- Derin Öğrenme (Yapay Sinir Ağları / YSA) modellerinin entegrasyonu.
- Web veya mobil kullanıcı arayüzü (UI/UX) / API uç noktası dağıtımı (Deployment - FastAPI/Flask).
- İkinci el cihazların kozmetik durumuna dayalı fiyatlandırması.

---

## 👥 Ekip ve Roller

| Üye | Rolü ve Sorumlulukları |
| :--- | :--- |
| **Mert Ali Dağ** | **Kodlama & Araştırmacı  :** model mimarilerinin (KNN, Random Forest) kurulması, kod yazımı ve performans metriklerinin analizi. rakip/benzer yaklaşımların incelenmesi, sonuçların raporlanması |
| **Gürkan Tural** | **Veri Analisti:** Veri temizleme, keşifçi veri analizi (EDA), donanım değişkenleri arası korelasyon incelemesi ve veri görselleştirme çalışmaları. |
| **Efe Erdem** | **Araştırmacı & Veri Analzi :** Veri ön işleme Sınıflandırma problemi için rakip/benzer yaklaşımların incelenmesi, sonuçların raporlanması ve proje dokümantasyonu desteği. |

---

## 🛠️ Kullanılan Teknolojiler ve Algoritmalar

### 🐍 Programlama Dili & Geliştirme Ortamı
- **Python 3.x:** Projenin temel geliştirme dili.
- **VS Code / Jupyter Notebook:** Kodlama, veri analizi ve deney ortamı.

### 📚 Temel Veri Bilimi Kütüphaneleri
- **Pandas & NumPy:** Veri setinin yüklenmesi, veri temizleme, matris ve vektör işlemleri.
- **Matplotlib & Seaborn:** Dağılım grafikleri, korelasyon matrisi ve Karmaşıklık Matrisi (`Confusion Matrix`) görselleştirmeleri.

### 🤖 Makine Öğrenmesi Algoritmaları (Scikit-Learn)
- **K-Nearest Neighbors (KNN):**
  - **Kullanım Amacı:** Mesafe tabanlı (Öklid) sınıflandırma.
  - **Veri Hazırlığı:** Özelliklerin eşit ağırlıkta mesafelendirilmesi için `StandardScaler` / `MinMaxScaler` ile ölçeklendirme uygulandı.
  - **Hiperparametre:** En yüksek doğruluğu veren $K$ komşu sayısı belirlendi.
- **Random Forest (Rastgele Orman):**
  - **Kullanım Amacı:** Ağaç tabanlı topluluk (`Ensemble` / `Bagging`) sınıflandırması.
  - **Veri Hazırlığı:** Karar ağacı yapısı sayesinde ek bir ölçeklendirme gerektirmeden eğitildi.
  - **Analiz Yeteneği:** Fiyat segmentini en çok etkileyen donanım bileşenlerini sıralamak için `Feature Importance` metriği kullanıldı.
