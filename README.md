# Mobile Price Range – Telefon Fiyat Segmenti Tahmini

Bu projede bir telefonun donanım özelliklerine (RAM, depolama, batarya, işlemci, ekran, kamera gibi) bakarak hangi fiyat grubunda olduğunu tahmin etmeye çalışıyoruz. Bunun için makine öğrenmesi modelleri kuruyor, birbirleriyle karşılaştırıyor ve en iyi çalışanı **TeknoTahmin** adında bir web sitesine koyuyoruz.

🌐 Site: **https://teknotahmin-0iji.onrender.com**

---

## Problem

Piyasada çok fazla telefon ve çok fazla özellik kombinasyonu var. Bir telefonun özelliklerine bakıp "bu telefon ucuz mu, orta mı, pahalı mı olmalı?" sorusunu cevaplamak kolay değil. Özellikler arasındaki ilişki de düz bir kurala oturmuyor: sadece RAM'e ya da sadece kameraya bakarak fiyatı doğru tahmin edemiyoruz.

Biz bu soruyu makine öğrenmesiyle çözmeye çalışıyoruz. Ana modellerimiz **KNN** ve **Random Forest**. Karşılaştırma için **XGBoost** ve **LightGBM** de deniyoruz.

---

## Projenin Amacı

- Telefonları özelliklerine göre **4 fiyat segmentine** ayırmak: `0: Düşük`, `1: Orta`, `2: Yüksek`, `3: Çok yüksek`.
- KNN ve Random Forest'ı karşılaştırmak: hangisi daha başarılı, hangisi ezberlemeye (overfitting) daha yatkın, hangisi daha hızlı.
- Fiyatı en çok hangi özelliğin belirlediğini bulmak (RAM mi, işlemci mi, ekran mı?).

---

## Kimler İşine Yarar?

- **Telefon almak isteyenler:** Baktığı telefonun özellikleri fiyatını hak ediyor mu, fikir edinebilir.
- **Alışveriş ve karşılaştırma siteleri:** Yeni eklenen telefonları otomatik olarak fiyat gruplarına ayırabilir.
- **Makine öğrenmesi öğrenen arkadaşlar:** Ölçeklendirmenin etkisini, hiperparametre aramayı ve farklı modellerin karşılaştırmasını bu proje üzerinden görebilir.

---

## Veri Seti

Projede iki veri seti kullanıyoruz. İlk analizi Kaggle'daki sentetik veri setiyle yapıyoruz (v1), sonra aynı adımları gerçek telefon verisiyle tekrarlıyoruz (v2). Neden değiştirdiğimizi en alttaki **"Veri Seti Neden Değişiyor?"** kısmında anlatıyoruz.

### 1. Eski veri seti (v1, sentetik)

| Bilgi | Değer |
| :--- | :--- |
| **Kaynak** | [Kaggle – *Mobile Price Classification*](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) |
| **Dosya** | `train.csv` |
| **Satır / Sütun** | 2000 satır, 21 sütun (20 özellik + 1 hedef) |
| **Eksik Değer** | Yok |
| **Hedef Değişken** | `price_range` (0, 1, 2, 3) |
| **Sınıf Dağılımı** | Dengeli, her sınıfta 500 örnek |

**Önemli değişkenler:** `battery_power` (mAh), `ram` (MB), `px_height` / `px_width` (piksel çözünürlüğü), `int_memory` (GB), `clock_speed` (GHz), `n_cores`, `fc` / `pc` (ön/arka kamera MP), `four_g`, `three_g`, `wifi`, `blue`, `dual_sim`, `touch_screen` (0/1).

### 2. Yeni veri seti (v2, gerçek telefonlar)

| Bilgi | Değer |
| :--- | :--- |
| **Kaynak** | [Kaggle – *Real World Smartphone's Dataset*](https://www.kaggle.com/datasets/abhijitdahatonde/real-world-smartphones-dataset) (Smartprix verisi), kullandığımız kopya: [Hugging Face](https://huggingface.co/datasets/jason1966/abhijitdahatonde_real-world-smartphones-dataset) |
| **Dosya** | `gercek_telefonlar.csv` |
| **Satır / Sütun** | 980 gerçek telefon, 22 sütun (marka, model, fiyat ve donanım özellikleri) |
| **Fiyat** | Gerçek satış fiyatı, Hindistan Rupisi (₹) |
| **Hedef Değişken** | `price_range`: fiyatı 4 eşit gruba (çeyrek) bölerek biz oluşturuyoruz |
| **Temizleme** | En pahalı %1'lik 10 lüks telefonu çıkarıyoruz, birkaç eksik değeri medyanla dolduruyoruz, geriye 970 telefon kalıyor |

**Kullandığımız özellikler (16):** `ram_capacity`, `internal_memory`, `battery_capacity`, `processor_speed`, `num_cores`, `screen_size`, `refresh_rate`, `resolution_height`, `resolution_width`, `primary_camera_rear`, `primary_camera_front`, `num_rear_cameras`, `5G_or_not`, `fast_charging_available`, `fast_charging`, `extended_memory_available`.

---

## Neleri Yapıyoruz, Neleri Yapmıyoruz?

### Yaptıklarımız
- **Veri temizliği:** Eksik değerlere, tekrar eden satırlara ve mantıksız değerlere (ör. ekran yüksekliğinin 0 olması) bakıyoruz, aykırı değerleri kutu grafikleriyle inceliyoruz.
- **Keşifçi veri analizi (EDA):** Sınıfların dağılımına, temel istatistiklere ve özellikler arasındaki korelasyona bakıyoruz.
- **Ölçeklendirme:** KNN için ölçeklendirmesiz, `StandardScaler` ve `MinMaxScaler` hâllerini karşılaştırıyoruz.
- **Model eğitimi:** Veriyi %80 eğitim, %20 test olarak ayırıyoruz (stratified), 5-fold cross-validation kullanıyoruz ve `GridSearchCV` ile en iyi parametreleri arıyoruz.
- **Ek modeller:** `XGBoost` ve `LightGBM`.
- **Değerlendirme:** `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`, öğrenme eğrisi, eğitim/test farkı (overfitting kontrolü) ve hata analizi.
- **Özellik önemi:** Random Forest'ın `Feature Importance` skorlarına bakıyor, sadece en önemli 5 özellikle de model kurup sonucu karşılaştırıyoruz.

### Yapmadıklarımız
- Fiyatları e-ticaret sitelerinden canlı olarak çekmiyoruz (web scraping yok).
- Kur, vergi ya da marka değeri (Apple, Samsung gibi) gibi donanım dışı şeyleri modele katmıyoruz.
- Derin öğrenme (yapay sinir ağları) kullanmıyoruz.
- Ayrı bir REST API (FastAPI / Flask) yazmıyoruz. Onun yerine Streamlit ile bir web sitesi yapıyoruz, aşağıdaki **"Web Sitesi: TeknoTahmin"** kısmına bakabilirsiniz.
- İkinci el ya da yenilenmiş telefonların fiyatına bakmıyoruz.

---

## Ekip ve Görev Dağılımı

| Üye | Ne yapıyor? |
| :--- | :--- |
| **Mert Ali Dağ** | **Kodlama, modeller ve web sitesi.** Veri ön işleme adımlarını, KNN ve Random Forest kodlarını yazıyor, parametre ayarlarını yapıyor ve sonuçları yorumluyor. XGBoost/LightGBM karşılaştırmasını, cross-validation'ı ve hata analizini ekliyor. Projeyi 980 gerçek telefonluk veri setine taşıyıp analizi bu veriyle tekrarlıyor (v2). **TeknoTahmin** sitesini (Streamlit, HTML/CSS) tasarlayıp kodluyor, Docker ile paketleyip Render'da yayınlıyor. GitHub deposunu, README'yi ve kanban tablosunu düzenliyor. |
| **Gürkan Tural** | **Veri analizi.** Ham veriyi temizliyor, keşifçi veri analizini (EDA) yapıyor, özellikler arasındaki korelasyonları inceliyor ve bulguları grafiklerle gösteriyor. EDA notebook'unu (`v1_kaggle_analizi/veri_inceleme_eda.ipynb`) hazırlıyor: satır/sütun sayıları, veri tipleri, özet istatistikler, fiyat segmentlerinin dağılımı, RAM, batarya ve ekran çözünürlüğünün fiyata etkisi ve korelasyon ısı haritası. Veri setinden örnek kayıtları (`Proje İçin Kullanılan Veri Seti Örneği`) derliyor ve veri setinin kaynağını belgeliyor. EDA'da çıkan sonuçlar (en belirleyici özelliklerin RAM, batarya ve çözünürlük olması) modelleme kısmına yön veriyor. |
| **Efe Erdem** | **Araştırma ve raporlama.** Literatürdeki benzer çalışmaları araştırıyor, sonuçları raporluyor ve proje belgelerini hazırlıyor. Literatür araştırmasını (`Literatur_Arastirmasi.md`) yazıyor: Kaggle'daki benzer mobil fiyat çalışmalarını ve akademik makaleleri inceliyor, kullanılan algoritmaları (SVM, KNN, Logistic Regression, ANN/DNN, ağaç tabanlı modeller) ve başarı oranlarını bir tabloda karşılaştırıyor. Ön işleme adımlarını anlatan belgeyi (`Veri_On_Isleme.md`) ve veri kalitesi sorunlarını inceleyen hata ayıklama raporunu (`Veri_seti_hata_ayıklama_raporu.md`) yazıyor. Belgelerin düzenli kalmasını sağlıyor. |

---

## Kullandığımız Teknolojiler

### Dil ve geliştirme ortamı
- **Python:** Veri bilimi kütüphaneleri çok olduğu ve okunması kolay olduğu için ana dilimiz Python.
- **VS Code / Jupyter Notebook:** Kodu adım adım çalıştırıp grafikleri anında görmek için kullanıyoruz.
- **HTML & CSS:** Sitedeki kartları, sonuç kutusunu, marka şeridini, koyu cam görünümlü tasarımı ve telefona uyumlu düzeni bunlarla yapıyoruz.
- **SVG:** Arka plandaki bulanık telefon çizimleri, telefon ikonları ve marka logoları SVG ile çiziliyor.
- **Dockerfile & TOML:** Sitenin sunucuda nasıl kurulacağını `Dockerfile`, tema ayarlarını `.streamlit/config.toml` belirliyor.

### Web sitesi ve yayınlama
- **Streamlit:** Python koduyla web sitesi yapmamızı sağlıyor. Açılır seçim kutuları, arama kutusu, filtreler, sekmeler ve sayfalama Streamlit'in hazır parçaları.
- **Docker:** Site her bilgisayarda aynı şekilde çalışsın diye uygulamayı `python:3.13-slim` tabanlı bir konteynere koyuyoruz.
- **Render:** Siteyi ücretsiz planda Render'da yayınlıyoruz (Frankfurt sunucusu).
- **Git & GitHub:** Kodu burada tutuyor ve paylaşıyoruz, Render da kodu buradan çekiyor. Görevleri GitHub Projects'teki kanban tablosunda takip ediyoruz.
- **Simple Icons:** Sitedeki marka logolarını buradan alıyoruz (açık lisanslı, CC0).

### Veri bilimi kütüphaneleri
- **Pandas & NumPy:** Tabloları okumak, eksik verileri doldurmak, filtrelemek ve hesap yapmak için.
- **Matplotlib & Seaborn:** Histogram, kutu grafikleri, korelasyon ısı haritası, confusion matrix, öğrenme eğrisi ve özellik önemi grafikleri için.
- **XGBoost & LightGBM:** Karşılaştırma için kullandığımız Gradient Boosting modelleri.
- **Joblib:** En iyi modeli `.pkl` dosyası olarak kaydetmek için.

### Makine öğrenmesi algoritmaları (Scikit-Learn)
- **KNN (K-Nearest Neighbors):**
  - **Nasıl çalışıyor:** Yeni bir telefonun en yakın $K$ komşusuna bakıyor ve çoğunluk hangi segmentteyse onu seçiyor. Yakınlığı Öklid mesafesiyle ölçüyor.
  - **Veri hazırlığı:** mAh ile GHz gibi farklı birimlerdeki özellikler mesafe hesabını bozmasın diye verileri `StandardScaler` / `MinMaxScaler` ile aynı ölçeğe getiriyoruz. Ölçekleyiciyi sadece eğitim verisine `fit` ediyoruz ki test verisinden bilgi sızmasın.
  - **Parametre arama:** 1 ile 29 arasındaki tek $K$ değerlerini 5-fold cross-validation ile (`GridSearchCV` + `Pipeline`) deniyor, en iyi komşu sayısını seçiyoruz.
- **Random Forest:**
  - **Nasıl çalışıyor:** Verinin farklı parçalarıyla çok sayıda karar ağacı eğitiyor ve bu ağaçların oylamasıyla karar veriyor.
  - **Veri hazırlığı:** Karar ağaçları eşik değerlerle çalıştığı için ölçeklendirmeye gerek yok, ham değerlerle eğitiyoruz.
  - **Parametre arama:** `GridSearchCV` ile `n_estimators`, `max_depth` ve `min_samples_split` kombinasyonlarını deniyoruz.
  - **Özellik önemi:** Her özelliğin ağaçlarda karışıklığı (Gini Impurity) ne kadar azalttığına bakarak hangi özelliklerin fiyatı en çok belirlediğini sıralıyoruz.

---

## UML Diyagramları

Projenin yapısını, adımların sırasını ve kullanıcıların siteyle ne yapabildiğini göstermek için üç UML diyagramı hazırlıyoruz. Gerçek telefon verisine (v2) ve web sitesine geçince diyagramları da güncelliyoruz. İlk hâlleri **v1**, güncel hâlleri **v2** olarak duruyor:

| Diyagram | v1 – İlk sürüm (sentetik veri, web sitesi yok) | v2 – Güncel sürüm (gerçek veri + TeknoTahmin) |
| :--- | :---: | :---: |
| Sınıf Diyagramı (Class) | [Görüntüle](https://drive.google.com/file/d/1wXQ1T78jBfvHYGDIdV7gmmJJrdiNILgY/view?usp=sharing) | [Görüntüle](https://drive.google.com/file/d/1nKgucbklsTJXGJYzTyyyrxNGbbuoFyd0/view?usp=sharing) |
| Aktivite Diyagramı (Activity) | [Görüntüle](https://drive.google.com/file/d/1sY0KN6_H-XqaEXJlFyjsmkQT3QytY3K5/view?usp=sharing) | [Görüntüle](https://drive.google.com/file/d/1lSw9-dR1bsSlzJepRA4KBatqPK6S_5-w/view?usp=sharing) |
| Kullanım Senaryosu Diyagramı (Use Case) | [Görüntüle](https://drive.google.com/file/d/1OTLalFktwd552cqEhQhItV4Ar7u9I8-s/view?usp=sharing) | [Görüntüle](https://drive.google.com/file/d/1PZXyBZt-nu3XMUGz2Q0yY06bAdKfJiSY/view?usp=sharing) |

Bütün diyagramlar Google Drive'daki **Diyagram Görselleri** klasöründe, linki olan herkes açabiliyor.

**Diyagramları hazırlayanlar:** Mert Ali Dağ · Gürkan Tural · Efe Erdem

### 1. Sınıf Diyagramı

Projenin parçalarını ve bu parçaların birbiriyle ilişkisini gösteriyor: veri yükleme, ön işleme, model eğitimi, değerlendirme ve web sitesi.

🖼️ **Diyagramı aç:** [v2 – güncel](https://drive.google.com/file/d/1nKgucbklsTJXGJYzTyyyrxNGbbuoFyd0/view?usp=sharing) · [v1 – ilk sürüm](https://drive.google.com/file/d/1wXQ1T78jBfvHYGDIdV7gmmJJrdiNILgY/view?usp=sharing)

**Sınıflar ne yapıyor:**

| Sınıf | Görevi | Koddaki karşılığı |
| :--- | :--- | :--- |
| `VeriYukleyici` | Seçilen veri setini (v1 sentetik ya da v2 gerçek telefonlar) dosyadan okuyor. | `pd.read_csv(...)` (1. bölüm) |
| `VeriOnIsleyici` | Eksik değerleri dolduruyor, en pahalı %1'lik lüks telefonları çıkarıyor, fiyatı 4 segmente bölüyor, veriyi %80/%20 ayırıyor ve KNN için ölçeklendiriyor. | 2, 3, 5 ve 6. bölümler |
| `KesifciAnaliz` | Sınıf dağılımı, korelasyon ısı haritası ve boxplot grafiklerini çiziyor. | 4 ve 17. bölümler |
| `ModelEgitici` | KNN, Random Forest, XGBoost ve LightGBM'i GridSearchCV ile en iyi parametrelerle eğitiyor. | 7, 8 ve 9. bölümler |
| `ModelDegerlendirici` | Accuracy/Precision/Recall/F1, confusion matrix, 5-fold CV, öğrenme eğrisi, özellik önemi ve hata analizini hesaplıyor. | 10–18. bölümler |
| `ModelKaydedici` | En iyi modeli `.pkl` olarak kaydediyor, gerekince geri yüklüyor. | `joblib.dump` (20. bölüm) |
| `Telefon` | Katalogdaki tek bir gerçek telefon. Fiyatını ₹'den TL'ye çeviriyor. | `gercek_telefonlar.csv` satırı |
| `Katalog` | Telefonları marka, segment, fiyat ve 5G'ye göre filtreliyor, sıralıyor, arıyor ve sayfalara bölüyor. | `web_arayuzu/app.py` katalog sekmesi |
| `TeknoTahminSitesi` | Fiyat tahmini, benzer telefonlar, karşılaştırma ve proje özetini gösteren web sitesi. | `web_arayuzu/app.py` |

**Oklar ne anlama geliyor:**
- **Düz oklar (→):** Verinin bir sınıftan diğerine geçişi. Ham veri önce ön işlemeye, sonra EDA'ya ve model eğitimine gidiyor. Eğitilen modeller değerlendiriliyor, en iyisi kaydediliyor.
- **İçi boş elmas (o--), toplama:** Katalog birçok telefondan oluşuyor ama telefonlar katalog olmadan da var (veri setinde duruyorlar).
- **İçi dolu elmas (*--), bileşim:** Katalog sitenin bir parçası, site olmadan katalog da yok.
- **Kesikli ok (..>), bağımlılık:** Site, tahmin yapmak için eğitim adımında çıkan Random Forest modeline ihtiyaç duyuyor.

### 2. Aktivite Diyagramı

Projenin baştan sona adımlarını gösteriyor: veri setini seçmekten modeli web sitesinde yayınlamaya kadar.

🖼️ **Diyagramı aç:** [v2 – güncel](https://drive.google.com/file/d/1lSw9-dR1bsSlzJepRA4KBatqPK6S_5-w/view?usp=sharing) · [v1 – ilk sürüm](https://drive.google.com/file/d/1sY0KN6_H-XqaEXJlFyjsmkQT3QytY3K5/view?usp=sharing)

**Adımlar:**
1. **Veri setini seçiyoruz:** Ya sentetik Kaggle verisiyle (v1) ya da 980 gerçek telefonla (v2) başlıyoruz. İki yol da aynı adımlardan geçiyor, böylece sonuçları karşılaştırabiliyoruz.
2. **Gerçek veriyi temizliyoruz:** v2'de birkaç eksik değeri medyanla dolduruyor ve fiyat dağılımını bozan en pahalı %1'lik lüks telefonları (ör. Vertu) çıkarıyoruz. Sonra telefonları fiyatlarına göre 4 eşit segmente ayırıyoruz. v1'de segmentler hazır geldiği için bu adıma gerek yok.
3. **EDA yapıyoruz:** Sınıf dağılımına, korelasyonlara ve aykırı değerlere grafiklerle bakıyoruz.
4. **Eğitim/test diye ayırıyoruz:** Veriyi segment oranlarını koruyarak (stratified) %80 eğitim, %20 test olarak bölüyoruz.
5. **Ölçeklendirmeye karar veriyoruz:** KNN mesafeye baktığı için ölçeklendirmeden etkileniyor, o yüzden StandardScaler ve MinMaxScaler'ı deniyoruz. Ağaç tabanlı modeller (Random Forest, XGBoost, LightGBM) ham değerlerle çalışıyor.
6. **Parametre arayıp eğitiyoruz:** GridSearchCV ile her model için en iyi parametreleri 5-fold cross-validation ile buluyor ve modelleri eğitiyoruz.
7. **Değerlendiriyoruz:** Test seti metriklerini, confusion matrix'i, cross-validation'ı, öğrenme eğrisini (overfitting kontrolü), özellik önemini ve hataları hesaplıyoruz.
8. **Kaydediyoruz:** En iyi modeli `.pkl` dosyası olarak saklıyoruz.
9. **Yayınlıyoruz:** Modeli TeknoTahmin sitesine koyuyor, Docker ile paketliyor ve Render'da herkesin açabileceği şekilde yayınlıyoruz.
10. **Kullanım:** Siteye giren kişi tahmin alıyor, katalogda geziyor ve telefonları karşılaştırıyor.

### 3. Kullanım Senaryosu Diyagramı

Sistemi kimlerin kullandığını ve ne yapabildiklerini gösteriyor.

🖼️ **Diyagramı aç:** [v2 – güncel](https://drive.google.com/file/d/1PZXyBZt-nu3XMUGz2Q0yY06bAdKfJiSY/view?usp=sharing) · [v1 – ilk sürüm](https://drive.google.com/file/d/1OTLalFktwd552cqEhQhItV4Ar7u9I8-s/view?usp=sharing)

| Aktör | Kim? | Neler yapabiliyor? |
| :--- | :--- | :--- |
| 👤 **Kullanıcı** | Telefon almayı düşünen ya da bir telefonun fiyatını merak eden herkes | Özellikleri girip **fiyat segmenti tahmini alıyor**, **katalogda** gerçek telefonları arıyor ve filtreliyor, **iki telefonu karşılaştırıyor**, proje sonuçlarına bakıyor. |
| 👤 **Veri Analisti / Araştırmacı** | **Gürkan Tural, Efe Erdem**: veri analizini ve araştırmayı yapan ekip üyeleri | **Veri setini yükleyip temizliyor**, **modelleri eğitip karşılaştırıyor**, grafik ve rapor hazırlıyor, sonuçları yorumluyor. |
| 👤 **Geliştirici** | **Mert Ali Dağ**: modeli ve siteyi hazırlayıp yayınlayan kişi | En iyi **modeli kaydediyor**, siteyi **Docker ile paketleyip Render'da yayınlıyor** ve güncel tutuyor. |

**Çizgiler ne anlama geliyor:**
- **Düz çizgiler:** Aktör o işi doğrudan başlatıyor.
- **«include» (kesikli oklar):** "Fiyat segmenti tahmini al" ve "Modelleri eğit ve karşılaştır" her zaman **kaydedilmiş modeli** kullanıyor. Eğitilmiş model olmadan tahmin yapılamıyor.
- **Gri kutu (sistem sınırı):** Kutunun içi TeknoTahmin'in yaptığı işler, aktörler kutunun dışında.

---

## Klasör Yapısı

```
Mobile-Price-Range/
├── README.md                      → Projenin tanıtımı ve sonuçları
├── v1_kaggle_analizi/             → İlk analiz (Kaggle sentetik veri seti) ve ona ait belgeler
├── v2_gercek_telefon_analizi/     → Gerçek telefon verisiyle analiz
└── web_arayuzu/                   → TeknoTahmin web sitesi (Render'da çalışan kod)
```

### `v1_kaggle_analizi/belgeler/` – Araştırma belgeleri (v1 veri setine ait)
| Dosya | Açıklama |
| :--- | :--- |
| `Literatur_Arastirmasi.md` | Benzer çalışmalar ve literatür taraması |
| `Veri_On_Isleme.md` | Ön işleme adımlarının açıklaması |
| `Veri_seti_hata_ayıklama_raporu.md` | Veri kalitesi / hata ayıklama raporu |
| `Proje İçin Kullanılan Veri Seti Örneği` | Veri setinin kaynağı ve örnek kayıtlar |

### `v1_kaggle_analizi/` – İlk analiz (sentetik veri seti)
| Dosya | Açıklama |
| :--- | :--- |
| `train.csv` | Veri seti (2000 satır, 21 sütun), kaynak: [Kaggle – Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) |
| `veri_inceleme_eda.ipynb` | Keşifçi veri analizi (EDA) notebook'u |
| `proje.ipynb` | KNN, Random Forest, XGBoost ve LightGBM kodu |
| `mobil_fiyat_projesi.py` | Aynı kodun tek dosyalık Python hâli |
| `sonuclar.csv` | Test setinde modellerin karşılaştırması |
| `cv_sonuclari.csv` | 5-fold cross-validation sonuçları |
| `en_iyi_model_rf.pkl`, `scaler.pkl` | Kaydedilen Random Forest modeli ve StandardScaler |
| `grafikler/` | Sınıf dağılımı, korelasyon, boxplot, confusion matrix, feature importance, KNN k grafiği, öğrenme eğrisi |

### `v2_gercek_telefon_analizi/` – Gerçek telefon verisiyle analiz
| Dosya | Açıklama |
| :--- | :--- |
| `gercek_telefonlar.csv` | 980 gerçek telefon ve gerçek fiyatları (Smartprix, Hindistan ₹), kaynak: [Kaggle – Real World Smartphone's Dataset](https://www.kaggle.com/datasets/abhijitdahatonde/real-world-smartphones-dataset) |
| `PROJE_gercek.ipynb` | Analiz adımlarının gerçek veriyle yapılmış hâli (20 bölüm) |
| `gercek_telefon_projesi.py` | Aynı kodun tek dosyalık Python hâli |
| `sonuclar_gercek.csv`, `cv_sonuclari_gercek.csv` | Gerçek veriyle model sonuçları |
| `grafikler_gercek/` | Gerçek veriyle çizilen grafikler |

### `web_arayuzu/` – TeknoTahmin web sitesi
| Dosya | Açıklama |
| :--- | :--- |
| `app.py` | Sitenin Streamlit kodu |
| `logolar.py` | Marka logoları (Simple Icons, tek renk SVG) |
| `gercek_telefonlar.csv` | Sitedeki modelin eğitildiği gerçek telefon verisi |
| `requirements.txt` | Sitenin ihtiyaç duyduğu kütüphaneler |
| `Dockerfile` | Sitenin Render'da Docker ile çalışması için kurulum dosyası |
| `.streamlit/config.toml` | Tema ve hız ayarları |

---

## Kurulum ve Çalıştırma

```bash
# 1. Depoyu indirin
git clone https://github.com/mertalidag14/Mobile-Price-Range.git
cd Mobile-Price-Range

# 2. Gerekli kütüphaneleri kurun
python -m pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm joblib notebook

# 3a. İlk analiz (sentetik veri seti)
cd v1_kaggle_analizi
jupyter notebook proje.ipynb          # ya da: python mobil_fiyat_projesi.py
cd ..

# 3b. Gerçek telefon verisiyle analiz
cd v2_gercek_telefon_analizi
jupyter notebook PROJE_gercek.ipynb   # ya da: python gercek_telefon_projesi.py
cd ..

# 4. Web sitesini bilgisayarda açmak için
cd web_arayuzu
python -m pip install -r requirements.txt
streamlit run app.py
```

VS Code kullanıyorsanız notebook'u açıp **Run All** demeniz yeterli.

---

## Sonuçlar (v1, sentetik veri seti)

Gerçek telefon verisiyle çıkan sonuçlar aşağıdaki **"Gerçek Telefon Verisiyle Analiz (v2)"** kısmında.

Test seti 400 örnekten oluşuyor (%20, stratified). Metrikler ağırlıklı (weighted) ortalama.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| KNN (k=5, ölçeklendirmesiz) | **0.9350** | **0.9346** | **0.9350** | **0.9347** |
| KNN (k=29, StandardScaler) | 0.6600 | 0.6688 | 0.6600 | 0.6623 |
| Random Forest | 0.8825 | 0.8818 | 0.8825 | 0.8819 |

### 5-Fold Cross-Validation (Accuracy)

Sadece tek bir test bölmesine güvenmemek için bütün modelleri 2000 örnek üzerinde 5-fold CV ile de deniyoruz.

| Model | CV Ortalama | CV Std |
| :--- | :---: | :---: |
| KNN (StandardScaler, CV ile seçilen k) | 0.6200 | ±0.0146 |
| Random Forest | 0.8755 | ±0.0099 |
| **XGBoost** | **0.9130** | ±0.0129 |
| LightGBM | 0.9085 | ±0.0096 |

XGBoost ve LightGBM, Random Forest'ı yaklaşık 3–4 puan geçiyor. Standart sapmalar düşük, yani sonuçlar kararlı.

### Neler görüyoruz?
- **En iyi sonucu ölçeklendirmesiz KNN veriyor** (%93,5). Cross-validation'da XGBoost %91,3, LightGBM %90,85, Random Forest %87,55 alıyor. Ölçeklendirilmiş KNN ise %66'da kalıyor.
- **Ölçeklendirme KNN'i bozuyor (beklemediğimiz bir sonuç):** Ölçeklendirmesiz veride `ram` (250–4000 arası) mesafe hesabını domine ediyor. RAM de fiyatı en iyi açıklayan özellik olduğu için KNN çok başarılı oluyor. StandardScaler bütün özelliklere eşit ağırlık verince fiyatla ilgisi olmayan 16 özellik mesafeyi karıştırıyor ve başarı %50'ye düşüyor.
- **RAM açık ara en önemli özellik:** fiyat segmentiyle korelasyonu **0,92**, Random Forest'taki önem skoru yaklaşık **0,48**. Ondan sonra `battery_power` (0,20), `px_width` (0,17) ve `px_height` (0,15) geliyor. Diğer özelliklerin fiyatla korelasyonu 0,05'in altında.
- **Hatalar sadece yan yana segmentlerde:** Random Forest 0 ile 3'ü hiç karıştırmıyor. En çok 1 ile 2'yi karıştırıyor (24 hata).
- **KNN neden düşük kalıyor:** Fiyatla ilgisi olmayan 16 özellik mesafe hesabını bozuyor. CV F1 skoru k büyüdükçe artıyor ve k≈19'dan sonra 0,59 civarında sabit kalıyor.
- **Overfitting:** Random Forest eğitimde %100, CV'de yaklaşık %87,5 alıyor. Öğrenme eğrisi hâlâ yükseldiği için daha fazla veriyle başarının artmasını bekliyoruz.
- **Veri seti tamamen dengeli:** her segmentte 500 örnek var.

### Hipotezlerimiz
| Hipotez | Sonuç | Neden |
| :--- | :---: | :--- |
| H1: Teknik özelliklerle 4 segment ayırt edilebilir | ✅ Kabul | Random Forest %88,25 doğruluk |
| H2: KNN ile RF arasında belirgin fark var | ✅ Kabul | Ölçeklendirmeye bağlı: ölçeklendirmesiz KNN RF'den 5 puan iyi, ölçeklendirilmiş KNN 22 puan kötü |
| H3: Ölçeklendirme KNN'i iyileştirir | ❌ Ret | Ölçeklendirmesiz %93,5, StandardScaler %50, MinMaxScaler %42,5 (k=5) |
| H4: RAM, batarya ve çözünürlük önemli | ✅ Kabul | Feature Importance'ta ilk 4 sırada |

---

## Gerçek Telefon Verisiyle Analiz (v2)

Sentetik veriden sonra modeli **980 gerçek telefonun** gerçek satış fiyatlarıyla da deniyoruz (Smartprix, Hindistan). En pahalı %1'lik lüks telefonları (ör. Vertu) çıkarıyoruz, kalan **970 telefonu** fiyatlarına göre 4 eşit segmente bölüyoruz: Düşük ₹3.499–12.999, Orta ₹12.999–19.990, Yüksek ₹19.990–34.999, Çok yüksek ₹34.999–163.980. Modelde 16 donanım özelliği kullanıyoruz (RAM, depolama, batarya, işlemci hızı, ekran, kamera, 5G, hızlı şarj vb.).

| Model | Accuracy | Precision | Recall | F1-Score | 5-Fold CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| KNN (k=3, StandardScaler) | 0.7010 | 0.6982 | 0.7010 | 0.6963 | 0.6186 |
| Random Forest | 0.7474 | 0.7441 | 0.7474 | 0.7430 | **0.7062** |
| **XGBoost** | **0.7680** | **0.7651** | **0.7680** | **0.7636** | 0.6794 |
| LightGBM | 0.7526 | 0.7495 | 0.7526 | 0.7462 | 0.6804 |

**Neler görüyoruz?**
- Gerçek veride başarı sentetik veriye göre daha düşük (%94 yerine %75 civarı). Çünkü gerçek fiyat sadece donanıma değil, marka değerine, tasarıma ve pazarlamaya da bağlı.
- Sentetik veride fiyatı neredeyse tek başına RAM belirliyor. Gerçek veride ise **işlemci hızı, depolama, RAM, hızlı şarj gücü ve ekran çözünürlüğü** birlikte önemli.
- Bu sefer ölçeklendirme KNN'e yardım ediyor (F1: 0.64 → 0.66), çünkü gerçek veride tek bir özellik baskın değil.
- Hatalar yine çoğunlukla yan yana segmentlerde (ör. Düşük ↔ Orta).
- Veriyi incelerken 39 telefonda ekran çözünürlüğünün genişlik ve yüksekliğinin ters girildiğini fark ediyoruz. Bunu düzeltmek bir sonraki adımımız.

---

## Proje Durumu ve Sonraki Adımlar

### Bitenler
- Keşifçi veri analizi, veri kalitesi ve mantıksal tutarlılık kontrolleri (v1 ve v2)
- KNN için ölçeklendirme karşılaştırması (ölçeklendirmesiz / StandardScaler / MinMaxScaler)
- KNN ve Random Forest'ın `GridSearchCV` ile ayarlanması
- `XGBoost` ve `LightGBM` ile karşılaştırma
- 5-fold cross-validation, öğrenme eğrisi ve overfitting kontrolü
- Feature Importance analizi ve en önemli 5 özellikle model denemesi
- Hata analizi ve eğitim süresi karşılaştırması
- En iyi modelin `.pkl` olarak kaydedilmesi
- Aynı analizin 980 gerçek telefonla tekrarlanması (v2)
- **TeknoTahmin** web sitesi (Streamlit + Docker, Render'da yayında)

### Sonraki adımlar
- `CatBoost` ve `SVM` modellerini de eklemek.
- `RandomizedSearchCV` ile daha geniş bir parametre aralığını daha hızlı taramak.
- Gerçek veride ters girilmiş 39 ekran çözünürlüğünü düzeltip modeli yeniden eğitmek.

---

## Web Sitesi: TeknoTahmin

Gerçek telefon verisiyle eğittiğimiz modeli **TeknoTahmin** adında bir Streamlit sitesine çeviriyoruz (canlı: **https://teknotahmin-0iji.onrender.com**). Sitede 4 sekme var:

- **🔮 Fiyat Tahmini:** Telefonun adını yazıp (ör. "iPhone") modeli seçiyorsunuz, sabit özellikleri kendiliğinden geliyor. RAM ve depolamayı seçince fiyat segmenti, modelin ne kadar emin olduğu, segmentin gerçek fiyat aralığı ve o varyantın gerçek fiyatı çıkıyor. Telefonun adını hatırlamıyorsanız RAM, depolama, batarya, kamera, ekran, işlemci, şarj ve 5G'yi listeden seçebiliyorsunuz. Yazdığınız isme ve seçtiğiniz özelliklere uyan gerçek telefonlar sağda kartlar hâlinde hemen listeleniyor (ör. "iPhone" yazıp 4 GB RAM seçince sadece 4 GB RAM'li iPhone'lar çıkıyor). "Hızlı dene" butonlarıyla tek tıkla örnek bir telefon da deneyebiliyorsunuz.
- **🛒 Telefon Kataloğu:** 970 gerçek telefon, gerçek fiyatları ve marka logolarıyla mağaza kartları hâlinde listeleniyor. Marka, segment, fiyat aralığı ve 5G filtreleri, sıralama, arama ve "modelin yanıldıkları" filtresi var. Kartlardaki "Karşılaştır" butonuyla iki telefonu seçip karşılaştırmaya geçebiliyorsunuz.
- **⚖️ Karşılaştır:** İki gerçek telefonu yan yana koyuyor (gerçek segment ve modelin tahmini dahil) ve özelliklerini bir radar grafikte gösteriyor.
- **📊 Proje Hakkında:** Yöntem, ekip, model sonuçları ve modelin grafikleri (özellik önemi, karışıklık matrisi).

**Model:** 10 donanım özelliğiyle eğitilmiş Random Forest, test doğruluğu yaklaşık **%76**.

### Ekran Görüntüleri

**🔮 Fiyat Tahmini:** Hızlı dene butonları, tahmin edilen segment, fiyat cetveli ve "Neden bu segment?" açıklaması

![TeknoTahmin – Fiyat Tahmini](https://drive.google.com/thumbnail?id=1jVVLqR3hSUd6k7DqG7uvSrHbXL4wDXqd&sz=w1600)


**🛒 Telefon Kataloğu:** 970 gerçek telefon, TL fiyatlar, marka logoları ve filtreler

![TeknoTahmin – Telefon Kataloğu](https://drive.google.com/thumbnail?id=1H00gzaTHNpeT5uFYHsX7p5rbm8FpNWHf&sz=w1600)


**⚖️ Karşılaştır:** İki telefonun yan yana karşılaştırması ve özellik haritası (radar grafik)

![TeknoTahmin – Karşılaştır](https://drive.google.com/thumbnail?id=1JCYjiNZqY0HPMuGOkD4iVuRtXl5neb3V&sz=w1600)


**📊 Proje Hakkında:** Yöntem, ekip ve modelin grafikleri (özellik önemi, karışıklık matrisi)

![TeknoTahmin – Proje Hakkında](https://drive.google.com/thumbnail?id=129iSuU-p2AsIGvmo2wR4eBQivA3mjSKx&sz=w1600)


> **Not:** Fiyatlar Smartprix (Hindistan) verisindeki Hindistan Rupisi (₹) fiyatları. Sitede bunları 1 ₹ ≈ 0,54 TL
> kuruyla (XE orta kur, 19 Kasım 2026) TL'ye çevirerek gösteriyoruz. Marka adları ve logoları sahiplerine
> ait (logolar: Simple Icons).

**Bilgisayarda çalıştırmak için:**
```bash
cd web_arayuzu
python -m pip install -r requirements.txt
streamlit run app.py
```

**Yayınlama:** Site [Render](https://render.com) üzerinde ücretsiz planda Docker ile çalışıyor (Root Directory: `web_arayuzu`).

---

## Veri Seti Neden Değişiyor?

Projeye Kaggle'daki **Mobile Price Classification** veri setiyle başlıyoruz (eski veri seti: [Kaggle – Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)). Bu veriyle yaptığımız analiz (v1) olduğu gibi duruyor: `v1_kaggle_analizi/` klasörü.

Çalıştıkça bu veri setinin bazı eksiklerini görüyoruz ve analizi gerçek telefon verisiyle tekrarlıyoruz (v2):

1. **Veri sentetik.** 2000 satırın hiçbiri gerçek bir telefon değil. Marka ve model adı yok, bu yüzden sonuçları gerçek telefonlarla karşılaştıramıyoruz.
2. **Gerçek fiyat yok.** Hedef değişken sadece 0–3 arası bir etiket. Bir segmentin kaç lira olduğunu söyleyemiyoruz.
3. **Sonuçlar gerçekçi değil.** Fiyat segmenti neredeyse tamamen RAM'e bağlı (korelasyon 0,92) ve modeller %93–94 gibi çok yüksek başarı veriyor. Gerçek hayatta fiyatı tek bir özellikten bu kadar net tahmin edemeyiz.
4. **Güncel değil.** RAM en fazla 4 GB, batarya en fazla yaklaşık 2000 mAh. Bugünün telefonları (8 GB+ RAM, 5000 mAh) bu aralığın tamamen dışında, o yüzden model güncel bir telefon için anlamlı bir tahmin yapamıyor.

**Yeni veri setiyle neler değişiyor?**
- 980 **gerçek** telefonu marka ve model adlarıyla, **gerçek satış fiyatlarıyla** kullanıyoruz.
- Segmentleri gerçek fiyatlara göre 4 eşit gruba bölerek oluşturuyoruz, böylece her segmentin fiyat aralığı belli.
- Başarı %94'ten yaklaşık **%75'e** düşüyor (en iyi: XGBoost %76,8). Bu düşüş beklenen bir şey: gerçek fiyat donanımın yanında marka değeri, tasarım ve pazarlama gibi modelin göremediği şeylere de bağlı. Bu yüzden v2 sonuçları gerçek hayatı daha dürüst gösteriyor.
- En önemli özellik artık tek başına RAM değil: **işlemci hızı, depolama, RAM, hızlı şarj gücü ve ekran çözünürlüğü** birlikte belirleyici.
- Web sitesi (TeknoTahmin) bu sayede gerçek telefonları gerçek fiyatlarıyla gösterebiliyor.

İki analiz de aynı adımları (EDA, ölçeklendirme karşılaştırması, KNN, Random Forest, XGBoost, LightGBM, cross-validation, feature importance, hata analizi) izlediği için iki veri setinin sonuçlarını doğrudan karşılaştırabiliyoruz.
