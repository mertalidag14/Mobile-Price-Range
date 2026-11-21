# Mobil Telefon Fiyat Segmenti Sınıflandırması: Kaggle Çalışmaları ve Literatür Taraması

Bu doküman, Kaggle üzerindeki **Mobile Price Classification** veri seti (Abhishek Sharma) kullanılarak geliştirilen üç temel notebook çalışmasının metodolojik incelemesini ve ilgili akademik literatürdeki bulguları özetlemektedir.

## 1. Veri Setinin Genel Yapısı ve Problem Tanımı

İncelenen tüm çalışmalar ve literatürdeki referans makaleler, **20 öznitelik (feature)** ve **2.000 eğitim örneğinden** oluşan standart **Mobile Price Classification** veri setini temel alır.

* **Problem Türü:** Çok Sınıflı Sınıflandırma (Multi-Class Classification)

* **Hedef Değişken (`price_range`):** 4 dengeli sınıftan oluşur (her sınıfta 500 örnek):

  * `0`: Düşük Maliyet (Low Cost)

  * `1`: Orta Maliyet (Medium Cost)

  * `2`: Yüksek Maliyet (High Cost)

  * `3`: Çok Yüksek Maliyet (Very High Cost)

* **Temel Donanım Öznitelikleri:** `ram`, `battery_power`, `px_height`, `px_width`, `int_memory`, `clock_speed`, `n_cores`, `mobile_wt`, `pc` (ana kamera), `fc` (ön kamera), `sc_h`, `sc_w`, `m_dep`, `talk_time` ve ikili (binary) bağlantı/donanım bayrakları (`blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi`).

## 2. İncelenen Kaggle Çalışmalarının Analizi

### 2.1. Vikram B. — *Mobile Price Prediction*

* **URL:** `https://www.kaggle.com/code/vikramb/mobile-price-prediction`

* **Odak Noktası:** Temel keşifçi veri analizi (EDA) ve çoklu klasik makine öğrenmesi algoritmalarının karşılaştırmalı kıyaslaması (benchmarking).

* **Metodoloji:**

  * Eksik veri kontrolü ve öznitelik dağılımlarının hedef sınıfa göre görselleştirilmesi (boxplot ve korelasyon matrisleri).

  * `ram` değişkeninin fiyat aralığı ile en yüksek pozitif korelasyona ($r > 0.90$) sahip olduğu, bunu batarya kapasitesi (`battery_power`) ve ekran çözünürlüğünün (`px_height`, `px_width`) izlediği ortaya konulmuştur.

  * **Kullanılan Modeller:** Logistic Regression, Linear Regression (yuvarlama ile), K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machines (SVM).

* **Temel Bulgu:** Doğrusal sınırlar çizebilen modeller (Logistic Regression ve Linear/RBF SVM), veri setindeki güçlü doğrusal/monotonik ilişkiler nedeniyle ağaç tabanlı modellere kıyasla daha yüksek doğruluk (accuracy) üretmiştir.

### 2.2. Gülşah Demiryürek — *Mobile Price Classification with SVM*

* **URL:** `https://www.kaggle.com/code/gulsahdemiryurek/mobile-price-classification-with-svm`

* **Odak Noktası:** Destek Vektör Makineleri (SVM) özelinde veri ön işleme, özellik ölçeklendirme (feature scaling) ve hiperparametre optimizasyonu.

* **Metodoloji:**

  * Veri ön işleme ve öznitelik korelasyonlarının incelenmesi.

  * Mesafe ve hiperdüzlem tabanlı bir algoritma olan SVM'in yüksek performans verebilmesi için özniteliklerin standardize edilmesi / normalize edilmesi.

  * Farklı çekirdek fonksiyonları (linear, rbf, poly) ve düzenlileştirme parametreleri ($C$, $\gamma$) üzerinde hiperparametre ayarlaması (GridSearchCV).

  * Karmaşıklık matrisi (Confusion Matrix) ve sınıflandırma raporu (Precision, Recall, F1-Score) ile hata analizi (hataların ağırlıklı olarak komşu sınıflar olan `1` ve `2` arasında gerçekleştiği gözlemlenmiştir).

* **Temel Bulgu:** Doğru ölçeklendirme ve hiperparametre optimizasyonu uygulandığında SVM algoritması bu veri setinde %95–%97 bandında yüksek genelleme başarısına ulaşmaktadır.

### 2.3. Melissa Monfared — *Mobile Price Prediction ▴ EDA & Classification*

* **URL:** `https://www.kaggle.com/code/melissamonfared/mobile-price-prediction-eda-classification`

* **Odak Noktası:** Kapsamlı görsel veri analizi (EDA), öznitelik mühendisliği (feature engineering) / seçimi ve modern sınıflandırıcıların uçtan uca pipeline içinde değerlendirilmesi.

* **Metodoloji:**

  * Tek değişkenli (univariate) ve çok değişkenli (multivariate) istatistiksel analizler; aykırı değer (outlier) ve sıfır değer kontrolleri (örneğin `px_height` veya `sc_w` sütunlarındaki $0$ değerlerinin tespiti).

  * Boyut/çözünürlük odaklı türetilmiş öznitelikler (ör. piksel alanı veya ekran alanı) ve öznitelik önem sıralamaları.

  * Ölçeklendirme (`StandardScaler` / `MinMaxScaler`) sonrası çoklu model eğitimi (Logistic Regression, SVM, Random Forest, Gradient Boosting / XGBoost, KNN) ve çapraz doğrulama (Cross-Validation).

* **Temel Bulgu:** Gürültülü veya düşük korelasyonlu ikili değişkenlerin (Bluetooth, 3G vb.) elenmesi veya boyut/ekran özelliklerinin birleştirilmesi model kararlılığını ve F1 skorlarını iyileştirmektedir.

## 3. Karşılaştırmalı Özet Tablosu

| Kriter | Vikram B. | Gülşah Demiryürek | Melissa Monfared | 
| ----- | ----- | ----- | ----- | 
| **Temel Yaklaşım** | Geniş Algoritma Kıyaslaması | Tek Model Derinlemesine Optimizasyon | Detaylı EDA + Pipeline Sınıflandırma | 
| **Ana Algoritma(lar)** | LR, SVM, KNN, RF, DT | Support Vector Machine (SVM) | LR, SVM, RF, KNN, Boosting | 
| **Ön İşleme / Ölçekleme** | Temel düzeyde | SVM için kritik (Standardizasyon) | Detaylı kontrol + Ölçekleme | 
| **Hiperparametre Ayarı** | Sınırlı / Varsayılan ağırlıklı | Kapsamlı (Kernel, $C$, $\gamma$) | Çapraz doğrulama destekli | 
| **Değerlendirme Metrikleri** | Accuracy | Accuracy, Confusion Matrix, F1 | Accuracy, Precision, Recall, F1, CV | 

## 4. İlgili Akademik Literatür Özeti

Bu Kaggle veri seti, makine öğrenmesi tabanlı fiyat segmentasyonu çalışmalarında standart bir kıyaslama (benchmark) veri seti olarak akademik literatürde yaygın biçimde kullanılmaktadır:

1. **Güvenç, E., Çetin, G., & Koçak, H. (2021).** *Comparison of KNN and DNN Classifiers Performance in Predicting Mobile Phone Price Ranges.* Artificial Intelligence Studies (AAIR), 1(1), 19–28.

   * **Bulgu:** K-En Yakın Komşu (KNN) ve Derin Sinir Ağları (DNN) karşılaştırılmış; doğrulama verisi üzerinde KNN **%93**, DNN ise **%94** sınıflandırma başarısı elde etmiştir.

2. **Nasser, I. M., & Al-Shawwa, M. (2019).** *Predicting Mobile Phone Price Range Using Artificial Neural Networks.* International Journal of Academic Information Systems Research (IJAISR), 3(2), 1–5.

   * **Bulgu:** Yapay Sinir Ağları (ANN) mimarisi ile 20 öznitelik üzerinden **%96.31** doğruluk oranına ulaşılmış ve `ram` değişkeninin en baskın belirleyici olduğu teyit edilmiştir.

3. **Al-Naymat, G., Al-Kasassbeh, M., & Al-Hawari, E. (2024).** *Optimizing Support Vector Machines for Mobile Phone Price Range Classification Using Hyperparameter Tuning.* Electronics, 13(14), 2784.

   * **Bulgu:** 5-katlı çapraz doğrulama (5-fold CV) ve hiperparametre optimizasyonu (HPO) ile desteklenen SVM modellerinin (`SVMOPT`), standart modellere (%90–%95) kıyasla **%97.50 – %97.67** doğruluk seviyesine çıkarak literatürdeki en yüksek sonuçları verdiği raporlanmıştır.

4. **Asim, M., & Khan, Z. (2018).** *Mobile Price Class Prediction Using Machine Learning Techniques.* International Journal of Computer Applications (IJCA), 179(29), 6–11.

   * **Bulgu:** 20 özniteliğin tamamı yerine ANOVA F-testi ve bilgi kazancı (Information Gain) ile seçilen en anlamlı 4–6 özniteliğin (`ram`, `battery_power`, `px_height`, `px_width`) kullanılması, özellikle ölçeklenmemiş SVM ve KNN modellerinde aşırı öğrenmeyi (overfitting) azaltarak test başarısını %9'a kadar artırabilmektedir.

## 5. Genel Literatür Çıkarımları (Sentez)

* **Dominant Değişken Etkisi:** Tüm Kaggle çalışmaları ve akademik makaleler, hedef sınıf varyansının büyük kısmının tek başına `ram` tarafından açıklandığı, ikincil belirleyicilerin ise `battery_power`, `px_width` ve `px_height` olduğu konusunda hemfikirdir.

* **Model Seçimi:** Veri seti 2.000 satırlık görece küçük ve doğrusal ayrılabilirliği yüksek bir yapıda olduğu için, **Lojistik Regresyon** ve **RBF/Linear Çekirdekli SVM**, karmaşık topluluk (ensemble) yöntemlerinden (Random Forest, XGBoost) daha yüksek veya onlara denk doğruluk sağlamaktadır.

* **Ölçeklendirme ve Özellik Seçimi:** Ağaç tabanlı olmayan modellerde (SVM, KNN, LR, MLP) `StandardScaler` kullanımı zorunludur. Düşük bilgi kazancına sahip ikili özniteliklerin (`blue`, `touch_screen`, `wifi` vb.) çıkarılması model genellemesini olumsuz etkilememekte, aksine gürültüyü azaltmaktadır.

## 6. Kaynakça

### Akademik Yayınlar

* Al-Naymat, G., Al-Kasassbeh, M., & Al-Hawari, E. (2024). Optimizing Support Vector Machines for Mobile Phone Price Range Classification Using Hyperparameter Tuning. *Electronics*, 13(14), 2784.

* Asim, M., & Khan, Z. (2018). Mobile Price Class Prediction Using Machine Learning Techniques. *International Journal of Computer Applications*, 179(29), 6–11.

* Güvenç, E., Çetin, G., & Koçak, H. (2021). Comparison of KNN and DNN Classifiers Performance in Predicting Mobile Phone Price Ranges. *Artificial Intelligence Studies*, 1(1), 19–28.

* Nasser, I. M., & Al-Shawwa, M. (2019). Predicting Mobile Phone Price Range Using Artificial Neural Networks. *International Journal of Academic Information Systems Research*, 3(2), 1–5.

### Veri Seti ve Kaggle Çalışmaları

* Sharma, A. (2018). *Mobile Price Classification* [Veri Seti]. Kaggle. `https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification`

* Vikram, B. (2018). *Mobile Price Prediction* [Kaggle Notebook]. `https://www.kaggle.com/code/vikramb/mobile-price-prediction`

* Demiryürek, G. (2020). *Mobile Price Classification with SVM* [Kaggle Notebook]. `https://www.kaggle.com/code/gulsahdemiryurek/mobile-price-classification-with-svm`

* Monfared, M. (2022). *Mobile Price Prediction ▴ EDA & Classification* [Kaggle Notebook]. `https://www.kaggle.com/code/melissamonfared/mobile-price-prediction-eda-classification`