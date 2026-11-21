# Kaggle Mobile Price Classification Çalışmaları: Ön Veri İşleme Karşılaştırması

Bu çalışma, Kaggle üzerindeki **"Mobile Price Classification"** veri setini kullanan üç farklı notebook'un uyguladığı ön veri işleme (data preprocessing) stratejilerini karşılaştırmaktadır.

---

## Ön Veri İşleme Adımları Karşılaştırma Tablosu

| Ön İşleme Adımı | 1. Vikram B. (`mobile-price-prediction`) | 2. Gülşah Demiryürek (`mobile-price-classification-with-svm`) | 3. Melissa Monfared (`mobile-price-prediction-eda-classification`) |
| :--- | :--- | :--- | :--- |
| **Veri Seti Yapısı** | 2000 satır, 21 sütun (20 öznitelik + `price_range` hedef değişkeni) | 2000 satır, 21 sütun (20 öznitelik + `price_range` hedef değişkeni) | 2000 satır, 21 sütun (20 öznitelik + `price_range` hedef değişkeni) |
| **Eksik Veri (Missing Values) Yönetimi** | Eksik değer kontrolü (`isnull().sum()`) yapılmıştır; veri setinde eksik değer bulunmadığı için doldurma veya silme uygulanmamıştır. | Eksik değer kontrolü yapılmıştır; temiz veri seti olduğu için herhangi bir tamamlama (imputation) işlemi uygulanmamıştır. | Eksik değer ve tekrarlayan kayıt (duplicate) kontrolü yapılmış, veri setinin temiz olduğu doğrulanmıştır. |
| **Aykırı Değer (Outlier) ve Mantıksal Hata Temizliği** | Dağılım grafikleri (box plot / histogram) ile aykırı değerler incelenmiş, doğrudan satır silme işlemi yapılmamıştır. | Aykırı değer silme işlemi uygulanmamış, tüm veri örnekleri korunmuştur. | Kutu grafikleri (boxplot) ile dağılımlar incelenmiş; `px_height` ve `sc_w` gibi sütunlardaki sıfır (0) değerleri analiz edilmiştir. |
| **Özellik Mühendisliği (Feature Engineering)** | Mevcut özelliklerin kombinasyonları ve **PCA (Temel Bileşenler Analizi)** uygulanarak hem ham veri hem de dönüştürülmüş veri karşılaştırılmıştır. | Ek özellik türetilmemiş, orijinal 20 öznitelik doğrudan kullanılmıştır. | Görselleştirme ve korelasyon odaklı analiz yapılmış, yüksek korelasyonlu (`ram`, `battery_power`, `px_width`, `px_height`) değişkenler öne çıkarılmıştır. |
| **Özellik Seçimi (Feature Selection)** | **ANOVA F-testi** ve korelasyon analizi ile en etkili öznitelikler belirlenmiş, model performansları hem tüm özelliklerle hem seçilmiş özelliklerle test edilmiştir. | Korelasyon matrisi incelenmiş ancak model aşamasında tüm özellikler korunmuştur. | Korelasyon ısı haritası (heatmap) ile hedef değişkene etki eden temel değişkenler belirlenmiştir. |
| **Ölçeklendirme (Feature Scaling)** | Mesafe ve hiperdüzlem tabanlı modeller (özellikle SVM ve KNN) için standartlaştırma/ölçeklendirme uygulanmıştır. | **SVM** algoritmasının mesafe/çekirdek duyarlılığı nedeniyle veriler ölçeklendirilmiştir (`MinMaxScaler` / `StandardScaler`). | Farklı sınıflandırma algoritmalarının (Logistic Regression, SVM, Tree-based modeller) karşılaştırılması sürecinde ölçeklendirme adımı kullanılmıştır. |
| **Kategorik Değişken Kodlama (Encoding)** | Tüm özellikler halihazırda sayısal veya ikili (0/1) formatta olduğu için ek One-Hot/Label Encoding gerekmemiştir. | İkili (binary) ve sayısal sütunlar hazır olduğu için ek encoding yapılmamıştır. | Değişken tipleri (`int64`, `float64`) kontrol edilmiş, ek encoding işlemine ihtiyaç duyulmamıştır. |
| **Veri Bölme (Train-Test Split)** | Eğitim ve test seti olarak ayrılmıştır (%80 eğitim - %20 test / %75 - %25). | `train_test_split` ile eğitim ve test kümelerine ayrılmıştır (`random_state` sabiti kullanılarak). | Eğitim ve doğrulama/test setlerine bölünerek modellerin genelleme başarısı ölçülmüştür. |

---

## Sonuç ve Değerlendirme

1. **Veri Setinin Doğası:** İncelenen üç çalışmada da kullanılan *Mobile Price Classification* veri seti, halihazırda eksik değer içermeyen ve tüm sütunları sayısal/ikili (0-1) yapıda olan temiz bir veri setidir. Bu nedenle çalışmaların hiçbirinde ağır veri temizliği (imputation veya encoding) gerekmemiştir.
2. **Algoritma Odaklı Ön İşleme Farkları:**
   * **Gülşah Demiryürek**, doğrudan **Destek Vektör Makineleri (SVM)** odaklı ilerlediği için özellikle özellik ölçeklendirme (scaling) adımına odaklanmış ve yüksek doğruluk elde etmiştir.
   * **Vikram B.**, ön işlemeyi en kapsamlı tutan çalışmadır; **PCA** ve **ANOVA F-testi** gibi boyut indirgeme/özellik seçimi tekniklerini deneyerek ham veri ile işlenmiş verinin model başarılarını (özellikle SVC ve ensemble modellerde) kıyaslamıştır.
   * **Melissa Monfared** ise keşifsel veri analizi (EDA) ve görselleştirme ağırlıklı bir ön inceleme yaparak özellikle `ram` ve ekran çözünürlüğü özelliklerinin fiyat aralığı (`price_range`) üzerindeki belirleyici etkisini ortaya koymuş ve çoklu sınıflandırma modellerini karşılaştırmıştır.
3. **Genel Çıkarım:** Bu veri setinde en yüksek sınıflandırma başarısı (%95 ve üzeri doğruluk oranları), verilerin uygun şekilde **ölçeklendirilmesi (scaling)** ve ardından **SVM (SVC)** veya **Lojistik Regresyon** gibi doğrusal/çekirdek tabanlı sınır çizen algoritmaların kullanılmasıyla elde edilmektedir.

---

## Kaynakça

1. **Vikram B.** *Mobile Price Prediction*. Kaggle Notebook. Erişim adresi: [https://www.kaggle.com/code/vikramb/mobile-price-prediction](https://www.kaggle.com/code/vikramb/mobile-price-prediction)
2. **Demiryürek, Gülşah.** *Mobile Price Classification with SVM*. Kaggle Notebook. Erişim adresi: [https://www.kaggle.com/code/gulsahdemiryurek/mobile-price-classification-with-svm](https://www.kaggle.com/code/gulsahdemiryurek/mobile-price-classification-with-svm)
3. **Monfared, Melissa.** *Mobile Price Prediction ▴ EDA & Classification*. Kaggle Notebook. Erişim adresi: [https://www.kaggle.com/code/melissamonfared/mobile-price-prediction-eda-classification](https://www.kaggle.com/code/melissamonfared/mobile-price-prediction-eda-classification)
4. **Sharma, Abhishek.** *Mobile Price Classification Dataset*. Kaggle Datasets. Erişim adresi: [https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)