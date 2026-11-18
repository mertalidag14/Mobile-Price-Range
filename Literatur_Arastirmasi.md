Mobil Telefon Fiyat Aralığı Sınıflandırması

Problem Tanımı, Veri Seti Analizi, Literatür Bulguları ve Uygulama Planı

1.  Problem Tanımı

Amaç, telefonun gerçek fiyatını tahmin etmek değil; teknik
özelliklerinden fiyat aralığını dört sınıftan birine atamaktır. Bu,
denetimli çok sınıflı sınıflandırma problemidir.

Kullanım gerekçesi literatürde iki taraflıdır: üretici yeni modelin
hangi fiyat bandına oturacağını belirler, alıcı da istediği özelliklerin
hangi banda denk geldiğini görür (Hassan vd., 2025; Kalaivani vd., AIP).

2.  Veri Seti

Kaynak: Mobile Price Classification (Abhishek Sharma).

train.csv: 2000 telefon, 20 özellik + price_range

test.csv: Etiket yok; kamu notebook'ları çoğu zaman eğitim dosyasını
bölerek değerlendirir

Veri Kalitesi: Eksik değer yok; veri temiz yüklenmiş kabul edilir (Chen)

Sınıf Dengesi: Sınıflar dengelidir (her bant yaklaşık 500 kayıt).
Accuracy bu yüzden yanıltıcı değildir; yine de precision, recall ve F1
birlikte raporlanır

3.  Literatürde Ortak Sonuçlar

RAM fiyat bandını belirleyen baskın özelliktir. Sistematik derlemede 25
çalışmanın 18'inde (%72) en güçlü öngörücü RAM'dir; onu dahili bellek
(%60) ve batarya (%48) izler (Naaman, Ahmed ve Kareem, 2025). Korelasyon
analizlerinde RAM--price_range ilişkisi yaklaşık 0,92'ye kadar çıkar.
Bluetooth, Wi-Fi, dokunmatik ekran, 3G/4G ve çekirdek sayısı bu sette
fiyatı neredeyse ayırmaz.

En tutarlı klasik modeller SVM ve lojistik regresyondur. Aynı Kaggle
setinde tekrarlanan sonuçlar aşağıdaki tabloda özetlenmiştir:

KNN sonuçları çalışmadan çalışmaya 0,41 ile 0,93 arasında değişir; k ve
ölçeklemeye çok duyarlıdır. Ağaç tabanlı modeller bu sette SVM'nin
gerisinde kalır, çünkü ayrım büyük ölçüde RAM üzerinden doğrusala
yakındır.

Özellik seçimi SVM'yi belirgin iyileştirir. Ki-kare ile seçilen ilk 10
özellik: RAM, piksel yüksekliği, batarya, piksel genişliği, ağırlık,
dahili bellek, ekran genişliği, konuşma süresi, ön kamera, ekran
yüksekliği (Kalaivani vd.). Yalnızca en güçlü dörtlü (RAM, batarya,
px_width, px_height) ile SVM yaklaşık 9 puan kazanırken lojistik
regresyon neredeyse aynı kalır.

İki ayrı problem karıştırılmamalıdır. Aralık sınıflandırmasında metrik
accuracy ve F1'dir. Gerçek fiyat tahmini (regresyon) ayrı veri ve metrik
ister: R², MAE, RMSE. Hindistan e-ticaret verisinde Gradient Boosting R²
≈ 0,98; Pakistan ikinci el pazarında Random Forest %97. Bu sayılar
Kaggle sınıflandırma skorlarıyla kıyaslanmaz.

4.  Yöntemlerde Tekrar Eden Düzen

Bölme çoğunlukla 70/30 veya 80/20; Hassan vd. üçlü bölme kullanır.

Sayısal özellikler farklı ölçekte olduğu için SVM, KNN ve lojistik
regresyondan önce standartlaştırma uygulanır.

İkili özellikler zaten 0/1'dir; one-hot gerekmez.

px_height veya sc_w sıfır olan kayıtlar vardır; piksel alanı (px_height
× px_width) türetilmiş özellik olarak kullanılır.

four_g = 1 iken three_g neredeyse her zaman 1'dir; ikisi birlikte bilgi
tekrarı yaratır.

5.  Literatürdeki Boşluklar ve Kısıtlar

Veri seti marka, çıkış yılı, 5G, işlemci modeli ve işletim sistemi
içermez. Aynı RAM'e sahip iki telefon piyasada çok farklı fiyatta
olabilir.

Kayıtlar güncel pazar taraması değildir; modeller bu dosyada yüksek
doğruluk verse de yeni bir mağaza verisine aynı skorla taşınmaz (Naaman
vd.; Hassan vd.).

Çoğu çalışma aynı 2000 satırı tekrarlar. %96--97 doğruluk, problemin
zorluğundan çok RAM'in sınıfı neredeyse tek başına ayırmasından gelir.

Derin öğrenme ve ensemble yöntemler bu küçük sette az denenmiştir. Sinir
ağı yüksek skor verir, eğitim maliyeti ise klasik modellere göre
belirgin fazladır; bu yüzden literatürde payı düşüktür.

6.  Bu Araştırmadan Çıkarılacak Çalışma Planı

Yalnızca train.csv üzerinde çalış; hedef sütunu olmayan test.csv'yi skor
için kullanma.

Sınıf dengesini ve RAM ile price_range korelasyonunu grafikleştir.

Karşılaştırma seti: Lojistik regresyon, SVM, Random Forest, KNN. Metrik:
accuracy, macro F1, karmaşıklık matrisi.

Aynı modelleri tüm özelliklerle ve yalnızca RAM, batarya, piksel boyutu,
dahili bellek ile tekrarla.

Komşu sınıfların (1↔2) daha çok karıştığını, uç sınıfların (0 ve 3) RAM
sayesinde daha kolay ayrıldığını raporla.

7.  Kaynakça

8.  Sharma, A. Mobile Price Classification. Kaggle.
    www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

9.  Kalaivani, K. S. vd. Predicting the price range of mobile phones
    using machine learning techniques. AIP Conference Proceedings,
    2387, 140010. https://doi.org/10.1063/5.0068608

10. Hassan, A., Huq, S. M., Foraji, M. K. H., & Kabir, M. H. (2025).
    Study on the performance of supervised machine learning algorithms
    in mobile price range classification. International Journal of
    Computer Applications. https://doi.org/10.5120/ijca2025924768

11. Chen, M. Mobile phone price prediction with feature reduction.
    Highlights in Science, Engineering and Technology.
    drpress.org/ojs/index.php/HSET/article/download/5440/5260

12. Naaman, D. W., Ahmed, B. T., & Kareem, O. S. (2025). Comparative
    review of machine learning models for mobile price prediction based
    on specifications: A systematic literature analysis. Digital Zone,
    16(2). https://doi.org/10.31849/digitalzone.v16i2.27096

| Sınıf \| Anlam \|

| --- \| --- \|

| 0 \| Düşük maliyet \|

| 1 \| Orta maliyet \|

| 2 \| Yüksek maliyet \|

| 3 \| Çok yüksek maliyet \|

| Veri Setindeki 20 Teknik Özellik battery_power, blue, clock_speed,
  dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc,
  px_height, px_width, ram, sc_h, sc_w, talk_time, three_g,
  touch_screen, wifi. \|

| --- \|

| Çalışma \| En İyi Model \| Sonuç \|

| --- \| --- \| --- \|

| Kalaivani vd. (AIP Conf. Proc.) \| SVM \| Özellik seçimi öncesi %95,
  Ki-kare ile ilk 10 özellikten sonra %97. Random Forest %83→%87,
  lojistik regresyon %76→%81 \|

| Hassan vd. (IJCA, 2025) \| SVM ve lojistik regresyon \| F1 yaklaşık
  %97. Karar ağacı ve Naive Bayes yaklaşık %81, KNN yaklaşık %54. Bölme:
  %70 eğitim, %15 doğrulama, %15 test \|

| Chen (HSET) \| MLP + Pearson korelasyonu \| %95,8. Az bileşenli PCA
  hafif iyileştirir; çok özellik atılınca performans düşer \|

| Naaman vd. derlemesi \| SVM en sık kullanılan algoritma \| Ortalama
  doğruluk yaklaşık %89,7. Sinir ağları uygulandığı çalışmalarda
  yaklaşık %95 civarı, fakat nadir kullanılır \|
