# TeknoTahmin - SHAP (Shapley değerleri) hesabı
#
# Bir tahminde her özelliğin ne kadar etkili olduğunu bulmak için "ortalama telefon"dan başlayıp
# özellikleri bu telefonun değerleriyle değiştiriyoruz. n özellik için 2^n kombinasyonun hepsi denendiği için
# (10 özellik -> 1024 satır) değerler yaklaşık değil, tam hesaplanıyor. Yöntem modelden bağımsız:
# predict_proba'sı olan her model (Random Forest, KNN, XGBoost, ZANEDUSCV, ortak karar) açıklanabiliyor.
from math import factorial

import numpy as np


def shapley_degerleri(tahmin_fonksiyonu, x, taban):
    """
    tahmin_fonksiyonu: (k, n) boyutlu diziyi alıp k tane sayı döndüren fonksiyon (ör. bir segmentin olasılığı)
    x: açıklanan telefonun n özelliği
    taban: karşılaştırma noktası ("ortalama telefon"), n özellik

    Döndürür: (tabandaki tahmin, her özelliğin katkısı, x'teki tahmin).
    Katkıların toplamı her zaman (x'teki tahmin - tabandaki tahmin) değerine eşittir.
    """
    x = np.asarray(x, dtype=float)
    taban = np.asarray(taban, dtype=float)
    n = len(x)
    # k. satırın j. biti 1 ise j. özellik x'ten, 0 ise tabandan alınıyor
    maskeler = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    boyut = maskeler.sum(axis=1)
    agirlik = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    f = np.asarray(tahmin_fonksiyonu(np.where(maskeler, x, taban)), dtype=float)
    katkilar = np.zeros(n)
    for i in range(n):
        olmayan = np.nonzero(~maskeler[:, i])[0]  # i. özelliğin eklenmediği kombinasyonlar
        katkilar[i] = np.sum(agirlik[boyut[olmayan]] * (f[olmayan | (1 << i)] - f[olmayan]))
    return float(f[0]), katkilar, float(f[-1])
