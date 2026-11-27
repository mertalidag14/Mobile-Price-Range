# Sitedeki SHAP grafiğinin hesabını (aciklama.py) kontrol ediyor
import numpy as np
import pytest

from aciklama import shapley_degerleri


def test_lineer_modelde_katki_agirlik_carpi_fark():
    # Lineer bir modelde her özelliğin katkısı tam olarak ağırlık * (değer - taban) olmalı
    agirliklar = np.array([2.0, -1.0, 0.5, 0.0])
    x, taban = np.array([3.0, 1.0, 4.0, 9.0]), np.array([1.0, 1.0, 0.0, 2.0])
    _, katkilar, _ = shapley_degerleri(lambda X: X @ agirliklar, x, taban)
    assert np.allclose(katkilar, agirliklar * (x - taban))


def test_katkilarin_toplami_farka_esit():
    # Etkileşimli (lineer olmayan) bir modelde de toplam = f(x) - f(taban)
    def f(X):
        return np.sin(X[:, 0]) * X[:, 1] + X[:, 2] ** 2 - X[:, 0] * X[:, 3]

    rng = np.random.default_rng(0)
    x, taban = rng.normal(size=5), rng.normal(size=5)
    bas, katkilar, son = shapley_degerleri(f, x, taban)
    assert katkilar.sum() == pytest.approx(son - bas)


def test_etkisiz_ozelligin_katkisi_sifir():
    _, katkilar, _ = shapley_degerleri(lambda X: X[:, 0] * X[:, 1], [2.0, 3.0, 7.0], [0.0, 0.0, 0.0])
    assert katkilar[2] == pytest.approx(0)


def test_simetrik_ozellikler_esit_katki_alir():
    _, katkilar, _ = shapley_degerleri(lambda X: X[:, 0] * X[:, 1], [2.0, 2.0], [0.0, 0.0])
    assert katkilar[0] == pytest.approx(katkilar[1])


def test_gercek_modelle_olasilik_katkilari():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, random_state=0)
    mdl = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    bas, katkilar, son = shapley_degerleri(lambda A: mdl.predict_proba(A)[:, 1], X[0], np.median(X, axis=0))
    assert son == pytest.approx(mdl.predict_proba(X[:1])[0, 1])
    assert katkilar.sum() == pytest.approx(son - bas)
