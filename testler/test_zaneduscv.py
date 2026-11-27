# Sitede kullanılan ZANEDUSCV algoritmasının (Mert Ali Dağ) temel davranışlarını kontrol ediyor
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, train_test_split

from zaneduscv import ZANEDUSCVClassifier, ZANEDUSCVRegressor


def test_siniflandirici_ogreniyor():
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5, n_classes=4,
                               n_clusters_per_class=1, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    mdl = ZANEDUSCVClassifier(random_state=0).fit(X_tr, y_tr)
    assert mdl.score(X_te, y_te) > 0.7


def test_olasiliklarin_toplami_bir():
    X, y = make_classification(n_samples=200, n_features=5, n_classes=3, n_informative=3, random_state=1)
    olas = ZANEDUSCVClassifier(random_state=0).fit(X, y).predict_proba(X)
    assert olas.shape == (200, 3)
    assert np.allclose(olas.sum(axis=1), 1)
    assert (olas >= 0).all()


def test_regresyon_ogreniyor():
    X, y = make_regression(n_samples=400, n_features=6, noise=5, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    assert ZANEDUSCVRegressor(random_state=0).fit(X_tr, y_tr).score(X_te, y_te) > 0.8


def test_scikit_learn_ile_uyumlu():
    # clone ve cross_val_score çalışıyorsa sklearn araçlarıyla (GridSearchCV, ensemble) kullanılabiliyor
    X, y = make_classification(n_samples=200, n_features=5, random_state=2)
    mdl = clone(ZANEDUSCVClassifier(random_state=0))
    skorlar = cross_val_score(mdl, X, y, cv=3)
    assert len(skorlar) == 3 and skorlar.mean() > 0.7


def test_ayni_tohumla_ayni_sonuc():
    X, y = make_classification(n_samples=150, n_features=5, random_state=3)
    a = ZANEDUSCVClassifier(random_state=0).fit(X, y).predict_proba(X)
    b = ZANEDUSCVClassifier(random_state=0).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)
