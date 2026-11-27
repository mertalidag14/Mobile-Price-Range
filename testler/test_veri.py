# Veri setinin sitenin beklediği hâlde olup olmadığını kontrol ediyor
import pandas as pd
import pytest

from conftest import WEB

OZELLIKLER = ["ram_capacity", "internal_memory", "battery_capacity", "processor_speed", "refresh_rate",
              "primary_camera_rear", "resolution_height", "screen_size", "5G_or_not", "fast_charging"]


@pytest.fixture(scope="module")
def veri():
    return pd.read_csv(WEB / "gercek_telefonlar.csv")


def test_telefon_sayisi(veri):
    # 980 gerçek telefon (sitede en pahalı %1 çıkarılınca 970 kalıyor)
    assert len(veri) == 980


def test_modelin_kullandigi_sutunlar_var(veri):
    for sutun in OZELLIKLER + ["price", "brand_name", "model"]:
        assert sutun in veri.columns, sutun


def test_fiyatlar_pozitif(veri):
    assert (veri["price"] > 0).all()


def test_ayni_telefon_iki_kez_yok(veri):
    assert not veri["model"].duplicated().any()


def test_ceyreklere_bolunce_segmentler_dengeli(veri):
    # Sitedeki gibi: en pahalı %1 çıkar, fiyatı 4 çeyreğe böl
    veri = veri[veri["price"] <= veri["price"].quantile(0.99)]
    sayilar = pd.qcut(veri["price"], 4, labels=False).value_counts()
    assert sayilar.max() - sayilar.min() <= 0.1 * len(veri) / 4
