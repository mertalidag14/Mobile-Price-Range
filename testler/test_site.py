# Siteyi tarayıcı olmadan çalıştırıp hata vermeden açıldığını kontrol ediyor (Streamlit'in AppTest aracı)
import pytest
from streamlit.testing.v1 import AppTest

from conftest import WEB


@pytest.fixture(scope="module")
def site():
    # İlk açılışta modeller eğitildiği için süre uzun tutuluyor
    return AppTest.from_file(str(WEB / "app.py"), default_timeout=900).run()


def test_site_hatasiz_aciliyor(site):
    assert not site.exception


def test_sekmeler_var(site):
    adlar = [s.label for s in site.tabs]
    for beklenen in ["Fiyat Tahmini", "Telefon Bul", "Telefon Kataloğu", "Karşılaştır", "Telefon Önerici", "Markalar",
                     "Hocayı Yen", "Hata Galerisi", "Proje Hakkında"]:
        assert any(beklenen in ad for ad in adlar), beklenen


def test_paylasma_linkiyle_tahmin_ve_shap():
    # Paylaş linkindeki gibi telefonu adresle seçince tahmin ve SHAP grafiği çıkmalı
    at = AppTest.from_file(str(WEB / "app.py"), default_timeout=900)
    at.query_params.update({"tel": "Apple iPhone 13", "ram": "4", "depo": "128", "model": "Random Forest"})
    at.run()
    assert not at.exception
    html = " ".join(str(m.value) for m in at.markdown)
    assert "tahmin-sonucu" in html
    assert "SHAP" in html
    assert "benzeyen 3 gerçek telefon" in html


def test_oyun_bir_tur_oynanabiliyor(site):
    # "Hocayı Yen" oyunu: başlat, bir segment seç, doğru cevap ve skor açılmalı
    site.button(key="oyun_basla").click().run()
    assert "oyun-kart" in " ".join(str(m.value) for m in site.markdown)
    site.button(key="oyun_s1").click().run()
    assert not site.exception
    html = " ".join(str(m.value) for m in site.markdown)
    assert "Doğru cevap" in html
    o = site.session_state["oyun"]
    assert o["cevap"] == 1 and 0 <= o["sen"] <= 1 and 0 <= o["yz"] <= 1


def test_hata_galerisi_dolu(site):
    html = " ".join(str(m.value) for m in site.markdown)
    assert html.count("hata-kart") > 10

