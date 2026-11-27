# TeknoTahmin - modelleri önceden eğitme
#
# Docker imajı hazırlanırken (Dockerfile) bir kez çalışıyor: siteyi tarayıcı olmadan baştan sona açıp bütün modelleri
# eğitiyor ve "hazir_modeller" klasörüne kaydediyor. Böylece Render'da site uykudan uyandığında modeller yeniden
# eğitilmiyor, diskten okunuyor ve açılış çok kısalıyor.
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

baslangic = time.time()
site = AppTest.from_file(str(Path(__file__).resolve().parent / "app.py"), default_timeout=1800).run()
if site.exception:
    raise SystemExit(f"Site açılırken hata: {site.exception}")
kayitlar = sorted(p.name for p in (Path(__file__).resolve().parent / "hazir_modeller").glob("*.joblib"))
print(f"{len(kayitlar)} model kaydedildi ({time.time() - baslangic:.0f} sn):", ", ".join(kayitlar))
