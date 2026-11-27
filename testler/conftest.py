# Testler web_arayuzu klasöründeki dosyaları (app.py, aciklama.py, zaneduscv) kullanıyor
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web_arayuzu"
sys.path.insert(0, str(WEB))
