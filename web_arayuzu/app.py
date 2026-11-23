# ============================================================
# TEKNOTAHMİN - GERÇEK TELEFONLARLA FİYAT SEGMENTİ TAHMİNİ (Streamlit)
# Veri: 970 gerçek akıllı telefon, Smartprix (Hindistan) fiyatları
# Çalıştırma: streamlit run app.py
# ============================================================
import base64
import math
import random
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from logolar import LOGOLAR

TABAN = Path(__file__).resolve().parent
SITE_ADI = "TeknoTahmin"
# Sitedeki modelin kullandığı 10 donanım özelliği
OZELLIKLER = ["ram_capacity", "internal_memory", "battery_capacity", "processor_speed", "refresh_rate",
              "primary_camera_rear", "resolution_height", "screen_size", "5G_or_not", "fast_charging"]
# (ad, ikon, renk)
SEGMENTLER = {
    0: ("Düşük fiyat", "💸", "#34d399"),
    1: ("Orta fiyat", "💰", "#60a5fa"),
    2: ("Yüksek fiyat", "💎", "#a78bfa"),
    3: ("Çok yüksek fiyat", "👑", "#f472b6"),
}
# Veri setindeki küçük harfli marka adlarının düzgün yazımı
MARKA_ADI = {"oneplus": "OnePlus", "iqoo": "iQOO", "lg": "LG", "zte": "ZTE", "poco": "POCO", "oppo": "OPPO",
             "tcl": "TCL", "blu": "BLU", "cat": "CAT", "jio": "Jio", "lyf": "LYF", "leeco": "LeEco", "letv": "LeTV"}

st.set_page_config(page_title=f"{SITE_ADI} - Telefon Fiyat Tahmini", page_icon="📱", layout="wide")


# Veri setindeki fiyatlar Hindistan Rupisi (₹). Sitede TL'ye çevrilerek gösteriliyor.
KUR = 0.54  # 1 ₹ ≈ 0,54 TL (XE orta kur, 19 Kasım 2026)


def rupi(x):
    return f"₹{x:,.0f}".replace(",", ".")


def tl(x):
    # ₹ cinsinden fiyatı TL'ye çevirip mağaza tarzı yazar (ör. 35.099 TL)
    return f"{x * KUR:,.0f}".replace(",", ".") + " TL"


def logo(marka, boyut=16):
    # Yazı şeklindeki logolar (Samsung, Sony...) ismi zaten gösterdiği için biraz daha küçük yükseklikte
    bilgi = LOGOLAR.get(marka.lower())
    if not bilgi:
        return ""
    yol, x, y, w, h, yazi = bilgi
    boy = boyut * (0.85 if yazi else 1)
    return (f'<svg width="{boy * w / h:.0f}" height="{boy:.0f}" viewBox="{x} {y} {w} {h}" aria-hidden="true">'
            f'<path d="{yol}"/></svg>')


# ------------------------------------------------------------
# Arka plan: markasız, çizilmiş telefonlar (logo kullanılmıyor)
# ------------------------------------------------------------
@st.cache_data
def arka_plan_svg():
    rnd = random.Random(7)
    renkler = [("#7c3aed", "#db2777"), ("#2563eb", "#06b6d4"), ("#db2777", "#f59e0b"),
               ("#059669", "#2563eb"), ("#9333ea", "#2563eb"), ("#e11d48", "#7c3aed")]
    parcalar = ['<rect width="1600" height="1000" fill="url(#zemin)"/>']
    tanimlar = ['<linearGradient id="zemin" x1="0" y1="0" x2="1" y2="1">'
                '<stop offset="0" stop-color="#0f1029"/><stop offset="1" stop-color="#1b0f2e"/></linearGradient>']
    # Telefonların arkasında renkli ışık parlamaları
    for j, (renk, x, y, yaricap) in enumerate([("#7c3aed", 260, 220, 420), ("#db2777", 1350, 260, 380),
                                               ("#2563eb", 900, 820, 460), ("#06b6d4", 120, 860, 300),
                                               ("#f59e0b", 1500, 900, 260)]):
        tanimlar.append(f'<radialGradient id="isik{j}"><stop offset="0" stop-color="{renk}" stop-opacity=".55"/>'
                        f'<stop offset="1" stop-color="{renk}" stop-opacity="0"/></radialGradient>')
        parcalar.append(f'<circle cx="{x}" cy="{y}" r="{yaricap}" fill="url(#isik{j})"/>')
    for i in range(16):
        r1, r2 = renkler[i % len(renkler)]
        tanimlar.append(f'<linearGradient id="g{i}" x1="0" y1="0" x2="1" y2="1">'
                        f'<stop offset="0" stop-color="{r1}"/><stop offset="1" stop-color="{r2}"/></linearGradient>')
        en = rnd.randint(120, 230)
        boy = int(en * 2.05)
        x = (i % 8) * 205 + rnd.randint(-50, 50)
        y = (i // 8) * 500 + rnd.randint(-120, 160)
        aci = rnd.randint(-30, 30)
        r = en * 0.16
        parcalar.append(
            f'<g transform="translate({x},{y}) rotate({aci} {en / 2} {boy / 2})">'
            f'<rect width="{en}" height="{boy}" rx="{r}" fill="#0b0b14" stroke="url(#g{i})" stroke-width="5"/>'
            f'<rect x="{en * 0.06}" y="{en * 0.06}" width="{en * 0.88}" height="{boy - en * 0.12}" '
            f'rx="{r * 0.8}" fill="url(#g{i})" opacity="0.85"/>'
            f'<rect x="{en * 0.38}" y="{en * 0.1}" width="{en * 0.24}" height="{en * 0.07}" rx="{en * 0.035}" fill="#0b0b14"/>'
            f'</g>'
        )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid slice">'
           f'<defs>{"".join(tanimlar)}</defs>{"".join(parcalar)}</svg>')
    return base64.b64encode(svg.encode()).decode()


@st.cache_data
def ikon_katmani_svg():
    # Bulanık telefonların üstünde duran net, soluk küçük süsler: yıldız, kamera merceği, şimşek, sinyal, pil, çip
    rnd = random.Random(21)
    sekiller = {
        "yildiz": lambda s: f'<path d="M0 {-s} Q0 0 {s} 0 Q0 0 0 {s} Q0 0 {-s} 0 Q0 0 0 {-s}Z" fill="#fff"/>',
        "mercek": lambda s: (f'<circle r="{s}" fill="none" stroke="#fff" stroke-width="2"/>'
                             f'<circle r="{s * .45}" fill="none" stroke="#fff" stroke-width="2"/>'),
        "simsek": lambda s: (f'<path d="M{s * .15} {-s} L{-s * .55} {s * .1} L{-s * .05} {s * .1} L{-s * .2} {s} '
                             f'L{s * .55} {-s * .15} L{s * .05} {-s * .15}Z" fill="none" stroke="#fff" stroke-width="2" '
                             f'stroke-linejoin="round"/>'),
        "sinyal": lambda s: "".join(f'<rect x="{-s + i * s * .7}" y="{s - (i + 1) * s * .5}" width="{s * .45}" '
                                    f'height="{(i + 1) * s * .5}" rx="2" fill="#fff"/>' for i in range(3)),
        "pil": lambda s: (f'<rect x="{-s}" y="{-s * .5}" width="{s * 1.8}" height="{s}" rx="4" fill="none" '
                          f'stroke="#fff" stroke-width="2"/><rect x="{s * .85}" y="{-s * .2}" width="{s * .25}" '
                          f'height="{s * .4}" rx="1" fill="#fff"/><rect x="{-s * .75}" y="{-s * .28}" '
                          f'width="{s * .9}" height="{s * .56}" rx="2" fill="#fff"/>'),
        "cip": lambda s: (f'<rect x="{-s * .6}" y="{-s * .6}" width="{s * 1.2}" height="{s * 1.2}" rx="4" fill="none" '
                          f'stroke="#fff" stroke-width="2"/>'
                          + "".join(f'<line x1="{k}" y1="{-s}" x2="{k}" y2="{-s * .6}" stroke="#fff" stroke-width="2"/>'
                                    f'<line x1="{k}" y1="{s * .6}" x2="{k}" y2="{s}" stroke="#fff" stroke-width="2"/>'
                                    for k in (-s * .3, 0, s * .3))),
        "nokta": lambda s: f'<circle r="{s * .25}" fill="#fff"/>',
    }
    adlar = list(sekiller)
    parcalar = []
    for i in range(46):
        ad = adlar[i % len(adlar)]
        x, y = rnd.randint(20, 1580), rnd.randint(20, 980)
        boyut = rnd.randint(9, 20) if ad != "nokta" else rnd.randint(8, 14)
        saydam = rnd.uniform(.08, .2) if ad != "yildiz" else rnd.uniform(.18, .35)
        parcalar.append(f'<g transform="translate({x},{y}) rotate({rnd.randint(-25, 25)})" opacity="{saydam:.2f}">'
                        f'{sekiller[ad](boyut)}</g>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid slice">'
           f'{"".join(parcalar)}</svg>')
    return base64.b64encode(svg.encode()).decode()


def telefon_ikonu(renk, boyut=64):
    return (f'<svg width="{boyut * 0.55:.0f}" height="{boyut}" viewBox="0 0 55 100">'
            f'<rect x="2" y="2" width="51" height="96" rx="10" fill="#0b0b14" stroke="{renk}" stroke-width="3"/>'
            f'<rect x="6" y="6" width="43" height="88" rx="7" fill="{renk}" opacity=".75"/>'
            f'<rect x="21" y="9" width="13" height="4" rx="2" fill="#0b0b14"/></svg>')


st.markdown(f"""
<style>
.stApp {{ background: #0f1029; }}
.stApp::before {{
content: ""; position: fixed; inset: -60px; z-index: 0;
background: url("data:image/svg+xml;base64,{arka_plan_svg()}") center / cover no-repeat;
filter: blur(5px) saturate(1.2);
}}
.stApp::after {{
content: ""; position: fixed; inset: -30px; z-index: 0; pointer-events: none;
background:
  url("data:image/svg+xml;base64,{ikon_katmani_svg()}") center / cover no-repeat,
  radial-gradient(circle, rgba(255,255,255,.05) 1px, transparent 1.5px) 0 0 / 26px 26px,
  radial-gradient(ellipse at 50% 40%, rgba(10,10,25,.2), rgba(10,10,25,.7));
}}
/* Not: arka plan bilerek hareketsiz. Bulanık cam kutunun arkası hareket edince tarayıcı her karede yeniden
   bulanıklaştırıyor ve sayfa kasıyordu. */
.block-container::before {{
content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1.5px; pointer-events: none;
background: linear-gradient(135deg, rgba(167,139,250,.7), rgba(244,114,182,.25) 40%, rgba(96,165,250,.6));
-webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite: xor; mask-composite: exclude;
}}
.logo {{ background-size: 200% auto !important; animation: parilti 6s linear infinite; }}
@keyframes parilti {{ to {{ background-position: 200% center; }} }}
@media (prefers-reduced-motion: reduce) {{ .logo {{ animation: none; }} }}
[data-testid="stAppViewContainer"] {{ position: relative; z-index: 1; background: transparent; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{
position: relative; max-width: 1150px; margin: 2.5rem auto; padding: 2rem 2.5rem 2rem !important;
background: rgba(18, 18, 38, .6); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
border: 1px solid rgba(255,255,255,.12); border-radius: 28px; box-shadow: 0 30px 80px rgba(0,0,0,.45);
}}
.nav {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .8rem;
padding-bottom: 1rem; margin-bottom: 1.2rem; border-bottom: 1px solid rgba(255,255,255,.1); }}
.logo {{ font-size: 1.6rem; font-weight: 900; letter-spacing: -.02em;
background: linear-gradient(90deg, #c4b5fd, #f9a8d4, #93c5fd, #c4b5fd); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.logo-alt {{ color: rgba(255,255,255,.6); font-size: .85rem; }}
.cipler {{ display: flex; gap: .5rem; flex-wrap: wrap; }}
.cip {{ padding: .3rem .8rem; border-radius: 999px; font-size: .82rem; color: #fff;
background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); }}
.baslik {{ font-size: 2rem; font-weight: 800; color: #fff; margin: .2rem 0 .3rem; }}
.alt {{ color: rgba(255,255,255,.72); font-size: 1rem; margin-bottom: 1rem; }}
.bolum {{ font-weight: 700; font-size: 1.05rem; margin: .4rem 0 .2rem; color: #fff; }}
.sonuc {{ padding: 1.4rem 1.6rem; border-radius: 22px;
background: linear-gradient(135deg, color-mix(in srgb, var(--renk) 28%, transparent), rgba(255,255,255,.04));
border: 1px solid color-mix(in srgb, var(--renk) 55%, transparent); }}
.sonuc .ust {{ color: rgba(255,255,255,.7); font-size: .85rem; letter-spacing: .04em; text-transform: uppercase; }}
.sonuc .ana {{ font-size: 1.9rem; font-weight: 800; color: #fff; margin: .2rem 0; }}
.sonuc .fiyat {{ font-size: 2.3rem; font-weight: 900; color: var(--renk); line-height: 1.1; margin-top: .6rem; }}
.sonuc .aralik {{ color: rgba(255,255,255,.75); font-size: .95rem; }}
.sonuc .guven {{ color: var(--renk); font-weight: 700; }}
.cubuk {{ display: grid; grid-template-columns: 140px 1fr 46px; align-items: center; gap: .7rem; margin: .45rem 0; }}
.cubuk .ad {{ color: rgba(255,255,255,.85); font-size: .9rem; }}
.cubuk .iz {{ height: 10px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; }}
.cubuk .dolu {{ height: 100%; border-radius: 99px; }}
.cubuk .yuzde {{ text-align: right; color: #fff; font-variant-numeric: tabular-nums; font-size: .9rem; }}
.izgara {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; margin-top: .6rem; }}
.kart {{ position: relative; padding: 1.1rem; border-radius: 20px; background: rgba(255,255,255,.05);
border: 1px solid rgba(255,255,255,.1); transition: transform .15s, border-color .15s; }}
.kart:hover {{ transform: translateY(-3px); border-color: color-mix(in srgb, var(--renk) 70%, transparent); }}
.kart .ust {{ display: flex; gap: .9rem; align-items: center; }}
.kart .isim {{ color: #fff; font-weight: 800; font-size: 1.05rem; }}
.rozet {{ display: inline-block; margin-top: .25rem; padding: .12rem .55rem; border-radius: 99px; font-size: .72rem;
font-weight: 700; color: var(--renk); background: color-mix(in srgb, var(--renk) 16%, transparent);
border: 1px solid color-mix(in srgb, var(--renk) 45%, transparent); }}
.kart .fiyat {{ font-size: 1.45rem; font-weight: 900; color: #fff; margin: .8rem 0 .5rem; }}
.kart ul {{ list-style: none; padding: 0; margin: 0; color: rgba(255,255,255,.78); font-size: .84rem; }}
.kart li {{ margin: .18rem 0; }}
.ozellikler {{ display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .7rem; }}
.ozellik {{ font-size: .7rem; padding: .1rem .45rem; border-radius: 6px; background: rgba(255,255,255,.08); color: rgba(255,255,255,.8); }}
.ozellik.yok {{ opacity: .35; text-decoration: line-through; }}
.tablo {{ width: 100%; border-collapse: collapse; margin-top: 1rem; color: #fff; font-size: .95rem; }}
.tablo th, .tablo td {{ padding: .6rem .8rem; border-bottom: 1px solid rgba(255,255,255,.08); text-align: center; }}
.tablo th:first-child, .tablo td:first-child {{ text-align: left; color: rgba(255,255,255,.65); }}
.tablo td.kazanan {{ color: #34d399; font-weight: 800; }}
.markalar {{ display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; margin: -.4rem 0 1.3rem; }}
.markalar .etiket {{ color: rgba(255,255,255,.55); font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; margin-right: .2rem; }}
.marka svg, .isim svg {{ fill: #fff; flex: none; }}
.isim {{ display: flex; align-items: center; gap: .4rem; }}
.marka {{ display: inline-flex; align-items: center; gap: .45rem; padding: .35rem .9rem; border-radius: 12px; font-weight: 800; font-size: .95rem; letter-spacing: .01em; color: #fff;
background: linear-gradient(135deg, rgba(167,139,250,.18), rgba(96,165,250,.10)); border: 1px solid rgba(255,255,255,.14); }}
.marka small {{ font-weight: 600; color: rgba(255,255,255,.55); margin-left: .35rem; font-size: .72rem; }}
.mini-izgara {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: .6rem; margin-top: .4rem; }}
.mini-kart {{ display: flex; gap: .7rem; align-items: center; padding: .6rem .7rem; border-radius: 14px;
background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); }}
.mini-kart:hover {{ border-color: color-mix(in srgb, var(--renk) 70%, transparent); }}
.mini-isim {{ font-weight: 800; color: #fff; font-size: .9rem; display: flex; gap: .3rem; align-items: center; line-height: 1.2; }}
.mini-isim svg {{ fill: #fff; flex: none; }}
.mini-bilgi {{ font-size: .74rem; color: rgba(255,255,255,.7); margin-top: .15rem; }}
.mini-fiyat {{ font-weight: 800; color: #fff; font-size: .95rem; margin-top: .2rem; }}
.mini-fiyat .rozet {{ font-size: .62rem; margin-left: .3rem; vertical-align: middle; }}
.not {{ color: rgba(255,255,255,.5); font-size: .78rem; text-align: center; margin-top: 1.6rem; }}
.model-tablo {{ margin-top: .3rem; font-size: .88rem; }}
.model-tablo th, .model-tablo td {{ padding: .45rem .6rem; }}
.model-tablo tr.secili td {{ background: rgba(167,139,250,.14); }}
.butce-puan {{ position: absolute; top: .8rem; right: .8rem; padding: .2rem .55rem; border-radius: 10px; font-weight: 900;
font-size: .85rem; color: #0f1029; background: linear-gradient(135deg, #c4b5fd, #f9a8d4); }}
.butce-neden {{ margin-top: .5rem; font-size: .78rem; color: rgba(255,255,255,.75); }}
.firsat {{ display: inline-block; margin-top: .35rem; padding: .1rem .5rem; border-radius: 99px; font-size: .7rem; font-weight: 800;
color: #34d399; background: rgba(52,211,153,.12); border: 1px solid rgba(52,211,153,.45); }}
.st-key-marka_seridi {{ align-items: center; flex-wrap: wrap; margin: -.4rem 0 1rem; }}
.etiket-marka {{ color: rgba(255,255,255,.55); font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; white-space: nowrap; }}
.st-key-marka_seridi button {{ border-radius: 12px; padding: .3rem .85rem; min-height: 0; color: #fff;
background: linear-gradient(135deg, rgba(167,139,250,.18), rgba(96,165,250,.10)); border: 1px solid rgba(255,255,255,.14);
transition: transform .15s, border-color .15s; }}
.st-key-marka_seridi button:hover {{ transform: translateY(-2px); border-color: rgba(196,181,253,.7); color: #fff; }}
.st-key-marka_seridi button p {{ font-weight: 800; font-size: .92rem; white-space: nowrap; }}
[data-testid="stButtonGroup"] [role="radiogroup"] {{ flex-wrap: wrap; overflow: visible; }}
a.marka:hover {{ transform: translateY(-2px); border-color: rgba(196,181,253,.7); }}
.cetvel {{ margin: 1rem 0 .4rem; }}
.cetvel .satir {{ position: relative; height: 1.25rem; }}
.cetvel .isaret {{ position: absolute; transform: translateX(-50%); font-size: .75rem; font-weight: 800; color: #fff; white-space: nowrap; }}
.cetvel .isaret.gercek {{ color: #34d399; }}
.cetvel .serit {{ display: flex; height: 12px; border-radius: 99px; overflow: hidden; }}
.cetvel .serit span {{ flex: 1; opacity: .9; }}
.cetvel .sinirlar {{ position: relative; height: 1.3rem; margin-top: .25rem; }}
.cetvel .sinirlar small {{ position: absolute; transform: translateX(-50%); color: rgba(255,255,255,.6); font-size: .7rem; white-space: nowrap; }}
.neden {{ margin-top: 1rem; padding: .8rem 1rem; border-radius: 14px; background: rgba(0,0,0,.18); border: 1px solid rgba(255,255,255,.1); }}
.neden-baslik {{ font-weight: 800; color: #fff; margin-bottom: .3rem; }}
.neden ul {{ list-style: none; padding: 0; margin: 0; }}
.neden li {{ display: flex; justify-content: space-between; gap: .6rem; padding: .3rem 0; color: rgba(255,255,255,.88); font-size: .9rem;
border-bottom: 1px dashed rgba(255,255,255,.08); }}
.neden li:last-child {{ border-bottom: 0; }}
.neden li i {{ color: rgba(255,255,255,.55); font-style: normal; }}
.neden .etki {{ color: var(--renk); font-weight: 800; white-space: nowrap; }}
.neden-not {{ color: rgba(255,255,255,.5); font-size: .72rem; margin-top: .4rem; }}
.radar {{ display: flex; flex-direction: column; align-items: center; }}
.lejant {{ display: flex; gap: 1.2rem; flex-wrap: wrap; justify-content: center; color: #fff; font-size: .9rem; margin-top: .3rem; }}
.lejant i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: .4rem; vertical-align: -1px; }}
.onem {{ display: grid; grid-template-columns: 120px 1fr 48px; gap: .6rem; align-items: center; margin: .35rem 0; color: #fff; font-size: .88rem; }}
.onem .iz {{ height: 10px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; }}
.onem .dolu {{ height: 100%; border-radius: 99px; background: linear-gradient(90deg, #a78bfa, #f472b6); }}
.onem .deger {{ text-align: right; color: rgba(255,255,255,.75); font-variant-numeric: tabular-nums; }}
.km {{ border-collapse: separate; border-spacing: 4px; color: #fff; font-size: .85rem; margin: 0 auto; }}
.km td, .km th {{ text-align: center; padding: .45rem .5rem; border-radius: 8px; min-width: 58px; }}
.km th {{ color: rgba(255,255,255,.65); font-weight: 600; font-size: .75rem; }}
[class*="st-key-gizlijs"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,.06); border-radius: 12px; padding: .4rem 1rem; }}
.stTabs [aria-selected="true"] {{ background: rgba(167,139,250,.25); }}
@media (max-width: 640px) {{
.block-container {{ margin: 1rem auto; padding: 1.4rem 1rem !important; border-radius: 20px; }}
.baslik {{ font-size: 1.6rem; }}
.cubuk {{ grid-template-columns: 105px 1fr 42px; }}
.markalar, .st-key-marka_seridi {{ flex-wrap: nowrap !important; overflow-x: auto; padding-bottom: .4rem; margin-bottom: .8rem; }}
.st-key-marka_seridi > div {{ flex: none; }}
.markalar .marka {{ flex: none; }}
.cipler {{ flex-wrap: nowrap; overflow-x: auto; }}
.cip {{ flex: none; }}
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Veri ve model
# ------------------------------------------------------------
@st.cache_data
def veri_yukle():
    df = pd.read_csv(TABAN / "gercek_telefonlar.csv")
    df["fast_charging"] = df["fast_charging"].fillna(0)
    # Ekran çözünürlüğü bazı telefonlarda ters girilmiş (genişlik > yükseklik, ör. 2400 × 1080): yerlerini değiştir
    ters = df["resolution_width"] > df["resolution_height"]
    df.loc[ters, ["resolution_width", "resolution_height"]] = df.loc[ters, ["resolution_height", "resolution_width"]].values
    for sutun in ["processor_speed", "num_cores", "battery_capacity", "primary_camera_front"]:
        df[sutun] = df[sutun].fillna(df[sutun].median())
    # Analiz koduyla aynı: en pahalı %1'lik lüks telefonlar çıkarılıyor
    df = df[df["price"] <= df["price"].quantile(0.99)].reset_index(drop=True)
    df["segment"], sinirlar = pd.qcut(df["price"], 4, labels=False, retbins=True)
    df["marka"] = [MARKA_ADI.get(m, m.title()) for m in df["brand_name"]]
    df["g5"] = df["5G_or_not"]  # "5G_or_not" adı itertuples'ta kullanılamıyor
    # Model adının başında marka tekrarlanıyorsa olduğu gibi bırak, yoksa ekle
    df["isim"] = [mo if mo.lower().startswith(b) else f"{ma} {mo}"
                  for mo, b, ma in zip(df["model"], df["brand_name"], df["marka"])]
    # "Samsung Galaxy A14 5G (6GB RAM + 128GB)" -> "Samsung Galaxy A14 5G": aynı modelin varyantlarını gruplamak için
    df["temel_model"] = df["isim"].str.replace(r"\s*\(.*\)\s*$", "", regex=True).str.strip()
    return df, list(sinirlar)


@st.cache_resource
def model_egit():
    df, _ = veri_yukle()
    X_tr, X_te, y_tr, y_te = train_test_split(df[OZELLIKLER], df["segment"], test_size=0.2,
                                              random_state=42, stratify=df["segment"])
    test_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42).fit(X_tr, y_tr)
    dogruluk = test_model.score(X_te, y_te)
    karisiklik = confusion_matrix(y_te, test_model.predict(X_te), labels=range(4))
    # Sitede kullanılan model tüm veriyle eğitiliyor
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42).fit(df[OZELLIKLER], df["segment"])
    return model, dogruluk, karisiklik


@st.cache_resource
def ek_modeller():
    # Sitedeki model seçici için KNN ve XGBoost (Random Forest ile aynı 10 özellik ve aynı eğitim/test bölmesi)
    df, _ = veri_yukle()
    X, y = df[OZELLIKLER], df["segment"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    knn_arama = GridSearchCV(Pipeline([("olcek", StandardScaler()), ("knn", KNeighborsClassifier())]),
                             {"knn__n_neighbors": list(range(1, 30, 2))}, cv=5, scoring="accuracy").fit(X_tr, y_tr)
    k = knn_arama.best_params_["knn__n_neighbors"]
    knn = Pipeline([("olcek", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
    xgb_ayar = dict(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                    random_state=42, n_jobs=1)
    sonuc = {}
    for ad, yeni in [(f"KNN (k={k})", lambda: knn), ("XGBoost", lambda: XGBClassifier(**xgb_ayar))]:
        dogruluk = yeni().fit(X_tr, y_tr).score(X_te, y_te)
        sonuc[ad] = (yeni().fit(X, y), dogruluk)  # sitede tüm veriyle eğitilmiş hâli kullanılıyor
    return sonuc


@st.cache_data
def katalog_tahminleri():
    # Katalogdaki her telefon için modelin tahmini (bir kez hesaplanıp saklanır)
    return model_egit()[0].predict(veri_yukle()[0][OZELLIKLER])


df, SINIRLAR = veri_yukle()
df = df.copy()  # önbellekteki tabloyu değiştirmemek için
model, DOGRULUK, KARISIKLIK = model_egit()
df["tahmin"] = katalog_tahminleri()
# Model seçici: ad -> (eğitilmiş model, test doğruluğu)
MODELLER = {"Random Forest": (model, DOGRULUK)} | ek_modeller()


def segment_araligi(s):
    return f"{tl(SINIRLAR[s])} – {tl(SINIRLAR[s + 1])}"


# Filtre aralıkları (alt, üst dahil)
KAMERA_ARALIK = {"12 MP ve altı": (0, 12.9), "13 – 48 MP": (13, 48.9), "50 – 64 MP": (50, 64.9),
                 "65 MP ve üzeri": (65, 1000)}
BATARYA_ARALIK = {"4000 mAh altı": (0, 3999), "4000 – 4999 mAh": (4000, 4999), "5000 – 5999 mAh": (5000, 5999),
                  "6000 mAh ve üzeri": (6000, 100000)}


def grup_secenekleri(sutun, gruplar, yaz, ondalik=0):
    # Her aralık için veri setindeki telefonların ortalamasını alır; seçenek etiketi -> modele verilecek değer
    secenek = {}
    for etiket, (alt, ust) in gruplar.items():
        degerler = df.loc[(df[sutun] >= alt) & (df[sutun] < ust), sutun]
        if not degerler.empty:
            ort = round(float(degerler.mean()), ondalik)
            secenek[f"{etiket} (ort. {yaz(ort)})"] = ort
    return secenek


EKRAN_SECENEK = grup_secenekleri("screen_size", {
    "Küçük, 6.0 inç altı": (0, 6.0), "Orta, 6.0 – 6.4 inç": (6.0, 6.45), "Büyük, 6.5 – 6.6 inç": (6.45, 6.65),
    "Çok büyük, 6.7 – 6.9 inç": (6.65, 7.0), "Katlanabilir / tablet, 7 inç ve üzeri": (7.0, 99)},
    lambda x: f"{x:g}\"", 2)
ISLEMCI_SECENEK = grup_secenekleri("processor_speed", {
    "Giriş seviyesi, 2.0 GHz altı": (0, 2.0), "Orta, 2.0 – 2.4 GHz": (2.0, 2.5),
    "Güçlü, 2.5 – 2.9 GHz": (2.5, 3.0), "Amiral gemisi, 3.0 GHz ve üzeri": (3.0, 99)},
    lambda x: f"{x:g} GHz", 2)
SARJ_SECENEK = {"Hızlı şarj yok": 0.0} | grup_secenekleri("fast_charging", {
    "Yavaş, 10 – 18 W": (1, 19), "Normal, 20 – 33 W": (19, 34), "Hızlı, 40 – 68 W": (34, 70),
    "Çok hızlı, 80 W ve üzeri": (70, 1000)}, lambda x: f"{x:.0f} W")
COZUNURLUK_SECENEK = grup_secenekleri("resolution_height", {
    "HD (düşük)": (0, 1000), "HD+": (1000, 2000), "Full HD+": (2000, 2900), "2K / QHD+ (en net)": (2900, 99999)},
    lambda x: f"{x:.0f} piksel")
YENILEME_SECENEK = sorted(df["refresh_rate"].unique())
BATARYA_GRUP = {"4000 mAh altı": (0, 4000), "4000 – 4499 mAh": (4000, 4500), "4500 – 4999 mAh": (4500, 5000),
                "5000 – 5499 mAh": (5000, 5500), "5500 – 5999 mAh": (5500, 6000), "6000 mAh ve üzeri": (6000, 10 ** 6)}
BATARYA_SECENEK = grup_secenekleri("battery_capacity", BATARYA_GRUP, lambda x: f"{x:.0f} mAh")
BATARYA_ARALIGI = {etiket: aralik for etiket, aralik in zip(BATARYA_SECENEK, BATARYA_GRUP.values())}
KAMERA_SECENEK = sorted(df["primary_camera_rear"].round().astype(int).unique())


def gb_yaz(x):
    return f"{x / 1024:g} TB" if x >= 1024 else f"{x:g} GB"


def mini_kart(r):
    # Tahmin ekranındaki görselli küçük telefon kartı (çizilmiş telefon + marka logosu)
    ad, ikon, renk = SEGMENTLER[r.segment]
    bilgi = LOGOLAR.get(r.brand_name)
    marka_logo = logo(r.marka, 13) if bilgi and not bilgi[5] else ""
    return (f'<div class="mini-kart" style="--renk:{renk}">{telefon_ikonu(renk, 54)}'
            f'<div><div class="mini-isim">{marka_logo}{r.isim}</div>'
            f'<div class="mini-bilgi">{r.ram_capacity:g} GB RAM · {gb_yaz(r.internal_memory)} · '
            f'{r.battery_capacity:.0f} mAh · {r.primary_camera_rear:g} MP</div>'
            f'<div class="mini-fiyat">{tl(r.price)}<span class="rozet">{ikon} {ad}</span></div></div></div>')


def sarj_yazisi(watt):
    return f"{watt:.0f} W şarj" if watt else "standart şarj"


# Özelliklerin sitede görünen adları: (ikon, ad, değer yazımı)
OZELLIK_ADI = {
    "ram_capacity": ("🧠", "RAM", lambda v: f"{v:g} GB"),
    "internal_memory": ("💾", "Depolama", gb_yaz),
    "battery_capacity": ("🔋", "Batarya", lambda v: f"{v:.0f} mAh"),
    "processor_speed": ("⚙️", "İşlemci", lambda v: f"{v:g} GHz"),
    "refresh_rate": ("🔄", "Yenileme hızı", lambda v: f"{v:g} Hz"),
    "primary_camera_rear": ("📷", "Ana kamera", lambda v: f"{v:g} MP"),
    "primary_camera_front": ("🤳", "Ön kamera", lambda v: f"{v:g} MP"),
    "resolution_height": ("📐", "Çözünürlük", lambda v: f"{v:.0f} piksel"),
    "screen_size": ("🖥️", "Ekran", lambda v: f"{v:g}\""),
    "5G_or_not": ("📶", "5G", lambda v: "var" if v else "yok"),
    "fast_charging": ("⚡", "Hızlı şarj", lambda v: f"{v:.0f} W" if v else "yok"),
}


def yuzdelik(sutun, deger):
    # Değerin veri setindeki yeri (0 = en düşük, 1 = en yüksek)
    s = df[sutun]
    return float((s < deger).mean() + (s == deger).mean() / 2)


def seviye_yazisi(sutun, deger):
    if sutun == "5G_or_not":
        return ""
    p = yuzdelik(sutun, deger)
    for esik, yazi in [(.8, "çok yüksek"), (.6, "yüksek"), (.4, "ortalama"), (.2, "düşük")]:
        if p >= esik:
            return yazi
    return "çok düşük"


def neden_verisi(girdi, sinif, mdl):
    # Her özelliği tek tek veri setinin ortancasıyla değiştirip modelin bu segmente olan güveninin
    # ne kadar düştüğüne bakar. En çok düşüren özellikler, tahmini en çok destekleyenlerdir.
    temel = mdl.predict_proba(girdi)[0][sinif]
    denemeler = pd.concat([girdi] * len(OZELLIKLER), ignore_index=True)
    for i, sutun in enumerate(OZELLIKLER):
        denemeler.loc[i, sutun] = df[sutun].median()
    etkiler = temel - mdl.predict_proba(denemeler)[:, sinif]
    sonuc = []
    for sutun, etki in sorted(zip(OZELLIKLER, etkiler), key=lambda x: -x[1])[:3]:
        if etki < 0.01:
            break
        ikon, ad, yaz = OZELLIK_ADI[sutun]
        deger = girdi.iloc[0][sutun]
        sonuc.append((ikon, ad, yaz(deger), seviye_yazisi(sutun, deger), float(etki)))
    return sonuc


def neden_bu_segment(girdi, sinif, mdl):
    satirlar = [f'<li><span>{ikon} <b>{ad}</b> {deger}' + (f' <i>({seviye})</i>' if seviye else '')
                + f'</span><span class="etki">+%{etki * 100:.0f}</span></li>'
                for ikon, ad, deger, seviye, etki in neden_verisi(girdi, sinif, mdl)]
    if not satirlar:
        return ""
    return ('<div class="neden"><div class="neden-baslik">🧐 Neden bu segment?</div><ul>' + "".join(satirlar)
            + '</ul><div class="neden-not">Sağdaki sayı: bu özellik ortalama bir telefondaki gibi olsaydı, modelin '
              'bu segmente olan güveni kaç puan düşerdi.</div></div>')


PDF_MOR, PDF_GRI, PDF_YAZI = (91, 33, 182), (90, 90, 110), (30, 27, 46)


def pdf_baslat(alt_baslik):
    # Ortak PDF başlangıcı: DejaVu fontu (Türkçe karakterler), başlık ve oluşturulma zamanı
    from datetime import datetime, timedelta, timezone
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_font("DejaVu", "", str(TABAN / "fontlar" / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(TABAN / "fontlar" / "DejaVuSans-Bold.ttf"))
    pdf.add_page()

    def yazi(metin, boyut=11, kalin=False, renk=PDF_YAZI, bosluk=6):
        pdf.set_font("DejaVu", "B" if kalin else "", boyut)
        pdf.set_text_color(*renk)
        pdf.multi_cell(0, bosluk, metin, new_x="LMARGIN", new_y="NEXT")

    zaman = datetime.now(timezone(timedelta(hours=3))).strftime("%d.%m.%Y %H:%M")
    yazi("TeknoTahmin", 22, True, PDF_MOR, 10)
    yazi(alt_baslik, 12, renk=PDF_GRI)
    yazi(f"Oluşturulma: {zaman} (Türkiye saati)", 9, renk=PDF_GRI)
    pdf.ln(4)
    return pdf, yazi


def pdf_bitir(pdf, yazi):
    pdf.ln(5)
    yazi("Not: Fiyatlar Smartprix (Hindistan) verisindeki Hindistan Rupisi fiyatlarıdır ve sitede "
         f"1 ₹ ≈ {KUR} TL kuruyla TL'ye çevriliyor. Tahminler yalnızca donanım özelliklerine dayanır, "
         "marka değeri modele dahil değildir.", 8, renk=PDF_GRI, bosluk=5)
    yazi("teknotahmin-0iji.onrender.com · github.com/mertalidag14/Mobile-Price-Range", 8, renk=PDF_GRI, bosluk=5)
    return bytes(pdf.output())


def pdf_tablo(pdf, basliklar, satirlar, genislikler, vurgular=None):
    # Basit tablo: başlık satırı mor zeminli, vurgulanan hücreler yeşil ve kalın
    pdf.set_font("DejaVu", "B", 9)
    pdf.set_fill_color(237, 233, 254)
    pdf.set_text_color(*PDF_MOR)
    for b, g in zip(basliklar, genislikler):
        pdf.cell(g, 7, b, border=1, fill=True)
    pdf.ln()
    for i, satir in enumerate(satirlar):
        for j, (h, g) in enumerate(zip(satir, genislikler)):
            vurgu = vurgular and (i, j) in vurgular
            pdf.set_font("DejaVu", "B" if vurgu or j == 0 else "", 9)
            pdf.set_text_color(*((5, 150, 105) if vurgu else PDF_YAZI))
            pdf.cell(g, 7, str(h), border=1)
        pdf.ln()


def karsilastirma_pdf(a, b, satir_bilgileri):
    # İki telefonun karşılaştırma raporu: satir_bilgileri = [(etiket, a_yazi, b_yazi, kazanan 0/1/None)]
    pdf, yazi = pdf_baslat("İki telefonun karşılaştırma raporu")
    yazi(f"{a.isim}  ile  {b.isim}", 14, True, bosluk=8)
    pdf.ln(2)
    satirlar = [(et, ay, by) for et, ay, by, _ in satir_bilgileri]
    vurgular = {(i, k + 1) for i, (_, _, _, k) in enumerate(satir_bilgileri) if k is not None}
    pdf_tablo(pdf, ["", a.isim[:38], b.isim[:38]], satirlar, [48, 71, 71], vurgular)
    yazi("Yeşil ve kalın değer, o özellikte daha iyi olan telefonu gösterir (fiyatta düşük olan daha iyi).",
         8, renk=PDF_GRI, bosluk=5)
    pdf.ln(3)
    yazi("Özet", 12, True, PDF_MOR)
    sayac = [sum(1 for *_, k in satir_bilgileri if k == t) for t in (0, 1)]
    yazi(f"{a.isim}: {sayac[0]} özellikte önde · {b.isim}: {sayac[1]} özellikte önde", 10)
    for t in (a, b):
        dogru = "doğru bildi" if t.tahmin == t.segment else "yanıldı"
        yazi(f"{t.isim}: gerçek segmenti {SEGMENTLER[t.segment][0]}, model {SEGMENTLER[t.tahmin][0]} dedi ({dogru}).", 10)
    return pdf_bitir(pdf, yazi)


def butce_pdf(butce, oncelik, toplam, telefonlar):
    # Bütçeye uygun en iyi telefonlar raporu: telefonlar = [(satır, puan, güçlü yanlar yazısı, fırsat mı)]
    pdf, yazi = pdf_baslat("Bütçeye göre telefon önerileri")
    yazi(f"Bütçe: {butce:,.0f} TL".replace(",", ".") + f"  ·  Öncelik: {oncelik}", 14, True, bosluk=8)
    yazi(f"Bu bütçeye {toplam} telefon uyuyor. Seçilen önceliğe göre en iyi {len(telefonlar)} tanesi:", 10)
    pdf.ln(2)
    for sira, (r, puan, guclu, firsat) in enumerate(telefonlar, 1):
        yazi(f"{sira}. {r.isim}  —  {puan:.0f}/100 puan", 12, True, PDF_MOR, 7)
        yazi(f"Fiyat: {tl(r.price)} ({rupi(r.price)})  ·  Segment: {SEGMENTLER[r.segment][0]}", 10)
        yazi(f"{r.ram_capacity:g} GB RAM · {gb_yaz(r.internal_memory)} · {r.battery_capacity:.0f} mAh · "
             f"{sarj_yazisi(r.fast_charging)} · {r.screen_size:g}\" {r.refresh_rate} Hz · "
             f"{r.primary_camera_rear:g} MP kamera · {r.processor_speed:g} GHz işlemci"
             + (" · 5G" if r.g5 else ""), 9, renk=PDF_GRI, bosluk=5)
        yazi(f"En güçlü yanları: {guclu}", 9, bosluk=5)
        if firsat:
            yazi("Fiyatına göre güçlü: model bu telefonu gerçek fiyatından daha üst bir segmentte görüyor.",
                 9, renk=(5, 150, 105), bosluk=5)
        pdf.ln(2)
    yazi("Puan: seçilen önceliğe göre ağırlıklandırılmış özelliklerin 970 telefon arasındaki yerinden hesaplanıyor "
         "(100 = o özelliklerde en iyiler).", 8, renk=PDF_GRI, bosluk=5)
    return pdf_bitir(pdf, yazi)


def tahmin_pdf(baslik, ozellik_listesi, model_adi, model_dogrulugu, olas, sinif, gercek_fiyat, nedenler):
    # Tahmin sonucunu tek sayfalık bir PDF rapora çevirir
    pdf, yazi = pdf_baslat("Telefon fiyat segmenti tahmin raporu")
    mor, gri = PDF_MOR, PDF_GRI
    yazi(baslik, 15, True, bosluk=8)
    pdf.ln(1)
    yazi("Özellikler", 12, True, mor)
    for etiket, deger in ozellik_listesi:
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(30, 27, 46)
        pdf.cell(55, 6, etiket)
        pdf.set_font("DejaVu", "", 10)
        pdf.cell(0, 6, deger, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    yazi("Tahmin", 12, True, mor)
    yazi(f"Tahmin edilen segment: {SEGMENTLER[sinif][0]}", 14, True, bosluk=8)
    yazi(f"Model güveni: %{olas[sinif] * 100:.0f}")
    yazi(f"Bu segmentteki telefonların fiyat aralığı: {segment_araligi(sinif)}")
    if gercek_fiyat is not None:
        gercek_seg = int(pd.cut([gercek_fiyat], SINIRLAR, labels=False, include_lowest=True)[0])
        yazi(f"Bu varyantın gerçek fiyatı: {tl(gercek_fiyat)} ({rupi(gercek_fiyat)}), "
             f"gerçek segmenti: {SEGMENTLER[gercek_seg][0]}")
    yazi(f"Kullanılan model: {model_adi} (test doğruluğu %{model_dogrulugu * 100:.1f})", renk=gri)
    pdf.ln(3)
    yazi("Segment olasılıkları", 12, True, mor)
    for i, p in enumerate(olas):
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(30, 27, 46)
        pdf.cell(45, 6, SEGMENTLER[i][0])
        pdf.set_fill_color(235, 230, 245)
        pdf.cell(100, 4, "", fill=True)
        x = pdf.get_x()
        pdf.set_x(x - 100)
        pdf.set_fill_color(124, 58, 237)
        pdf.cell(max(100 * p, 0.1), 4, "", fill=True)
        pdf.set_x(x + 3)
        pdf.cell(0, 6, f"%{p * 100:.0f}", new_x="LMARGIN", new_y="NEXT")
    if nedenler:
        pdf.ln(3)
        yazi("Neden bu segment?", 12, True, mor)
        for _, ad, deger, seviye, etki in nedenler:
            yazi(f"- {ad}: {deger}" + (f" ({seviye})" if seviye else "") + f"  ->  güvene katkısı +%{etki * 100:.0f}", 10)
        yazi("Katkı: bu özellik ortalama bir telefondaki gibi olsaydı modelin bu segmente olan güveni kaç puan düşerdi.",
             8, renk=gri, bosluk=5)
    return pdf_bitir(pdf, yazi)


def fiyat_cetveli(olas, gercek_fiyat=None):
    # 4 segmentli renkli çubuk: tahminin ve (varsa) gerçek fiyatın yeri
    beklenen = sum(i * p for i, p in enumerate(olas))
    tahmin_yeri = (beklenen + .5) / 4 * 100
    serit = "".join(f'<span style="background:{SEGMENTLER[i][2]}"></span>' for i in range(4))
    sinirlar = "".join(f'<small style="left:{k * 25}%">{tl(SINIRLAR[k])}</small>' for k in (1, 2, 3))
    gercek = ""
    if gercek_fiyat is not None:
        gercek_yeri = min(max(yuzdelik("price", gercek_fiyat) * 100, 2), 98)
        gercek = (f'<div class="satir"><div class="isaret gercek" style="left:{gercek_yeri:.1f}%">'
                  f'Gerçek fiyat ▼</div></div>')
    return (f'<div class="cetvel"><div class="satir"><div class="isaret" style="left:{min(max(tahmin_yeri, 6), 94):.1f}%">'
            f'Tahmin ▼</div></div>{gercek}<div class="serit">{serit}</div>'
            f'<div class="sinirlar"><small style="left:0;transform:none">Ucuz</small>{sinirlar}'
            f'<small style="left:100%;transform:translateX(-100%)">Pahalı</small></div></div>')


RADAR_EKSENLERI = [("RAM", "ram_capacity"), ("Depolama", "internal_memory"), ("Batarya", "battery_capacity"),
                   ("Şarj", "fast_charging"), ("İşlemci", "processor_speed"), ("Ekran", "screen_size"),
                   ("Yenileme", "refresh_rate"), ("Kamera", "primary_camera_rear")]


def radar_svg(telefonlar):
    # telefonlar: [(isim, satır, renk)]. Her eksen, özelliğin veri setindeki yeri (0-1).
    m, r, n = 210, 140, len(RADAR_EKSENLERI)

    def nokta(i, oran):
        aci = -math.pi / 2 + 2 * math.pi * i / n
        return m + r * oran * math.cos(aci), m + r * oran * math.sin(aci)

    parca = []
    for halka in (.25, .5, .75, 1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (nokta(i, halka) for i in range(n)))
        parca.append(f'<polygon points="{pts}" fill="none" stroke="rgba(255,255,255,.15)"/>')
    for i, (ad, _) in enumerate(RADAR_EKSENLERI):
        x, y = nokta(i, 1)
        parca.append(f'<line x1="{m}" y1="{m}" x2="{x:.1f}" y2="{y:.1f}" stroke="rgba(255,255,255,.12)"/>')
        lx, ly = nokta(i, 1.17)
        parca.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="rgba(255,255,255,.8)" font-size="13" '
                     f'text-anchor="middle" dominant-baseline="middle">{ad}</text>')
    for _, satir, renk in telefonlar:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                       (nokta(i, max(yuzdelik(s, satir[s]), .04)) for i, (_, s) in enumerate(RADAR_EKSENLERI)))
        parca.append(f'<polygon points="{pts}" fill="{renk}" fill-opacity=".22" stroke="{renk}" stroke-width="2.5"/>')
    lejant = "".join(f'<span><i style="background:{renk}"></i>{isim}</span>' for isim, _, renk in telefonlar)
    return (f'<div class="radar"><svg viewBox="0 0 420 420" width="100%" style="max-width:420px">{"".join(parca)}'
            f'</svg><div class="lejant">{lejant}</div></div>')


def js_calistir(kod):
    # Sayfaya küçük bir JavaScript ekler (sekme değiştirme, kaydırma). Yalnızca kendi yazdığımız sabit kod çalışır.
    sayac = st.session_state.get("_js_sayac", 0) + 1
    st.session_state["_js_sayac"] = sayac
    with st.container(key=f"gizlijs_{sayac}"):
        # Not: Streamlit'in HTML temizleyicisi "<" ardından harf/rakam gelen betikleri siler, kodda "a < b" gibi boşluk bırakılmalı
        # Her betik kendi kapsamında çalışır (aynı değişken adları çakışmasın)
        st.html(f"<script>(()=>{{{kod}}})();</script>", unsafe_allow_javascript=True)


def sekmeye_gec(sira):
    js_calistir("let d=0;const f=()=>{const t=document.querySelectorAll('[role=\"tab\"]');"
                f"if(t.length>{sira}){{const e=t[{sira}];const o={{bubbles:true,cancelable:true,pointerId:1,"
                "pointerType:'mouse',isPrimary:true,button:0};"
                "e.dispatchEvent(new PointerEvent('pointerdown',o));e.dispatchEvent(new MouseEvent('mousedown',o));"
                "e.dispatchEvent(new PointerEvent('pointerup',o));e.dispatchEvent(new MouseEvent('mouseup',o));"
                "e.click();e.scrollIntoView({behavior:'smooth',block:'center'});}"
                "else if(d++ < 40)setTimeout(f,100)};f();")


# ------------------------------------------------------------
# Üst menü ve marka şeridi
# ------------------------------------------------------------
st.markdown(f"""
<div class="nav">
<div><div class="logo">📱 {SITE_ADI}</div><div class="logo-alt">Yapay zekâ ile gerçek telefonların fiyat segmenti tahmini</div></div>
<div class="cipler"><span class="cip">{len(df)} gerçek telefon</span><span class="cip">{df['marka'].nunique()} marka</span>
<span class="cip">Random Forest</span><span class="cip">%{DOGRULUK * 100:.0f} test doğruluğu</span></div>
</div>
""", unsafe_allow_html=True)


def marka_tikla(marka):
    # Sayfa yeniden yüklenmeden katalog o markaya göre süzülür ve katalog sekmesine geçilir
    st.session_state["k_marka"] = [marka]
    st.session_state["_sekmeye_git"] = 1


def logo_maskesi(marka):
    # Marka logosunu butonun içine CSS maskesi olarak koymak için tek renk SVG (base64)
    yol, x, y, w, h, yazi = LOGOLAR[marka.lower()]
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {w} {h}"><path d="{yol}"/></svg>'
    return base64.b64encode(svg.encode()).decode(), w / h, yazi


# Paylaşılan bir tahmin linkiyle gelindiyse (?tel=...&ram=...&depo=...&model=...) aynı tahmini hazırla
if "tel" in st.query_params:
    _tel = st.query_params.get("tel")
    if _tel in set(df["temel_model"]):
        st.session_state.update(t_isim=_tel, t_model=_tel)
        for _anahtar, _durum in [("ram", "t_ram"), ("depo", "t_depo")]:
            try:
                st.session_state[_durum] = int(float(st.query_params.get(_anahtar)))
            except (TypeError, ValueError):
                pass
        if st.query_params.get("model") in MODELLER:
            st.session_state["t_algo"] = st.query_params.get("model")
    st.query_params.clear()

ilk_markalar = df["marka"].value_counts().head(12)
marka_css = []
with st.container(horizontal=True, key="marka_seridi", gap="small"):
    st.markdown('<span class="etiket-marka">Popüler markalar</span>', unsafe_allow_html=True, width="content")
    for i, (m, n) in enumerate(ilk_markalar.items()):
        etiket = f"{m}  {n}"
        if m.lower() in LOGOLAR:
            veri, oran, yazi = logo_maskesi(m)
            boy = 13 if yazi else 16
            if yazi:  # yazı logosu: marka adı logoda zaten var, sadece adet yazılsın
                etiket = f"{n}"
            marka_css.append(
                f'.st-key-mb_{i} button::before {{ content: ""; flex: none; width: {boy * oran:.0f}px; height: {boy}px; '
                f'margin-right: .45rem; background: #fff; -webkit-mask: url("data:image/svg+xml;base64,{veri}") '
                f'center / contain no-repeat; mask: url("data:image/svg+xml;base64,{veri}") center / contain no-repeat; }}')
        st.button(etiket, key=f"mb_{i}", on_click=marka_tikla, args=(m,), help=f"{m} telefonlarını göster")
st.html("<style>" + "".join(marka_css) + "</style>")  # sadece stil: sayfada yer kaplamaz

sekme_tahmin, sekme_katalog, sekme_karsilastir, sekme_butce, sekme_hakkinda = st.tabs(
    ["🔮 Fiyat Tahmini", "🛒 Telefon Kataloğu", "⚖️ Karşılaştır", "💰 Bütçeme Göre", "📊 Proje Hakkında"])
# Sekme geçişi en başta gönderiliyor: sayfanın geri kalanı hazırlanırken sekme hemen açılıyor (donma hissi olmasın)
if "_sekmeye_git" in st.session_state:
    sekmeye_gec(st.session_state.pop("_sekmeye_git"))

# ------------------------------------------------------------
# 1) Tahmin
# ------------------------------------------------------------
with sekme_tahmin:
    st.markdown('<div class="baslik">Telefonun hangi fiyat segmentinde?</div>'
                '<div class="alt">Telefonunun modelini seç, sabit özellikleri otomatik gelsin. Modelini hatırlamıyorsan '
                'aklındaki özellikleri (RAM, depolama, batarya, kamera) yaz, uyan telefonları bul.</div>',
                unsafe_allow_html=True)

    def gb(x):
        return f"{x / 1024:g} TB" if x >= 1024 else f"{x:g} GB"

    sol, sag = st.columns([1.1, 1], gap="large")
    gercek_fiyat = None
    tahmin_imzasi = None
    with sol:
        modeller = sorted(df["temel_model"].unique(), key=str.lower)
        isim_arama = st.text_input("📱 Telefon adı (biliyorsan yaz, bilmiyorsan boş bırak)",
                                   placeholder="ör. iPhone, Galaxy S23, Redmi Note", key="t_isim")
        # Hızlı deneme: tek tıkla modeli ve en ucuz varyantını seçip tahmini gösterir
        ornekler = [m for m in ["Apple iPhone 13", "Samsung Galaxy S23", "Xiaomi Redmi Note 12",
                                "Google Pixel 7 5G", "OnePlus 11 5G", "Samsung Galaxy A14 5G"] if m in set(modeller)]

        def ornek_sec():
            secilen = st.session_state.get("t_ornek")
            if not secilen:
                return
            v = df[df["temel_model"] == secilen].sort_values("price").iloc[0]
            st.session_state.update(t_isim=secilen, t_model=secilen, t_ram=int(v.ram_capacity),
                                    t_depo=int(v.internal_memory), t_ornek=None)

        st.pills("⚡ Hızlı dene", ornekler, key="t_ornek", on_change=ornek_sec)
        kelimeler = isim_arama.lower().split()
        eslesen_modeller = [m for m in modeller if all(k in m.lower() for k in kelimeler)] if kelimeler else []
        if st.session_state.get("t_model") not in eslesen_modeller:
            st.session_state.pop("t_model", None)
        secili = None
        if kelimeler and not eslesen_modeller:
            st.warning("Bu isimde telefon bulunamadı. Yazımı kontrol et ya da sadece markayı yaz (ör. Samsung).")
        elif eslesen_modeller:
            secili = st.selectbox(f"Hangi model? ({len(eslesen_modeller)} model bulundu, sağda görselleriyle listeleniyor)",
                                  eslesen_modeller, index=0 if len(eslesen_modeller) == 1 else None, key="t_model",
                                  placeholder="Modeli seç ya da aşağıdan özellik seçerek listeyi daralt")
        varyantlar = df[df["temel_model"] == secili] if secili else df
        # Seçenekler sabit tutuluyor: model seçilince yapılan RAM/depolama seçimi sıfırlanmasın
        ram_secenek = sorted(set(df["ram_capacity"]) | {4, 6, 8, 12, 16})
        depo_secenek = sorted(set(df["internal_memory"]) | {64, 128, 256, 512, 1024})

        st.markdown('<div class="bolum">⚙️ Özellikler</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        ram = c1.selectbox("RAM", ram_secenek, index=None, format_func=gb, placeholder="RAM seç / yaz", key="t_ram")
        hafiza = c2.selectbox("Depolama", depo_secenek, index=None, format_func=gb, placeholder="Depolama seç / yaz", key="t_depo")

        if secili:
            # Model seçildi: sabit özellikler veri setinden gelir
            sabit = varyantlar.iloc[0]
            batarya, islemci, sarj = sabit.battery_capacity, sabit.processor_speed, sabit.fast_charging
            g5, ekran, yenileme = bool(sabit["5G_or_not"]), sabit.screen_size, sabit.refresh_rate
            cozunurluk, kamera = sabit.resolution_height, sabit.primary_camera_rear
            st.markdown(
                '<div class="bolum">📋 Bu modelin sabit özellikleri</div>'
                '<div class="ozellikler" style="margin-top:.3rem">'
                + "".join(f'<span class="ozellik" style="font-size:.85rem;padding:.25rem .6rem">{y}</span>'
                          for y in [f"🔋 {batarya:.0f} mAh batarya", f"📷 {kamera:g} MP ana kamera",
                                    f"🤳 {sabit.primary_camera_front:g} MP ön kamera",
                                    f"⚙️ {islemci:g} GHz işlemci", f"⚡ {sarj_yazisi(sarj)}",
                                    f"🖥️ {ekran:g}\" ekran", f"📐 {sabit.resolution_width}×{cozunurluk} piksel",
                                    f"🔄 {yenileme} Hz", "📶 5G var" if g5 else "📶 5G yok"])
                + '</div>', unsafe_allow_html=True)
            varyant_yazi = ", ".join(f"{gb(r)} RAM + {gb(d)}" for r, d in
                                     zip(varyantlar["ram_capacity"], varyantlar["internal_memory"]))
            st.caption(f"Veri setindeki varyantlar: {varyant_yazi}")
            if ram is not None and hafiza is not None:
                eslesen = varyantlar[(varyantlar["ram_capacity"] == ram) & (varyantlar["internal_memory"] == hafiza)]
                if not eslesen.empty:
                    gercek_fiyat = eslesen.iloc[0].price
            hazir = ram is not None and hafiza is not None
        else:
            # Model yok: kullanıcı aklındaki özellikleri yazar
            c3, c4 = st.columns(2)
            batarya_sec = c3.selectbox("Batarya", list(BATARYA_SECENEK), index=None, placeholder="Seç")
            kamera = c4.selectbox("Ana kamera", KAMERA_SECENEK, index=None, format_func=lambda x: f"{x} MP",
                                  placeholder="Seç")
            batarya = BATARYA_SECENEK.get(batarya_sec)
            with st.expander("Diğer özellikler (biliyorsan seç, tahmin için gerekli)"):
                c5, c6 = st.columns(2)
                islemci_sec = c5.selectbox("İşlemci", list(ISLEMCI_SECENEK), index=None, placeholder="Seç")
                sarj_sec = c6.selectbox("Hızlı şarj", list(SARJ_SECENEK), index=None, placeholder="Seç")
                c7, c8 = st.columns(2)
                ekran_sec = c7.selectbox("Ekran boyutu", list(EKRAN_SECENEK), index=None, placeholder="Seç")
                yenileme = c8.selectbox("Yenileme hızı", YENILEME_SECENEK, index=None, format_func=lambda x: f"{x} Hz",
                                        placeholder="Seç")
                c9, c10 = st.columns(2)
                cozunurluk_sec = c9.selectbox("Ekran çözünürlüğü", list(COZUNURLUK_SECENEK), index=None, placeholder="Seç")
                g5_sec = c10.selectbox("5G", ["Var", "Yok"], index=None, placeholder="Seç")
                st.caption("Seçeneklerdeki değerler, veri setinde o gruba giren telefonların ortalamasıdır.")
            islemci = ISLEMCI_SECENEK.get(islemci_sec)
            sarj = SARJ_SECENEK.get(sarj_sec)
            ekran = EKRAN_SECENEK.get(ekran_sec)
            cozunurluk = COZUNURLUK_SECENEK.get(cozunurluk_sec)
            g5 = g5_sec == "Var"
            hazir = None not in (ram, hafiza, islemci, batarya, sarj, g5_sec, ekran, yenileme, cozunurluk, kamera)
            st.caption("Telefonunun adını hatırlamıyorsan aklındaki özellikleri seç, uyan telefonlar sağda listelenir. "
                       "Diğer özellikleri de seçersen model fiyat segmentini de tahmin eder.")

    with sag:
        # 1) Tahmin (model seçilip RAM/depolama seçildiyse ya da tüm özellikler yazıldıysa)
        if hazir:
            girdi = pd.DataFrame([[ram, hafiza, batarya, islemci, yenileme, kamera, cozunurluk, ekran, int(g5), sarj]],
                                 columns=OZELLIKLER)
            # Model seçici: aynı telefon için farklı algoritmaların tahminine bakılabiliyor
            if st.session_state.get("t_algo") not in MODELLER:
                st.session_state["t_algo"] = "Random Forest"
            algo = st.segmented_control("🤖 Tahmini yapan model", list(MODELLER), key="t_algo",
                                        help="Analizde karşılaştırdığımız modellerden birini seç. Hepsi aynı 10 özellikle "
                                             "ve aynı veriyle eğitildi.") or "Random Forest"
            secili_model, secili_dogruluk = MODELLER[algo]
            olas = secili_model.predict_proba(girdi)[0]
            sinif = int(olas.argmax())
            ad, ikon, renk = SEGMENTLER[sinif]
            cubuklar = "".join(
                f'<div class="cubuk"><span class="ad">{SEGMENTLER[i][1]} {SEGMENTLER[i][0]}</span>'
                f'<div class="iz"><div class="dolu" style="width:{p * 100:.1f}%;background:{SEGMENTLER[i][2]}"></div></div>'
                f'<span class="yuzde">%{p * 100:.0f}</span></div>'
                for i, p in enumerate(olas))
            gercek = ""
            if gercek_fiyat is not None:
                gercek_seg = int(pd.cut([gercek_fiyat], SINIRLAR, labels=False, include_lowest=True)[0])
                gercek = (f'<div class="aralik" style="margin-top:.5rem">🏷️ Seçtiğin varyantın gerçek fiyatı: '
                          f'<b>{tl(gercek_fiyat)}</b> (gerçek segmenti: {SEGMENTLER[gercek_seg][0]})</div>')
            st.markdown(
                f'<div class="sonuc" id="tahmin-sonucu" style="--renk:{renk}">'
                f'<div class="ust">Tahmin edilen segment</div>'
                f'<div class="ana">{ikon} {ad}</div>'
                f'<div class="guven">{algo} güveni: %{olas[sinif] * 100:.0f}</div>'
                f'<div class="fiyat">{segment_araligi(sinif)}</div>'
                f'<div class="aralik">Bu segmentteki telefonların gerçek fiyat aralığı</div>{gercek}'
                f'{fiyat_cetveli(olas, gercek_fiyat)}'
                f'<div style="margin-top:.6rem">{cubuklar}</div>'
                f'{neden_bu_segment(girdi, sinif, secili_model)}</div>', unsafe_allow_html=True)
            tahmin_imzasi = tuple(girdi.iloc[0].round(3))

            # Üç model aynı telefon için ne diyor?
            satirlar = ""
            for ad_m, (mdl, dog) in MODELLER.items():
                o = mdl.predict_proba(girdi)[0]
                s = int(o.argmax())
                satirlar += (f'<tr{" class=secili" if ad_m == algo else ""}><td>{ad_m}</td>'
                             f'<td style="color:{SEGMENTLER[s][2]};font-weight:800">{SEGMENTLER[s][1]} {SEGMENTLER[s][0]}</td>'
                             f'<td>%{o[s] * 100:.0f}</td><td>%{dog * 100:.0f}</td></tr>')
            st.markdown('<div class="bolum" style="margin-top:1rem">🤖 Üç model ne diyor?</div>'
                        '<table class="tablo model-tablo"><tr><th>Model</th><th>Tahmin</th><th>Güven</th>'
                        f'<th>Test doğruluğu</th></tr>{satirlar}</table>', unsafe_allow_html=True)

            # Paylaşma linki ve PDF rapor
            baslik = (f"{secili} ({gb(ram)} RAM + {gb(hafiza)})" if secili else
                      f"Özellikleri seçilen telefon ({gb(ram)} RAM + {gb(hafiza)})")
            ozellik_listesi = [("RAM", gb(ram)), ("Depolama", gb(hafiza)), ("Batarya", f"{batarya:.0f} mAh"),
                               ("İşlemci hızı", f"{islemci:g} GHz"), ("Hızlı şarj", sarj_yazisi(sarj)),
                               ("Ekran boyutu", f"{ekran:g} inç"), ("Ekran yüksekliği", f"{cozunurluk:.0f} piksel"),
                               ("Yenileme hızı", f"{yenileme} Hz"), ("Ana kamera", f"{kamera:g} MP"),
                               ("5G", "Var" if g5 else "Yok")]
            p1, p2 = st.columns(2)
            p1.download_button("📄 PDF olarak indir",
                               # PDF sadece butona basınca hazırlanıyor (her yenilemede değil, site hızlı kalsın)
                               data=lambda a=(baslik, ozellik_listesi, algo, secili_dogruluk, olas, sinif, gercek_fiyat,
                                              girdi, secili_model): tahmin_pdf(*a[:7], neden_verisi(a[7], a[5], a[8])),
                               file_name=f"TeknoTahmin_{(secili or 'tahmin').replace(' ', '_')}.pdf",
                               mime="application/pdf", on_click="ignore", use_container_width=True)
            if secili:
                with p2.popover("🔗 Paylaş", use_container_width=True):
                    taban_url = (st.context.url or "https://teknotahmin-0iji.onrender.com/").split("?")[0]
                    link = (f"{taban_url}?tel={quote(secili)}&ram={ram:g}&depo={hafiza:g}&model={quote(algo)}")
                    st.caption("Bu linki açan kişi aynı telefonu ve aynı tahmini görür:")
                    st.code(link, language=None, wrap_lines=True)
            else:
                p2.caption("Paylaşma linki, listeden bir telefon modeli seçildiğinde açılıyor.")

        # 2) Yazılan isme ve seçilen özelliklere uyan gerçek telefonlar (görselli kartlar, anında süzülür)
        girilen = {k: v for k, v in {"ram_capacity": ram, "internal_memory": hafiza, "battery_capacity": batarya,
                                     "primary_camera_rear": kamera}.items() if v is not None}
        if kelimeler or girilen:
            if secili:
                uyan = varyantlar
            elif kelimeler:
                uyan = df[df["temel_model"].isin(eslesen_modeller)]
            else:
                uyan = df
            if "ram_capacity" in girilen:
                uyan = uyan[uyan["ram_capacity"] == ram]
            if "internal_memory" in girilen:
                uyan = uyan[uyan["internal_memory"] == hafiza]
            if "battery_capacity" in girilen and not secili:  # seçilen batarya aralığındakiler
                alt, ust = BATARYA_ARALIGI[batarya_sec]
                uyan = uyan[(uyan["battery_capacity"] >= alt) & (uyan["battery_capacity"] < ust)]
            if "primary_camera_rear" in girilen and not secili:
                uyan = uyan[uyan["primary_camera_rear"].round() == kamera]
            if uyan.empty:
                st.info("Bu isme ve özelliklere uyan telefon bulunamadı. Bir seçimi kaldırmayı ya da değiştirmeyi dene.")
            else:
                if not hazir and not secili:
                    dagilim = uyan["segment"].value_counts(normalize=True).reindex(range(4), fill_value=0)
                    en_cok = int(dagilim.idxmax())
                    ad, ikon, renk = SEGMENTLER[en_cok]
                    cubuklar = "".join(
                        f'<div class="cubuk"><span class="ad">{SEGMENTLER[i][1]} {SEGMENTLER[i][0]}</span>'
                        f'<div class="iz"><div class="dolu" style="width:{p * 100:.1f}%;background:{SEGMENTLER[i][2]}"></div></div>'
                        f'<span class="yuzde">%{p * 100:.0f}</span></div>' for i, p in dagilim.items())
                    st.markdown(
                        f'<div class="sonuc" style="--renk:{renk}">'
                        f'<div class="ust">Bu özelliklerde {len(uyan)} telefon var</div>'
                        f'<div class="ana">{ikon} Çoğu {ad.lower()}</div>'
                        f'<div class="fiyat" style="font-size:1.6rem">{tl(uyan["price"].min())} – {tl(uyan["price"].max())}</div>'
                        f'<div class="aralik">Bu telefonların gerçek fiyat aralığı</div>'
                        f'<div style="margin-top:1rem">{cubuklar}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bolum" style="margin-top:1.2rem">📱 Uyan telefonlar ({len(uyan)})</div>',
                            unsafe_allow_html=True)
                gosterilecek = uyan.sort_values(["avg_rating", "price"], ascending=[False, True],
                                                na_position="last").head(12)
                st.markdown('<div class="mini-izgara">' + "".join(mini_kart(r) for r in gosterilecek.itertuples())
                            + '</div>', unsafe_allow_html=True)
                if len(uyan) > 12:
                    st.caption("En beğenilen 12 tanesi gösteriliyor. RAM, depolama gibi özellikleri seçerek listeyi daralt.")

        # 3) Hiçbir şey girilmediyse yönlendirme
        if not hazir and not girilen and not kelimeler:
            st.markdown(
                '<div class="sonuc" style="--renk:#a78bfa">'
                '<div class="ust">Tahmin edilen segment</div>'
                '<div class="ana">🔮 Henüz tahmin yok</div>'
                '<div class="aralik" style="margin-top:.5rem">'
                + ("Şimdi RAM ve depolamayı seç, tahmin burada görünecek." if secili else
                   "Telefon modelini yaz ya da modeli hatırlamıyorsan RAM, depolama, batarya, kamera gibi "
                   "özellikleri gir. Uyan telefonlar ve tahmin burada görünecek.")
                + '</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 2) Katalog
# ------------------------------------------------------------
SAYFA_BOYUTU = 24


def kart_html(r):
    ad, ikon, renk = SEGMENTLER[r.segment]
    dogru = r.tahmin == r.segment
    tahmin_rozeti = (f'<span class="ozellik">🤖 Model: {SEGMENTLER[r.tahmin][0]} {"✓" if dogru else "✗"}</span>')
    ozellikler = "".join(f'<span class="ozellik{"" if var else " yok"}">{etiket}</span>'
                         for etiket, var in [("5G", r.g5), ("Hızlı şarj", r.fast_charging_available),
                                             ("Hafıza kartı", r.extended_memory_available)])
    bilgi = LOGOLAR.get(r.brand_name)
    marka_logo = logo(r.marka, 15) if bilgi and not bilgi[5] else ""
    return (
        f'<div class="kart" style="--renk:{renk}">'
        f'<div class="ust">{telefon_ikonu(renk)}<div><div class="isim">{marka_logo}{r.isim}</div>'
        f'<span class="rozet">{ikon} {ad}</span></div></div>'
        f'<div class="fiyat">{tl(r.price)} <small style="font-size:.75rem;font-weight:600;color:rgba(255,255,255,.5)">'
        f'({rupi(r.price)})</small></div>'
        f'<ul><li>🧠 {r.ram_capacity:g} GB RAM · 💾 {r.internal_memory:g} GB</li>'
        f'<li>🔋 {r.battery_capacity:.0f} mAh · ⚡ {sarj_yazisi(r.fast_charging)}</li>'
        f'<li>🖥️ {r.screen_size:g}" · {r.resolution_width}×{r.resolution_height} · {r.refresh_rate} Hz</li>'
        f'<li>📷 {r.primary_camera_rear:g} MP · ⚙️ {r.processor_speed:g} GHz</li>'
        + (f'<li>⭐ {r.avg_rating:g}/10 kullanıcı puanı</li>' if pd.notna(r.avg_rating) else '')
        + '</ul>'
        f'<div class="ozellikler">{ozellikler}{tahmin_rozeti}</div></div>'
    )


def karsilastirmaya_ekle(sira):
    secilenler = [s for s in st.session_state.get("kars", []) if s != sira] + [sira]
    if len(secilenler) == 2:
        # İki telefon seçildi: karşılaştırma sekmesi güncellensin diye tüm sayfa yenilenecek
        st.session_state.update(k1=secilenler[0], k2=secilenler[1], kars=[], _sekmeye_git=2, _tam_yenile=True)
    else:
        st.session_state["kars"] = secilenler


@st.fragment
def katalog_kartlari(gosterilen):
    # Kartlar ayrı bir parça: "Karşılaştır"a basınca sadece bu kısım yenilenir, sayfanın tamamı değil (donmayı önler)
    if st.session_state.pop("_tam_yenile", False):
        st.rerun(scope="app")
    bekleyen = st.session_state.get("kars", [])
    if bekleyen:
        u1, u2 = st.columns([4, 1])
        u1.info(f"⚖️ **{df.iloc[bekleyen[0]].isim}** seçildi. Karşılaştırmak için bir telefon daha seç.")
        u2.button("Vazgeç", on_click=lambda: st.session_state.update(kars=[]), use_container_width=True)
    if gosterilen.empty:
        st.info("Bu filtrelere uyan telefon yok. Filtreleri gevşetmeyi dene.")
        return
    satirlar_listesi = list(gosterilen.itertuples())
    for bas in range(0, len(satirlar_listesi), 3):
        kolonlar = st.columns(3)
        for kol, r in zip(kolonlar, satirlar_listesi[bas:bas + 3]):
            with kol:
                st.markdown(kart_html(r), unsafe_allow_html=True)
                secili_mi = r.Index in bekleyen
                st.button("✓ Seçildi" if secili_mi else "⚖️ Karşılaştır", key=f"kb_{r.Index}",
                          on_click=karsilastirmaya_ekle, args=(r.Index,), disabled=secili_mi,
                          use_container_width=True)


with sekme_katalog:
    st.markdown('<div class="baslik">Telefon Kataloğu</div>'
                f'<div class="alt">{len(df)} gerçek telefon, gerçek özellikleri ve satış fiyatlarıyla '
                '(Smartprix, Hindistan). Her kartta modelin tahmini ve doğru bilip bilmediği de yazıyor.</div>',
                unsafe_allow_html=True)
    filtre_kol, liste_kol = st.columns([1, 3.1], gap="large")

    # ---- Sol: tek "Filtrele" paneli (Trendyol tarzı) ----
    with filtre_kol:
        with st.container(border=True):
            st.markdown('<div class="bolum">🔎 Filtrele</div>', unsafe_allow_html=True)
            arama = st.text_input("Model ara", placeholder="ör. iPhone 14")
            tum_markalar = df["marka"].value_counts().index.tolist()
            marka_sec = st.multiselect("Marka", tum_markalar, placeholder="Tüm markalar", key="k_marka")
            segment_sec = st.multiselect("Fiyat segmenti", list(SEGMENTLER), placeholder="Tüm segmentler",
                                         format_func=lambda s: f"{SEGMENTLER[s][1]} {SEGMENTLER[s][0]}")
            ram_sec = st.multiselect("RAM", sorted(df["ram_capacity"].unique()), placeholder="Tümü",
                                     format_func=lambda x: f"{x:g} GB")
            depo_sec = st.multiselect("Depolama", sorted(df["internal_memory"].unique()), placeholder="Tümü",
                                      format_func=lambda x: f"{x / 1024:g} TB" if x >= 1024 else f"{x:g} GB")
            kamera_sec = st.multiselect("Ana kamera", list(KAMERA_ARALIK), placeholder="Tümü")
            batarya_sec = st.multiselect("Batarya", list(BATARYA_ARALIK), placeholder="Tümü")
            st.markdown('<div style="font-size:.9rem;margin-top:.2rem">Fiyat aralığı (TL)</div>',
                        unsafe_allow_html=True)
            en_yuksek_tl = int(df["price"].max() * KUR) + 1000
            p1, p2 = st.columns(2)
            tl_min = p1.number_input("En az", 0, en_yuksek_tl, None, 1000, placeholder="En az",
                                     label_visibility="collapsed")
            tl_max = p2.number_input("En çok", 0, en_yuksek_tl, None, 1000, placeholder="En çok",
                                     label_visibility="collapsed")
            sadece_5g = st.checkbox("5G destekli")
            hizli_sarj = st.checkbox("Hızlı şarj")
            hafiza_karti = st.checkbox("Hafıza kartı desteği")
            yanlislar = st.checkbox("Modelin yanıldıkları")

    liste = df[df["price"].between((tl_min or 0) / KUR, (tl_max or en_yuksek_tl) / KUR)]
    if kamera_sec:
        liste = liste[pd.concat([liste["primary_camera_rear"].between(*KAMERA_ARALIK[k]) for k in kamera_sec],
                                axis=1).any(axis=1)]
    if batarya_sec:
        liste = liste[pd.concat([liste["battery_capacity"].between(*BATARYA_ARALIK[b]) for b in batarya_sec],
                                axis=1).any(axis=1)]
    if marka_sec:
        liste = liste[liste["marka"].isin(marka_sec)]
    if segment_sec:
        liste = liste[liste["segment"].isin(segment_sec)]
    if ram_sec:
        liste = liste[liste["ram_capacity"].isin(ram_sec)]
    if depo_sec:
        liste = liste[liste["internal_memory"].isin(depo_sec)]
    if sadece_5g:
        liste = liste[liste["5G_or_not"] == 1]
    if hizli_sarj:
        liste = liste[liste["fast_charging_available"] == 1]
    if hafiza_karti:
        liste = liste[liste["extended_memory_available"] == 1]
    if yanlislar:
        liste = liste[liste["tahmin"] != liste["segment"]]
    if arama.strip():
        kelimeler = arama.lower().split()
        liste = liste[liste["isim"].str.lower().apply(lambda ad: all(k in ad for k in kelimeler))]

    # ---- Sağ: sıralama ve kartlar ----
    with liste_kol:
        SIRALAMA = {
            "Önerilen": (["avg_rating", "price"], [False, True]),
            "En düşük fiyat": (["price"], [True]),
            "En yüksek fiyat": (["price"], [False]),
            "En çok beğenilen (puan)": (["avg_rating"], [False]),
            "En yüksek RAM": (["ram_capacity", "price"], [False, True]),
            "En yüksek depolama": (["internal_memory", "price"], [False, True]),
            "En uzun batarya": (["battery_capacity"], [False]),
            "En iyi kamera": (["primary_camera_rear"], [False]),
        }
        b1, b2, b3 = st.columns([1.6, 1.4, 1])
        b1.markdown(f'<div class="alt" style="margin:1.9rem 0 0"><b>{len(liste)}</b> telefon bulundu</div>',
                    unsafe_allow_html=True)
        siralama = b2.selectbox("Sırala", list(SIRALAMA))
        sutunlar, artanlar = SIRALAMA[siralama]
        liste = liste.sort_values(sutunlar, ascending=artanlar, na_position="last")
        toplam_sayfa = max(1, -(-len(liste) // SAYFA_BOYUTU))
        sayfa = b3.number_input(f"Sayfa (/{toplam_sayfa})", 1, toplam_sayfa, 1)
        gosterilen = liste.iloc[(sayfa - 1) * SAYFA_BOYUTU: sayfa * SAYFA_BOYUTU]

        katalog_kartlari(gosterilen)

# ------------------------------------------------------------
# 3) Karşılaştırma
# ------------------------------------------------------------
with sekme_karsilastir:
    st.markdown('<div class="baslik">İki telefonu karşılaştır</div>'
                '<div class="alt">Katalogdan iki telefon seç, hangisinin nerede önde olduğunu gör.</div>',
                unsafe_allow_html=True)
    isimler = df["isim"].tolist()

    def varsayilan(aranan):
        return next((i for i, ad in enumerate(isimler) if aranan.lower() in ad.lower()), 0)

    # Varsayılanlar; katalogdan "Karşılaştır" ile gelindiyse seçilen telefonlar kullanılır
    st.session_state.setdefault("k1", varsayilan("iPhone 14"))
    st.session_state.setdefault("k2", varsayilan("Galaxy S23"))
    k1, k2 = st.columns(2)
    a = df.iloc[k1.selectbox("1. telefon", range(len(isimler)), key="k1", format_func=lambda i: isimler[i])]
    b = df.iloc[k2.selectbox("2. telefon", range(len(isimler)), key="k2", format_func=lambda i: isimler[i])]

    # (etiket, sütun, birim, büyük olan mı iyi?)
    satirlar = [("Fiyat", "price", "", False), ("RAM", "ram_capacity", " GB", True),
                ("Depolama", "internal_memory", " GB", True), ("Batarya", "battery_capacity", " mAh", True),
                ("Hızlı şarj", "fast_charging", " W", True), ("İşlemci hızı", "processor_speed", " GHz", True),
                ("Ekran boyutu", "screen_size", "\"", True), ("Yenileme hızı", "refresh_rate", " Hz", True),
                ("Ana kamera", "primary_camera_rear", " MP", True), ("Ön kamera", "primary_camera_front", " MP", True)]

    def hucre(deger, diger, sutun, birim, buyuk_iyi):
        kazanan = deger != diger and ((deger > diger) == buyuk_iyi)
        yazi = tl(deger) if sutun == "price" else f"{deger:g}{birim}"
        return f'<td class="{"kazanan" if kazanan else ""}">{yazi}</td>'

    govde = "".join(f"<tr><td>{et}</td>{hucre(a[s], b[s], s, br, iyi)}{hucre(b[s], a[s], s, br, iyi)}</tr>"
                    for et, s, br, iyi in satirlar)
    ust_satir = "".join(
        f'<tr><td>{et}</td><td>{SEGMENTLER[a[s]][1]} {SEGMENTLER[a[s]][0]}</td>'
        f'<td>{SEGMENTLER[b[s]][1]} {SEGMENTLER[b[s]][0]}</td></tr>'
        for et, s in [("Gerçek segment", "segment"), ("Modelin tahmini", "tahmin")])
    tablo_kol, radar_kol = st.columns([1.35, 1], gap="large")
    with tablo_kol:
        st.markdown(f'<table class="tablo"><tr><th></th><th>{a.isim}</th><th>{b.isim}</th></tr>'
                    f'{ust_satir}{govde}</table>', unsafe_allow_html=True)
        st.caption("Yeşil değer o özellikte daha iyi olan telefonu gösterir (fiyatta düşük olan daha iyi). "
                   "Katalogdaki kartlarda \"⚖️ Karşılaştır\" butonuyla da telefon seçebilirsin.")
        # Karşılaştırma raporu (PDF)
        pdf_satirlari = [("Gerçek segment", SEGMENTLER[a.segment][0], SEGMENTLER[b.segment][0], None),
                         ("Modelin tahmini", SEGMENTLER[a.tahmin][0], SEGMENTLER[b.tahmin][0], None)]
        for et, s, br, iyi in satirlar:
            ya = tl(a[s]) if s == "price" else f"{a[s]:g}{br}"
            yb = tl(b[s]) if s == "price" else f"{b[s]:g}{br}"
            kazanan = None if a[s] == b[s] else (0 if (a[s] > b[s]) == iyi else 1)
            pdf_satirlari.append((et, ya, yb, kazanan))
        st.download_button("📄 Karşılaştırmayı PDF olarak indir", data=lambda a=a, b=b, s=pdf_satirlari: karsilastirma_pdf(a, b, s),
                           file_name=f"TeknoTahmin_karsilastirma_{a.isim[:25]}_{b.isim[:25]}.pdf".replace(" ", "_"),
                           mime="application/pdf", key="pdf_kars", on_click="ignore")
    with radar_kol:
        st.markdown('<div class="bolum" style="margin-top:1rem">🕸️ Özellik haritası</div>', unsafe_allow_html=True)
        st.markdown(radar_svg([(a.isim, a, "#60a5fa"), (b.isim, b, "#f472b6")]), unsafe_allow_html=True)
        st.caption("Her eksen, o özelliğin 970 telefon arasındaki yerini gösterir: merkez en düşük, dış çember en yüksek.")

# ------------------------------------------------------------
# 4) Bütçeye göre öneri
# ------------------------------------------------------------
# Önceliğe göre özellik ağırlıkları. Her özellik, 970 telefon arasındaki yerine (0-1) göre puanlanıyor.
ONCELIKLER = {
    "⚖️ Dengeli": {"processor_speed": .25, "ram_capacity": .2, "internal_memory": .15, "primary_camera_rear": .1,
                   "battery_capacity": .1, "refresh_rate": .1, "fast_charging": .1},
    "🚀 Performans": {"processor_speed": .45, "ram_capacity": .3, "refresh_rate": .15, "internal_memory": .1},
    "📷 Kamera": {"primary_camera_rear": .5, "primary_camera_front": .2, "processor_speed": .15, "ram_capacity": .15},
    "🔋 Batarya": {"battery_capacity": .55, "fast_charging": .3, "processor_speed": .15},
    "💾 Depolama": {"internal_memory": .55, "ram_capacity": .25, "processor_speed": .2},
}


@st.cache_data
def ozellik_yuzdelikleri():
    # Her özelliğin 970 telefon arasındaki yeri (0 = en düşük, 1 = en yüksek)
    tablo, _ = veri_yukle()
    sutunlar = {s for agirlik in ONCELIKLER.values() for s in agirlik}
    return pd.DataFrame({s: tablo[s].rank(pct=True) for s in sutunlar})


def butce_ayarla(miktar):
    st.session_state["b_butce"] = miktar


with sekme_butce:
    st.markdown('<div class="baslik">Bütçeme göre en iyi telefonlar</div>'
                '<div class="alt">Bütçeni ve senin için en önemli şeyi seç, o fiyata kadar olan gerçek telefonları '
                'özelliklerine göre puanlayıp sıralayalım.</div>', unsafe_allow_html=True)
    b1, b2 = st.columns([1, 1.4], gap="large")
    with b1:
        butce = st.number_input("💰 Bütçen (TL)", min_value=1000, max_value=100000, value=None, step=1000,
                                placeholder="ör. 15000", key="b_butce")
        hk = st.columns(5)
        for kol, miktar in zip(hk, [7500, 10000, 15000, 20000, 30000]):
            kol.button(f"{miktar / 1000:g} bin".replace(".", ","), key=f"b_{miktar}", on_click=butce_ayarla, args=(miktar,),
                       use_container_width=True)
    with b2:
        oncelik = st.segmented_control("🎯 Senin için en önemli olan", list(ONCELIKLER), default="⚖️ Dengeli",
                                       key="b_oncelik") or "⚖️ Dengeli"
        e1, e2 = st.columns(2)
        b_5g = e1.checkbox("5G olsun", key="b_5g")
        b_marka = e2.multiselect("Marka (isteğe bağlı)", df["marka"].value_counts().index.tolist(),
                                 placeholder="Tüm markalar", key="b_marka")

    if butce is None:
        st.markdown('<div class="sonuc" style="--renk:#a78bfa;margin-top:1rem"><div class="ana">💰 Bütçeni seç</div>'
                    '<div class="aralik">Yukarıya bütçeni yaz ya da hazır bütçelerden birine bas. Önceliğini de seçersen '
                    'o bütçedeki en iyi telefonları puanlarıyla sıralarız.</div></div>', unsafe_allow_html=True)
    else:
        aday = df[df["price"] * KUR <= butce]
        if b_5g:
            aday = aday[aday["5G_or_not"] == 1]
        if b_marka:
            aday = aday[aday["marka"].isin(b_marka)]
        if aday.empty:
            st.info("Bu bütçede uyan telefon yok. Bütçeyi biraz artırmayı ya da filtreleri kaldırmayı dene.")
        else:
            agirlik = ONCELIKLER[oncelik]
            yuzdelik = ozellik_yuzdelikleri().loc[aday.index]
            puan = sum(yuzdelik[s] * w for s, w in agirlik.items()) * 100
            aday = aday.assign(puan=puan.round(0)).sort_values(["puan", "price"], ascending=[False, True])
            st.markdown(f'<div class="alt" style="margin-top:.8rem"><b>{len(aday)}</b> telefon bu bütçeye uyuyor. '
                        f'{oncelik.split(" ", 1)[1]} önceliğine göre en iyi 6 tanesi:</div>', unsafe_allow_html=True)
            kartlar = []
            for r in aday.head(6).itertuples():
                guclu = sorted(agirlik, key=lambda s: -yuzdelik.loc[r.Index, s])[:2]
                neden = " · ".join(f"{OZELLIK_ADI[s][0]} {OZELLIK_ADI[s][1]} {OZELLIK_ADI[s][2](df.at[r.Index, s])}"
                                   for s in guclu)
                firsat = ('<span class="firsat">🔥 Fiyatına göre güçlü: model daha üst segment diyor</span>'
                          if r.tahmin > r.segment else "")
                # Katalog kartına puan rozeti ve "en güçlü yanları" satırı ekleniyor
                html = kart_html(r)
                html = html[:-len("</div>")] + f'<div class="butce-neden">En güçlü yanları: {neden}</div>{firsat}</div>'
                html = html.replace('style="--renk:', 'style="padding-top:2.2rem;--renk:', 1)
                html = html.replace('<div class="ust">', f'<span class="butce-puan">{r.puan:.0f}/100</span><div class="ust">', 1)
                kartlar.append(html)
            st.markdown(f'<div class="izgara">{"".join(kartlar)}</div>', unsafe_allow_html=True)
            st.caption("Puan: seçtiğin önceliğe göre ağırlıklandırılmış özelliklerin, 970 telefon arasındaki yerinden "
                       "hesaplanıyor (100 = o özelliklerde en iyiler). \"Fiyatına göre güçlü\" rozeti, modelin "
                       "telefonun donanımına bakıp gerçek fiyatından daha üst bir segment tahmin ettiği telefonlarda çıkıyor.")
            # Bütçe raporu (PDF): en iyi 10 telefon
            pdf_telefonlar = []
            for r in aday.head(10).itertuples():
                guclu = sorted(agirlik, key=lambda s: -yuzdelik.loc[r.Index, s])[:2]
                guclu_yazi = ", ".join(f"{OZELLIK_ADI[s][1]} {OZELLIK_ADI[s][2](df.at[r.Index, s])}" for s in guclu)
                pdf_telefonlar.append((r, r.puan, guclu_yazi, r.tahmin > r.segment))
            st.download_button("📄 Bu listeyi PDF olarak indir (en iyi 10 telefon)",
                               data=lambda a=(butce, oncelik.split(" ", 1)[1], len(aday), pdf_telefonlar): butce_pdf(*a),
                               file_name=f"TeknoTahmin_butce_{int(butce)}_TL.pdf", mime="application/pdf",
                               key="pdf_butce", on_click="ignore")

# ------------------------------------------------------------
# 5) Proje hakkında
# ------------------------------------------------------------
with sekme_hakkinda:
    st.markdown('<div class="baslik">Proje Hakkında</div>', unsafe_allow_html=True)
    sinir_metni = "; ".join(f"{SEGMENTLER[i][0]}: {segment_araligi(i)}" for i in range(4))
    h1, h2 = st.columns(2, gap="large")
    with h1:
        st.markdown(f"""
#### 🎯 Amaç
Bir telefonun donanım özelliklerine (RAM, depolama, işlemci, batarya, ekran, kamera, 5G, hızlı şarj) bakarak
hangi **fiyat segmentinde** olduğunu makine öğrenmesiyle tahmin etmek.

#### 📁 Veri seti
- **{len(df)} gerçek akıllı telefon**, {df['marka'].nunique()} marka, gerçek satış fiyatlarıyla
  (Kaggle "Real World Smartphone's Dataset", Smartprix, Hindistan).
- Fiyatlar sitede **1 ₹ ≈ {KUR} TL** kuruyla (XE orta kur, 19 Kasım 2026) TL'ye çevrildi. Bunlar Hindistan
  fiyatlarıdır, Türkiye'de vergiler yüzünden genelde daha yüksektir.
- Projeye önce Kaggle'ın sentetik *Mobile Price Classification* veri setiyle başlandı (v1), gerçekçi sonuç
  için gerçek telefon verisine geçildi (v2).

#### 🧹 Veri ön işleme
- Ekran çözünürlüğü ters girilmiş 39 telefonda (ör. 2400 × 1080) genişlik ve yükseklik yer değiştirildi.
- Eksik değerler medyanla dolduruldu, hızlı şarj bilgisi olmayanlar 0 W kabul edildi.
- Fiyatı uç değerde olan en pahalı %1'lik lüks telefonlar (ör. Vertu) çıkarıldı.
- Telefonlar fiyatlarına göre **4 eşit gruba (çeyreklere)** bölündü: {sinir_metni}
- Veri %80 eğitim / %20 test olarak, segment oranları korunarak (stratified) ayrıldı.

#### 🔬 Yöntem
- **KNN** için StandardScaler / MinMaxScaler ölçeklendirmesi karşılaştırıldı.
- Tüm modellerde en iyi parametreler **GridSearchCV** ile aranıp **5-fold cross-validation** ile doğrulandı.
- Değerlendirme: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, öğrenme eğrisi, özellik önemi,
  hata analizi.
""")
    with h2:
        st.markdown(f"""
#### 🤖 Karşılaştırılan modeller (test seti)

| Model | Accuracy | F1-Score | 5-Fold CV |
| :--- | :---: | :---: | :---: |
| KNN (k=3, StandardScaler) | %70.6 | 0.699 | %63.5 |
| Random Forest | %75.8 | 0.753 | **%71.1** |
| **XGBoost** | **%76.3** | **0.758** | %67.8 |
| LightGBM | %73.7 | 0.728 | %67.9 |

- **Bu sitedeki model:** 10 donanım özelliğiyle eğitilmiş Random Forest, test doğruluğu
  **%{DOGRULUK * 100:.1f}**. Cross-validation'da en kararlı model olduğu için seçildi.
- **En belirleyici özellikler:** işlemci hızı, depolama, RAM, hızlı şarj gücü, ekran çözünürlüğü.
- Model fiyatın sadece donanımdan tahmin edilebilen kısmını öğrenir. Marka değeri (ör. Apple) modele dahil
  değildir, bu yüzden bazı telefonlarda yanılır. Katalogda **"Modelin yanıldıkları"** filtresiyle görebilirsin.

#### 🛠️ Kullanılan yazılımlar ve teknolojiler
- **Python** · **Pandas, NumPy** (veri işleme) · **Matplotlib, Seaborn** (grafikler)
- **Scikit-learn** (KNN, Random Forest, GridSearchCV) · **XGBoost, LightGBM**
- **Jupyter Notebook, VS Code** (geliştirme ortamı)
- **Streamlit, HTML/CSS, SVG** (bu web sitesi) · **Docker** + **Render** (yayınlama)
- **Git & GitHub** (sürüm kontrolü, kanban tablosu)

#### 👥 Ekip
Mert Ali Dağ · Gürkan Tural · Efe Erdem

[📂 Projenin GitHub sayfası](https://github.com/mertalidag14/Mobile-Price-Range)
""")

    # Grafikler: sitedeki modelden canlı hesaplanır
    st.markdown('<div class="bolum" style="margin-top:1.4rem;font-size:1.25rem">📈 Modelin grafikleri</div>',
                unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown('<div class="bolum">Özellik önemi (Random Forest)</div>', unsafe_allow_html=True)
        onemler = sorted(zip(OZELLIKLER, model.feature_importances_), key=lambda x: -x[1])
        en_buyuk = onemler[0][1]
        st.markdown("".join(
            f'<div class="onem"><span>{OZELLIK_ADI[s][0]} {OZELLIK_ADI[s][1]}</span>'
            f'<div class="iz"><div class="dolu" style="width:{o / en_buyuk * 100:.0f}%"></div></div>'
            f'<span class="deger">%{o * 100:.1f}</span></div>' for s, o in onemler), unsafe_allow_html=True)
        st.caption("Modelin karar verirken her özelliğe ne kadar dayandığı. Toplamı %100.")
    with g2:
        st.markdown('<div class="bolum">Karışıklık matrisi (test seti)</div>', unsafe_allow_html=True)
        en_cok = KARISIKLIK.max()
        kisa = ["Düşük", "Orta", "Yüksek", "Çok yük."]
        hucreler = ""
        for i in range(4):
            hucreler += f'<tr><th>{SEGMENTLER[i][1]} {kisa[i]}</th>'
            for j in range(4):
                n = KARISIKLIK[i, j]
                renk = "52,211,153" if i == j else "244,114,182"
                hucreler += (f'<td style="background:rgba({renk},{.12 + .75 * n / en_cok:.2f});'
                             f'font-weight:{800 if i == j else 500}">{n}</td>')
            hucreler += "</tr>"
        st.markdown(f'<table class="km"><tr><th>Gerçek ↓ / Tahmin →</th>'
                    + "".join(f"<th>{SEGMENTLER[j][1]} {kisa[j]}</th>" for j in range(4))
                    + f'</tr>{hucreler}</table>', unsafe_allow_html=True)
        st.caption(f"Köşegendeki yeşil kutular doğru tahminler ({KARISIKLIK.trace()} / {KARISIKLIK.sum()}). "
                   "Hatalar çoğunlukla komşu segmentler arasında.")

st.markdown('<div class="not">Telefon bilgileri ve fiyatları: Smartprix (Hindistan) verisinden alınan Kaggle veri seti. '
            f'TL fiyatları Hindistan fiyatlarının 1 ₹ ≈ {KUR} TL kuruyla çevrilmiş hâlidir. '
            'Marka adları ve logoları sahiplerine aittir (logolar: Simple Icons). · Okul projesi</div>',
            unsafe_allow_html=True)

# Sayfa içi yönlendirmeler (görünmez küçük betikler)
# Telefonda yeni bir tahmin çıkınca sonuç kartına kaydır (form uzun olduğu için kart aşağıda kalıyor)
if tahmin_imzasi is not None and st.session_state.get("_son_tahmin") != tahmin_imzasi:
    st.session_state["_son_tahmin"] = tahmin_imzasi
    js_calistir("let d=0;const f=()=>{const w=window;const e=w.document.getElementById('tahmin-sonucu');"
                "if(e){if(w.innerWidth < 768)e.scrollIntoView({behavior:'smooth',block:'start'});}"
                "else if(d++ < 30)setTimeout(f,100)};f();")