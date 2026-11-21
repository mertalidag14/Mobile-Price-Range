# ============================================================
# TEKNOTAHMİN - GERÇEK TELEFONLARLA FİYAT SEGMENTİ TAHMİNİ (Streamlit)
# Veri: 970 gerçek akıllı telefon, Smartprix (Hindistan) fiyatları
# Çalıştırma: streamlit run app.py
# ============================================================
import base64
import random
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
content: ""; position: fixed; inset: 0; z-index: 0;
background: radial-gradient(ellipse at 50% 40%, rgba(10,10,25,.2), rgba(10,10,25,.7));
}}
[data-testid="stAppViewContainer"] {{ position: relative; z-index: 1; background: transparent; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{
max-width: 1150px; margin: 2.5rem auto; padding: 2rem 2.5rem 2rem !important;
background: rgba(18, 18, 38, .6); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
border: 1px solid rgba(255,255,255,.12); border-radius: 28px; box-shadow: 0 30px 80px rgba(0,0,0,.45);
}}
.nav {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .8rem;
padding-bottom: 1rem; margin-bottom: 1.2rem; border-bottom: 1px solid rgba(255,255,255,.1); }}
.logo {{ font-size: 1.6rem; font-weight: 900; letter-spacing: -.02em;
background: linear-gradient(90deg, #c4b5fd, #f9a8d4, #93c5fd); -webkit-background-clip: text; background-clip: text; color: transparent; }}
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
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,.06); border-radius: 12px; padding: .4rem 1rem; }}
.stTabs [aria-selected="true"] {{ background: rgba(167,139,250,.25); }}
@media (max-width: 640px) {{
.block-container {{ margin: 1rem auto; padding: 1.4rem 1rem !important; border-radius: 20px; }}
.baslik {{ font-size: 1.6rem; }}
.cubuk {{ grid-template-columns: 105px 1fr 42px; }}
.markalar {{ flex-wrap: nowrap; overflow-x: auto; padding-bottom: .4rem; margin-bottom: .8rem; }}
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
    # Sitede kullanılan model tüm veriyle eğitiliyor
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42).fit(df[OZELLIKLER], df["segment"])
    return model, dogruluk


df, SINIRLAR = veri_yukle()
model, DOGRULUK = model_egit()
df["tahmin"] = model.predict(df[OZELLIKLER])


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


def marka_cipi(marka, adet):
    bilgi = LOGOLAR.get(marka.lower())
    if bilgi and bilgi[5]:  # yazı logosu: logo ismi zaten gösteriyor
        return f'<span class="marka" title="{marka}">{logo(marka, 15)}<small>{adet}</small></span>'
    return f'<span class="marka">{logo(marka)}{marka}<small>{adet}</small></span>'


ilk_markalar = df["marka"].value_counts().head(12)
st.markdown('<div class="markalar"><span class="etiket">Popüler markalar</span>'
            + "".join(marka_cipi(m, n) for m, n in ilk_markalar.items())
            + '</div>', unsafe_allow_html=True)

sekme_tahmin, sekme_katalog, sekme_karsilastir, sekme_hakkinda = st.tabs(
    ["🔮 Fiyat Tahmini", "🛒 Telefon Kataloğu", "⚖️ Karşılaştır", "📊 Proje Hakkında"])

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
    with sol:
        modeller = sorted(df["temel_model"].unique(), key=str.lower)
        isim_arama = st.text_input("📱 Telefon adı (biliyorsan yaz, bilmiyorsan boş bırak)",
                                   placeholder="ör. iPhone, Galaxy S23, Redmi Note")
        kelimeler = isim_arama.lower().split()
        eslesen_modeller = [m for m in modeller if all(k in m.lower() for k in kelimeler)] if kelimeler else []
        secili = None
        if kelimeler and not eslesen_modeller:
            st.warning("Bu isimde telefon bulunamadı. Yazımı kontrol et ya da sadece markayı yaz (ör. Samsung).")
        elif eslesen_modeller:
            secili = st.selectbox(f"Hangi model? ({len(eslesen_modeller)} model bulundu, sağda görselleriyle listeleniyor)",
                                  eslesen_modeller, index=0 if len(eslesen_modeller) == 1 else None,
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
            olas = model.predict_proba(girdi)[0]
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
                f'<div class="sonuc" style="--renk:{renk}">'
                f'<div class="ust">Tahmin edilen segment</div>'
                f'<div class="ana">{ikon} {ad}</div>'
                f'<div class="guven">Model güveni: %{olas[sinif] * 100:.0f}</div>'
                f'<div class="fiyat">{segment_araligi(sinif)}</div>'
                f'<div class="aralik">Bu segmentteki telefonların gerçek fiyat aralığı</div>{gercek}'
                f'<div style="margin-top:1rem">{cubuklar}</div></div>', unsafe_allow_html=True)

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
            marka_sec = st.multiselect("Marka", tum_markalar, placeholder="Tüm markalar")
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
        if gosterilen.empty:
            st.info("Bu filtrelere uyan telefon yok. Filtreleri gevşetmeyi dene.")
        else:
            st.markdown(f'<div class="izgara">{"".join(kart_html(r) for r in gosterilen.itertuples())}</div>',
                        unsafe_allow_html=True)

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

    k1, k2 = st.columns(2)
    a = df.iloc[k1.selectbox("1. telefon", range(len(isimler)), index=varsayilan("iPhone 14"),
                             format_func=lambda i: isimler[i])]
    b = df.iloc[k2.selectbox("2. telefon", range(len(isimler)), index=varsayilan("Galaxy S23"),
                             format_func=lambda i: isimler[i])]

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
    st.markdown(f'<table class="tablo"><tr><th></th><th>{a.isim}</th><th>{b.isim}</th></tr>'
                f'{ust_satir}{govde}</table>', unsafe_allow_html=True)
    st.caption("Yeşil değer o özellikte daha iyi olan telefonu gösterir (fiyatta düşük olan daha iyi).")

# ------------------------------------------------------------
# 4) Proje hakkında
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
| KNN (k=3, StandardScaler) | %70.1 | 0.696 | %61.9 |
| Random Forest | %74.7 | 0.743 | **%70.6** |
| **XGBoost** | **%76.8** | **0.764** | %67.9 |
| LightGBM | %75.3 | 0.746 | %68.0 |

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

st.markdown('<div class="not">Telefon bilgileri ve fiyatları: Smartprix (Hindistan) verisinden alınan Kaggle veri seti. '
            f'TL fiyatları Hindistan fiyatlarının 1 ₹ ≈ {KUR} TL kuruyla çevrilmiş hâlidir. '
            'Marka adları ve logoları sahiplerine aittir (logolar: Simple Icons). · Okul projesi</div>',
            unsafe_allow_html=True)