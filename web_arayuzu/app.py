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
.not {{ color: rgba(255,255,255,.5); font-size: .78rem; text-align: center; margin-top: 1.6rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,.06); border-radius: 12px; padding: .4rem 1rem; }}
.stTabs [aria-selected="true"] {{ background: rgba(167,139,250,.25); }}
@media (max-width: 640px) {{
.block-container {{ margin: 1rem auto; padding: 1.4rem 1rem !important; border-radius: 20px; }}
.baslik {{ font-size: 1.6rem; }}
.cubuk {{ grid-template-columns: 105px 1fr 42px; }}
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
                '<div class="alt">Özellikleri seç, gerçek telefonlarla eğitilmiş model fiyat segmentini söylesin.</div>',
                unsafe_allow_html=True)

    def secenekler(sutun):
        return sorted(df[sutun].dropna().unique().tolist())

    def ortanca(sutun):
        degerler = secenekler(sutun)
        med = df[sutun].median()
        return min(degerler, key=lambda v: abs(v - med))

    sol, sag = st.columns([1.1, 1], gap="large")
    with sol:
        st.markdown('<div class="bolum">⚙️ Donanım</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        ram = c1.select_slider("RAM (GB)", secenekler("ram_capacity"), ortanca("ram_capacity"))
        hafiza = c2.select_slider("Depolama (GB)", secenekler("internal_memory"), ortanca("internal_memory"))
        islemci = st.slider("İşlemci hızı (GHz)", 1.2, 3.2, 2.3, 0.05)
        batarya = st.slider("Batarya (mAh)", 2000, 7000, 5000, 100)
        c3, c4 = st.columns(2)
        sarj = c3.slider("Hızlı şarj (W, 0 = yok)", 0, 150, 33, 1)
        g5 = c4.radio("5G", ["Var", "Yok"], horizontal=True) == "Var"
        st.markdown('<div class="bolum">🖥️ Ekran ve kamera</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        ekran = c5.slider("Ekran boyutu (inç)", 4.0, 7.5, 6.6, 0.1)
        yenileme = c6.select_slider("Yenileme hızı (Hz)", [60, 90, 120, 144, 165], 120)
        c7, c8 = st.columns(2)
        cozunurluk = c7.select_slider("Ekran yüksekliği (piksel)", [960, 1280, 1600, 2340, 2400, 2460, 2800, 3200], 2400)
        kamera = c8.select_slider("Ana kamera (MP)", [8, 12, 13, 48, 50, 64, 108, 200], 50)

    girdi = pd.DataFrame([[ram, hafiza, batarya, islemci, yenileme, kamera, cozunurluk, ekran, int(g5), sarj]],
                         columns=OZELLIKLER)
    olas = model.predict_proba(girdi)[0]
    sinif = int(olas.argmax())
    ad, ikon, renk = SEGMENTLER[sinif]
    cubuklar = "".join(
        f'<div class="cubuk"><span class="ad">{SEGMENTLER[i][1]} {SEGMENTLER[i][0]}</span>'
        f'<div class="iz"><div class="dolu" style="width:{p * 100:.1f}%;background:{SEGMENTLER[i][2]}"></div></div>'
        f'<span class="yuzde">%{p * 100:.0f}</span></div>'
        for i, p in enumerate(olas)
    )
    with sag:
        st.markdown(
            f'<div class="sonuc" style="--renk:{renk}">'
            f'<div class="ust">Tahmin edilen segment</div>'
            f'<div class="ana">{ikon} {ad}</div>'
            f'<div class="guven">Model güveni: %{olas[sinif] * 100:.0f}</div>'
            f'<div class="fiyat">{segment_araligi(sinif)}</div>'
            f'<div class="aralik">Bu segmentteki telefonların gerçek fiyat aralığı</div>'
            f'<div style="margin-top:1rem">{cubuklar}</div></div>',
            unsafe_allow_html=True)

        # Seçilen özelliklere en yakın gerçek telefonlar
        fark = ((df["ram_capacity"] - ram).abs() / 2 + (df["internal_memory"] - hafiza).abs() / 64
                + (df["battery_capacity"] - batarya).abs() / 1000 + (df["processor_speed"] - islemci).abs() * 2
                + (df["refresh_rate"] - yenileme).abs() / 30 + (df["5G_or_not"] - int(g5)).abs())
        benzer = df.assign(fark=fark).nsmallest(4, "fark")
        st.markdown('<div class="bolum" style="margin-top:1.2rem">🔎 Bu özelliklere en yakın gerçek telefonlar</div>',
                    unsafe_allow_html=True)
        st.markdown("".join(
            f'<div class="cubuk" style="grid-template-columns:1fr auto"><span class="ad">{r.isim} · '
            f'{r.ram_capacity:g} GB / {r.internal_memory:g} GB</span><span class="yuzde">{tl(r.price)}</span></div>'
            for r in benzer.itertuples()), unsafe_allow_html=True)

# ------------------------------------------------------------
# 2) Katalog
# ------------------------------------------------------------
SAYFA_BOYUTU = 24


def sarj_yazisi(watt):
    return f"{watt:.0f} W şarj" if watt else "standart şarj"


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
        f'<li>📷 {r.primary_camera_rear:g} MP · ⚙️ {r.processor_speed:g} GHz</li></ul>'
        f'<div class="ozellikler">{ozellikler}{tahmin_rozeti}</div></div>'
    )


with sekme_katalog:
    st.markdown('<div class="baslik">Telefon Kataloğu</div>'
                f'<div class="alt">{len(df)} gerçek telefon, gerçek özellikleri ve satış fiyatlarıyla '
                '(Smartprix, Hindistan). Her kartta modelin tahmini ve doğru bilip bilmediği de yazıyor.</div>',
                unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.5, 1.3, 1, 1])
    tum_markalar = df["marka"].value_counts().index.tolist()
    marka_sec = f1.multiselect("Marka", tum_markalar, placeholder="Tüm markalar")
    segment_sec = f2.multiselect("Segment", list(SEGMENTLER), placeholder="Tüm segmentler",
                                 format_func=lambda s: f"{SEGMENTLER[s][1]} {SEGMENTLER[s][0]}")
    siralama = f3.selectbox("Sırala", ["Fiyat (artan)", "Fiyat (azalan)", "RAM (yüksek)", "Batarya (yüksek)",
                                        "Kamera (yüksek)"])
    arama = f4.text_input("Model ara", placeholder="ör. iPhone 14")
    c1, c2, c3 = st.columns([2, 1, 1])
    en_yuksek_tl = int(df["price"].max() * KUR) + 1000
    tl_aralik = c1.slider("Fiyat aralığı (TL)", 0, en_yuksek_tl, (0, en_yuksek_tl), 500)
    fiyat_aralik = (tl_aralik[0] / KUR, tl_aralik[1] / KUR)
    sadece_5g = c2.checkbox("Sadece 5G")
    yanlislar = c3.checkbox("Modelin yanıldıkları")

    liste = df[df["price"].between(*fiyat_aralik)]
    if marka_sec:
        liste = liste[liste["marka"].isin(marka_sec)]
    if segment_sec:
        liste = liste[liste["segment"].isin(segment_sec)]
    if sadece_5g:
        liste = liste[liste["5G_or_not"] == 1]
    if yanlislar:
        liste = liste[liste["tahmin"] != liste["segment"]]
    if arama.strip():
        kelimeler = arama.lower().split()
        liste = liste[liste["isim"].str.lower().apply(lambda ad: all(k in ad for k in kelimeler))]
    sutun, artan = {"Fiyat (artan)": ("price", True), "Fiyat (azalan)": ("price", False),
                    "RAM (yüksek)": ("ram_capacity", False), "Batarya (yüksek)": ("battery_capacity", False),
                    "Kamera (yüksek)": ("primary_camera_rear", False)}[siralama]
    liste = liste.sort_values(sutun, ascending=artan)

    toplam_sayfa = max(1, -(-len(liste) // SAYFA_BOYUTU))
    b1, b2 = st.columns([3, 1])
    b1.markdown(f'<div class="alt" style="margin:1.9rem 0 0">{len(liste)} telefon bulundu</div>',
                unsafe_allow_html=True)
    sayfa = b2.number_input(f"Sayfa (toplam {toplam_sayfa})", 1, toplam_sayfa, 1)
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
    st.markdown(f"""
- **Veri seti:** {len(df)} gerçek akıllı telefon ve gerçek satış fiyatları (Smartprix, Hindistan, ₹).
  Sitede fiyatlar **1 ₹ ≈ {KUR} TL** kuruyla (XE orta kur, 19 Kasım 2026) TL'ye çevrildi. Bunlar Hindistan
  fiyatlarıdır, Türkiye'deki satış fiyatları vergiler yüzünden genelde daha yüksektir.
  Kaynak: Kaggle "Real World Smartphone's Dataset". En pahalı %1'lik lüks telefonlar (ör. Vertu) çıkarıldı.
- **Fiyat segmentleri:** Telefonlar fiyatlarına göre 4 eşit gruba bölündü. {sinir_metni}
- **Analiz sonuçları (test seti):** KNN %70.1, Random Forest %74.7, XGBoost %76.8, LightGBM %75.3
- **Bu sitedeki model:** 10 donanım özelliğiyle eğitilmiş Random Forest, test doğruluğu %{DOGRULUK * 100:.1f}
- **En belirleyici özellikler:** işlemci hızı, depolama, RAM, hızlı şarj gücü ve ekran çözünürlüğü
- Model fiyatın sadece donanımdan tahmin edilebilen kısmını öğrenir. Marka değeri (ör. Apple) modele dahil
  değildir, bu yüzden bazı telefonlarda yanılır. Katalogda "Modelin yanıldıkları" filtresiyle bunları görebilirsin.
""")

st.markdown('<div class="not">Telefon bilgileri ve fiyatları: Smartprix (Hindistan) verisinden alınan Kaggle veri seti. '
            f'TL fiyatları Hindistan fiyatlarının 1 ₹ ≈ {KUR} TL kuruyla çevrilmiş hâlidir. '
            'Marka adları ve logoları sahiplerine aittir (logolar: Simple Icons). · Okul projesi</div>',
            unsafe_allow_html=True)
