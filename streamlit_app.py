"""B2B Kitap Dağıtımı — Akıllı Karar Destek Paneli.

ISL 533 Veri Madenciliği · Ödev I
Satış, Pazarlama ve Tedarik Zinciri departmanları için Streamlit gösterge paneli.
Çalıştırma:  streamlit run streamlit_app.py
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import forecasting as fc
from src.charts import GRID, MUTED, SEQ_BLUE, SERIES, STATUS, num, pct, style, tl, tr_month_axis
from src.data import apply_filters, clean, load_raw, missing_profile, previous_period

st.set_page_config(page_title="B2B Kitap Dağıtımı · Karar Destek Paneli", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {background:#fff; border:1px solid #e8e7e3; border-radius:12px; padding:14px 16px;}
      [data-testid="stMetricLabel"] p {font-size:0.82rem; color:#52514e;}
      [data-testid="stMetricValue"] {font-size:1.55rem;}
      .insight {background:#f4f8fe; border-left:4px solid #2a78d6; border-radius:8px; padding:10px 14px; margin-bottom:8px; color:#0b0b0b;}
      .action  {background:#f3faf5; border-left:4px solid #008300; border-radius:8px; padding:10px 14px; margin-bottom:8px; color:#0b0b0b;}
      .warn    {background:#fff8eb; border-left:4px solid #c98500; border-radius:8px; padding:10px 14px; margin-bottom:8px; color:#0b0b0b;}
      .subtle  {color:#52514e; font-size:0.9rem;}
      h3 {margin-top: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================== Veri katmanı (önbellekli)
class Filters(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    cities: tuple
    categories: tuple
    series: tuple
    ctypes: tuple


@st.cache_resource(show_spinner="Veri yükleniyor ve temizleniyor…")
def get_data():
    raw = load_raw()
    df, report = clean(raw)
    return raw, df, report


@st.cache_resource(max_entries=6, show_spinner=False)
def filtered(f: Filters) -> pd.DataFrame:
    _, df, _ = get_data()
    return apply_filters(df, f.start, f.end, f.cities, f.categories, f.series, f.ctypes)


@st.cache_data(max_entries=6, show_spinner=False)
def cached_rfm(f: Filters) -> pd.DataFrame:
    return an.rfm_table(filtered(f))


@st.cache_data(max_entries=12, show_spinner="Kümeleme modeli eğitiliyor…")
def cached_kmeans(f: Filters, k: int):
    return an.kmeans_segments(cached_rfm(f), k)


@st.cache_data(max_entries=6, show_spinner=False)
def cached_abc(f: Filters, a_cut: int, b_cut: int, x_cut: float, y_cut: float) -> pd.DataFrame:
    return an.abc_xyz(filtered(f), a_cut, b_cut, x_cut, y_cut)


@st.cache_data(max_entries=6, show_spinner=False)
def cached_lift(f: Filters) -> pd.DataFrame:
    return an.category_lift(filtered(f))


@st.cache_data(max_entries=24, show_spinner="Tahmin modelleri karşılaştırılıyor…")
def cached_compare(f: Filters, level: str, item: str, value: str, test_weeks: int):
    y = fc.weekly_series(_slice(filtered(f), level, item), value)
    if len(y) < test_weeks + fc.MIN_TRAIN_WEEKS:
        return y, pd.DataFrame()
    return y, fc.compare_models(y, test_weeks)


def _slice(df: pd.DataFrame, level: str, item: str) -> pd.DataFrame:
    if level == "Kategori":
        return df[df["Category"] == item]
    if level == "Şehir":
        return df[df["City"] == item]
    return df


# =============================================================== Kenar çubuğu
raw, data, report = get_data()

with st.sidebar:
    st.markdown("## 📚 Karar Destek Paneli")
    st.caption("B2B kitap dağıtımcısı · 2020 işlem verisi")
    st.markdown("### Filtreler")
    dmin, dmax = data["InvoiceDate"].min().date(), data["InvoiceDate"].max().date()
    preset = st.selectbox("Dönem", ["Tüm yıl", "Son çeyrek (Q4)", "İlk yarı (H1)", "İkinci yarı (H2)", "Özel aralık"])
    ranges = {
        "Tüm yıl": (dmin, dmax),
        "Son çeyrek (Q4)": (pd.Timestamp("2020-10-01").date(), dmax),
        "İlk yarı (H1)": (dmin, pd.Timestamp("2020-06-30").date()),
        "İkinci yarı (H2)": (pd.Timestamp("2020-07-01").date(), dmax),
    }
    if preset == "Özel aralık":
        picked = st.date_input("Tarih aralığı", (dmin, dmax), min_value=dmin, max_value=dmax, format="DD.MM.YYYY")
        start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (dmin, dmax))
    else:
        start, end = ranges[preset]

    city_opts = data.groupby("City", observed=True)["Revenue"].sum().sort_values(ascending=False).index.tolist()
    cat_opts = data.groupby("Category", observed=True)["Revenue"].sum().sort_values(ascending=False).index.tolist()
    cities = st.multiselect("Şehir", city_opts, placeholder="Tümü")
    categories = st.multiselect("Kategori", cat_opts, placeholder="Tümü")
    with st.expander("Gelişmiş filtreler"):
        series = st.multiselect("Fatura serisi", sorted(data["Series"].cat.categories), placeholder="Tümü",
                                help="Fatura numarasının ilk 3 harfi (ör. BAD, BSR) — satış kanalı/şube serisi.")
        ctypes = st.multiselect("Müşteri tipi", sorted(data["CustomerType"].cat.categories), placeholder="Tümü",
                                help="Müşteri kodunun ilk harfine göre gruplama.")

    F = Filters(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59),
                tuple(cities), tuple(categories), tuple(series), tuple(ctypes))
    df = filtered(F)
    p_start, p_end = previous_period(data, F.start.normalize(), F.end.normalize())
    prev = apply_filters(data, p_start, p_end + pd.Timedelta(hours=23, minutes=59), F.cities, F.categories, F.series, F.ctypes)

    st.divider()
    st.markdown(f"**{num(len(df))}** satır · **{num(df['InvoiceNo'].nunique())}** fatura seçili")
    st.download_button("⬇️ Filtrelenmiş veriyi indir (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       "b2b_filtrelenmis.csv", "text/csv", width="stretch")
    st.caption("ISL 533 Veri Madenciliği · Ödev I")

if df.empty:
    st.warning("Seçili filtrelerle eşleşen kayıt yok. Lütfen filtreleri genişletin.")
    st.stop()

# =============================================================== Başlık + KPI şeridi
st.title("B2B Kitap Dağıtımı — Akıllı Karar Destek Paneli")
compare_txt = (f"Karşılaştırma: önceki eşit dönem ({p_start:%d.%m.%Y} – {p_end:%d.%m.%Y})" if not prev.empty
               else "Karşılaştırma: önceki dönem verisi yok (veri seti yalnızca 2020'yi kapsıyor) — değişim oranları için daha kısa bir dönem seçin")
st.markdown(f"<span class='subtle'>Dönem: <b>{F.start:%d.%m.%Y} – {F.end:%d.%m.%Y}</b> · {compare_txt}</span>", unsafe_allow_html=True)

k, kp = an.kpis(df), (an.kpis(prev) if not prev.empty else None)


def delta(key: str, as_points: bool = False):
    if not kp:
        return None
    if as_points:
        return f"{(k[key] - kp[key]) * 100:+.1f} puan".replace(".", ",")
    ch = an.pct_change(k[key], kp[key])
    return None if ch is None else f"{ch:+.1f}%".replace(".", ",")


c = st.columns(6)
c[0].metric("Net Ciro", tl(k["revenue"]), delta("revenue"), help="Σ Miktar × Net Fiyat")
c[1].metric("Sipariş (Fatura)", num(k["orders"]), delta("orders"))
c[2].metric("Aktif Müşteri", num(k["customers"]), delta("customers"))
c[3].metric("Ort. Sipariş Değeri", tl(k["aov"], 0), delta("aov"), help="Net ciro / fatura sayısı")
c[4].metric("Satılan Adet", num(k["units"]), delta("units"))
c[5].metric("Ort. İskonto", pct(k["discount"] * 100), delta("discount", as_points=True), delta_color="inverse",
            help="1 − Net ciro / Liste fiyatı cirosu")

tabs = st.tabs(["🏠 Yönetici Özeti", "💰 Satış", "🎯 Müşteri & Pazarlama", "📦 Tedarik Zinciri", "🔮 Tahmin", "🧹 Veri Kalitesi & Yöntem"])


def render_kmeans(f: Filters, n_customers: int) -> None:
    """K-Means bölümü: k seçimi, silüet eğrisi, küme dağılımı ve profilleri."""
    max_k = min(8, n_customers - 1)
    km1, km2 = st.columns([1, 2.2])
    with km1:
        kk = st.slider("Küme sayısı (k)", 2, max_k, min(4, max_k))
        km, sil, elbow = cached_kmeans(f, kk)
        st.metric("Silüet skoru", f"{sil:.3f}".replace(".", ","), help="−1 ile 1 arası; yüksek = daha iyi ayrışmış kümeler")
        fig = go.Figure()
        fig.add_scatter(x=elbow["k"], y=elbow["Silhouette"], mode="lines+markers", line=dict(color=SERIES[0]), marker=dict(size=8),
                        hovertemplate="k=%{x}<br>Silüet: %{y:.3f}<extra></extra>", name="Silüet")
        fig.add_vline(x=kk, line_dash="dot", line_color=MUTED)
        fig.update_layout(xaxis_title="k", yaxis_title="Silüet", showlegend=False)
        st.plotly_chart(style(fig, 240, "Uygun k seçimi"), width="stretch")
    with km2:
        prof = km.groupby("Cluster").agg(Müşteri=("CustomerID", "count"), Recency=("Recency", "median"),
                                         Frekans=("Frequency", "median"), Ciro=("Monetary", "median"), ToplamCiro=("Monetary", "sum")).reset_index()
        cl_colors = {f"Küme {i + 1}": SERIES[i] for i in range(kk)}
        fig = px.scatter(km, x="Frequency", y="Monetary", color="Cluster", log_x=True, log_y=True, color_discrete_map=cl_colors,
                         category_orders={"Cluster": sorted(cl_colors)}, hover_data={"CustomerID": True, "Recency": True, "City": True},
                         opacity=0.75)
        fig.update_traces(marker=dict(size=8, line=dict(color="white", width=1)))
        fig.update_layout(xaxis_title="Fatura sayısı (log)", yaxis_title="Net ciro ₺ (log)")
        st.plotly_chart(style(fig, 380, "Kümeler: frekans × ciro"), width="stretch")
    st.dataframe(prof, hide_index=True, width="stretch", column_config={
        "Recency": st.column_config.NumberColumn("Medyan gün (R)", format="%.0f"),
        "Frekans": st.column_config.NumberColumn("Medyan fatura (F)", format="%.0f"),
        "Ciro": st.column_config.NumberColumn("Medyan ciro (M, ₺)", format="localized"),
        "ToplamCiro": st.column_config.NumberColumn("Toplam ciro (₺)", format="localized"),
    })


# =============================================================== 1) Yönetici özeti
with tabs[0]:
    left, right = st.columns([1.7, 1])
    with left:
        m = an.time_series(df, "M")
        mp = an.time_series(df, "M", "GrossRevenue")
        fig = go.Figure()
        fig.add_bar(x=m["InvoiceDate"], y=m["Revenue"], name="Net ciro", marker_color=SERIES[0],
                    hovertemplate="%{x|%b %Y}<br>Net ciro: ₺%{y:,.0f}<extra></extra>")
        fig.add_scatter(x=mp["InvoiceDate"], y=mp["GrossRevenue"], name="Liste fiyatıyla ciro", mode="lines+markers",
                        line=dict(color=MUTED, dash="dot"), marker=dict(size=8),
                        hovertemplate="%{x|%b %Y}<br>Liste: ₺%{y:,.0f}<extra></extra>")
        fig.update_yaxes(tickprefix="₺")
        fig = style(fig, 380, "Aylık net ciro ve iskonto öncesi ciro")
        st.plotly_chart(tr_month_axis(fig, m["InvoiceDate"]), width="stretch")
    with right:
        st.markdown("### 💡 Otomatik içgörüler")
        for tip in an.generate_insights(df, prev):
            st.info(tip, icon="💡")

    a, b, cc = st.columns(3)
    with a:
        t = an.top_n(df, "Category", 10).iloc[::-1]
        fig = px.bar(t, x="Revenue", y="Category", orientation="h", custom_data=["Share"])
        fig.update_traces(marker_color=SERIES[0], hovertemplate="%{y}<br>₺%{x:,.0f} · pay %{customdata[0]:.1f}%<extra></extra>")
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        fig.update_xaxes(nticks=4, tickprefix="₺")
        fig.update_yaxes(ticklabelstandoff=4, tickfont_size=11)
        st.plotly_chart(style(fig, 380, "En çok ciro getiren 10 kategori"), width="stretch")
    with b:
        t = an.top_n(df, "City", 10).iloc[::-1]
        fig = px.bar(t, x="Revenue", y="City", orientation="h", custom_data=["Customers"])
        fig.update_traces(marker_color=SERIES[0], hovertemplate="%{y}<br>₺%{x:,.0f} · %{customdata[0]} müşteri<extra></extra>")
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        fig.update_xaxes(nticks=4, tickprefix="₺")
        st.plotly_chart(style(fig, 380, "Şehirlere göre ciro (ilk 10)"), width="stretch")
    with cc:
        s = df.groupby("Series", observed=True)["Revenue"].sum().sort_values(ascending=False).reset_index()
        fig = px.pie(s, names="Series", values="Revenue", hole=0.6, color="Series",
                     color_discrete_sequence=SERIES)
        fig.update_traces(texttemplate=[f"{l}<br>%{{percent}}" if v / s["Revenue"].sum() > 0.02 else "" for l, v in zip(s["Series"], s["Revenue"])],
                          textposition="outside", marker_line_color="white", marker_line_width=2,
                          hovertemplate="%{label}<br>₺%{value:,.0f} (%{percent})<extra></extra>")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 380, "Fatura serisine göre ciro payı"), width="stretch")


# =============================================================== 2) Satış
with tabs[1]:
    st.markdown("### Satış performansı")
    g1, g2 = st.columns([3, 1])
    with g2:
        freq = st.radio("Zaman ölçeği", ["Günlük", "Haftalık", "Aylık"], index=1, horizontal=True)
        metric = st.radio("Ölçü", ["Net ciro", "Adet", "Fatura sayısı"], horizontal=True)
        breakdown = st.selectbox("Kırılım", ["Yok", "Fatura serisi", "Müşteri tipi", "İlk 5 kategori"])
    fmap = {"Günlük": "D", "Haftalık": "W-MON", "Aylık": "MS"}[freq]
    with g1:
        dd = df.copy()
        if breakdown == "İlk 5 kategori":
            top5 = an.top_n(df, "Category", 5)["Category"].tolist()
            dd["Grp"] = np.where(dd["Category"].isin(top5), dd["Category"].astype(str), "Diğer")
        elif breakdown == "Fatura serisi":
            dd["Grp"] = dd["Series"].astype(str)
        elif breakdown == "Müşteri tipi":
            dd["Grp"] = dd["CustomerType"].astype(str)
        else:
            dd["Grp"] = "Toplam"
        agg = {"Net ciro": ("Revenue", "sum"), "Adet": ("Quantity", "sum"), "Fatura sayısı": ("InvoiceNo", "nunique")}[metric]
        ts = dd.groupby([pd.Grouper(key="InvoiceDate", freq=fmap, label="left", closed="left"), "Grp"], observed=True).agg(v=agg).reset_index()
        order = ts.groupby("Grp")["v"].sum().sort_values(ascending=False).index.tolist()
        if "Diğer" in order:
            order = [o for o in order if o != "Diğer"] + ["Diğer"]
        cmap = {g: SERIES[i % len(SERIES)] for i, g in enumerate([o for o in order if o != "Diğer"])}
        cmap["Diğer"] = MUTED
        fig = px.line(ts, x="InvoiceDate", y="v", color="Grp", category_orders={"Grp": order}, color_discrete_map=cmap)
        fig.update_traces(hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.0f}<extra>%{fullData.name}</extra>")
        fig.update_layout(xaxis_title=None, yaxis_title=metric, hovermode="x unified", showlegend=breakdown != "Yok")
        st.plotly_chart(style(fig, 400, f"{freq} {metric.lower()} trendi"), width="stretch")

    h1, h2 = st.columns(2)
    with h1:
        hm = an.weekday_month_heatmap(df)
        hm.index = [an.DAYS_TR[i] for i in hm.index]
        hm.columns = [an.month_tr(c, short=True) for c in hm.columns]
        fig = px.imshow(hm / 1e3, color_continuous_scale=SEQ_BLUE, aspect="auto", labels=dict(color="₺ bin"))
        fig.update_traces(hovertemplate="%{y} · %{x}<br>₺%{z:,.0f} bin<extra></extra>", xgap=2, ygap=2)
        st.plotly_chart(style(fig, 360, "Gün × Ay ciro ısı haritası (₺ bin)"), width="stretch")
    with h2:
        p = an.pareto(df, "CustomerID")
        fig = go.Figure()
        fig.add_scatter(x=p["Rank%"], y=p["CumShare%"], mode="lines", line=dict(color=SERIES[0]), name="Kümülatif pay",
                        hovertemplate="Müşterilerin ilk %{x:.0f}%'i → cironun %{y:.1f}%'i<extra></extra>")
        fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=MUTED, dash="dot"), name="Eşit dağılım", hoverinfo="skip")
        share20 = p.loc[p["Rank%"] <= 20, "CumShare%"].max()
        fig.add_annotation(x=20, y=share20, text=f"İlk %20 → %{share20:.0f}", showarrow=True, arrowhead=0, ax=40, ay=30)
        fig.update_xaxes(title="Müşteri yüzdesi (cirosuna göre sıralı)", ticksuffix="%")
        fig.update_yaxes(title="Kümülatif ciro payı", ticksuffix="%")
        st.plotly_chart(style(fig, 360, "Müşteri yoğunlaşması (Pareto / Lorenz eğrisi)"), width="stretch")

    st.markdown("### Ürün ve kategori detayı")
    t1, t2 = st.columns([1.2, 1])
    with t1:
        cat = df.groupby("Category", observed=True).agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Discount=("DiscountRate", "mean")).reset_index()
        cat = cat.nlargest(30, "Revenue")
        fig = px.treemap(cat, path=["Category"], values="Revenue", color="Discount", color_continuous_scale=SEQ_BLUE,
                         labels={"Discount": "Ort. iskonto"})
        fig.update_traces(hovertemplate="%{label}<br>₺%{value:,.0f}<br>Ort. iskonto: %{color:.1%}<extra></extra>",
                          marker_line_color="white", marker_line_width=2)
        st.plotly_chart(style(fig, 420, "İlk 30 kategori — alan: ciro, renk: ortalama iskonto"), width="stretch")
    with t2:
        n = st.slider("Gösterilecek ürün sayısı", 5, 50, 15)
        prod = df.groupby(["StockCode", "Description"], observed=True).agg(
            Ciro=("Revenue", "sum"), Adet=("Quantity", "sum"), Müşteri=("CustomerID", "nunique"), İskonto=("DiscountRate", "mean")
        ).reset_index().nlargest(n, "Ciro")
        st.dataframe(
            prod.drop(columns="StockCode").rename(columns={"Description": "Ürün"}),
            hide_index=True, width="stretch", height=380,
            column_config={
                "Ciro": st.column_config.ProgressColumn("Ciro (₺)", format="₺%.0f", min_value=0, max_value=float(prod["Ciro"].max())),
                "Adet": st.column_config.NumberColumn(format="%d"),
                "İskonto": st.column_config.NumberColumn("Ort. iskonto", format="percent"),
            },
        )

    st.markdown("### Fiyat ve iskonto etkisi")
    bins = pd.cut(df["DiscountRate"], [0, .2, .3, .35, .4, .45, .5, 1], include_lowest=True)
    disc = df.groupby(bins, observed=True).agg(Satır=("Quantity", "size"), OrtAdet=("Quantity", "mean"), Ciro=("Revenue", "sum")).reset_index()
    disc["Aralık"] = disc["DiscountRate"].apply(lambda i: f"%{i.left * 100:.0f}–{i.right * 100:.0f}")
    d1, d2 = st.columns(2)
    with d1:
        fig = px.bar(disc, x="Aralık", y="Ciro")
        fig.update_traces(marker_color=SERIES[0], hovertemplate="İskonto %{x}<br>₺%{y:,.0f}<extra></extra>")
        fig.update_layout(xaxis_title="İskonto aralığı", yaxis_title=None)
        st.plotly_chart(style(fig, 320, "İskonto aralığına göre ciro"), width="stretch")
    with d2:
        fig = px.bar(disc, x="Aralık", y="OrtAdet")
        fig.update_traces(marker_color=SERIES[1], hovertemplate="İskonto %{x}<br>Satır başına %{y:.2f} adet<extra></extra>")
        fig.update_layout(xaxis_title="İskonto aralığı", yaxis_title=None)
        st.plotly_chart(style(fig, 320, "İskonto arttıkça satır başına adet artıyor mu?"), width="stretch")


# =============================================================== 3) Müşteri & pazarlama
with tabs[2]:
    rfm = cached_rfm(F)
    st.markdown("### RFM segmentasyonu")
    st.caption("R: son alışverişten bu yana geçen gün · F: fatura sayısı · M: net ciro. Her boyut 1–5 arası puanlanır.")
    seg = rfm.groupby("Segment").agg(Müşteri=("CustomerID", "count"), Ciro=("Monetary", "sum"),
                                     OrtRecency=("Recency", "mean"), OrtFrekans=("Frequency", "mean")).reset_index()
    seg["Ciro Payı"] = seg["Ciro"] / seg["Ciro"].sum()
    seg_order = [s for s, _ in an.RFM_SEGMENTS] + ["Diğer"]
    seg_colors = {s: SERIES[i] for i, s in enumerate(seg_order[:8])}
    seg_colors["Diğer"] = MUTED
    s1, s2 = st.columns([1, 1.1])
    with s1:
        segp = seg.set_index("Segment").reindex([s for s in seg_order if s in seg["Segment"].values]).reset_index()
        fig = go.Figure()
        fig.add_bar(y=segp["Segment"], x=segp["Müşteri"] / segp["Müşteri"].sum() * 100, orientation="h", name="Müşteri payı",
                    marker_color=SERIES[0], hovertemplate="%{y}<br>Müşterilerin %{x:.1f}%'i<extra></extra>")
        fig.add_bar(y=segp["Segment"], x=segp["Ciro Payı"] * 100, orientation="h", name="Ciro payı",
                    marker_color=SERIES[1], hovertemplate="%{y}<br>Cironun %{x:.1f}%'i<extra></extra>")
        fig.update_layout(barmode="group", yaxis=dict(autorange="reversed", title=None), xaxis=dict(ticksuffix="%", title=None))
        st.plotly_chart(style(fig, 400, "Segmentlerin müşteri ve ciro payı"), width="stretch")
    with s2:
        st.dataframe(
            segp[["Segment", "Müşteri", "Ciro", "Ciro Payı", "OrtRecency", "OrtFrekans"]],
            hide_index=True, width="stretch",
            column_config={
                "Ciro": st.column_config.NumberColumn("Ciro (₺)", format="localized"),
                "Ciro Payı": st.column_config.NumberColumn(format="percent"),
                "OrtRecency": st.column_config.NumberColumn("Ort. gün", format="%.0f"),
                "OrtFrekans": st.column_config.NumberColumn("Ort. fatura", format="%.1f"),
            },
        )
        actions = {
            "Şampiyonlar": "Özel fiyat/erken erişim; referans programı.",
            "Riskli": "Satış temsilcisi ziyareti, kişiye özel iskonto teklifi.",
            "Kaybedilmek Üzere": "Yeniden kazanım kampanyası (yeni çıkanlar bülteni).",
            "Potansiyel Sadıklar": "Çapraz satış: birlikte alınan kategorilerden paket öner.",
            "Uykuda / Kayıp": "Düşük maliyetli e-posta kampanyası; yanıt yoksa portföyden çıkar.",
        }
        for sname, act in actions.items():
            if sname in seg["Segment"].values:
                st.markdown(f"<div class='action'><b>{sname}:</b> {act}</div>", unsafe_allow_html=True)

    st.markdown("### Makine öğrenmesi ile müşteri kümeleme (K-Means)")
    if len(rfm) < 20:
        st.warning("Kümeleme için seçili filtrede en az 20 müşteri gerekli. Filtreleri genişletin.")
    else:
        render_kmeans(F, len(rfm))

    st.markdown("### Müşteri elde tutma (kohort analizi)")
    coh = an.cohort_retention(df)
    coh.index = [an.month_tr(i, short=True) for i in coh.index]
    fig = px.imshow(coh.round(0), color_continuous_scale=SEQ_BLUE, text_auto=".0f", aspect="auto",
                    labels=dict(x="İlk alımdan sonraki ay", y="İlk alım ayı (kohort)", color="Elde tutma %"))
    fig.update_traces(hovertemplate="Kohort %{y} · +%{x}. ay<br>Aktif oran: %{z:.0f}%<extra></extra>", xgap=2, ygap=2)
    st.plotly_chart(style(fig, 420, "Kohortların aylara göre aktif kalma oranı (%)"), width="stretch")

    ch1, ch2 = st.columns([1.3, 1])
    with ch1:
        st.markdown("### ⚠️ Kayıp riski taşıyan değerli müşteriler")
        st.caption("Kural: en az 3 faturası olan, en az 30 gündür alım yapmayan ve son alışverişinden bu yana olağan alışveriş aralığının **2 katından** fazla süre geçen müşteriler.")
        risk = an.churn_risk(rfm, df)
        st.metric("Risk altındaki aylık ciro", tl(risk["RevenueAtRisk"].sum()), f"{len(risk)} müşteri", delta_color="off")
        st.dataframe(
            risk[["CustomerID", "City", "Segment", "Recency", "MedianGap", "DelayRatio", "Monetary"]].head(50),
            hide_index=True, width="stretch", height=320,
            column_config={
                "CustomerID": "Müşteri", "City": "Şehir",
                "Recency": st.column_config.NumberColumn("Son alımdan beri (gün)"),
                "MedianGap": st.column_config.NumberColumn("Olağan aralık (gün)", format="%.0f"),
                "DelayRatio": st.column_config.NumberColumn("Gecikme katı", format="%.1fx"),
                "Monetary": st.column_config.NumberColumn("Yıllık ciro (₺)", format="localized"),
            },
        )
    with ch2:
        st.markdown("### 🧺 Çapraz satış fırsatları")
        st.caption("Lift > 1: iki kategori aynı faturada tesadüften daha sık birlikte alınıyor.")
        lift = cached_lift(F)
        if not lift.empty:
            rules = an.top_rules(lift, 8)
            fig = px.bar(rules.iloc[::-1], x="Lift", y=rules.iloc[::-1]["Kategori A"] + " + " + rules.iloc[::-1]["Kategori B"], orientation="h")
            fig.update_traces(marker_color=SERIES[2], hovertemplate="%{y}<br>Lift: %{x:.2f}<extra></extra>")
            fig.add_vline(x=1, line_dash="dot", line_color=MUTED)
            fig.update_layout(yaxis_title=None, xaxis_title="Lift")
            st.plotly_chart(style(fig, 380, "En güçlü kategori birliktelikleri"), width="stretch")


# =============================================================== 4) Tedarik zinciri
with tabs[3]:
    st.markdown("### ABC–XYZ envanter sınıflandırması")
    st.caption("ABC: ürünün ciro katkısı (Pareto). XYZ: haftalık talebin değişkenliği (CV = std/ortalama). AX ürünler kolay planlanır, CZ ürünler sipariş üzerine tedarik adayıdır.")
    with st.expander("Sınıflandırma eşiklerini ayarla"):
        e1, e2, e3, e4 = st.columns(4)
        a_cut = e1.number_input("A sınıfı: kümülatif ciro ≤ %", 50, 95, 80, 5)
        b_cut = e2.number_input("B sınıfı: kümülatif ciro ≤ %", a_cut + 1, 99, 95, 1)
        x_cut = e3.number_input("X: CV ≤", 0.1, 3.0, 1.0, 0.1)
        y_cut = e4.number_input("Y: CV ≤", x_cut + 0.1, 6.0, 2.0, 0.1)
    abc = cached_abc(F, a_cut, b_cut, x_cut, y_cut)
    mat = abc.pivot_table(index="ABC", columns="XYZ", values="StockCode", aggfunc="count", fill_value=0).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    rev = abc.pivot_table(index="ABC", columns="XYZ", values="Revenue", aggfunc="sum", fill_value=0).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    x1, x2 = st.columns([1, 1.4])
    with x1:
        share = rev / rev.values.sum() * 100
        text = [[f"{num(mat.iloc[i, j])} ürün<br>{pct(share.iloc[i, j])} ciro" for j in range(3)] for i in range(3)]
        fig = go.Figure(go.Heatmap(z=share.values, x=["X (stabil)", "Y (dalgalı)", "Z (düzensiz)"], y=["A", "B", "C"],
                                   text=text, texttemplate="%{text}", colorscale=SEQ_BLUE, xgap=3, ygap=3,
                                   hovertemplate="%{y}%{x}<br>%{text}<extra></extra>", colorbar=dict(title="Ciro %")))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 360, "ABC–XYZ matrisi (ürün sayısı · ciro payı)"), width="stretch")
    with x2:
        policy = pd.DataFrame({
            "Sınıf": ["AX / AY", "AZ", "BX / BY", "BZ", "CX / CY", "CZ"],
            "Önerilen politika": [
                "Sürekli gözden geçirme, yüksek servis seviyesi (%98), otomatik yenileme",
                "Yüksek emniyet stoğu; talep sinyallerini yakından izle",
                "Periyodik gözden geçirme (haftalık), servis seviyesi %95",
                "Merkez depoda tut, şehirlere sipariş üzerine sevk",
                "Düşük stok, aylık toplu sipariş",
                "Stoksuz / sipariş üzerine tedarik; portföy temizliği adayı",
            ],
        })
        st.dataframe(policy, hide_index=True, width="stretch")
        st.markdown(f"<div class='insight'>Ürünlerin <b>%{(abc['ABC'] == 'A').mean() * 100:.0f}</b>'i cironun <b>%{a_cut}</b>'ini getiriyor. "
                    f"Bu {num((abc['ABC'] == 'A').sum())} A-sınıfı ürün stok bulunurluğunda öncelikli olmalı.</div>", unsafe_allow_html=True)

    st.markdown("### Ürün bazında talep ve stok politikası hesaplayıcı")
    top_a = abc[abc["ABC"] == "A"].head(300)
    labels = [f"{d}  ·  {c}" for d, c in zip(top_a["Description"].fillna("(adsız ürün)"), top_a["Class"])]
    sel = st.selectbox("Ürün seçin (A sınıfı, ciroya göre sıralı)", range(len(top_a)), format_func=lambda i: str(labels[i])) if len(top_a) else None
    if sel is not None:
        item = top_a.iloc[sel]
        sp1, sp2 = st.columns([2, 1])
        with sp2:
            lt = st.slider("Tedarik süresi (hafta)", 0.5, 8.0, 2.0, 0.5)
            sl = st.select_slider("Hedef servis seviyesi", [0.90, 0.95, 0.97, 0.98, 0.99], value=0.95, format_func=lambda v: f"%{v * 100:.0f}")
            pol = an.stock_policy(item["MeanWeekly"], item["StdWeekly"], lt, sl)
            st.metric("Ort. haftalık talep", f"{num(item['MeanWeekly'], 1)} adet")
            st.metric("Emniyet stoğu", f"{num(np.ceil(pol['safety_stock']))} adet", help=f"z={pol['z']:.2f} × σ × √(L+R)")
            st.metric("Yeniden sipariş noktası", f"{num(np.ceil(pol['reorder_point']))} adet", help="Ortalama talep × L + emniyet stoğu")
        with sp1:
            wk = df[df["StockCode"] == item["StockCode"]].groupby("Week")["Quantity"].sum()
            wk = wk.reindex(sorted(df["Week"].unique()), fill_value=0)
            fig = go.Figure()
            fig.add_bar(x=wk.index, y=wk.values, marker_color=SERIES[0], name="Haftalık adet",
                        hovertemplate="%{x|%d.%m.%Y} haftası<br>%{y:,.0f} adet<extra></extra>")
            fig.add_hline(y=pol["reorder_point"], line_dash="dash", line_color=STATUS["warning"],
                          annotation_text="Yeniden sipariş noktası", annotation_position="top left")
            fig.add_hline(y=item["MeanWeekly"], line_dash="dot", line_color=MUTED, annotation_text="Ortalama", annotation_position="bottom left")
            fig.update_layout(xaxis_title=None, yaxis_title="Adet", showlegend=False)
            st.plotly_chart(style(fig, 380, f"Haftalık talep — {item['Description']}"), width="stretch")

    st.markdown("### Bölgesel talep dağılımı ve yavaş dönen ürünler")
    r1, r2 = st.columns(2)
    with r1:
        top_cities = an.top_n(df, "City", 8)["City"].tolist()
        top_cats = an.top_n(df, "Category", 8)["Category"].tolist()
        cm = df[df["City"].isin(top_cities) & df["Category"].isin(top_cats)].pivot_table(
            index="City", columns="Category", values="Quantity", aggfunc="sum", fill_value=0, observed=True)
        cm = cm.reindex(index=top_cities, columns=top_cats)
        cm_share = cm.div(cm.sum(axis=1), axis=0) * 100
        fig = px.imshow(cm_share.round(1), color_continuous_scale=SEQ_BLUE, aspect="auto", text_auto=".0f",
                        labels=dict(color="Şehir içi pay %"))
        fig.update_traces(hovertemplate="%{y} · %{x}<br>Şehir adedinin %{z:.1f}%'i<extra></extra>", xgap=2, ygap=2)
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(style(fig, 400, "Şehir × kategori talep karması (adet payı %)"), width="stretch")
    with r2:
        days_idle = st.slider("Son satıştan bu yana en az (gün)", 30, 180, 90, 15)
        last = df["InvoiceDate"].max()
        slow = abc[(last - abc["LastSale"]).dt.days >= days_idle].copy()
        slow["Gün"] = (last - slow["LastSale"]).dt.days
        st.metric("Yavaş dönen / hareketsiz ürün", num(len(slow)), f"{pct(len(slow) / max(len(abc), 1) * 100)} portföy", delta_color="off")
        st.dataframe(slow.nlargest(100, "Revenue")[["Description", "Category", "Units", "Revenue", "Gün"]], hide_index=True,
                     width="stretch", height=300, column_config={
                         "Description": "Ürün", "Category": "Kategori", "Units": "Toplam adet",
                         "Revenue": st.column_config.NumberColumn("Ciro (₺)", format="localized"),
                         "Gün": st.column_config.NumberColumn("Satışsız gün")})


# =============================================================== 5) Tahmin
with tabs[4]:
    st.markdown("### Haftalık satış tahmini")
    st.caption("Modeller son N hafta saklanarak (holdout) geri test edilir; en düşük MAPE'ye sahip model önerilir. "
               "Not: 2020 verisi pandemi kapanmalarını içerdiği için tahminler yön gösterici niteliktedir.")
    f1, f2, f3, f4 = st.columns(4)
    level = f1.selectbox("Tahmin düzeyi", ["Toplam", "Kategori", "Şehir"])
    if level == "Kategori":
        item = f2.selectbox("Kategori", an.top_n(df, "Category", 30)["Category"].tolist())
    elif level == "Şehir":
        item = f2.selectbox("Şehir", an.top_n(df, "City", 20)["City"].tolist())
    else:
        item = "Toplam"
        f2.text_input("Kapsam", "Tüm filtreli veri", disabled=True)
    value = f3.radio("Hedef değişken", ["Revenue", "Quantity"], format_func=lambda v: "Net ciro" if v == "Revenue" else "Adet", horizontal=True)
    horizon = f4.slider("Tahmin ufku (hafta)", 4, 16, 8)
    test_weeks = st.slider("Geri test penceresi (hafta)", 4, 12, 8)

    y, comp = cached_compare(F, level, str(item), value, test_weeks)
    if comp.empty:
        st.warning(f"Tahmin için yeterli haftalık veri yok ({len(y)} hafta; en az {test_weeks + fc.MIN_TRAIN_WEEKS} gerekli). Daha geniş bir tarih aralığı seçin veya geri test penceresini küçültün.")
    else:
        best = comp.iloc[0]["Model"]
        model = st.selectbox("Model", fc.MODELS, index=fc.MODELS.index(best), help="Varsayılan: geri testte en iyi sonuç veren model")
        bt, met = fc.backtest(y, test_weeks, model)
        fut = fc.forecast(y, horizon, model, test_weeks)
        unit = "₺" if value == "Revenue" else ""
        fig = go.Figure()
        fig.add_scatter(x=y.index, y=y.values, mode="lines", name="Gerçekleşen", line=dict(color=SERIES[0]),
                        hovertemplate="%{x|%d.%m.%Y}<br>" + unit + "%{y:,.0f}<extra>Gerçekleşen</extra>")
        fig.add_scatter(x=bt["Tarih"], y=bt["Tahmin"], mode="lines+markers", name="Geri test tahmini",
                        line=dict(color=SERIES[1], dash="dot"), marker=dict(size=8),
                        hovertemplate="%{x|%d.%m.%Y}<br>" + unit + "%{y:,.0f}<extra>Geri test</extra>")
        fig.add_scatter(x=list(fut["Tarih"]) + list(fut["Tarih"][::-1]), y=list(fut["Üst"]) + list(fut["Alt"][::-1]),
                        fill="toself", fillcolor="rgba(235,104,52,0.15)", mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"), name="%80 tahmin aralığı", hoverinfo="skip")
        fig.add_scatter(x=fut["Tarih"], y=fut["Tahmin"], mode="lines+markers", name="İleri tahmin",
                        line=dict(color=SERIES[1]), marker=dict(size=8),
                        hovertemplate="%{x|%d.%m.%Y}<br>" + unit + "%{y:,.0f}<extra>Tahmin</extra>")
        fig.add_vrect(x0=bt["Tarih"].min() - pd.Timedelta(days=3), x1=bt["Tarih"].max() + pd.Timedelta(days=3),
                      fillcolor=GRID, opacity=0.35, line_width=0, annotation_text="Geri test", annotation_position="top left")
        fig.update_layout(hovermode="x unified", xaxis_title=None, yaxis_title="Net ciro (₺)" if value == "Revenue" else "Adet")
        st.plotly_chart(style(fig, 440, f"{item} — {horizon} haftalık tahmin ({model})"), width="stretch")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAPE (geri test)", pct(met["MAPE"]), help="Ortalama mutlak yüzde hata")
        m2.metric("MAE", (tl(met["MAE"]) if value == "Revenue" else num(met["MAE"])))
        m3.metric("Sapma (bias)", pct(met["Bias%"]), help="Pozitif: model fazla tahmin ediyor")
        total = fut["Tahmin"].sum()
        m4.metric(f"Önümüzdeki {horizon} hafta toplamı", tl(total) if value == "Revenue" else f"{num(total)} adet")

        cmp1, cmp2 = st.columns([1.3, 1])
        with cmp1:
            st.markdown("#### Model karşılaştırması")
            st.dataframe(comp, hide_index=True, width="stretch", column_config={
                "MAE": st.column_config.NumberColumn(format="%.0f"), "RMSE": st.column_config.NumberColumn(format="%.0f"),
                "MAPE": st.column_config.NumberColumn("MAPE %", format="%.1f"), "Bias%": st.column_config.NumberColumn("Sapma %", format="%.1f")})
        with cmp2:
            st.markdown("#### Karar desteği")
            last4 = y.iloc[-4:].mean()
            nxt4 = fut["Tahmin"].iloc[:4].mean()
            chg = (nxt4 - last4) / last4 * 100 if last4 else 0
            box = "action" if chg >= 0 else "warn"
            st.markdown(f"<div class='{box}'>Önümüzdeki 4 haftanın ortalaması, son 4 haftaya göre <b>%{chg:+.1f}</b> "
                        f"{'artış' if chg >= 0 else 'düşüş'} öngörüyor.</div>".replace("%+", "%").replace(".", ","), unsafe_allow_html=True)
            st.markdown("<div class='insight'><b>Satış:</b> hedefleri tahmin aralığının alt sınırına göre, kampanyaları üst sınıra göre planlayın.<br>"
                        "<b>Tedarik:</b> sipariş miktarını tahmin + emniyet stoğu ile belirleyin.</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Tahmin tablosunu indir (CSV)", fut.to_csv(index=False).encode("utf-8-sig"), "tahmin.csv", "text/csv")


# =============================================================== 6) Veri kalitesi & yöntem
with tabs[5]:
    st.markdown("### Veri ön işleme özeti")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Ham satır", num(report.raw_rows))
    q2.metric("Temiz satır", num(len(data)), f"{num(len(data) - report.raw_rows)} satır", delta_color="off")
    q3.metric("Ham sütun → analiz sütunu", f"{raw.shape[1]} → {data.shape[1]}")
    q4.metric("Kapsanan dönem", f"{data['InvoiceDate'].min():%d.%m} – {data['InvoiceDate'].max():%d.%m.%Y}")

    st.markdown("#### Temizlik günlüğü")
    st.dataframe(report.to_frame(), hide_index=True, width="stretch")

    v1, v2 = st.columns(2)
    with v1:
        mp_ = missing_profile(raw)
        fig = px.bar(mp_, x="Sütun", y="Eksik", text="Eksik")
        fig.update_traces(marker_color=SERIES[0], textposition="outside", hovertemplate="%{x}<br>%{y:,} eksik<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(style(fig, 320, "Ham veride eksik değer sayısı"), width="stretch")
    with v2:
        qd = data["Quantity"].clip(upper=data["Quantity"].quantile(0.999))
        fig = px.histogram(qd, nbins=60, log_y=True)
        fig.update_traces(marker_color=SERIES[0], hovertemplate="%{x} adet<br>%{y:,} satır<extra></extra>")
        thr = data.loc[data["IsBulkOrder"], "Quantity"].min()
        fig.add_vline(x=thr, line_dash="dash", line_color=STATUS["warning"], annotation_text="Toplu sipariş eşiği (%99)")
        fig.update_layout(showlegend=False, xaxis_title="Satır başına adet", yaxis_title="Satır sayısı (log)")
        st.plotly_chart(style(fig, 320, "Miktar dağılımı ve aykırı değer eşiği"), width="stretch")

    st.markdown("#### Değişken sözlüğü")
    st.dataframe(pd.DataFrame([
        ("InvoiceNo", "Fatura numarası; ilk 3 harf seri (BAD, BSR, DPF, DPE, MUH)", "Ham"),
        ("CustomerID", "Müşteri (bayi) kodu", "Ham"),
        ("InvoiceDate", "Fatura tarihi", "Ham → datetime"),
        ("StockCode / Description", "Ürün ISBN/barkodu ve adı", "Ham"),
        ("Category / City", "Ürün kategorisi, müşteri şehri", "Ham → eksikler dolduruldu"),
        ("Quantity, NetPrice, UnitPrice", "Adet, iskontolu birim fiyat, liste fiyatı", "Ham"),
        ("Revenue", "Quantity × NetPrice", "Türetilmiş"),
        ("GrossRevenue", "Quantity × UnitPrice", "Türetilmiş"),
        ("DiscountRate", "1 − NetPrice / UnitPrice", "Türetilmiş"),
        ("Series, CustomerType", "Fatura serisi, müşteri kodu öneki", "Türetilmiş"),
        ("Month, Week, Weekday", "Zaman kırılımları", "Türetilmiş"),
        ("IsBulkOrder", "Satır adedi %99 persentilin üzerinde mi", "Türetilmiş (bayrak)"),
    ], columns=["Alan", "Açıklama", "Kaynak"]), hide_index=True, width="stretch")

    with st.expander("Kullanılan yöntemler"):
        st.markdown(
            """
- **RFM analizi:** Her müşteri için Recency/Frequency/Monetary değerleri beşliklere bölünür ve kural tabanlı segmentlere atanır.
- **K-Means kümeleme:** RFM değerleri log dönüşümü ve standartlaştırma sonrası kümelenir; k seçimi silüet skoruyla desteklenir.
- **Kohort analizi:** Müşteriler ilk alım aylarına göre gruplanıp sonraki aylardaki aktiflik oranı izlenir.
- **Sepet analizi (lift):** Kategorilerin aynı faturada birlikte bulunma olasılığı, bağımsızlık varsayımına göre oranlanır.
- **ABC–XYZ:** Pareto ile ciro katkısı, haftalık talebin değişim katsayısı (CV) ile öngörülebilirlik sınıflanır.
- **Stok politikası:** Emniyet stoğu = z · σ_haftalık · √(L + R); yeniden sipariş noktası = μ · L + emniyet stoğu.
- **Tahmin:** Holt sönümlü trend (statsmodels), Random Forest / Gradient Boosting / Ridge (scikit-learn, 1–4 hafta gecikme + hareketli istatistik + mevsimsel sin/cos özellikleri, özyinelemeli çok adımlı tahmin) ve hareketli ortalama referans modeli; holdout geri testi ile MAPE/MAE/RMSE karşılaştırması.
            """
        )

    st.markdown("#### Veri gezgini")
    q = st.text_input("Ürün adında ara", placeholder="ör. Puzzle, Sabahattin Ali…")
    view = df if not q else df[df["Description"].str.contains(q, case=False, na=False)]
    st.dataframe(view.head(1000), width="stretch", height=320, hide_index=True)
    st.caption(f"{num(len(view))} eşleşen satırdan ilk 1.000 tanesi gösteriliyor (filtreler uygulanmış).")
