"""Analitik hesaplamalar: KPI, RFM, kümeleme, kohort, ABC-XYZ, stok politikası, sepet analizi."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import safe_div


MONTHS_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
DAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


def month_tr(ts: pd.Timestamp, short: bool = False) -> str:
    name = MONTHS_TR[ts.month - 1]
    return f"{name[:3]} {ts.year}" if short else f"{name} {ts.year}"


# ---------------------------------------------------------------- KPI
def kpis(df: pd.DataFrame) -> dict:
    orders = df["InvoiceNo"].nunique()
    revenue = df["Revenue"].sum()
    gross = df["GrossRevenue"].sum()
    return {
        "revenue": revenue,
        "orders": orders,
        "customers": df["CustomerID"].nunique(),
        "units": df["Quantity"].sum(),
        "aov": safe_div(revenue, orders),
        "discount": 1 - safe_div(revenue, gross) if gross else np.nan,
        "skus": df["StockCode"].nunique(),
        "lines_per_order": safe_div(len(df), orders),
    }


def pct_change(cur: float, prev: float) -> float | None:
    if prev is None or prev == 0 or np.isnan(prev):
        return None
    return (cur - prev) / prev * 100


def time_series(df: pd.DataFrame, freq: str, value: str = "Revenue") -> pd.DataFrame:
    """freq: 'D', 'W', 'M'. Eksik dönemler 0 ile doldurulur."""
    rule = {"D": "D", "W": "W-MON", "M": "MS"}[freq]
    if df.empty:
        return pd.DataFrame(columns=["InvoiceDate", value])
    s = df.set_index("InvoiceDate")[value].resample(rule, label="left", closed="left").sum()
    return s.reset_index()


def top_n(df: pd.DataFrame, by: str, n: int = 10, value: str = "Revenue") -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(
        Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"), Customers=("CustomerID", "nunique"),
    )
    g["Share"] = g["Revenue"] / g["Revenue"].sum() * 100
    return g.sort_values(value, ascending=False).head(n).reset_index()


def pareto(df: pd.DataFrame, by: str) -> pd.DataFrame:
    s = df.groupby(by, observed=True)["Revenue"].sum().sort_values(ascending=False)
    out = pd.DataFrame({"Revenue": s.values})
    out["Rank%"] = (np.arange(1, len(s) + 1)) / len(s) * 100
    out["CumShare%"] = out["Revenue"].cumsum() / out["Revenue"].sum() * 100
    return out


def weekday_month_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="Weekday", columns="Month", values="Revenue", aggfunc="sum", fill_value=0, observed=True)


# ---------------------------------------------------------------- RFM & segmentasyon
RFM_SEGMENTS = [
    ("Şampiyonlar", lambda r, f: (r >= 4) & (f >= 4)),
    ("Sadık Müşteriler", lambda r, f: (r >= 3) & (f >= 4)),
    ("Potansiyel Sadıklar", lambda r, f: (r >= 4) & (f.between(2, 3))),
    ("Yeni Müşteriler", lambda r, f: (r >= 4) & (f <= 1)),
    ("Riskli", lambda r, f: (r <= 2) & (f >= 3)),
    ("Kaybedilmek Üzere", lambda r, f: (r == 3) & (f <= 3)),
    ("Uykuda / Kayıp", lambda r, f: (r <= 2) & (f <= 2)),
]


def _quintile(s: pd.Series) -> pd.Series:
    """1–5 puan. Eşit değerlerde sıra bozulmasın diye 'first' rank; az müşteride de çalışır."""
    return np.ceil(s.rank(method="first", pct=True) * 5).clip(1, 5).astype(int)


def rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID", observed=True).agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        City=("City", lambda s: s.mode().iat[0]),
        CustomerType=("CustomerType", "first"),
    )
    # Eşit değerler çok olduğundan rank üzerinden beşliklere bölünür
    rfm["R"] = 6 - _quintile(rfm["Recency"])
    rfm["F"] = _quintile(rfm["Frequency"])
    rfm["M"] = _quintile(rfm["Monetary"])
    rfm["Segment"] = "Diğer"
    for name, rule in reversed(RFM_SEGMENTS):  # öncelikli kural en son yazılır
        rfm.loc[rule(rfm["R"], rfm["F"]), "Segment"] = name
    return rfm.reset_index()


def kmeans_segments(rfm: pd.DataFrame, k: int, seed: int = 42) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Log dönüşümlü + standartlaştırılmış RFM üzerinde K-Means. (etiketli tablo, silüet, elbow)"""
    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].to_numpy())
    X = StandardScaler().fit_transform(X)
    elbow = []
    for kk in range(2, min(9, len(X))):
        m = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(X)
        elbow.append({"k": kk, "Inertia": m.inertia_, "Silhouette": silhouette_score(X, m.labels_)})
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    out = rfm.copy()
    # Kümeleri ortalama ciroya göre sırala ki etiketler anlamlı olsun
    order = out.assign(c=model.labels_).groupby("c")["Monetary"].mean().sort_values(ascending=False).index
    mapping = {c: f"Küme {i + 1}" for i, c in enumerate(order)}
    out["Cluster"] = pd.Series(model.labels_).map(mapping).values
    return out, silhouette_score(X, model.labels_), pd.DataFrame(elbow)


def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Müşterinin ilk alım ayına göre aylık elde tutma oranı (%)."""
    d = df[["CustomerID", "Month"]].drop_duplicates()
    first = d.groupby("CustomerID", observed=True)["Month"].transform("min")
    d = d.assign(Cohort=first, Age=((d["Month"].dt.year - first.dt.year) * 12 + d["Month"].dt.month - first.dt.month))
    tbl = d.pivot_table(index="Cohort", columns="Age", values="CustomerID", aggfunc="nunique", observed=True)
    return tbl.div(tbl[0], axis=0) * 100


def churn_risk(rfm: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Alışveriş aralığına göre 'beklenenden uzun süredir gelmeyen' değerli müşteriler."""
    gaps = (
        df[["CustomerID", "InvoiceDate"]].drop_duplicates().sort_values("InvoiceDate")
        .groupby("CustomerID", observed=True)["InvoiceDate"].apply(lambda s: s.diff().dt.days.median())
        .rename("MedianGap")
    )
    out = rfm.merge(gaps, left_on="CustomerID", right_index=True, how="left")
    out = out[(out["Frequency"] >= 3) & out["MedianGap"].notna()]
    out["DelayRatio"] = out["Recency"] / out["MedianGap"].clip(lower=1)
    out = out[(out["DelayRatio"] >= 2) & (out["Recency"] >= 30)]
    out["RevenueAtRisk"] = out["Monetary"] / 12  # aylık ortalama ciro
    return out.sort_values("RevenueAtRisk", ascending=False)


def category_lift(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """En büyük kategoriler arasında aynı faturada birlikte görülme 'lift' değeri."""
    cats = df.groupby("Category", observed=True)["Revenue"].sum().nlargest(top).index
    basket = (
        df[df["Category"].isin(cats)][["InvoiceNo", "Category"]].drop_duplicates()
        .assign(v=1).pivot_table(index="InvoiceNo", columns="Category", values="v", fill_value=0, observed=True)
    )
    n = len(basket)
    if n == 0:
        return pd.DataFrame()
    support = basket.mean()
    co = basket.T.dot(basket) / n
    arr = (co.to_numpy() / np.outer(support, support)).copy()
    np.fill_diagonal(arr, np.nan)
    names = [str(c) for c in basket.columns]
    return pd.DataFrame(arr, index=names, columns=names)


def top_rules(lift: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    rows = [(a, b, lift.loc[a, b]) for a, b in combinations(lift.index, 2)]
    return pd.DataFrame(rows, columns=["Kategori A", "Kategori B", "Lift"]).nlargest(n, "Lift")


# ---------------------------------------------------------------- Tedarik zinciri
def abc_xyz(df: pd.DataFrame, a_cut: float = 80, b_cut: float = 95, x_cut: float = 1.0, y_cut: float = 2.0) -> pd.DataFrame:
    """ABC: ciro payı (Pareto). XYZ: haftalık talebin değişim katsayısı (CV).

    Kitap talebi SKU düzeyinde çok kesikli olduğundan klasik 0.5/1.0 CV eşikleri
    neredeyse tüm ürünleri Z yapar; varsayılan eşikler 1.0/2.0 seçilmiştir.
    """
    weeks = pd.Index(sorted(df["Week"].unique()))
    wk = df.pivot_table(index="StockCode", columns="Week", values="Quantity", aggfunc="sum", fill_value=0, observed=True)
    wk = wk.reindex(columns=weeks, fill_value=0)
    stats = df.groupby("StockCode", observed=True).agg(
        Description=("Description", "first"), Category=("Category", "first"),
        Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), LastSale=("InvoiceDate", "max"),
    )
    stats = stats[stats["Revenue"] > 0]
    stats["MeanWeekly"] = wk.mean(axis=1).reindex(stats.index)
    stats["StdWeekly"] = wk.std(axis=1).reindex(stats.index)
    stats["CV"] = stats["StdWeekly"] / stats["MeanWeekly"]
    stats = stats.sort_values("Revenue", ascending=False)
    cum = stats["Revenue"].cumsum() / stats["Revenue"].sum() * 100
    stats["ABC"] = np.where(cum <= a_cut, "A", np.where(cum <= b_cut, "B", "C"))
    stats["XYZ"] = pd.cut(stats["CV"], [-np.inf, x_cut, y_cut, np.inf], labels=["X", "Y", "Z"]).astype(str)
    stats["Class"] = stats["ABC"] + stats["XYZ"]
    return stats.reset_index()


def stock_policy(mean_weekly: float, std_weekly: float, lead_time_weeks: float, service: float, review_weeks: float = 1) -> dict:
    """Emniyet stoğu ve yeniden sipariş noktası (periyodik gözden geçirme, normal talep varsayımı)."""
    z = norm.ppf(service)
    horizon = lead_time_weeks + review_weeks
    ss = z * std_weekly * np.sqrt(horizon)
    rop = mean_weekly * lead_time_weeks + ss
    order_up_to = mean_weekly * horizon + ss
    return {"z": z, "safety_stock": ss, "reorder_point": rop, "order_up_to": order_up_to}


# ---------------------------------------------------------------- Otomatik içgörüler
def generate_insights(df: pd.DataFrame, prev: pd.DataFrame) -> list[str]:
    tips: list[str] = []
    if df.empty:
        return tips
    k, kp = kpis(df), kpis(prev) if not prev.empty else None
    if kp:
        ch = pct_change(k["revenue"], kp["revenue"])
        if ch is not None:
            word = "arttı 📈" if ch > 0 else "azaldı 📉"
            tips.append(f"Ciro bir önceki eşit uzunluktaki döneme göre **%{abs(ch):.1f} {word}**.")
    p = pareto(df, "CustomerID")
    share20 = p.loc[p["Rank%"] <= 20, "CumShare%"].max()
    tips.append(f"Müşterilerin en büyük **%20'si** cironun **%{share20:.0f}'ini** oluşturuyor — müşteri yoğunlaşma riski {'yüksek' if share20 > 75 else 'orta'}.")
    cat = top_n(df, "Category", 1).iloc[0]
    tips.append(f"En çok ciro getiren kategori **{cat['Category']}** (pay: %{cat['Share']:.1f}).")
    city = top_n(df, "City", 2)
    if len(city):
        tips.append(f"**{city.iloc[0]['City']}** cironun **%{city.iloc[0]['Share']:.0f}'ini** üretiyor; coğrafi çeşitlendirme fırsatı mevcut.")
    m = time_series(df, "M")
    if len(m) >= 3:
        best, worst = m.loc[m["Revenue"].idxmax()], m.loc[m["Revenue"].idxmin()]
        tips.append(f"En güçlü ay **{month_tr(best['InvoiceDate'])}**, en zayıf ay **{month_tr(worst['InvoiceDate'])}** (zirve/dip oranı {best['Revenue'] / max(worst['Revenue'], 1):.1f}x).")
    dw = df.groupby("Weekday")["Revenue"].sum()
    tips.append(f"Haftanın en yoğun günü **{DAYS_TR[dw.idxmax()]}** — sevkiyat kapasitesi bu güne göre planlanmalı.")
    return tips
