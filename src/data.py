"""Veri yükleme ve ön işleme katmanı.

Ham Excel dosyası (664 bin satır) Streamlit Cloud'da yavaş okunduğu için
`prepare_data.py` ile bir kez Parquet'e çevrilir. Uygulama önce Parquet'i,
bulamazsa Excel'i okur. Temizleme kuralları her iki durumda da burada,
şeffaf bir "temizlik günlüğü" üretilerek uygulanır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXCEL_PATH = DATA_DIR / "B2B_Transaction_Data_.xlsx"
PARQUET_PATH = DATA_DIR / "transactions_raw.parquet"

RAW_DTYPES = {
    "InvoiceNo": "string",
    "CustomerID": "string",
    "StockCode": "string",
    "Description": "string",
    "Category": "string",
    "City": "string",
}

CUSTOMER_TYPES = {"A": "A-Tipi Bayi", "P": "P-Tipi Müşteri", "S": "S-Tipi (Toptan)", "D": "D-Tipi"}
UNKNOWN_CITY = "BİLİNMİYOR"
UNKNOWN_CATEGORY = "Kategorisiz"
MIN_VALID_PRICE = 0.01  # TL; bunun altındaki birim fiyatlar sembolik/promosyon kaydıdır


@dataclass
class CleaningReport:
    """Her temizlik adımında kaç satırın etkilendiğini tutar."""

    raw_rows: int = 0
    steps: list[dict] = field(default_factory=list)

    def log(self, step: str, rule: str, affected: int, action: str) -> None:
        self.steps.append({"Adım": step, "Kural": rule, "Etkilenen Satır": int(affected), "Uygulanan İşlem": action})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def load_raw() -> pd.DataFrame:
    """Ham veriyi Parquet (hızlı) veya Excel (yedek) kaynağından okur."""
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    df = pd.read_excel(EXCEL_PATH, dtype={"StockCode": str})
    return df.astype(RAW_DTYPES)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Ham işlem verisini analize hazır hale getirir."""
    rep = CleaningReport(raw_rows=len(raw))
    df = raw.copy()

    # 1) Tam tekrar eden satırlar
    dup = df.duplicated()
    rep.log("1. Tekrarlar", "Tüm sütunları aynı olan satırlar", dup.sum(), "Silindi (mükerrer kayıt)")
    df = df[~dup]

    # 2) Eksik miktar
    miss_q = df["Quantity"].isna()
    rep.log("2. Eksik değer", "Quantity boş", miss_q.sum(), "Silindi (miktar bilinmeden ciro hesaplanamaz)")
    df = df[~miss_q]

    # 3) Geçersiz miktar / fiyat (iade-düzeltme ve sembolik fiyatlı kayıtlar)
    bad_q = df["Quantity"] <= 0
    rep.log("3. Geçersiz değer", "Quantity ≤ 0 (iade/düzeltme)", bad_q.sum(), "Silindi")
    df = df[~bad_q]
    bad_p = (df["NetPrice"] <= 0) | (df["UnitPrice"] < MIN_VALID_PRICE)
    rep.log("3. Geçersiz değer", f"NetPrice ≤ 0 veya UnitPrice < {MIN_VALID_PRICE} TL", bad_p.sum(), "Silindi")
    df = df[~bad_p]
    over = df["NetPrice"] > df["UnitPrice"]
    rep.log("3. Geçersiz değer", "NetPrice > UnitPrice (negatif iskonto)", over.sum(), "NetPrice, UnitPrice ile sınırlandı")
    df.loc[over, "NetPrice"] = df.loc[over, "UnitPrice"]

    # 4) Eksik kategorik alanlar
    miss_city = df["City"].isna()
    rep.log("4. Eksik değer", "City boş", miss_city.sum(), f"'{UNKNOWN_CITY}' olarak dolduruldu")
    df["City"] = df["City"].fillna(UNKNOWN_CITY)
    miss_cat = df["Category"].isna()
    rep.log("4. Eksik değer", "Category boş", miss_cat.sum(), f"'{UNKNOWN_CATEGORY}' olarak dolduruldu")
    df["Category"] = df["Category"].fillna(UNKNOWN_CATEGORY)

    # 5) Metin standardizasyonu
    for col in ["Description", "Category", "City"]:
        df[col] = df[col].str.strip()

    # 6) Tip dönüşümleri ve türetilmiş alanlar
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    bad_date = df["InvoiceDate"].isna()
    rep.log("5. Tip dönüşümü", "InvoiceDate → datetime (hatalı tarih)", bad_date.sum(), "Silindi")
    df = df[~bad_date]

    df["Revenue"] = df["Quantity"] * df["NetPrice"]
    df["GrossRevenue"] = df["Quantity"] * df["UnitPrice"]
    df["DiscountRate"] = (1 - df["NetPrice"] / df["UnitPrice"]).clip(0, 1)
    df["Series"] = df["InvoiceNo"].str[:3]
    df["CustomerType"] = df["CustomerID"].str[0].map(CUSTOMER_TYPES).fillna("Diğer")
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    df["Weekday"] = df["InvoiceDate"].dt.dayofweek
    rep.log("6. Özellik türetme", "Revenue, GrossRevenue, DiscountRate, Series, CustomerType, Month, Week", len(df), "Yeni sütunlar eklendi")

    # 7) Aykırı değer işaretleme (silinmez; B2B'de büyük toplu siparişler gerçektir)
    # Miktar dağılımı çok sağa çarpık (medyan 1) olduğundan IQR yerine %99 persentil kullanılır.
    upper = df["Quantity"].quantile(0.99)
    df["IsBulkOrder"] = df["Quantity"] > upper
    rep.log("7. Aykırı değer", f"Quantity > %99 persentil (= {upper:.0f} adet)", df["IsBulkOrder"].sum(), "Silinmedi, 'Toplu sipariş' olarak işaretlendi")

    for col in ["Category", "City", "Series", "CustomerType", "CustomerID", "StockCode"]:
        df[col] = df[col].astype("category")
    df["Description"] = df["Description"].astype("string")
    return df.reset_index(drop=True), rep


def missing_profile(raw: pd.DataFrame) -> pd.DataFrame:
    """Ham verideki eksik değer profilini döndürür."""
    out = raw.isna().sum().rename("Eksik").to_frame()
    out["Oran (%)"] = out["Eksik"] / len(raw) * 100
    out["Benzersiz"] = raw.nunique()
    out["Tip"] = raw.dtypes.astype(str)
    return out.reset_index(names="Sütun")


def apply_filters(df: pd.DataFrame, start, end, cities, categories, series, ctypes) -> pd.DataFrame:
    """Kenar çubuğu filtrelerini uygular. Boş seçim = filtre yok."""
    mask = (df["InvoiceDate"] >= pd.Timestamp(start)) & (df["InvoiceDate"] <= pd.Timestamp(end))
    if cities:
        mask &= df["City"].isin(cities)
    if categories:
        mask &= df["Category"].isin(categories)
    if series:
        mask &= df["Series"].isin(series)
    if ctypes:
        mask &= df["CustomerType"].isin(ctypes)
    return df[mask]


def previous_period(df_all: pd.DataFrame, start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Seçili dönemle aynı uzunluktaki bir önceki dönemin sınırları."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    span = end - start
    return start - span - pd.Timedelta(days=1), start - pd.Timedelta(days=1)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else np.nan
