"""Ham Excel dosyasını hızlı okunan Parquet formatına çevirir.

Kullanım:  python prepare_data.py

Excel (40 MB) Streamlit Cloud'da ~1 dakikada okunurken Parquet birkaç saniyede
okunur. Bu betik veriyi DEĞİŞTİRMEZ; yalnızca format dönüşümü yapar. Tüm temizlik
kuralları uygulama içinde `src/data.py::clean` fonksiyonunda uygulanır.
"""
import pandas as pd

from src.data import EXCEL_PATH, PARQUET_PATH, RAW_DTYPES

if __name__ == "__main__":
    df = pd.read_excel(EXCEL_PATH, dtype={"StockCode": str}).astype(RAW_DTYPES)
    df.to_parquet(PARQUET_PATH, index=False, compression="zstd")
    print(f"{len(df):,} satır → {PARQUET_PATH} ({PARQUET_PATH.stat().st_size / 1e6:.1f} MB)")
