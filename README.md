# 📚 B2B Kitap Dağıtımı — Akıllı Karar Destek Paneli

**ISL 533 Veri Madenciliği · Ödev I** — Python Streamlit ile akıllı gösterge paneli

**Proje Ekibi:**
Mehmet Bütkül – 507251074
Mert Eren Kuru – 507251077
Merve Şanlı – 507241096


Bir B2B kitap dağıtımcısının 2020 yılına ait 664 bin satırlık işlem verisi üzerinde, **satış, pazarlama ve tedarik zinciri** departmanları için geliştirilmiş etkileşimli bir karar destek sistemi.

🔗 **Canlı panel:** _https://b2b-dashboard-uvrrb7pubnnnm6ohb9denf.streamlit.app_

## Panel yapısı

| Sekme | Hedef kullanıcı | İçerik |
|---|---|---|
| 🏠 Yönetici Özeti | Üst yönetim | 6 KPI (önceki dönemle karşılaştırmalı), aylık ciro, otomatik içgörüler, kategori/şehir/seri dağılımı |
| 💰 Satış | Satış | Günlük/haftalık/aylık trend ve kırılımlar, gün×ay ısı haritası, Pareto (müşteri yoğunlaşması), kategori treemap, en çok satan ürünler, iskonto etkisi |
| 🎯 Müşteri & Pazarlama | Pazarlama / CRM | RFM segmentasyonu ve aksiyon önerileri, **K-Means** kümeleme (silüet analizi), kohort elde tutma, kayıp riski listesi, kategori birliktelik (lift) analizi |
| 📦 Tedarik Zinciri | Tedarik / Depo | ABC–XYZ envanter matrisi, ürün bazında **emniyet stoğu ve yeniden sipariş noktası** hesaplayıcı, şehir×kategori talep karması, yavaş dönen ürünler |
| 🔮 Tahmin | Tüm departmanlar | Haftalık ciro/adet tahmini (Holt, Random Forest, Gradient Boosting, Ridge, hareketli ortalama), holdout geri testi, MAPE/MAE/RMSE karşılaştırması, %80 tahmin aralığı |
| 🧹 Veri Kalitesi & Yöntem | Analist / değerlendirici | Temizlik günlüğü, eksik değer profili, aykırı değer eşiği, değişken sözlüğü, yöntem açıklamaları, veri gezgini |

Kenar çubuğundaki **dönem, şehir, kategori, fatura serisi ve müşteri tipi** filtreleri tüm sekmelere uygulanır. Filtrelenmiş veri ve tahminler CSV olarak indirilebilir.

## Veri ön işleme

1. Tam tekrar eden 6.213 satır silindi.
2. Miktarı boş olan 6 satır silindi.
3. Geçersiz kayıtlar (miktar ≤ 0, net fiyat ≤ 0, sembolik birim fiyat < 0,01 TL) temizlendi.
4. Boş şehir ve kategori alanları "BİLİNMİYOR" / "Kategorisiz" olarak dolduruldu.
5. Tarih alanı datetime tipine çevrildi; Revenue, GrossRevenue, DiscountRate, Series, CustomerType, Month, Week alanları türetildi.
6. Toplu siparişler %99 persentil eşiğiyle **işaretlendi, silinmedi**, çünkü B2B'de büyük siparişler gerçek taleptir.

Tüm kurallar `src/data.py` içindedir. Her adımda etkilenen satır sayısı panelde raporlanır.

## Proje yapısı

```
├── streamlit_app.py          # Ana uygulama (arayüz ve sekmeler)
├── requirements.txt          # Bağımlılıklar
├── prepare_data.py           # Excel → Parquet dönüşümü (yalnızca format)
├── data/
│   ├── B2B_Transaction_Data_.xlsx   # Orijinal veri seti
│   └── transactions_raw.parquet     # Hızlı yükleme için aynı verinin Parquet kopyası
├── src/
│   ├── data.py               # Yükleme, temizlik, filtreleme
│   ├── analytics.py          # KPI, RFM, K-Means, kohort, ABC-XYZ, stok politikası, lift, içgörüler
│   ├── forecasting.py        # Zaman serisi tahmin modelleri ve geri test
│   └── charts.py             # Renk paleti, grafik teması, Türkçe sayı biçimleri
└── .streamlit/config.toml    # Tema
```

## Yerelde çalıştırma

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Parquet dosyası yoksa uygulama Excel'den okur. Bu yaklaşık 40 saniye sürer. Parquet'i yeniden üretmek için `python prepare_data.py` komutunu çalıştırın.
