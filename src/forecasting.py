"""Haftalık talep/ciro tahmini: istatistiksel ve makine öğrenmesi modelleri + geri test."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MODELS = ["Holt (Sönümlü Trend)", "Random Forest (gecikme özellikleri)", "Gradient Boosting (gecikme özellikleri)", "Ridge Regresyon (gecikme özellikleri)", "Hareketli Ortalama (referans)"]
LAGS = (1, 2, 3, 4)
MIN_TRAIN_WEEKS = 12  # gecikme özellikleri + anlamlı eğitim için alt sınır


def weekly_series(df: pd.DataFrame, value: str = "Revenue") -> pd.Series:
    """Pazartesi başlangıçlı haftalık toplam. Veri aralığının dışına taşan
    (yarım kalmış) ilk/son hafta, tahmini yanıltmaması için atılır."""
    s = df.set_index("InvoiceDate")[value].resample("W-MON", label="left", closed="left").sum()
    first, last = df["InvoiceDate"].min().normalize(), df["InvoiceDate"].max().normalize()
    full = (s.index >= first - pd.Timedelta(days=2)) & (s.index + pd.Timedelta(days=5) <= last)
    s = s[full] if full.sum() >= 12 else s  # kısa filtrelerde kırpma yapma
    return s.asfreq("7D", fill_value=0)


def _features(history: list[float], date: pd.Timestamp) -> list[float]:
    woy = date.isocalendar().week
    f = [history[-l] for l in LAGS]
    f += [np.mean(history[-4:]), np.std(history[-4:]), np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)]
    return f


def _ml_model(name: str):
    if name.startswith("Random"):
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
    if name.startswith("Gradient"):
        return GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
    return Ridge(alpha=1.0)


def _fit_predict(y: pd.Series, horizon: int, model: str) -> np.ndarray:
    future_idx = pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    if model.startswith("Holt"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y.values, trend="add", damped_trend=True, initialization_method="estimated").fit()
        return np.maximum(fit.forecast(horizon), 0)
    if model.startswith("Hareketli"):
        return np.repeat(y.iloc[-4:].mean(), horizon)

    vals, X, t = y.tolist(), [], []
    for i in range(max(LAGS), len(vals)):
        X.append(_features(vals[:i], y.index[i]))
        t.append(vals[i])
    reg = _ml_model(model).fit(np.array(X), np.array(t))
    hist, preds = vals.copy(), []
    for d in future_idx:  # özyinelemeli (recursive) çok adımlı tahmin
        p = max(float(reg.predict(np.array([_features(hist, d)]))[0]), 0)
        preds.append(p)
        hist.append(p)
    return np.array(preds)


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    mask = actual != 0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err[mask] / actual[mask])) * 100) if mask.any() else np.nan,
        "Bias%": float(err.sum() / actual.sum() * -100) if actual.sum() else np.nan,
    }


def backtest(y: pd.Series, test_weeks: int, model: str) -> tuple[pd.DataFrame, dict]:
    train, test = y.iloc[:-test_weeks], y.iloc[-test_weeks:]
    pred = _fit_predict(train, test_weeks, model)
    frame = pd.DataFrame({"Tarih": test.index, "Gerçek": test.values, "Tahmin": pred})
    return frame, metrics(test.values, pred)


def compare_models(y: pd.Series, test_weeks: int) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        _, met = backtest(y, test_weeks, m)
        rows.append({"Model": m, **met})
    return pd.DataFrame(rows).sort_values("MAPE")


def forecast(y: pd.Series, horizon: int, model: str, test_weeks: int) -> pd.DataFrame:
    """Tüm veriyle yeniden eğitip ileri tahmin yapar; aralık geri test hatasından türetilir."""
    bt, _ = backtest(y, test_weeks, model)
    resid_sd = float(np.std(bt["Gerçek"] - bt["Tahmin"], ddof=1)) if len(bt) > 1 else 0.0
    pred = _fit_predict(y, horizon, model)
    idx = pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    widen = np.sqrt(np.arange(1, horizon + 1))
    return pd.DataFrame({
        "Tarih": idx, "Tahmin": pred,
        "Alt": np.maximum(pred - 1.28 * resid_sd * widen, 0),  # ~%80 aralık
        "Üst": pred + 1.28 * resid_sd * widen,
    })
