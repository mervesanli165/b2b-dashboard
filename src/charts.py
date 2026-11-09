"""Grafik teması, renk paleti ve sayı biçimlendirme yardımcıları."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Renk körlüğüne göre doğrulanmış kategorik sıra (sabit sıra; asla döngüyle üretilmez)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
MUTED = "#8a8984"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e8e7e3"
SURFACE = "#ffffff"
STATUS = {"good": "#008300", "warning": "#c98500", "critical": "#d03b3b"}

pio.templates["b2b"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TEXT_2),
        title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left", y=0.98, yanchor="top", yref="container"),
        colorway=SERIES,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=10, r=10, t=56, b=10),
        xaxis=dict(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=TEXT)),
        hovermode="closest",
        bargap=0.25,
        separators=",.",
    )
)
pio.templates.default = "plotly_white+b2b"


def style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height)
    has_legend = fig.layout.showlegend is not False and sum(1 for t in fig.data if getattr(t, "showlegend", False) is not False and t.type in ("scatter", "bar")) > 1
    if has_legend:  # başlık ile lejant çakışmasın: lejant başlığın altına, grafik alanının üstüne
        fig.update_layout(margin_t=86, legend=dict(y=1.0, yanchor="bottom", yref="paper"))
    if title is not None:
        fig.update_layout(title_text=title)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line_width=2)
    fig.update_xaxes(selector=dict(type="date"), tickformat="%d.%m.%y")
    return fig


def tr_month_axis(fig: go.Figure, dates) -> go.Figure:
    """Aylık eksende Türkçe ay kısaltmaları (Plotly varsayılanı İngilizcedir)."""
    months = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
    dates = list(dates)
    fig.update_xaxes(tickmode="array", tickvals=dates, ticktext=[months[d.month - 1] for d in dates])
    return fig


def tl(x: float, digits: int = 1) -> str:
    """Türkçe kısa para biçimi: ₺29,4 Mn / ₺812 B."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    ax = abs(x)
    if ax >= 1e6:
        s = f"{x / 1e6:.{digits}f} Mn"
    elif ax >= 1e3:
        s = f"{x / 1e3:.{digits}f} B"
    else:
        s = f"{x:.{digits}f}"
    return "₺" + s.replace(".", ",")


def num(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"%{x:.{digits}f}".replace(".", ",")
