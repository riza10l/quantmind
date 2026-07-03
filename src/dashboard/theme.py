"""
QuantMind Dashboard Theme
==========================
Design system for the Streamlit dashboard: global CSS, color tokens,
and small HTML component helpers (badges, metric cards, asset header).

Visual identity: Soft UI Dashboard (Creative Tim) — light surface,
white cards with soft shadows, gradient accents, Open Sans typography,
green/red financial semantics preserved.
"""

from __future__ import annotations

# ---- Color tokens (Soft UI Dashboard palette) ----
BG = "#f8f9fa"
SURFACE = "#ffffff"
BORDER = "#e9ecef"
TEXT = "#344767"
MUTED = "#67748e"
GREEN = "#17ad37"
RED = "#ea0606"
AMBER = "#f59e0b"
ACCENT = "#cb0c9f"
BLUE = "#2152ff"

# Soft UI signature gradients
GRAD_PRIMARY = "linear-gradient(310deg,#7928ca,#ff0080)"
GRAD_BLUE = "linear-gradient(310deg,#2152ff,#21d4fd)"
GRAD_GREEN = "linear-gradient(310deg,#17ad37,#98ec2d)"
GRAD_RED = "linear-gradient(310deg,#ea0606,#ff667c)"
GRAD_DARK = "linear-gradient(310deg,#141727,#3a416f)"

SHADOW_SOFT_XL = "0 20px 27px 0 rgba(0,0,0,0.05)"
SHADOW_SOFT_MD = "0 4px 7px -1px rgba(0,0,0,.11),0 2px 4px -1px rgba(0,0,0,.07)"

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="Open Sans, sans-serif", color=TEXT, size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor=BORDER, zeroline=False),
    yaxis=dict(gridcolor=BORDER, zeroline=False),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT)),
)

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Open Sans', sans-serif; color: #344767; }

/* Hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }

.stApp { background: #f8f9fa; }

/* Metric cards — white, soft shadow, no border */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 0;
    border-radius: 1rem;
    box-shadow: 0 20px 27px 0 rgba(0,0,0,0.05);
    padding: 14px 16px;
}
div[data-testid="stMetric"] label {
    color: #67748e; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.02em;
}
div[data-testid="stMetricValue"] { font-weight: 700; color: #344767; }

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: #67748e;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #cb0c9f; }
div[data-baseweb="tab-highlight"] { background-color: #cb0c9f; }

/* Tables */
div[data-testid="stDataFrame"] {
    border: 0;
    border-radius: 1rem;
    box-shadow: 0 20px 27px 0 rgba(0,0,0,0.05);
    overflow: hidden;
}

/* Sidebar — white card look */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e9ecef;
    box-shadow: 0 4px 7px -1px rgba(0,0,0,.11);
}

/* Primary buttons — Soft UI gradient */
button[kind="primary"], div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(310deg,#7928ca,#ff0080) !important;
    border: 0 !important;
    border-radius: 0.5rem !important;
    box-shadow: 0 4px 7px -1px rgba(0,0,0,.11),0 2px 4px -1px rgba(0,0,0,.07);
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.02em;
}

/* Expanders and inputs pick up soft radius */
div[data-testid="stExpander"] {
    border: 0; border-radius: 1rem;
    box-shadow: 0 20px 27px 0 rgba(0,0,0,0.05);
    background: #ffffff;
}

hr { border-color: #e9ecef; }
</style>
"""

_CARD = (f"background:{SURFACE};border:0;border-radius:1rem;"
         f"box-shadow:{SHADOW_SOFT_XL};")


def badge(text: str, color: str = ACCENT, filled: bool = False) -> str:
    """Small status badge (inline HTML)."""
    if filled:
        return (
            f'<span style="background:{color}1a;color:{color};'
            f'border-radius:0.5rem;padding:3px 10px;font-size:0.68rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.02em;'
            f'margin-right:6px;white-space:nowrap;">{text}</span>'
        )
    return (
        f'<span style="color:{color};border:1px solid {BORDER};background:{SURFACE};'
        f'border-radius:0.5rem;padding:3px 10px;font-size:0.68rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.02em;box-shadow:{SHADOW_SOFT_MD};'
        f'margin-right:6px;white-space:nowrap;">{text}</span>'
    )


def signal_badge(label: str) -> str:
    """Buy/Sell/Neutral badge with financial color semantics."""
    color = {"BUY": GREEN, "STRONG BUY": GREEN, "SELL": RED, "STRONG SELL": RED}.get(label.upper(), MUTED)
    return badge(label.upper(), color, filled=True)


def pct_html(value: float, suffix: str = "%") -> str:
    """Green/red colored percentage."""
    color = GREEN if value >= 0 else RED
    sign = "+" if value >= 0 else ""
    return f'<span style="color:{color};font-weight:700;">{sign}{value:.2f}{suffix}</span>'


def asset_header(symbol: str, source: str, timeframe: str, price: float,
                 change_abs: float, change_pct: float, last_updated: str) -> str:
    """Soft UI stat-card style asset header block."""
    color = GREEN if change_pct >= 0 else RED
    sign = "+" if change_pct >= 0 else ""
    return f"""
<div style="{_CARD}padding:18px 22px;margin-bottom:8px;">
  <div style="color:{MUTED};font-size:0.7rem;font-weight:700;letter-spacing:0.04em;
              text-transform:uppercase;">
    {symbol} &nbsp;·&nbsp; {source.upper()} &nbsp;·&nbsp; {timeframe}
  </div>
  <div style="display:flex;align-items:baseline;gap:14px;margin-top:4px;">
    <span style="font-size:2.1rem;font-weight:800;color:{TEXT};">{price:,.2f}</span>
    <span style="font-size:1.05rem;font-weight:700;color:{color};">
      {sign}{change_abs:,.2f} ({sign}{change_pct:.2f}%)
    </span>
  </div>
  <div style="color:{MUTED};font-size:0.72rem;margin-top:2px;">Last updated: {last_updated}</div>
</div>
"""


def panel_title(text: str, sub: str = "") -> str:
    sub_html = f'<div style="color:{MUTED};font-size:0.78rem;margin-top:2px;">{sub}</div>' if sub else ""
    return (
        f'<div style="margin:6px 0 10px 0;">'
        f'<div style="font-size:1.05rem;font-weight:700;color:{TEXT};">{text}</div>{sub_html}</div>'
    )


def disclaimer() -> str:
    return (
        f'<div style="{_CARD}border-left:3px solid {AMBER};'
        f'padding:10px 14px;color:{MUTED};font-size:0.76rem;margin-top:10px;">'
        f'QuantMind is a research and simulation tool. Backtest results are historical and do not '
        f'guarantee future performance. Use paper trading or testnet before considering live execution.'
        f'</div>'
    )


def pipeline_step(name: str, status: str, detail: str = "") -> str:
    """One step of the research pipeline status strip."""
    color = {"ok": GREEN, "warn": AMBER, "off": MUTED}.get(status, MUTED)
    grad = {"ok": GRAD_GREEN, "warn": GRAD_PRIMARY, "off": GRAD_DARK}.get(status, GRAD_DARK)
    dot = (f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
           f'background:{grad};margin-right:4px;"></span>')
    detail_html = f'<div style="color:{MUTED};font-size:0.68rem;margin-left:13px;">{detail}</div>' if detail else ""
    return (
        f'<div style="flex:1;{_CARD}box-shadow:{SHADOW_SOFT_MD};'
        f'padding:10px 12px;min-width:120px;margin-bottom:6px;">'
        f'{dot}<span style="font-size:0.78rem;font-weight:700;color:{TEXT};">{name}</span>'
        f'{detail_html}</div>'
    )
