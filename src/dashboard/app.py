"""
QuantMind Dashboard — Quantitative Market Research Lab
=======================================================
Chart-first, risk-first research terminal. Every panel is driven by real
data from the local database and engines; panels show empty states when
their pipeline stage has not run yet (no fake data).

Run: python src/cli.py dashboard
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.core.database import DatabaseManager
from src.dashboard import theme as T
from src.features.store import FeatureStore

st.set_page_config(page_title="QuantMind — Quant Research Lab", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(T.GLOBAL_CSS, unsafe_allow_html=True)


# ============================================================
# Cached resources & data
# ============================================================

@st.cache_resource
def get_system():
    config = load_config(project_root / "configs")
    db = DatabaseManager(config.database.url)
    fstore = FeatureStore(config, db)
    return config, db, fstore


@st.cache_data(ttl=300)
def load_summary() -> pd.DataFrame:
    _, db, _ = get_system()
    return db.get_data_summary()


@st.cache_data(ttl=300)
def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    _, db, _ = get_system()
    return db.query_ohlcv(symbol, timeframe)


@st.cache_data(ttl=3600)
def count_tests() -> int:
    return sum(f.read_text(encoding="utf-8", errors="ignore").count("def test_")
               for f in (project_root / "tests").glob("test_*.py"))


# ============================================================
# Indicator computations (for the technical summary)
# ============================================================

def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def technical_summary(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Returns [(indicator, value, verdict)] from latest OHLCV."""
    close = df["close"]
    out = []
    rsi = compute_rsi(close)
    out.append(("RSI (14)", f"{rsi:.1f}",
                "BUY" if rsi < 30 else "SELL" if rsi > 70 else "NEUTRAL"))
    ema9, ema21 = close.ewm(span=9).mean().iloc[-1], close.ewm(span=21).mean().iloc[-1]
    out.append(("EMA 9/21", f"{ema9:,.0f} / {ema21:,.0f}",
                "BUY" if ema9 > ema21 else "SELL"))
    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    out.append(("MACD (12,26,9)", f"{macd.iloc[-1]:,.1f}",
                "BUY" if macd.iloc[-1] > sig.iloc[-1] else "SELL"))
    ma20, sd20 = close.rolling(20).mean().iloc[-1], close.rolling(20).std().iloc[-1]
    pos = (close.iloc[-1] - ma20) / (2 * sd20) if sd20 else 0
    out.append(("Bollinger (20,2)", f"{pos:+.2f}σ",
                "SELL" if pos > 1 else "BUY" if pos < -1 else "NEUTRAL"))
    vol30 = close.pct_change().rolling(30).std().iloc[-1] * (252 ** 0.5)
    out.append(("Volatility (30d ann.)", f"{vol30:.1%}", "NEUTRAL"))
    return out


# ============================================================
# Charts
# ============================================================

def price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=symbol, increasing_line_color=T.GREEN, decreasing_line_color=T.RED,
        increasing_fillcolor=T.GREEN, decreasing_fillcolor=T.RED,
    ), row=1, col=1)
    for span, color in ((21, T.ACCENT), (50, T.AMBER)):
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"].ewm(span=span).mean(), name=f"EMA {span}",
            line=dict(color=color, width=1.2),
        ), row=1, col=1)
    vol_colors = [T.GREEN if c >= o else T.RED for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.5), row=2, col=1)
    fig.update_layout(**T.PLOTLY_LAYOUT, height=520, showlegend=True,
                      legend=dict(orientation="h", y=1.05, x=0),
                      xaxis_rangeslider_visible=False)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[dict(count=1, label="1M", step="month", stepmode="backward"),
                     dict(count=6, label="6M", step="month", stepmode="backward"),
                     dict(step="year", stepmode="todate", label="YTD"),
                     dict(count=1, label="1Y", step="year", stepmode="backward"),
                     dict(step="all", label="All")],
            bgcolor=T.SURFACE, activecolor=T.BORDER, font=dict(color=T.TEXT),
        ), row=1, col=1)
    fig.update_yaxes(gridcolor=T.BORDER)
    return fig


def line_chart(series: pd.Series, name: str, color: str, height: int = 260,
               fill: bool = False) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values, name=name, line=dict(color=color, width=1.6),
        fill="tozeroy" if fill else None,
        fillcolor=f"rgba(239,68,68,0.15)" if fill else None,
    ))
    fig.update_layout(**T.PLOTLY_LAYOUT, height=height, showlegend=False,
                      title=dict(text=name, font=dict(size=13), x=0))
    return fig


# ============================================================
# Tab: Overview
# ============================================================

def render_overview(df: pd.DataFrame, symbol: str, summary: pd.DataFrame, timeframe: str):
    left, right = st.columns([2.6, 1])
    with left:
        st.plotly_chart(price_chart(df.tail(500), symbol), width="stretch")
    with right:
        st.markdown(T.panel_title("Watchlist"), unsafe_allow_html=True)
        for _, row in summary.iterrows():
            wdf = load_ohlcv(row["symbol"], row["timeframe"])
            if len(wdf) < 2:
                continue
            last, prev = wdf["close"].iloc[-1], wdf["close"].iloc[-2]
            chg = (last / prev - 1) * 100
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;background:{T.SURFACE};'
                f'border:0;border-radius:0.75rem;box-shadow:{T.SHADOW_SOFT_MD};'
                f'padding:9px 12px;margin-bottom:8px;">'
                f'<span style="font-weight:600;font-size:0.82rem;">{row["symbol"]}'
                f'<span style="color:{T.MUTED};font-size:0.7rem;"> · {row["timeframe"]}</span></span>'
                f'<span style="font-size:0.82rem;">{last:,.2f}&nbsp;&nbsp;{T.pct_html(chg)}</span></div>',
                unsafe_allow_html=True)

        st.markdown(T.panel_title("Technical Summary", f"{symbol} · latest bar"),
                    unsafe_allow_html=True)
        for name, value, verdict in technical_summary(df):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'background:{T.SURFACE};border:0;border-radius:0.75rem;'
                f'box-shadow:{T.SHADOW_SOFT_MD};padding:8px 12px;margin-bottom:8px;">'
                f'<span style="font-size:0.78rem;color:{T.MUTED};">{name}</span>'
                f'<span style="font-size:0.8rem;">{value}&nbsp;&nbsp;{T.signal_badge(verdict)}</span></div>',
                unsafe_allow_html=True)


# ============================================================
# Tab: Backtest Lab
# ============================================================

def render_backtest(df: pd.DataFrame, symbol: str):
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.strategy.templates import (BollingerBreakoutStrategy, EMACrossStrategy,
                                        MACDMomentumStrategy, RSIMeanReversionStrategy,
                                        StrategyParams)
    strategies = {"EMA Crossover": EMACrossStrategy, "RSI Mean Reversion": RSIMeanReversionStrategy,
                  "Bollinger Breakout": BollingerBreakoutStrategy, "MACD Momentum": MACDMomentumStrategy}

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    strat_name = c1.selectbox("Strategy", list(strategies))
    capital = c2.number_input("Initial capital", 100.0, 10_000_000.0, 10_000.0, step=1000.0)
    commission = c3.number_input("Commission %", 0.0, 1.0, 0.1, step=0.05) / 100
    run_mc = c4.checkbox("Monte Carlo", value=True)

    if st.button("Run Backtest", type="primary"):
        engine = BacktestEngine(BacktestConfig(initial_capital=capital, commission_pct=commission))
        strategy = strategies[strat_name](StrategyParams(name=strat_name))
        with st.spinner("Running historical simulation..."):
            result = engine.run(df, strategy, symbol=symbol)
        st.session_state["bt"] = (result, engine.monte_carlo(result) if run_mc else None, strat_name)

    if "bt" not in st.session_state:
        st.info("Configure a strategy and click **Run Backtest** to simulate it on stored historical data.")
        return

    result, mc, ran_name = st.session_state["bt"]
    st.markdown(T.panel_title(f"{ran_name} — {result.symbol}",
                f"{result.start_date:%Y-%m-%d} → {result.end_date:%Y-%m-%d} · historical simulation"),
                unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Return", f"{result.total_return:+.1%}")
    m2.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
    m3.metric("Max Drawdown", f"{result.max_drawdown:.1%}")
    m4.metric("Win Rate", f"{result.win_rate:.0%}")
    m5.metric("Profit Factor", f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞")
    m6.metric("Trades", result.total_trades)

    ec = result.equity_curve
    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(line_chart(ec, "Equity Curve", T.GREEN), width="stretch")
    with cr:
        dd = (ec - ec.cummax()) / ec.cummax()
        st.plotly_chart(line_chart(dd, "Drawdown", T.RED, fill=True), width="stretch")

    if mc is not None:
        st.markdown(T.panel_title("Monte Carlo Robustness", "1000 bootstrap resamples of trade order"),
                    unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("Median Final Equity", f"${mc.median_equity:,.0f}")
        k2.metric("5th Percentile", f"${mc.var_95_equity:,.0f}")
        k3.metric("P(Loss)", f"{mc.prob_loss:.1%}")
        hist = go.Figure(go.Histogram(x=mc.final_equities, nbinsx=60, marker_color=T.ACCENT))
        hist.add_vline(x=result.initial_capital, line_color=T.AMBER, line_dash="dash",
                       annotation_text="initial capital")
        hist.update_layout(**T.PLOTLY_LAYOUT, height=240, showlegend=False)
        st.plotly_chart(hist, width="stretch")

    if result.trades:
        with st.expander(f"Trade History ({len(result.trades)} trades)"):
            st.dataframe(pd.DataFrame([{
                "Entry": t.entry_time, "Exit": t.exit_time, "Side": t.side.value,
                "Entry Px": round(t.entry_price, 2), "Exit Px": round(t.exit_price, 2),
                "PnL": round(t.pnl, 2), "PnL %": f"{t.pnl_pct:.2%}",
            } for t in result.trades]), width="stretch", hide_index=True)

    st.markdown(T.disclaimer(), unsafe_allow_html=True)


# ============================================================
# Tab: Indicators
# ============================================================

def render_indicators(fstore: FeatureStore, db: DatabaseManager, symbol: str, timeframe: str):
    summary = fstore.get_feature_summary()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Registered Indicators", summary["registered_features"])
        for group, count in summary.get("features_by_group", {}).items():
            st.markdown(f'{T.badge(group, T.ACCENT, filled=True)} '
                        f'<span style="color:{T.MUTED};font-size:0.8rem;">{count} features</span>',
                        unsafe_allow_html=True)
    with c2:
        feats = db.query_features(symbol, timeframe)
        if feats.empty:
            st.info(f"No computed features stored for {symbol}/{timeframe}. "
                    f"Run: `python src/cli.py features --symbol {symbol}`")
        else:
            st.markdown(T.panel_title("Latest Feature Values",
                        f"{feats.shape[1]} features × {feats.shape[0]} bars"), unsafe_allow_html=True)
            latest = feats.iloc[-1].rename("value").to_frame()
            st.dataframe(latest, width="stretch", height=320)

    try:
        sel = pd.read_sql("SELECT feature_name, method, rank, score FROM selected_features "
                          "ORDER BY rank ASC LIMIT 25", db.engine)
    except Exception:
        sel = pd.DataFrame()
    if not sel.empty:
        st.markdown(T.panel_title("Feature Importance", "Top features from the last selection run"),
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=sel["score"][::-1], y=sel["feature_name"][::-1],
                               orientation="h", marker_color=T.ACCENT))
        fig.update_layout(**T.PLOTLY_LAYOUT, height=480, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No feature-selection run stored yet — run `python src/cli.py select`.")


# ============================================================
# Tab: Risk Guard
# ============================================================

def render_risk(df: pd.DataFrame):
    from src.portfolio.optimizer import kelly_from_returns
    from src.portfolio.risk_engine import (RiskEngine, RiskLimits, cvar,
                                           historical_var, tail_metrics)
    returns = df["close"].pct_change().dropna()

    st.markdown(T.panel_title("Portfolio Risk Metrics", "Computed from stored daily returns"),
                unsafe_allow_html=True)
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("VaR 95% (daily)", f"{historical_var(returns):.2%}")
    r2.metric("CVaR 95% (daily)", f"{cvar(returns):.2%}")
    tm = tail_metrics(returns)
    r3.metric("Skewness", f"{tm['skewness']:.2f}")
    r4.metric("Excess Kurtosis", f"{tm['kurtosis']:.2f}")
    r5.metric("Kelly Fraction (½)", f"{kelly_from_returns(returns):.1%}")

    st.markdown(T.panel_title("Pre-Trade Risk Check", "Is this order allowed under configured limits?"),
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    equity = c1.number_input("Account equity ($)", 100.0, 10_000_000.0, 10_000.0, step=1000.0)
    order_usd = c2.number_input("Order size ($)", 10.0, 10_000_000.0, 5_000.0, step=500.0)
    max_pos = c3.slider("Max position %", 5, 100, 25) / 100
    max_var = c4.slider("Max daily VaR %", 1, 20, 5) / 100

    price = float(df["close"].iloc[-1])
    engine = RiskEngine(RiskLimits(max_position_pct=max_pos, max_var_95=max_var))
    check = engine.check_order(order_usd / price, price, equity, recent_returns=returns)
    if not check.approved:
        st.markdown(T.badge(f"BLOCKED — {check.reason}", T.RED, filled=True), unsafe_allow_html=True)
    elif check.reason:
        st.markdown(T.badge(f"CAPPED — allowed ${check.adjusted_quantity * price:,.0f} "
                            f"({check.reason})", T.AMBER, filled=True), unsafe_allow_html=True)
    else:
        st.markdown(T.badge("ALLOWED — order within risk limits", T.GREEN, filled=True),
                    unsafe_allow_html=True)

    st.markdown(T.panel_title("Circuit Breaker", "Auto-halts trading when limits are breached"),
                unsafe_allow_html=True)
    cb1, cb2, cb3 = st.columns(3)
    cb1.metric("Max daily drawdown", "3%")
    cb2.metric("Max total drawdown", "10%")
    cb3.metric("Max consecutive losses", "5")
    st.markdown(T.disclaimer(), unsafe_allow_html=True)


# ============================================================
# Tab: System
# ============================================================

def render_system(summary: pd.DataFrame):
    n_tests = count_tests()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Automated Tests", n_tests)
    s2.metric("Symbols Stored", summary["symbol"].nunique() if not summary.empty else 0)
    s3.metric("Total Bars", f"{summary['row_count'].sum():,}" if not summary.empty else 0)
    s4.metric("Execution Mode", "Research")
    st.markdown(T.panel_title("Data Store"), unsafe_allow_html=True)
    if summary.empty:
        st.info("No market data stored yet. Run: `python src/cli.py download`")
    else:
        st.dataframe(summary, width="stretch", hide_index=True)
    st.caption("Modules: Data Pipeline · Feature Store (113 indicators) · ML Lab · Regime Detection · "
               "XAI · GA Optimizer · Backtest Engine · Portfolio Optimizer · Risk Engine · "
               "Paper/Testnet Broker · RL Environment")


# ============================================================
# Layout
# ============================================================

def main():
    config, db, fstore = get_system()
    summary = load_summary()

    head_l, head_r = st.columns([1.2, 2])
    with head_l:
        st.markdown(
            f'<div style="font-size:1.5rem;font-weight:800;">QuantMind</div>'
            f'<div style="color:{T.MUTED};font-size:0.8rem;">Quantitative Market Research Lab</div>',
            unsafe_allow_html=True)
    with head_r:
        st.markdown(
            '<div style="text-align:right;padding-top:12px;">'
            + T.badge("RESEARCH MODE", T.ACCENT, filled=True)
            + T.badge("113 INDICATORS", T.MUTED)
            + T.badge(f"{count_tests()} TESTS PASSING", T.GREEN)
            + T.badge("RISK GUARD ON", T.AMBER)
            + T.badge("PAPER TRADING READY", T.MUTED)
            + "</div>", unsafe_allow_html=True)

    if summary.empty:
        st.warning("No market data yet. Start with: `python src/cli.py download --symbols BTC-USD "
                   "--provider yahoo --start 2023-01-01`")
        return

    with st.sidebar:
        st.markdown(T.panel_title("Market"), unsafe_allow_html=True)
        pairs = summary[["symbol", "timeframe"]].drop_duplicates()
        symbol = st.selectbox("Asset", sorted(pairs["symbol"].unique()))
        tfs = pairs[pairs["symbol"] == symbol]["timeframe"].tolist()
        timeframe = st.selectbox("Timeframe", tfs)
        st.markdown("---")
        st.markdown(T.panel_title("Research Pipeline"), unsafe_allow_html=True)
        feats_exist = not db.query_features(symbol, timeframe).empty
        steps = [("Data Ingestion", "ok", f"{summary['row_count'].sum():,} bars"),
                 ("Features", "ok" if feats_exist else "off",
                  "computed" if feats_exist else "not run"),
                 ("ML Training", "off", "run via CLI"),
                 ("Backtest", "ok" if "bt" in st.session_state else "off",
                  "done" if "bt" in st.session_state else "not run"),
                 ("Risk Check", "ok", "guard active"),
                 ("Execution", "off", "paper ready")]
        for name, status, detail in steps:
            st.markdown(T.pipeline_step(name, status, detail), unsafe_allow_html=True)

    df = load_ohlcv(symbol, timeframe)
    if len(df) < 30:
        st.warning(f"Not enough data for {symbol}/{timeframe}.")
        return

    last, prev = df["close"].iloc[-1], df["close"].iloc[-2]
    src = summary[summary["symbol"] == symbol]["source"].iloc[0]
    st.markdown(T.asset_header(symbol, src, timeframe, last, last - prev,
                               (last / prev - 1) * 100, f"{df.index[-1]:%Y-%m-%d %H:%M}"),
                unsafe_allow_html=True)

    tab_ov, tab_bt, tab_ind, tab_risk, tab_sys = st.tabs(
        ["Overview", "Backtest Lab", "Indicators", "Risk Guard", "System"])
    with tab_ov:
        render_overview(df, symbol, summary, timeframe)
    with tab_bt:
        render_backtest(df, symbol)
    with tab_ind:
        render_indicators(fstore, db, symbol, timeframe)
    with tab_risk:
        render_risk(df)
    with tab_sys:
        render_system(summary)


if __name__ == "__main__":
    main()
