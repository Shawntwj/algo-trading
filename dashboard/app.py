# DEPRECATED: this Streamlit dashboard is superseded by the React + FastAPI stack.
# Keep for reference until Task 5 (live runner) ships. New UI lives in /frontend, API in /api.
# See README for `make dev`.
from __future__ import annotations

import sys
from pathlib import Path

# Allow running with `streamlit run dashboard/app.py` from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtest import compare, polars_to_wide, run_backtest, sweep
from config import load_settings
from data import list_tickers, load_bars
from strategies import REGISTRY


st.set_page_config(page_title="Algo Research", layout="wide")
st.title("Equities Signal Research")

settings = load_settings()


# ─── Sidebar: data + strategy selection ─────────────────────────────────────
with st.sidebar:
    st.header("Data")
    try:
        available = list_tickers() or settings.universe
    except Exception as exc:
        st.error(f"ClickHouse unavailable: {exc}")
        available = settings.universe

    tickers = st.multiselect("Tickers", options=available, default=available[:3])
    start = st.date_input("Start", value=pd.Timestamp(settings.backfill_start).date())
    end = st.date_input("End", value=pd.Timestamp(settings.end_date).date())
    interval = st.selectbox("Interval", options=settings.intervals, index=0)

    st.header("Strategy")
    strategy_name = st.selectbox("Strategy", options=list(REGISTRY.keys()))
    strategy_cls = REGISTRY[strategy_name]

    st.header("Costs")
    commission = st.number_input("Commission (per side)", value=settings.costs.commission, step=0.0001, format="%.4f")
    slippage = st.number_input("Slippage (per side)", value=settings.costs.slippage, step=0.0001, format="%.4f")

    st.header("Parameters")
    defaults = strategy_cls.default_params()
    grid_defaults = strategy_cls.param_grid()
    mode = st.radio("Mode", options=["Single", "Sweep"], horizontal=True)

    selected_params: dict = {}
    sweep_grid: dict = {}
    for param, default in defaults.items():
        options = grid_defaults.get(param, [default])
        if mode == "Single":
            selected_params[param] = st.select_slider(param, options=options, value=default) \
                if len(options) > 1 else st.number_input(param, value=default)
        else:
            sweep_grid[param] = st.multiselect(param, options=options, default=options)

    run = st.button("Run", type="primary", use_container_width=True)


# ─── Main ───────────────────────────────────────────────────────────────────
if not run:
    st.info("Configure inputs in the sidebar and press **Run**.")
    st.stop()

if not tickers:
    st.warning("Pick at least one ticker.")
    st.stop()

with st.spinner("Loading bars from ClickHouse…"):
    df = load_bars(tickers, start=str(start), end=str(end), interval=interval)
    if df.is_empty():
        st.error("No bars in ClickHouse for that selection. Run a backfill first.")
        st.stop()
    wide = polars_to_wide(df)

st.success(f"Loaded {len(wide):,} bars across {len(tickers)} tickers.")

# ─── Run backtest(s) ────────────────────────────────────────────────────────
freq = "1D" if interval == "1d" else interval

if mode == "Single":
    strat = strategy_cls(**selected_params)
    results = [run_backtest(wide, strat, commission=commission, slippage=slippage, freq=freq)]
else:
    results = sweep(
        wide,
        strategy_cls,
        grid={k: v for k, v in sweep_grid.items() if v},
        commission=commission,
        slippage=slippage,
        freq=freq,
    )

if not results:
    st.warning("Sweep produced no valid combos.")
    st.stop()

# ─── Metrics table ──────────────────────────────────────────────────────────
st.subheader("Metrics")
metrics_df = compare(results)
st.dataframe(metrics_df, use_container_width=True)

# ─── Equity curves overlay ──────────────────────────────────────────────────
st.subheader("Equity curves (portfolio total)")
fig_eq = go.Figure()
for r in results:
    eq = r.equity_curve()
    if isinstance(eq, pd.DataFrame):
        eq = eq.sum(axis=1)
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name=r.label(), mode="lines"))
fig_eq.update_layout(height=420, hovermode="x unified", legend_orientation="h")
st.plotly_chart(fig_eq, use_container_width=True)

# ─── Best run: price + entry/exit markers ───────────────────────────────────
st.subheader("Best run — price with entry/exit markers")
best_label = metrics_df.iloc[0]["label"]
best = next(r for r in results if r.label() == best_label)
strat_obj = REGISTRY[best.strategy_name](**best.params)
sig = strat_obj.generate_signals(wide)

ticker_for_chart = st.selectbox("Chart ticker", options=tickers, index=0)
close_series = wide["close"][ticker_for_chart]
entries = sig.entries[ticker_for_chart]
exits = sig.exits[ticker_for_chart]

fig_px = go.Figure()
fig_px.add_trace(go.Scatter(x=close_series.index, y=close_series.values, name="close", mode="lines"))
fig_px.add_trace(go.Scatter(
    x=close_series.index[entries.values], y=close_series[entries.values].values,
    mode="markers", marker_symbol="triangle-up", marker_color="green", marker_size=10, name="entry",
))
fig_px.add_trace(go.Scatter(
    x=close_series.index[exits.values], y=close_series[exits.values].values,
    mode="markers", marker_symbol="triangle-down", marker_color="red", marker_size=10, name="exit",
))
fig_px.update_layout(height=420, hovermode="x unified")
st.plotly_chart(fig_px, use_container_width=True)

# ─── Heatmap (sweep mode, exactly 2 swept params) ───────────────────────────
if mode == "Sweep":
    swept = [p for p, vs in sweep_grid.items() if vs and len(vs) > 1]
    if len(swept) == 2:
        st.subheader(f"Sharpe heatmap — {swept[0]} × {swept[1]}")
        pa, pb = swept
        pivot = (
            metrics_df
            .pivot_table(index=f"param_{pa}", columns=f"param_{pb}", values="sharpe", aggfunc="mean")
        )
        fig_hm = px.imshow(
            pivot, color_continuous_scale="RdYlGn", origin="lower", aspect="auto",
            labels={"color": "Sharpe"},
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    elif len(swept) > 2:
        st.info("Heatmap shown only when exactly 2 params are swept. Pick 2 to visualize.")
