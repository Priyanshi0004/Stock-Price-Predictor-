"""
app.py -- Streamlit interface for the Stock Forecasting Study.
Live ARIMA forecast + honest model evaluation.
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import load_data
from forecast import arima_forecast
from garch_model import forecast_volatility

st.set_page_config(page_title="Stock Forecasting Study", layout="wide")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "metrics.json")


@st.cache_data
def get_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


st.title("📈 Apple Stock Forecasting — A Time-Series Study")
st.caption("ARIMA(5,1,0) · Linear Regression · SVR · Random Forest · LSTM — "
           "with honest evaluation against a naive baseline.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Forecast", "📊 Model Comparison",
    "🎯 The Honest Finding", "🌊 Volatility (GARCH)"
])
# ---------------- TAB 1: LIVE FORECAST ----------------
with tab1:
    st.subheader("Live 30-day ARIMA forecast")
    st.write("Pulls current Apple data from Yahoo Finance and forecasts ahead. "
             "Note the forecast is nearly flat and the uncertainty band widens — "
             "the honest signature of a random-walk series.")

    days = st.slider("Forecast horizon (trading days)", 5, 60, 30)

    with st.spinner("Fetching live data and fitting ARIMA..."):
        df, mean, ci = arima_forecast(days=days)

    last_date = df["Date"].iloc[-1]
    last_price = df["Close"].iloc[-1]
    st.metric("Last close", f"${last_price:.2f}", help=f"as of {last_date.date()}")

    fig, ax = plt.subplots(figsize=(11, 5))
    recent = df.tail(180)
    ax.plot(recent["Date"], recent["Close"], label="Actual (last 180 days)")
    future_idx = pd.date_range(last_date, periods=days + 1, freq="B")[1:]
    ax.plot(future_idx, mean, "--", color="orange", label="Forecast")
    ax.fill_between(future_idx, ci[:, 0], ci[:, 1], color="orange", alpha=0.2,
                    label="95% confidence")
    ax.legend(); ax.set_ylabel("Price ($)")
    st.pyplot(fig)

# ---------------- TAB 2: MODEL COMPARISON ----------------
with tab2:
    r = get_results()
    st.subheader(f"Five models — headline ({r['n_features_15']} features)")

    df15 = pd.DataFrame(r["classical_15"]).T[["RMSE", "R2", "RMSE_reduction_%"]]
    st.dataframe(df15.style.format("{:.3f}"))

    st.markdown(
        f"**LSTM:** R² {r['lstm_80']['R2']} (RMSE {r['lstm_80']['RMSE']}) on a strict "
        f"80/20 split; R² {r['lstm_95']['R2']} (RMSE {r['lstm_95']['RMSE']}) on 95/5, "
        "matching the original study."
    )

    st.subheader(f"Leakage check — causal features only ({r['n_features_12']} features)")
    st.write("Same-day High/Low bound the closing price. Removing them shows how much "
             "of the headline accuracy was look-ahead leakage:")
    df12 = pd.DataFrame(r["classical_12"]).T[["RMSE", "R2"]]
    st.dataframe(df12.style.format("{:.3f}"))

# ---------------- TAB 3: THE HONEST FINDING ----------------
with tab3:
    r = get_results()
    st.subheader("Does ARIMA actually beat doing nothing?")

    comp = pd.DataFrame({
        "ARIMA(5,1,0)": r["arima"],
        "Persistence (tomorrow = today)": r["persistence"],
    }).T[["RMSE", "MAE", "R2"]]
    st.dataframe(comp.style.format("{:.4f}"))

    st.markdown(f"""
    **The finding.** ARIMA's R² of {r['arima']['R2']} looks excellent — but a naive
    "tomorrow = today" baseline scores essentially the same (RMSE {r['persistence']['RMSE']}
    vs {r['arima']['RMSE']}). The high R² reflects that daily prices barely move, not
    forecasting skill.

    **Directional accuracy: {r['arima_directional_accuracy']}%** — about a coin flip.
    On the metric that matters for trading (up vs down), the model has no edge.

    This is consistent with the **weak-form efficient market hypothesis**: past prices
    don't predict future direction. The value of this project isn't a profitable model —
    it's a rigorous demonstration of *why* daily prices resist this class of model.
    """)

st.divider()
st.caption("Evaluation on data capped at 2021 for reproducibility · live forecast uses current data.")

# ---------------- TAB 4: VOLATILITY (GARCH) ----------------
with tab4:
    st.subheader("What actually works: forecasting volatility")
    st.write("Direction is unpredictable (see the previous tab). But *volatility* — "
             "how much the price swings — **is** predictable, because of volatility "
             "clustering: big moves follow big moves. GARCH captures exactly this.")

    vdays = st.slider("Volatility forecast horizon (trading days)", 5, 60, 30, key="vol")

    with st.spinner("Fitting GARCH(1,1) on live returns..."):
        vdf, hist_vol, fc_vol, res = forecast_volatility(days=vdays)

    beta = res.params.get("beta[1]", float("nan"))
    alpha = res.params.get("alpha[1]", float("nan"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Current annualised volatility", f"{hist_vol[-1]:.1f}%")
    col2.metric("Volatility persistence (β)", f"{beta:.2f}",
                help="Fraction of today's volatility carried to tomorrow. "
                     "High β = strong clustering.")
    col3.metric("Shock decay (α+β)", f"{alpha + beta:.3f}",
                help="Near 1 means volatility shocks are long-lived.")

    # Plot: recent realised volatility + the forward forecast
    fig, ax = plt.subplots(figsize=(11, 5))
    recent_dates = vdf["Date"].iloc[-len(hist_vol):].tail(250)
    recent_vol = hist_vol[-250:]
    ax.plot(recent_dates, recent_vol, label="Historical volatility (annualised %)")

    last_date = vdf["Date"].iloc[-1]
    future_idx = pd.date_range(last_date, periods=vdays + 1, freq="B")[1:]
    ax.plot(future_idx, fc_vol, "--", color="crimson", label="GARCH forecast")
    ax.set_ylabel("Annualised volatility (%)")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"""
    **What this shows.** GARCH(1,1) estimates a volatility persistence (β) of
    **{beta:.2f}** — meaning {beta*100:.0f}% of today's volatility carries into
    tomorrow. That high number *is* volatility clustering, quantified.

    Unlike the price forecast, this is a **genuinely useful** result. A risk manager
    can't tell you if Apple rises or falls tomorrow — but GARCH tells them roughly
    how much it will move (~{hist_vol[-1]:.0f}% annualised), which is exactly what's
    needed for options pricing, position sizing, and Value-at-Risk.

    *Extension:* a GJR-GARCH variant would also capture the leverage effect —
    downward shocks raising volatility more than upward ones.
    """)