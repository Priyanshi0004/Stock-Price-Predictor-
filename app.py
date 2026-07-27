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

st.set_page_config(page_title="Stock Forecasting Study", layout="wide")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "metrics.json")


@st.cache_data
def get_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


st.title("📈 Apple Stock Forecasting — A Time-Series Study")
st.caption("ARIMA(5,1,0) · Linear Regression · SVR · Random Forest · LSTM — "
           "with honest evaluation against a naive baseline.")

tab1, tab2, tab3 = st.tabs(["🔮 Live Forecast", "📊 Model Comparison", "🎯 The Honest Finding"])

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