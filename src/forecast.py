"""
forecast.py
Functions the app calls to produce forward-looking forecasts from LIVE data.
Separate from training/evaluation, which runs on the capped 2021 data.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from data import load_data


def arima_forecast(days: int = 30, order=(5, 1, 0)):
    """
    Fit ARIMA on all LIVE data and forecast the next `days` trading days.
    Returns (history_df, forecast_values, conf_int).
    """
    df = load_data(eval_mode=False)          # live data through today
    series = df["Close"].values

    model_fit = ARIMA(series, order=order).fit()

    fc = model_fit.get_forecast(steps=days)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)             # 95% interval

    return df, mean, ci


if __name__ == "__main__":
    df, mean, ci = arima_forecast(days=30)
    print(f"Last actual close: {df['Close'].iloc[-1]:.2f} on {df['Date'].iloc[-1].date()}")
    print(f"30-day forecast (first 5):")
    for i in range(5):
        print(f"  Day {i+1}: {mean[i]:.2f}   [{ci[i,0]:.2f}, {ci[i,1]:.2f}]")