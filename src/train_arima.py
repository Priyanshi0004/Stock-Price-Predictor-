"""
train_arima.py
ARIMA(5,1,0) with walk-forward one-step-ahead forecasting.

Reports the model against a naive PERSISTENCE baseline (tomorrow = today)
and DIRECTIONAL ACCURACY (did we predict up vs down), because R2 on price
levels overstates skill on random-walk series.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import load_data


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def directional_accuracy(actual, pred, last_train_value):
    """
    Fraction of days where the model got the direction (up/down) right.
    Compares each prediction's move vs the previous ACTUAL value.
    """
    prev_actual = np.concatenate([[last_train_value], actual[:-1]])
    actual_dir = np.sign(actual - prev_actual)
    pred_dir   = np.sign(pred   - prev_actual)
    return np.mean(actual_dir == pred_dir) * 100


def run_arima(order=(5, 1, 0), save_model=True):
    df = load_data(eval_mode=True)
    series = df["Close"].values

    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]

    # --- Fit once, then walk forward updating state (refit=False) ---
    model_fit = ARIMA(train, order=order).fit()

    predictions = []
    for t in range(len(test)):
        yhat = model_fit.forecast()[0]          # predict next day
        predictions.append(yhat)
        model_fit = model_fit.append([test[t]], refit=False)   # reveal truth

    predictions = np.array(predictions)

    # --- ARIMA metrics ---
    arima_metrics = evaluate(test, predictions)

    # --- Persistence baseline: tomorrow = today ---
    naive = np.concatenate([[train[-1]], test[:-1]])
    naive_metrics = evaluate(test, naive)

    # --- Directional accuracy ---
    arima_dir = directional_accuracy(test, predictions, train[-1])
    

    if save_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump({"order": order, "last_train": train[-1]},
                    "models/arima_meta.pkl")

    return arima_metrics, naive_metrics, arima_dir


if __name__ == "__main__":
    arima_m, naive_m, arima_dir = run_arima()

    table = pd.DataFrame({
        "ARIMA(5,1,0)": arima_m,
        "Persistence (naive)": naive_m,
    }).T

    print("=" * 60)
    print("ARIMA(5,1,0) walk-forward  vs  naive persistence")
    print("=" * 60)
    print(table.round(4))

    print("\nDirectional accuracy (did we predict up vs down correctly?):")
    print(f"  ARIMA: {arima_dir:.2f}%  (≈50% = chance, as efficient-market theory predicts)")