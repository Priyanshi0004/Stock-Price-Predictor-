"""
train_ml.py  (Part 1: classical models + baseline)

Trains and evaluates:
  - Baseline (DummyRegressor, predicts training mean) -- the naive reference
  - Linear Regression
  - Support Vector Regression (SVR)
  - Random Forest

Reports metrics for the 15-feature study (headline) AND the 12-feature
causal set (leakage check). Chronological split throughout (shuffle=False).
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import load_data
from features import build_features


def evaluate(y_true, y_pred):
    """Return the standard metric set as a dict."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def run_classical(causal_only: bool):
    """Train the four classical models on one feature set, return a results table."""
    df = load_data(eval_mode=True)                      # capped at 2021 for CV parity
    X, y, cols = build_features(df, causal_only=causal_only)

    # Chronological 80/20 split -- NO shuffling
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    results = {}

    # --- Baseline: predict the training mean for every test day ---
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    results["Baseline (mean)"] = evaluate(y_test, dummy.predict(X_test))

    # --- Linear Regression: scale inputs so large-magnitude columns
    #     (year, Volume) don't dominate and destabilise extrapolation ---
    lr_scaler = MinMaxScaler().fit(X_train)
    lr = LinearRegression().fit(lr_scaler.transform(X_train), y_train)
    results["Linear Regression"] = evaluate(y_test, lr.predict(lr_scaler.transform(X_test)))

    # --- SVR: needs scaled inputs (distance-based). Fit scaler on TRAIN only. ---
    scaler = MinMaxScaler().fit(X_train)
    svr = SVR(kernel="linear").fit(scaler.transform(X_train), y_train)
    results["SVR"] = evaluate(y_test, svr.predict(scaler.transform(X_test)))

    # --- Random Forest: predict the CHANGE, not the level, so it can extrapolate ---
    y_train_delta = y_train.values - X_train["lag_1"].values
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_delta)
    rf_pred = rf.predict(X_test) + X_test["lag_1"].values
    results["Random Forest"] = evaluate(y_test, rf_pred)

    return pd.DataFrame(results).T, len(cols)

    # ---------------------------------------------------------------------------
# LSTM (the 5th model) -- deep learning on Close-price sequences
# ---------------------------------------------------------------------------
import os
import joblib

def run_lstm(save_model: bool = True, train_frac: float = 0.8):
    """
    Train an LSTM on 60-day windows of Close price.
    Uses a chronological 80/20 split. Scaler is fit on TRAIN ONLY (no leak).
    Saves the trained model + scaler so the app can load them instantly.
    """
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    df = load_data(eval_mode=True)
    close = df[["Close"]].values                     # shape (N, 1)

       # Chronological split point (parameterised so we can test sensitivity)
    split = int(len(close) * train_frac)

    # Fit scaler on TRAINING data only, then apply to everything
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close[:split])
    scaled = scaler.transform(close)

    # Build 60-day sequences: 60 days in -> next day out
    window = 60
    def make_sequences(series):
        Xs, ys = [], []
        for i in range(window, len(series)):
            Xs.append(series[i - window:i, 0])
            ys.append(series[i, 0])
        return np.array(Xs), np.array(ys)

    # Train sequences from train portion; test sequences from the join onward
    X_train, y_train = make_sequences(scaled[:split])
    X_test,  y_test  = make_sequences(scaled[split - window:])   # include lead-in

    # LSTM expects 3D input: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # --- Model ---
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(window, 1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=32, epochs=50,
              validation_split=0.1, callbacks=[es], verbose=1)

    # Predict, then invert scaling back to dollars
    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = evaluate(actual, pred)

    if save_model:
        os.makedirs("models", exist_ok=True)
        model.save("models/lstm_model.keras")
        joblib.dump(scaler, "models/lstm_scaler.pkl")
        print("Saved LSTM model + scaler to models/")

    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("HEADLINE: 15-feature study (matches CV)")
    print("=" * 60)
    table15, n15 = run_classical(causal_only=False)
    print(f"({n15} features)\n")
    print(table15.round(4))

    # RMSE reduction vs baseline -- this is the CV's "98%" claim
    base_rmse = table15.loc["Baseline (mean)", "RMSE"]
    table15["RMSE_reduction_%"] = (base_rmse - table15["RMSE"]) / base_rmse * 100
    print("\nRMSE reduction vs baseline:")
    print(table15[["RMSE", "RMSE_reduction_%"]].round(2))

    print("\n" + "=" * 60)
    print("HONESTY CHECK: 12-feature causal set (no same-day leak)")
    print("=" * 60)
    table12, n12 = run_classical(causal_only=True)
    print(f"({n12} features)\n")
    print(table12.round(4))

    print("\n" + "=" * 60)
    print("LSTM -- split sensitivity comparison")
    print("=" * 60)

    print("\n[80/20 split -- honest, longer test window; this model is saved]")
    m_80 = run_lstm(save_model=True, train_frac=0.8)
    print(pd.Series(m_80).round(4))

    print("\n[95/5 split -- matches original study conditions]")
    m_95 = run_lstm(save_model=False, train_frac=0.95)
    print(pd.Series(m_95).round(4))