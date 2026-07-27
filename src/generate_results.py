"""
generate_results.py
Runs all evaluations once and saves results to results/metrics.json
so the app can display them instantly without recomputing.
Run this whenever you want to refresh the reported numbers.
"""

import json
import os
import pandas as pd

from train_ml import run_classical, run_lstm
from train_arima import run_arima

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.json")


def main():
    print("Running classical models (15-feature headline)...")
    table15, n15 = run_classical(causal_only=False)
    base_rmse = table15.loc["Baseline (mean)", "RMSE"]
    table15["RMSE_reduction_%"] = (base_rmse - table15["RMSE"]) / base_rmse * 100

    print("Running classical models (12-feature causal check)...")
    table12, n12 = run_classical(causal_only=True)

    print("Running LSTM (80/20 headline)...")
    lstm_80 = run_lstm(save_model=True, train_frac=0.8)
    print("Running LSTM (95/5, matches original study)...")
    lstm_95 = run_lstm(save_model=False, train_frac=0.95)

    print("Running ARIMA walk-forward + persistence...")
    arima_m, naive_m, arima_dir = run_arima(save_model=True)

    results = {
        "classical_15": table15.round(4).to_dict(orient="index"),
        "classical_12": table12.round(4).to_dict(orient="index"),
        "n_features_15": n15,
        "n_features_12": n12,
        "lstm_80": {k: round(v, 4) for k, v in lstm_80.items()},
        "lstm_95": {k: round(v, 4) for k, v in lstm_95.items()},
        "arima": {k: round(v, 4) for k, v in arima_m.items()},
        "persistence": {k: round(v, 4) for k, v in naive_m.items()},
        "arima_directional_accuracy": round(arima_dir, 2),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved all results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()