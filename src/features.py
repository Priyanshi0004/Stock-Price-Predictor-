"""
features.py
Builds the model inputs (features) from raw price data.

Two kinds of features (15 total), matching the original study:
  - Calendar features (6): day_of_week, month, quarter, year,
    week_of_year, day_of_year
  - Lag features (5): previous 1-5 days' closing prices
  - Same-day market columns (4): Open, High, Low, Volume

Note on leakage: Open/High/Low are same-day values. They are included
to reproduce the original study, but High/Low mathematically bound Close,
which inflates scores. build_features() can exclude them via causal_only=True
for an honest, deployable comparison.
"""

import pandas as pd

CALENDAR = ["day_of_week", "month", "quarter", "year", "week_of_year", "day_of_year"]
LAGS = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_5"]
SAME_DAY = ["Open", "High", "Low", "Volume"]   # High/Low leak the target


def build_features(df: pd.DataFrame, causal_only: bool = False):
    """
    Add calendar + lag features to the price DataFrame.

    Returns (X, y):
      X = feature columns, y = Close (the target).

    causal_only=False -> all 15 features (reproduces the CV study)
    causal_only=True  -> drops same-day Open/High/Low (leakage-free)
    """
    data = df.copy()

    # --- Calendar features ---
    data["day_of_week"]  = data["Date"].dt.dayofweek
    data["month"]        = data["Date"].dt.month
    data["quarter"]      = data["Date"].dt.quarter
    data["year"]         = data["Date"].dt.year
    data["week_of_year"] = data["Date"].dt.isocalendar().week.astype(int)
    data["day_of_year"]  = data["Date"].dt.dayofyear

    # --- Lag features: previous 1-5 days' Close ---
    for i in range(1, 6):
        data[f"lag_{i}"] = data["Close"].shift(i)

    # Drop the first 5 rows that have no lag history
    data = data.dropna().reset_index(drop=True)

    # --- Choose feature set ---
    if causal_only:
        feature_cols = CALENDAR + LAGS + ["Volume"]      # 12 features, no leak
    else:
        feature_cols = SAME_DAY + CALENDAR + LAGS        # 15 features (CV study)

    X = data[feature_cols]
    y = data["Close"]
    return X, y, feature_cols


if __name__ == "__main__":
    from data import load_data

    df = load_data(eval_mode=True)

    X, y, cols = build_features(df, causal_only=False)
    print(f"Full study : {len(cols)} features -> {cols}")
    print(f"X shape: {X.shape}, y shape: {y.shape}\n")

    Xc, yc, colsc = build_features(df, causal_only=True)
    print(f"Causal only: {len(colsc)} features -> {colsc}")
    print(f"X shape: {Xc.shape}")