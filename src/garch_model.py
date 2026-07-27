"""
garch_model.py
GARCH(1,1) volatility forecasting.

Unlike the price models (which fail to predict direction), GARCH predicts
VOLATILITY -- how much the price will swing -- which IS predictable due to
volatility clustering (big moves follow big moves).
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from arch import arch_model
from data import load_data


def compute_returns(prices):
    """Daily percentage log returns (x100 so GARCH's optimizer converges well)."""
    log_ret = np.diff(np.log(prices)) * 100
    return log_ret


def fit_garch(eval_mode: bool = False):
    """Fit GARCH(1,1) on live returns; return the fitted result + returns series."""
    df = load_data(eval_mode=eval_mode)
    returns = compute_returns(df["Close"].values)

    model = arch_model(returns, vol="Garch", p=1, q=1, dist="t")
    res = model.fit(disp="off")
    return res, returns, df


def forecast_volatility(days: int = 30, eval_mode: bool = False):
    """
    Forecast next `days` of volatility (annualised %).
    Returns (df, historical_vol, forecast_vol).
    """
    res, returns, df = fit_garch(eval_mode=eval_mode)

    # In-sample conditional volatility (daily %), annualised (~252 trading days)
    hist_vol = res.conditional_volatility * np.sqrt(252)

    # Forecast variance ahead, convert to annualised volatility
    fc = res.forecast(horizon=days, reindex=False)
    fc_var = fc.variance.values[-1]                 # daily variance path
    fc_vol = np.sqrt(fc_var) * np.sqrt(252)         # annualised %

    return df, hist_vol, fc_vol, res


if __name__ == "__main__":
    df, hist_vol, fc_vol, res = forecast_volatility(days=30)

    print("GARCH(1,1) model summary:")
    print(res.summary())

    print(f"\nCurrent annualised volatility: {hist_vol[-1]:.1f}%")
    print(f"30-day volatility forecast (first 5 days, annualised %):")
    for i in range(5):
        print(f"  Day {i+1}: {fc_vol[i]:.1f}%")