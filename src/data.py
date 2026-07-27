import os 
import pandas as pd
import yfinance as yf

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "AAPL.csv")
def load_data(ticker: str = "AAPL", start: str = "1980-12-12",
              end: str = None, eval_mode: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of daily prices with columns:
    Date, Open, High, Low, Close, Adj Close, Volume — sorted oldest to newest.

    eval_mode=True caps data at 2021-12-31 to reproduce the original study's
    metrics (the figures reported on the CV). eval_mode=False uses all data
    up to today, for the live app's forecasts.
    """
    # In eval mode, freeze the end date so results are reproducible
    if eval_mode:
        end = "2021-12-31"
    
    # 1. Try live data
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if df is not None and not df.empty:
            df = df.reset_index()
            # yfinance sometimes returns multi-level columns; flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            _save_cache(df)
            print(f"Loaded {len(df)} rows live from Yahoo Finance.")
    except Exception as e:
        print(f"Live fetch failed ({e}); falling back to cache.")

    # 2. Fall back to cached CSV
    if df is None or df.empty:
        if os.path.exists(CACHE_PATH):
            df = pd.read_csv(CACHE_PATH)
            print(f"Loaded {len(df)} rows from cache.")
        else:
            raise FileNotFoundError(
                "No live data and no cached CSV. Connect to the internet once "
                "to build the cache, or place AAPL.csv in the data/ folder."
            )

    # 3. Clean up — guarantee correct types and chronological order
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna().drop_duplicates()

    return df


def _save_cache(df: pd.DataFrame) -> None:
    """Save a copy so we can work offline later."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)


# Lets us test this file on its own
if __name__ == "__main__":
    print("=== EVAL MODE (capped at 2021) ===")
    eval_data = load_data(eval_mode=True)
    print(eval_data.shape, "| last date:", eval_data["Date"].max().date())

    print("\n=== LIVE MODE (through today) ===")
    live_data = load_data(eval_mode=False)
    print(live_data.shape, "| last date:", live_data["Date"].max().date())