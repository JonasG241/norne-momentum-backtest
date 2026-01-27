"""Data fetching utilities."""
from typing import Optional

import pandas as pd


def download_adj_close(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download adjusted close prices for `ticker` and return a DataFrame with columns
    ['Date', 'price'] where 'Date' is a datetime index reset as a column.

    Raises ValueError if no data is returned.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} in the given period")

    # Try to robustly select the adjusted close series. yfinance can return
    # different column shapes (Series, DataFrame, or MultiIndex columns).
    adj = None
    if "Adj Close" in df.columns:
        adj = df["Adj Close"]
    else:
        # MultiIndex columns or unexpected column names: try to find a match
        cols = getattr(df, "columns", [])
        if isinstance(cols, pd.MultiIndex):
            matches = [c for c in cols if c[0] == "Adj Close"]
            if matches:
                adj = df[matches[0]]
        if adj is None:
            # fallback: search for a column name containing 'Adj Close'
            possible = [c for c in cols if "Adj Close" in str(c)]
            if possible:
                adj = df[possible[0]]

    if adj is None:
        raise ValueError(f"Adjusted Close column not found for {ticker}")

    # Ensure we have a Series (if it's a single-column DataFrame, extract it)
    if isinstance(adj, pd.DataFrame) and adj.shape[1] == 1:
        adj = adj.iloc[:, 0]

    adj = adj.dropna()
    adj.name = "price"
    return adj.reset_index()
