"""Simple moving-average based strategy functions."""
from typing import Optional

import pandas as pd


def ma_signals(prices: pd.DataFrame, k_short: int, k_mid: Optional[int] = None, k_long: Optional[int] = None, pos_mode: str = "binary") -> pd.DataFrame:
    """Compute moving averages and a simple signal DataFrame.

    If `k_mid` or `k_long` are not provided they are set to 2*k_short and 4*k_short.

    The returned DataFrame is a copy of `prices` with columns added:
    - `ma_short`, `ma_mid`, `ma_long`
    - `score` (sum of the three pairwise comparisons)
        - `pos` (position used by the backtest). `pos_mode` controls the format:
            - `'binary'` (default): `pos = (score >= 2).astype(int)`
            - `'raw'`: `pos = score` (this matches the original notebook where `pos = score.shift(1)`)
    """
    df = prices.copy()
    if k_mid is None:
        k_mid = max(2 * k_short, k_short + 1)
    if k_long is None:
        k_long = max(4 * k_short, k_mid + 1)

    df["ma_short"] = df["price"].rolling(window=k_short, min_periods=1).mean()
    df["ma_mid"] = df["price"].rolling(window=k_mid, min_periods=1).mean()
    df["ma_long"] = df["price"].rolling(window=k_long, min_periods=1).mean()

    df["score"] = (
        (df["ma_short"] > df["ma_mid"]).astype(int)
        + (df["ma_short"] > df["ma_long"]).astype(int)
        + (df["ma_mid"] > df["ma_long"]).astype(int)
    )

    if pos_mode == "binary":
        df["pos"] = (df["score"] >= 2).astype(int)
    elif pos_mode == "raw":
        # keep raw score (0..3); downstream code typically shifts the position
        df["pos"] = df["score"]
    else:
        raise ValueError("pos_mode must be 'binary' or 'raw'")
    return df
