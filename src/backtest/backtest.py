"""Backtesting helpers and evaluation over parameter grid."""
from typing import Iterable, List, Tuple

import pandas as pd


def run_backtest(prices: pd.DataFrame, pos_col: str = "pos", rf_daily: float = 0.0) -> pd.DataFrame:
    """Given a `prices` DataFrame with a `price` column and a position column `pos_col`,
    compute returns and equities for strategy and buy-and-hold. Returns a DataFrame
    containing added columns: `ret`, `strat_ret`, `strat_equity`, `hold_equity`.
    """
    df = prices.copy()
    # compute returns and fill initial NaN with 0 so equity series are well-defined
    df["ret"] = df["price"].pct_change().fillna(0.0)

    # get position (use previous day's signal). Coerce to numeric and fill missing.
    pos = pd.to_numeric(df[pos_col].shift(1), errors="coerce").fillna(0.0)

    # strategy return: invest fraction `pos` in asset, remainder in rf_daily
    df["strat_ret"] = pos * df["ret"] + (1 - pos) * rf_daily
    df["strat_ret"] = df["strat_ret"].fillna(0.0)

    # cumulative equity starting from 1.0
    df["strat_equity"] = (1 + df["strat_ret"]).cumprod()
    df["hold_equity"] = (1 + df["ret"]).cumprod()
    return df


def evaluate_k_list(prices: pd.DataFrame, k_values: Iterable[int], strategy_fn, rf_daily: float = 0.0) -> pd.DataFrame:
    """Evaluate a set of `k_values` using `strategy_fn(prices, k)`.

    `strategy_fn` must accept `(prices, k)` and return a DataFrame containing a `pos` column.

    Returns a DataFrame with columns: `k`, `final_strat`, `final_hold`, `return_ratio`.
    """
    results: List[Tuple[int, float, float, float]] = []
    for k in k_values:
        sig_df = strategy_fn(prices, k)
        out = run_backtest(sig_df, pos_col="pos", rf_daily=rf_daily)
        # explicitly take the last non-null values from the equity columns
        final_strat = float(out["strat_equity"].dropna().iloc[-1])
        final_hold = float(out["hold_equity"].dropna().iloc[-1])
        return_ratio = final_strat / final_hold if final_hold != 0 else float("nan")
        results.append((k, final_strat, final_hold, return_ratio))

    results_df = pd.DataFrame(results, columns=["k", "final_strat", "final_hold", "return_ratio"])
    results_df = results_df.sort_values(by="return_ratio", ascending=False).reset_index(drop=True)
    return results_df
