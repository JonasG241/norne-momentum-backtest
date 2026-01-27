"""Run parameter search for moving-average strategy over a short list of tickers.

Usage (from repo root):
    python src/backtest/run_find_best_k.py

The script prints a small table of results for each ticker and the best `k` found
by `return_ratio` (final strategy equity divided by final buy&hold equity).
"""
from typing import List
import traceback

from src.data.fetch import download_adj_close
from src.strategies.moving_average import ma_signals
from src.backtest.backtest import evaluate_k_list


def find_best_k_for_ticker(ticker: str, k_values: List[int], start: str = "2000-01-01", end=None):
    prices = download_adj_close(ticker, start=start, end=end)
    # debug: show basic info to help trace errors
    print(f"Debug: downloaded prices for {ticker} -> type={type(prices)}, columns={list(prices.columns)}")
    print(prices.head(3).to_string(index=False))
    # Use 'raw' pos_mode to match notebook behaviour where `pos = score.shift(1)`
    # notebook uses rf_annual = 0.03 and rf_daily = (1+rf_annual)**(1/252)-1
    rf_annual = 0.03
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    # In the notebook the moving averages are fixed to 50/100/200 and `k` is a
    # threshold on `score` (number of MA pairwise wins). Implement that here
    # so results match the notebook.
    def threshold_strategy(df, k_threshold: int):
        d = df.copy()
        d["ma50"] = d["price"].rolling(window=50, min_periods=1).mean()
        d["ma100"] = d["price"].rolling(window=100, min_periods=1).mean()
        d["ma200"] = d["price"].rolling(window=200, min_periods=1).mean()
        d["score"] = (
            (d["ma50"] > d["ma100"]).astype(int)
            + (d["ma50"] > d["ma200"]).astype(int)
            + (d["ma100"] > d["ma200"]).astype(int)
        )
        # return unshifted signal so run_backtest will shift by 1
        d["pos"] = (d["score"] >= k_threshold).astype(int)
        return d

    results = evaluate_k_list(prices, k_values, strategy_fn=threshold_strategy, rf_daily=rf_daily)

    best_row = results.iloc[0]
    print(f"Ticker: {ticker} | Best k: {int(best_row['k'])} | Return ratio: {best_row['return_ratio']:.6f}")
    print(results.head(10).to_string(index=False))
    return results


def main():
    # Short list of tickers to try. Edit as needed.
    tickers = ["AKER.OL", "EQNR.OL", "NHY.OL"]
    # Example k values to search. Adjust range/resolution as desired.
    k_values = [1, 2, 3]

    for t in tickers:
        try:
            find_best_k_for_ticker(t, k_values)
        except Exception:
            print(f"Error processing {t}:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
