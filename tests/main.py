from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from setup import Stock, StockUniverse, Backtest
from strategies import GoldenCross


def filename_to_stockname(path: Path) -> str:
    # "Arctic Bioscience AS Stock Price History.csv" -> "Arctic Bioscience AS"
    name = path.stem
    return name.replace(" Stock Price History", "").strip()


def main():
    # tests/main.py -> project root is one level up from tests/
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "raw"

    if not data_dir.exists():
        raise RuntimeError(f"Data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {data_dir}")

    stocks = []
    for fp in csv_files:
        stocks.append(
            Stock(
                name=filename_to_stockname(fp),
                filename=str(fp),
                excange="OSE",
            )
        )

    print(f"Loaded {len(stocks)} stocks from {data_dir}")

    universe = StockUniverse(stocks)

    strategy = GoldenCross(allocation_per_position=0.10)  # 10% per position
    bt = Backtest(universe=universe, strategy=strategy, initial_capital=100_000)

    equity, trades = bt.run(pd.Timestamp("2018-01-01"), pd.Timestamp("2024-12-31"))

    print("\n--- Equity (tail) ---")
    print(equity.tail())

    print("\n--- Trades (tail) ---")
    print(trades.tail(20) if len(trades) else "No trades generated.")

    equity["TotalValue"].plot()
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Total Value")
    plt.show()


if __name__ == "__main__":
    main()
