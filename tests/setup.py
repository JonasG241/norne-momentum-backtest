# setup.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas_market_calendars as mcal

import strategies as strat
import helping_fuctions as hf


class Stock:
    def __init__(
        self,
        name: str,
        filename: str,
        fill_method: str = "average",
        max_stale_days: int = 5,
        excange: str = "OSE",
    ):
        self.name = name
        self.filename = filename
        self.fill_method = fill_method
        self.max_stale_days = max_stale_days
        self.exchange = excange

        self.rawdata = pd.read_csv(self.filename)
        self.data = self.preprocess_data()

    def preprocess_data(self) -> pd.DataFrame:
        dt = self.rawdata.copy()

        # --- Robust column stripping (Investing.com exports often have whitespace) ---
        dt.columns = [c.strip() for c in dt.columns]

        # --- Parse Date ---
        if "Date" not in dt.columns:
            raise ValueError(f"{self.name}: Expected 'Date' column in CSV. Found: {dt.columns.tolist()}")

        dt["Date"] = pd.to_datetime(dt["Date"], errors="coerce")
        dt = dt.dropna(subset=["Date"])
        dt = dt.sort_values("Date")

        # --- Align to trading calendar if requested ---
        if self.fill_method in ("average", "no_trade"):
            dt = hf.align_missing_dates_as_nan(dt, self.exchange)
        elif self.fill_method == "average_past_n":
            dt = dt.copy()
        else:
            raise ValueError(f"Unknown fill_method: {self.fill_method}")

        # --- Ensure DateTimeIndex ---
        if "Date" in dt.columns:
            dt.columns = [c.strip() for c in dt.columns]
            dt["Date"] = pd.to_datetime(dt["Date"], errors="coerce")
            dt = dt.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

        dt.index = pd.DatetimeIndex(dt.index).tz_localize(None)
        dt = dt.sort_index()

        # --- Normalize columns (Close naming differs across sources) ---
        dt = self.normalize_columns(dt)

        # --- Ensure Close is numeric ---
        dt = self.ensure_close_numeric(dt)

        # --- Add indicators (stored in Stock.data) ---
        dt = self.add_indicators(dt)

        return dt

    def normalize_columns(self, dt: pd.DataFrame) -> pd.DataFrame:
        dt = dt.copy()
        dt.columns = [c.strip() for c in dt.columns]

        # Common Investing.com variations for close/last price column
        candidates = ["Close", "Close/Last", "Price", "Last", "Adj Close", "AdjClose"]

        if "Close" not in dt.columns:
            for c in candidates:
                if c in dt.columns:
                    dt = dt.rename(columns={c: "Close"})
                    break

        return dt

    def ensure_close_numeric(self, dt: pd.DataFrame) -> pd.DataFrame:
        """
        Investing.com sometimes exports numbers with commas, spaces, etc.
        Convert Close to numeric safely.
        """
        dt = dt.copy()

        if "Close" not in dt.columns:
            raise ValueError(f"{self.name}: Expected a close column. Columns: {dt.columns.tolist()}")

        if not pd.api.types.is_numeric_dtype(dt["Close"]):
            s = dt["Close"].astype(str).str.strip()

            # Remove thousands separators and spaces
            # (If your data uses European decimals like '12,34' this might need tweaking,
            # but Investing.com usually uses '.' decimals and ',' thousands.)
            s = s.str.replace(",", "", regex=False)
            s = s.str.replace(" ", "", regex=False)

            dt["Close"] = pd.to_numeric(s, errors="coerce")

        return dt

    def add_indicators(self, dt: pd.DataFrame) -> pd.DataFrame:
        if "Close" not in dt.columns:
            raise ValueError(f"{self.name}: Expected 'Close' column in CSV/data")

        dt = dt.copy()
        dt["MA_50"] = dt["Close"].rolling(50, min_periods=50).mean()
        dt["MA_100"] = dt["Close"].rolling(100, min_periods=100).mean()
        dt["MA_200"] = dt["Close"].rolling(200, min_periods=200).mean()
        return dt

    def get_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        start_date = pd.Timestamp(start_date).tz_localize(None)
        end_date = pd.Timestamp(end_date).tz_localize(None)
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        return self.data.loc[mask].copy()


class StockUniverse:
    def __init__(self, stocks: list[Stock]):
        self.stocks = stocks

    def visualize_missing_data(self):
        n = len(self.stocks)
        ncols = 5
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
        axes = axes.flatten()

        for idx, stock in enumerate(self.stocks):
            ax = axes[idx]

            dt = stock.rawdata.copy()
            dt.columns = [c.strip() for c in dt.columns]
            dt["Date"] = pd.to_datetime(dt["Date"], errors="coerce")
            dt = dt.dropna(subset=["Date"]).sort_values("Date")

            start_date = dt["Date"].iloc[0]
            end_date = dt["Date"].iloc[-1]

            schedule = mcal.get_calendar(stock.exchange).schedule(start_date=start_date, end_date=end_date)
            trading_days = schedule.index.tz_localize(None).normalize()

            dates = pd.DatetimeIndex(dt["Date"].unique()).tz_localize(None).normalize()
            missing_days = trading_days.difference(dates)

            years = range(start_date.year, end_date.year + 1)
            grid = np.full((len(years), 366), 0, dtype=np.uint8)

            trading_set = set(trading_days.date)
            missing_set = set(missing_days.date)

            for r, y in enumerate(years):
                jan1 = pd.Timestamp(y, 1, 1)
                for d in range(1, 367):
                    day = jan1 + pd.Timedelta(days=d - 1)
                    if day < start_date or day > end_date:
                        continue
                    if day.date() in trading_set:
                        grid[r, d - 1] = 2 if day.date() in missing_set else 1

            cmap = ListedColormap(["lightgray", "white", "red"])
            ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)

            ax.set_title(stock.name, fontsize=9)
            ax.set_xlabel("Day of year")
            ax.set_ylabel("Year")
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(list(years), fontsize=8)

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


class Portifolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.holdings: dict[str, int] = {}
        self.total_value = float(initial_capital)

    def buy_holdings(self, stock_name: str, quantity: int, price: float):
        quantity = int(quantity)
        price = float(price)

        if quantity <= 0:
            return

        cost = quantity * price
        if cost > self.cash:
            raise ValueError(f"Not enough cash to buy {quantity} of {stock_name} at {price}")

        self.holdings[stock_name] = self.holdings.get(stock_name, 0) + quantity
        self.cash -= cost

    def sell_holdings(self, stock_name: str, quantity: int, price: float):
        quantity = int(quantity)
        price = float(price)

        if quantity <= 0:
            return

        if self.holdings.get(stock_name, 0) < quantity:
            raise ValueError(f"Not enough holdings to sell {quantity} of {stock_name}")

        self.holdings[stock_name] -= quantity
        if self.holdings[stock_name] == 0:
            del self.holdings[stock_name]

        self.cash += quantity * price

    def mark_to_market(self, price_by_stock: dict[str, float]) -> float:
        holdings_value = 0.0
        for name, qty in self.holdings.items():
            px = price_by_stock.get(name, None)
            if px is None or not np.isfinite(px):
                continue
            holdings_value += qty * float(px)

        self.total_value = self.cash + holdings_value
        return self.total_value


class Backtest:
    def __init__(self, universe: StockUniverse, strategy: strat.Strategy, initial_capital: float = 100000):
        self.universe = universe
        self.strategy = strategy
        self.portofilio = Portifolio(initial_capital=initial_capital)

    def run(self, start_date: pd.Timestamp, end_date: pd.Timestamp, portifolio: Portifolio | None = None):
        universe = self.universe
        portifolio = portifolio if portifolio is not None else self.portofilio

        start_date = pd.Timestamp(start_date).tz_localize(None)
        end_date = pd.Timestamp(end_date).tz_localize(None)

        stock_data_map: dict[str, pd.DataFrame] = {}
        all_dates = set()

        for stock in universe.stocks:
            sdt = stock.get_data(start_date, end_date)
            sdt.index = pd.DatetimeIndex(sdt.index).tz_localize(None)
            stock_data_map[stock.name] = sdt
            all_dates.update(sdt.index)

        timeline = pd.DatetimeIndex(sorted(all_dates))
        timeline = timeline[(timeline >= start_date) & (timeline <= end_date)]

        equity_curve = []
        trades = []

        for time in timeline:
            price_by_stock = {}

            for stock in universe.stocks:
                stock_data = stock_data_map[stock.name]
                if time not in stock_data.index:
                    continue

                close = stock_data.at[time, "Close"]
                if pd.isna(close):
                    continue

                price_by_stock[stock.name] = float(close)

            portifolio.mark_to_market(price_by_stock)

            for stock in universe.stocks:
                stock_data = stock_data_map[stock.name]

                if time not in stock_data.index:
                    continue

                close = stock_data.at[time, "Close"]
                if pd.isna(close):
                    continue

                signal = self.strategy.generate_signal(stock_data.loc[:time])

                if signal == "buy":
                    if portifolio.holdings.get(stock.name, 0) > 0:
                        continue

                    alloc = float(self.strategy.allocation_per_position)
                    if not (0.0 < alloc <= 1.0):
                        raise ValueError("allocation_per_position must be in (0, 1]")

                    target_cash = portifolio.total_value * alloc
                    qty = int(target_cash // float(close))

                    max_affordable = int(portifolio.cash // float(close))
                    qty = min(qty, max_affordable)

                    if qty > 0:
                        portifolio.buy_holdings(stock.name, qty, float(close))
                        trades.append({"Date": time, "Stock": stock.name, "Side": "BUY", "Qty": qty, "Price": float(close)})

                elif signal == "sell":
                    qty = int(portifolio.holdings.get(stock.name, 0))
                    if qty > 0:
                        portifolio.sell_holdings(stock.name, qty, float(close))
                        trades.append({"Date": time, "Stock": stock.name, "Side": "SELL", "Qty": qty, "Price": float(close)})

            portifolio.mark_to_market(price_by_stock)

            equity_curve.append({
                "Date": time,
                "TotalValue": portifolio.total_value,
                "Cash": portifolio.cash,
                "NumPositions": len(portifolio.holdings),
            })

        equity_df = pd.DataFrame(equity_curve).set_index("Date")
        trades_df = pd.DataFrame(trades)

        return equity_df, trades_df
