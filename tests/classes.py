import pandas as pd
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import strategies as strat
import helping_fuctions as hf





class Stock():
    def __init__(self, name: str, filename: str, fill_method: str = "average", max_stale_days: int = 5, excange = "OSE"):
        self.name = name
        self.filename = filename
        self.fill_method = fill_method
        self.max_stale_days = max_stale_days
        self.exchange = excange
        

        self.rawdata = pd.read_csv(self.filename)
        self.data = self.preprocess_data()

    def preprocess_data(self):
        dt = self.rawdata.copy()
        dt["Date"] = pd.to_datetime(dt["Date"])
        dt = dt.sort_values("Date")

        if self.fill_method == "average":
            return hf.align_missing_dates_as_nan(dt, self.exchange)

        if self.fill_method == "no_trade":
            return hf.align_missing_dates_as_nan(dt, self.exchange)

        if self.fill_method == "average_past_n":
            return dt.set_index("Date")

        raise ValueError(f"Unknown fill_method: {self.fill_method}")
    
    def get_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        dt = self.data.copy()
        mask = (dt.index >= start_date) & (dt.index <= end_date)
        return dt.loc[mask]




class StockUniverse():
    def __init__(self, stocks: list[Stock]): 
        self.stocks = stocks

    def visualize_missing_data(self):
        n = len(self.stocks)
        ncols = 5
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.5*nrows))
        axes = axes.flatten()

        for idx, stock in enumerate(self.stocks):
            ax = axes[idx]

            dt = stock.rawdata
            dt["Date"] = pd.to_datetime(dt["Date"])
            dt = dt.sort_values("Date")

            start_date = dt["Date"].iloc[0]
            end_date   = dt["Date"].iloc[-1]

            schedule = mcal.get_calendar(stock.exchange).schedule(start_date=start_date, end_date=end_date)
            trading_days = schedule.index

            dates = pd.DatetimeIndex(dt["Date"].unique())

            trading_days = trading_days.tz_localize(None).normalize()
            dates = dates.tz_localize(None).normalize()

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

            ax.imshow(grid, aspect="auto", interpolation="nearest",
                      cmap=cmap, vmin=0, vmax=2)

            ax.set_title(stock.name, fontsize=9)
            ax.set_xlabel("Day of year")
            ax.set_ylabel("Year")
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(list(years), fontsize=8)

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


class Portifolio():
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {}
        self.total_value = initial_capital

    def buy_holdings(self, stock_name: str, quantity: int, price: float):
        if stock_name in self.holdings:
            self.holdings[stock_name] += quantity
        else:
            self.holdings[stock_name] = quantity
        self.cash -= quantity * price
        self.total_value = self.cash + sum(qty * price for qty, price in self.holdings.items())

    def sell_holdings(self, stock_name: str, quantity: int, price: float):
        if stock_name in self.holdings and self.holdings[stock_name] >= quantity:
            self.holdings[stock_name] -= quantity
            self.cash += quantity * price
            self.total_value = self.cash + sum(qty * price for qty, price in self.holdings.items())
        else:
            raise ValueError("Not enough holdings to sell")


class Backtest():
    def __init__(self, universe: StockUniverse, strategy: strat.Strategy, initial_capital: float = 100000):
        self.universe = universe
        self.strategy = strategy
        self.portofilio = Portifolio(initial_capital=initial_capital)

    def run(self, start_date: pd.Timestamp, end_date: pd.Timestamp, portifolio: Portifolio):

        universe = self.universe
        portifolio = portifolio

        for time in pd.date_range(start_date, end_date):        
            for stock in universe.stocks:
                stock_data = stock.get_data(start_date, end_date)
                signal = self.strategy.generate_signal(time, stock_data)
                if signal == "buy":
                    portifolio.buy_holdings(stock.name, quantity=self.strategy.allocation_per_position, price=stock_data.loc[time, "Close"])
                if signal == "sell":
                    portifolio.sell_holdings(stock.name, quantity=self.strategy.allocation_per_position, price=stock_data.loc[time, "Close"])
                if signal == "hold":
                    continue




        




