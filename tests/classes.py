import pandas as pd
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import strategies as strat
import helping_fuctions as hf

name_datasets = [
    "Arctic Bioscience AS Stock Price History.csv",
    "Arcticzymes Tech Stock Price History.csv",
    "Circio Holding Stock Price History.csv",
    "Codelab Capital AS Stock Price History.csv",
    "ContextVision AB Stock Price History.csv",
    "Exact Therapeutics AS Stock Price History.csv",
    "Gentian Diagnostics Stock Price History.csv",
    "INIFY Laboratories Stock Price History.csv",
    "Lifecare Stock Price History.csv",
    "Lytix Biopharma AS Stock Price History.csv",
    "Medistim Stock Price History.csv",
    "Navamedic Stock Price History.csv",
    "Nordhealth AS Stock Price History.csv",
    "Nykode Therapeutics Stock Price History.csv",
    "Observe Medical Stock Price History.csv",
    "Omda AS Stock Price History.csv",
    "Oncoinvent Stock Price History.csv",
    "PCI Biotech Stock Price History.csv",
    "Photocure Stock Price History.csv",
    "SoftOx Solutions Stock Price History.csv",
    "Thor Medical Stock Price History.csv",
    "Vistin Pharma ASA Stock Price History.csv",
    "Zelluna Stock Price History.csv",
]

print(hf.__file__)

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

            ax.set_title(i.replace("Stock Price History.csv", "").strip(), fontsize=9)
            ax.set_xlabel("Day of year")
            ax.set_ylabel("Year")
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(list(years), fontsize=8)

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()





class Backtest():

    def __init__(self, universe: StockUniverse, strategy: strat.Strategy):
        self.universe = universe
        self.strategy = strategy

    def run(self, start_date: pd.Timestamp, end_date: pd.Timestamp, initial_capital: float):
        pass



stocks = []
for i in name_datasets:
    stock = Stock(
        name=i.replace("Stock Price History.csv", "").strip(),
        filename=f"norne-momentum-backtest/data/raw/{i}",
        fill_method="average"
    )
    stocks.append(stock)

universe = StockUniverse(stocks)
universe.visualize_missing_data()




