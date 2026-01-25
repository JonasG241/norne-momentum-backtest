import pandas as pd
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

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

ose = mcal.get_calendar("OSE")



def visualize_missing_data(datasets):
    for i in datasets:
        dt = pd.read_csv(f"norne-momentum-backtest/data/raw/{i}")
        dt["Date"] = pd.to_datetime(dt["Date"])
        dt = dt.sort_values("Date")

        start_date = dt["Date"].iloc[0]
        end_date   = dt["Date"].iloc[-1]

        schedule = ose.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index

        dates = pd.DatetimeIndex(dt["Date"].unique())

        # normalize to avoid timezone/time mismatches
        trading_days = trading_days.tz_localize(None).normalize()
        dates = dates.tz_localize(None).normalize()

        missing_days = trading_days.difference(dates)

        print(i)
        print("Trading days in file:", len(dates))
        print("Expected trading days:", len(trading_days))
        print("True missing trading days:", len(missing_days))
        print("-" * 40)

        years = range(start_date.year, end_date.year + 1)
        grid = np.full((len(years), 366), 0, dtype=np.uint8)  # 0 gray, 1 white, 2 red

        trading_set = set(trading_days.date)
        missing_set = set(missing_days.date)

        for r, y in enumerate(years):
            jan1 = pd.Timestamp(y, 1, 1)
            for d in range(1, 367):
                day = jan1 + pd.Timedelta(days=d - 1)

                if day < start_date or day > end_date:
                    continue

                day_key = day.date()
                if day_key in trading_set:
                    grid[r, d - 1] = 2 if day_key in missing_set else 1

        cmap = ListedColormap(["lightgray", "white", "red"])

        plt.figure()
        plt.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
        plt.yticks(range(len(list(years))), list(years))
        plt.xlabel("Day of year (1–366)")
        plt.ylabel("Year")
        plt.title(i.replace("Stock Price History.csv", "").strip()
                + f" — Missing trading days (missing dates = {len(missing_days)})")
        plt.show()


visualize_missing_data(name_datasets)