import pandas as pd
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import strategies as strat
import helping_fuctions as hf
import classes as cls

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

stocks = []
for i in name_datasets:
    stock = cls.Stock(
        name=i.replace("Stock Price History.csv", "").strip(),
        filename=f"norne-momentum-backtest/data/raw/{i}",
        fill_method="average"
    )
    stocks.append(stock)

universe = cls.StockUniverse(stocks)
universe.visualize_missing_data()



