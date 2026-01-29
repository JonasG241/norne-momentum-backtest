import pandas as pd


class Strategy:
    def __init__(self, equal_weight: bool = True, allocation_per_position: float = 0.10):
        """
        allocation_per_position (Option B):
            Fraction of portfolio value to allocate on a BUY (0 < alloc <= 1).
        """
        self.equal_weight = equal_weight
        self.allocation_per_position = allocation_per_position

    def generate_signal(self, stock_data: pd.DataFrame) -> str:
        return "hold"

    @staticmethod
    def calculate_moving_average(dt: pd.DataFrame, number_of_days: int) -> pd.Series:
        if "Close" not in dt.columns:
            raise ValueError("calculate_moving_average: dt must contain 'Close'")
        return dt["Close"].rolling(window=number_of_days, min_periods=number_of_days).mean()


class GoldenCross(Strategy):
    """
    Uses precomputed MA_50 and MA_200 stored in stock_data.
    BUY when MA_50 crosses above MA_200
    SELL when MA_50 crosses below MA_200
    """

    def generate_signal(self, stock_data: pd.DataFrame) -> str:
        if len(stock_data) < 2:
            return "hold"

        if "MA_50" not in stock_data.columns or "MA_200" not in stock_data.columns:
            return "hold"

        prev = stock_data.iloc[-2]
        curr = stock_data.iloc[-1]

        # Need both MAs available
        if pd.isna(prev["MA_50"]) or pd.isna(prev["MA_200"]) or pd.isna(curr["MA_50"]) or pd.isna(curr["MA_200"]):
            return "hold"

        # Golden cross
        if prev["MA_50"] <= prev["MA_200"] and curr["MA_50"] > curr["MA_200"]:
            return "buy"

        # Death cross
        if prev["MA_50"] >= prev["MA_200"] and curr["MA_50"] < curr["MA_200"]:
            return "sell"

        return "hold"


class SimpleScore(Strategy):
    def generate_signal(self, stock_data: pd.DataFrame) -> str:
        return "hold"
