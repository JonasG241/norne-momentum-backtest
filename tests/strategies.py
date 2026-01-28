import pandas as pd


class Strategy:
    
    def __init__(self, equal_weight: bool = True, allocation_per_position: float = 0.1 ):
        self.equal_weight = equal_weight
        self.allocation_per_position = allocation_per_position


    def generate_signal(self, stock_data: pd.DataFrame):
        return "hold"
    
    @staticmethod
    def calculate_moving_average(dt: pd.DataFrame, number_of_days: int):
        pass




class GoldenCross(Strategy):
    def generate_signal(self, stock_data: pd.DataFrame):
        pass 




class SimpleScore(Strategy):
    def generate_signal(self, stock_data):
        pass

