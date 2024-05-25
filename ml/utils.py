import pandas as pd


class TimeSeriesStatsCalculator:
    def __init__(self, df):
        self.df = df

    def get_range(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(by='timestamp')

        range_of_series = self.df['value'].max() - self.df['value'].min()

        return range_of_series

    def get_mean(self):
        return self.df['value'].mean()

    def get_median(self):
        return self.df['value'].median()

    def get_std(self):
        return self.df['value'].std()

    def get_var(self):
        return self.get_std() / self.get_mean()

    def get_correlation(self):
        return self.df.corr().abs()
