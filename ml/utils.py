import pandas as pd


def getRange(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    range_of_series = df['value'].max() - df['value'].min()

    return range_of_series


def getMean(df):
    return df['value'].mean()


def getMedian(df):
    return df['value'].median()


def getStd(df):
    return df['value'].std()


def getVar(df):
    return getStd(df) / getMean(df)
