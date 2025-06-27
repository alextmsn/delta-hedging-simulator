import pandas as pd

def load_csv_price_series(filepath, column='Close'):
    df = pd.read_csv(filepath)
    return df[column].dropna().tolist()

def resample_series(series, every_n):
    return [series[i] for i in range(0, len(series), every_n)]

def percent_change(series):
    return [100 * (series[i+1] - series[i]) / series[i] for i in range(len(series)-1)]
