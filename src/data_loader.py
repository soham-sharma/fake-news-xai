import pandas as pd

def load_data(path):
    print("loading data...")
    return pd.read_csv(path)
