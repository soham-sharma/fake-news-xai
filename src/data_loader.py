import pandas as pd

def load_data(true_path, fake_path):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    true_df['label'] = 0
    fake_df['label'] = 1
    df = pd.concat([true_df, fake_df])
    # basic cleaning
    df['text'] = df['text'].str.lower()
    return df
