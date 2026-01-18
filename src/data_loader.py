import pandas as pd
import string

def load_data(true_path, fake_path):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    true_df['label'] = 0
    fake_df['label'] = 1
    df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)
    
    df['text'] = df['text'].str.lower()
    # Removing punctuation as well
    df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return df
