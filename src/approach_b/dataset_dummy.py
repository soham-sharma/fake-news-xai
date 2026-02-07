# Playing around with huggingface dataset formatting
from datasets import Dataset
import pandas as pd
df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})
dataset = Dataset.from_pandas(df)
print(dataset[0])
