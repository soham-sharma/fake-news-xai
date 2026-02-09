import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTClassifier:
    def __init__(self):
        self.model_name = "distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        
    def train(self):
        # TODO: training loop
        pass

    def tokenize(self, texts, max_length=256):
        import pandas as pd
        from datasets import Dataset
        text_list = pd.Series(texts).astype(str).tolist()
        dataset = Dataset.from_dict({"text": text_list})
        
        def _tokenize(batch):
            return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=max_length)
            
        return dataset.map(_tokenize, batched=True).remove_columns(["text"])

    def _build_training_arguments(self):
        from transformers import TrainingArguments
        return TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            eval_strategy="epoch"
        )
