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
