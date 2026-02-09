# refactoring train.py heavily to support evaluation metrics properly
import yaml
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTClassifier:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        # rest omitted for now, adding metrics next
