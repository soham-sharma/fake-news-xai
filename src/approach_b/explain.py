import shap
import torch

class BERTShapExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.masker = shap.maskers.Text(tokenizer)
    
    def _predict_proba_wrapper(self, texts):
        # need to bypass huggingface dataset overhead for shap permutations
        pass
