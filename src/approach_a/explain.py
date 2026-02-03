import shap
import numpy as np

class TreeExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)
        self.explainer = shap.TreeExplainer(model)
        
    def _coerce_shap_values(self, values):
        # dealing with new shap API format
        pass
