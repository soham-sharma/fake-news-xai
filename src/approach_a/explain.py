import shap
import matplotlib.pyplot as plt

class TreeExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        
    def get_shap(self, X):
        return self.explainer.shap_values(X)
    
    def test_waterfall(self, X_sample):
        # working out the bug where shap expected values is an array
        pass
