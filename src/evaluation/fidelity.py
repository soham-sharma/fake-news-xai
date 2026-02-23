class FidelityEvaluator:
    def __init__(self, model_predict_fn, feature_type):
        self.model_predict_fn = model_predict_fn
        self.feature_type = feature_type
        
    def sufficiency_score( self, X_samples, shap_values, top_k=5):
        # wait why is my python throwing error with this argument word 
        pass

    def random_baseline(self, X_samples):
        # masking baseline
        pass
