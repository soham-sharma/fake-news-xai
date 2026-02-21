import pandas as pd
import numpy as np

class ComparisonMetrics:
    def __init__(self):
        self.results_dir = "results/"
        
    def accuracy_comparison(self, dt_metrics, bert_metrics):
        # TODO: table comparing basic accuracy
        pass

    def feature_overlap(self, top_dt_features, top_bert_features):
        """calculate jaccard overlap"""
        # dummy logic
        return 0.5
