import yaml
import logging
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeExplainer:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
    def train(self, X_train, y_train, depth):
        # Needs cross validation
        pass
