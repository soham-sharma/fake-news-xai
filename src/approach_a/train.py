import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeExplainer:
    def __init__(self, max_depth=5):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        
    def train(self, X, y):
        print("Training decision tree...")
        self.model.fit(X, y)
