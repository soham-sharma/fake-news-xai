import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeExplainer:
    def __init__(self, max_depth=5):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        
    def train(self, X, y):
        print("Training decision tree...")
        self.model.fit(X, y)
    def evaluate(self, X, y):
        from sklearn.metrics import accuracy_score
        preds = self.model.predict(X)
        print("Accuracy:", accuracy_score(y, preds))
