from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import pandas as pd

class FeatureEngineer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def extract_sentiment(self, texts):
        print("extracting sentiment...")
        return [TextBlob(str(t)).sentiment.polarity for t in texts]
    
    def extract_readability(self, texts):
        import textstat
        return [textstat.flesch_kincaid_grade(str(t)) for t in texts]
