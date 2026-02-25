import nltk
from nltk.corpus import wordnet

class NearDuplicatePairGenerator:
    def __init__(self, df):
        self.df = df
        
    def generate_pairs(self):
        # generating pairs to test model explainability stability
        pass
