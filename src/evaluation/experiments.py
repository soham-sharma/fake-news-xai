import nltk
from nltk.corpus import wordnet

class NearDuplicatePairGenerator:
    def __init__(self, df):
        self.df = df
        
    def generate_pairs(self):
        # generating pairs to test model explainability stability
        pass

    def compute_tfidf_similarity(self, text1, text2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        pass
