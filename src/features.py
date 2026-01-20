from sklearn.feature_extraction.text import TfidfVectorizer

def get_basic_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    features = vectorizer.fit_transform(corpus)
    return features, vectorizer
