"""Feature engineering utilities for fake news explainability experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import textstat
from scipy.sparse import csr_matrix, hstack, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


class FeatureEngineer:
    """Builds text and metadata features for downstream ML models."""

    def __init__(
        self,
        max_tfidf_features: int = 5000,
        vectorizer_output_path: str = "data/processed/tfidf_vectorizer.joblib",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.max_tfidf_features = max_tfidf_features
        self.vectorizer_output_path = Path(vectorizer_output_path)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _to_text_series(texts: Iterable[str]) -> pd.Series:
        return pd.Series(list(texts), dtype="object").fillna("").astype(str)

    def extract_tfidf(
        self,
        texts: Iterable[str],
        max_features: int,
        fit: bool = True,
    ) -> tuple[spmatrix, TfidfVectorizer]:
        """Extract TF-IDF features and persist the fitted vectorizer."""
        text_series = self._to_text_series(texts)

        if fit:
            self.logger.info("Fitting TF-IDF vectorizer with max_features=%s", max_features)
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            tfidf_matrix = self.vectorizer.fit_transform(text_series)
            self.vectorizer_output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.vectorizer, self.vectorizer_output_path)
            self.logger.info("Saved TF-IDF vectorizer to %s", self.vectorizer_output_path)
            return tfidf_matrix, self.vectorizer

        if self.vectorizer is None:
            if self.vectorizer_output_path.exists():
                self.vectorizer = joblib.load(self.vectorizer_output_path)
                self.logger.info("Loaded TF-IDF vectorizer from %s", self.vectorizer_output_path)
            else:
                raise ValueError(
                    "No fitted TF-IDF vectorizer found. Use fit=True first or provide a saved vectorizer."
                )

        self.logger.info("Transforming text using existing TF-IDF vectorizer")
        return self.vectorizer.transform(text_series), self.vectorizer

    def extract_sentiment(self, texts: Iterable[str]) -> pd.DataFrame:
        """Extract TextBlob polarity and subjectivity features."""
        self.logger.info("Extracting sentiment features")
        text_series = self._to_text_series(texts)
        sentiments = text_series.apply(lambda txt: TextBlob(txt).sentiment)
        return pd.DataFrame(
            {
                "sentiment_polarity": sentiments.apply(lambda s: s.polarity).astype(float),
                "sentiment_subjectivity": sentiments.apply(lambda s: s.subjectivity).astype(float),
            }
        )

    def extract_readability(self, texts: Iterable[str]) -> pd.Series:
        """Compute Flesch-Kincaid grade level via textstat."""
        self.logger.info("Extracting readability scores")
        text_series = self._to_text_series(texts)

        def _score(text: str) -> float:
            try:
                return float(textstat.flesch_kincaid_grade(text))
            except Exception:
                return np.nan

        return text_series.apply(_score).rename("readability_fk_grade")

    def extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata-style features from text content."""
        self.logger.info("Extracting metadata features")
        if "full_text" in df.columns:
            text_series = self._to_text_series(df["full_text"])
        elif "text" in df.columns:
            text_series = self._to_text_series(df["text"])
        else:
            raise ValueError("Input DataFrame must contain either 'full_text' or 'text'.")

        def _caps_ratio(text: str) -> float:
            letters = [ch for ch in text if ch.isalpha()]
            if not letters:
                return 0.0
            uppercase = sum(1 for ch in letters if ch.isupper())
            return uppercase / len(letters)

        return pd.DataFrame(
            {
                "text_length": text_series.apply(lambda x: len(x.split())).astype(float),
                "has_author": text_series.apply(lambda x: int(("By " in x) or ("by " in x))).astype(float),
                "exclamation_count": text_series.str.count("!").astype(float),
                "question_count": text_series.str.count(r"\?").astype(float),
                "caps_ratio": text_series.apply(_caps_ratio).astype(float),
            }
        )

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None,
        fit_tfidf: bool = True,
    ) -> tuple[spmatrix, list[str]]:
        """Combine TF-IDF, sentiment, readability, and metadata into one matrix."""
        self.logger.info("Building combined feature matrix")
        if "full_text" in df.columns:
            text_series = self._to_text_series(df["full_text"])
        elif {"title", "text"}.issubset(df.columns):
            text_series = self._to_text_series(df["title"]) + " " + self._to_text_series(df["text"])
        elif "text" in df.columns:
            text_series = self._to_text_series(df["text"])
        else:
            raise ValueError("Input DataFrame must contain 'full_text' or text fields ('title', 'text').")

        if tfidf_vectorizer is not None:
            self.vectorizer = tfidf_vectorizer

        tfidf_matrix, vectorizer = self.extract_tfidf(
            text_series,
            max_features=self.max_tfidf_features,
            fit=fit_tfidf,
        )
        sentiment_df = self.extract_sentiment(text_series)
        readability_series = self.extract_readability(text_series)
        metadata_df = self.extract_metadata(df.assign(full_text=text_series))

        numeric_df = pd.concat(
            [
                sentiment_df,
                readability_series.to_frame(),
                metadata_df,
            ],
            axis=1,
        ).fillna(0.0)

        dense_sparse = csr_matrix(numeric_df.to_numpy(dtype=float))
        X = hstack([tfidf_matrix, dense_sparse], format="csr")

        feature_names = list(vectorizer.get_feature_names_out()) + list(numeric_df.columns)
        self.logger.info("Built feature matrix with shape %s", X.shape)
        return X, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    test_df = pd.DataFrame(
        {
            "title": [
                "Breaking Update",
                "Research Report",
                "By Editor: Market Brief",
            ],
            "text": [
                "By Alice! This is AMAZING news?!",
                "A calm and factual analysis of economic indicators.",
                "by John the report highlights uncertainty and mixed evidence.",
            ],
            "full_text": [
                "Breaking Update By Alice! This is AMAZING news?!",
                "Research Report A calm and factual analysis of economic indicators.",
                "By Editor: Market Brief by John the report highlights uncertainty and mixed evidence.",
            ],
        }
    )

    engineer = FeatureEngineer(max_tfidf_features=50)

    tfidf_X, vec = engineer.extract_tfidf(test_df["full_text"], max_features=50, fit=True)
    assert tfidf_X.shape[0] == len(test_df)
    assert hasattr(vec, "vocabulary_")
    assert engineer.vectorizer_output_path.exists()

    sentiment = engineer.extract_sentiment(test_df["full_text"])
    assert list(sentiment.columns) == ["sentiment_polarity", "sentiment_subjectivity"]
    assert sentiment.shape == (len(test_df), 2)

    readability = engineer.extract_readability(test_df["full_text"])
    assert readability.shape[0] == len(test_df)

    metadata = engineer.extract_metadata(test_df)
    expected_meta_cols = [
        "text_length",
        "has_author",
        "exclamation_count",
        "question_count",
        "caps_ratio",
    ]
    assert list(metadata.columns) == expected_meta_cols
    assert metadata.shape == (len(test_df), len(expected_meta_cols))

    X, names = engineer.build_feature_matrix(test_df, fit_tfidf=False)
    assert X.shape[0] == len(test_df)
    assert X.shape[1] == len(names)
    assert set(expected_meta_cols).issubset(set(names))

    print("FeatureEngineer unit tests passed.")
