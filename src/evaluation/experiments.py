"""Experiment utilities for stability testing with near-duplicate text pairs."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NearDuplicatePairGenerator:
    """Generate semantically similar article pairs for explanation-stability experiments."""

    def __init__(self, df: pd.DataFrame, n_pairs: int = 50, seed: int = 42) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        if "label" not in df.columns:
            raise ValueError("Input DataFrame must include a 'label' column.")

        if "full_text" in df.columns:
            self.text_col = "full_text"
        elif "text" in df.columns:
            self.text_col = "text"
        else:
            raise ValueError("Input DataFrame must include 'full_text' or 'text'.")

        self.df = df.copy()
        self.n_pairs = int(n_pairs)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self._stopwords = set()
        self._pairs_cache: list[tuple[str, str, int]] = []

        self.lexical_map = {
            "says": "stated",
            "said": "reported",
            "saying": "stating",
            "shows": "indicates",
            "showed": "indicated",
            "asks": "questions",
            "claims": "asserts",
            "according to": "as reported by",
        }

        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self) -> None:
        resources = [
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
            ("corpora/stopwords", "stopwords"),
        ]
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                self.logger.info("Downloading NLTK resource: %s", resource_name)
                nltk.download(resource_name, quiet=True)

        try:
            self._stopwords = set(stopwords.words("english"))
        except LookupError:
            self._stopwords = set()

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.title()
        return replacement

    def _get_synonym(self, word: str) -> str | None:
        candidates: set[str] = set()
        for synset in wordnet.synsets(word):
            if synset is None:
                continue
            for lemma in synset.lemmas():
                synonym = lemma.name().replace("_", " ").strip()
                if " " in synonym:
                    continue
                if not synonym.isalpha():
                    continue
                if synonym.lower() == word.lower():
                    continue
                candidates.add(synonym)

        if not candidates:
            return None

        synonyms = sorted(candidates)
        return self.rng.choice(synonyms)

    def _synonym_substitution(self, text: str, min_replace: int = 2, max_replace: int = 3) -> str:
        tokens = re.findall(r"\w+|\W+", text)
        candidate_indices: list[int] = []

        for idx, token in enumerate(tokens):
            if not token.isalpha():
                continue
            token_lower = token.lower()
            if token_lower in self._stopwords:
                continue
            if len(token_lower) <= 3:
                continue
            if self._get_synonym(token) is None:
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return text

        replace_count = min(len(candidate_indices), self.rng.randint(min_replace, max_replace))
        chosen_indices = self.rng.sample(candidate_indices, replace_count)

        for idx in chosen_indices:
            original = tokens[idx]
            synonym = self._get_synonym(original)
            if synonym is None:
                continue
            tokens[idx] = self._match_case(original, synonym)

        return "".join(tokens)

    def _sentence_reordering(self, text: str) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 2:
            return text

        first_sentence = sentences[0]
        remaining = sentences[1:]
        self.rng.shuffle(remaining)
        return " ".join([first_sentence] + remaining)

    def _minor_lexical_variation(self, text: str) -> str:
        varied = text
        items = list(self.lexical_map.items())
        self.rng.shuffle(items)

        max_changes = min(3, len(items))
        change_count = self.rng.randint(1, max_changes)
        selected = items[:change_count]

        for source, target in selected:
            pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)

            def _replace(match: re.Match[str]) -> str:
                return self._match_case(match.group(0), target)

            varied = pattern.sub(_replace, varied)

        return varied

    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(matrix.getrow(0), matrix.getrow(1))[0, 0]
        return float(similarity)

    def _create_variant(self, original_text: str) -> tuple[str, float]:
        """Apply all requested strategies while preserving high lexical similarity."""
        variant = self._synonym_substitution(original_text)
        variant = self._sentence_reordering(variant)
        variant = self._minor_lexical_variation(variant)

        similarity = self.compute_tfidf_similarity(original_text, variant)
        if similarity >= 0.7:
            return variant, similarity

        # Fallback: lighter edits when substitutions/reordering are too strong.
        fallback = self._minor_lexical_variation(original_text)
        fallback_similarity = self.compute_tfidf_similarity(original_text, fallback)
        return fallback, fallback_similarity

    def generate_pairs(self) -> list[tuple[str, str, int]]:
        self.logger.info("Generating near-duplicate pairs (target=%s)", self.n_pairs)

        valid_df = self.df.dropna(subset=[self.text_col, "label"]).copy()
        valid_df[self.text_col] = valid_df[self.text_col].astype(str)
        valid_df = valid_df[valid_df[self.text_col].str.strip() != ""]
        if valid_df.empty:
            self.logger.warning("No valid rows available for pair generation.")
            self._pairs_cache = []
            return []

        sample_n = min(self.n_pairs, len(valid_df))
        sampled = valid_df.sample(n=sample_n, random_state=self.seed).reset_index(drop=True)

        pairs: list[tuple[str, str, int]] = []
        for _, row in sampled.iterrows():
            original_text = str(row[self.text_col])
            label = int(row["label"])

            variant_text, similarity = self._create_variant(original_text)
            if similarity < 0.7:
                self.logger.debug("Skipping one pair with similarity %.3f < 0.7", similarity)
                continue

            pairs.append((original_text, variant_text, label))

        self._pairs_cache = pairs
        self.logger.info("Generated %s near-duplicate pairs", len(pairs))
        return pairs

    def save_pairs(self, path: str | Path) -> Path:
        if not self._pairs_cache:
            self.logger.info("Pair cache empty; generating pairs before save")
            self.generate_pairs()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for original_text, variant_text, label in self._pairs_cache:
            rows.append(
                {
                    "original_text": original_text,
                    "variant_text": variant_text,
                    "label": int(label),
                    "tfidf_similarity": self.compute_tfidf_similarity(original_text, variant_text),
                }
            )

        pairs_df = pd.DataFrame(rows)
        pairs_df.to_csv(output_path, index=False)
        self.logger.info("Saved near-duplicate pairs to %s", output_path)
        return output_path

    @staticmethod
    def load_pairs(path: str | Path) -> list[tuple[str, str, int]]:
        pair_path = Path(path)
        if not pair_path.exists():
            raise FileNotFoundError(f"Pair file does not exist: {pair_path}")

        pair_df = pd.read_csv(pair_path)
        required_cols = {"original_text", "variant_text", "label"}
        missing = required_cols.difference(set(pair_df.columns))
        if missing:
            raise ValueError(f"Missing required pair columns: {sorted(missing)}")

        pairs: list[tuple[str, str, int]] = []
        for _, row in pair_df.iterrows():
            pairs.append(
                (
                    str(row["original_text"]),
                    str(row["variant_text"]),
                    int(row["label"]),
                )
            )
        return pairs

    @classmethod
    def load_or_generate_pairs(
        cls,
        df: pd.DataFrame,
        path: str | Path,
        n_pairs: int = 50,
        seed: int = 42,
    ) -> list[tuple[str, str, int]]:
        pair_path = Path(path)
        if pair_path.exists():
            loaded_pairs = cls.load_pairs(pair_path)
            if len(loaded_pairs) >= n_pairs:
                return loaded_pairs[:n_pairs]

        generator = cls(df=df, n_pairs=n_pairs, seed=seed)
        generated_pairs = generator.generate_pairs()
        generator.save_pairs(pair_path)
        return generated_pairs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    root = Path(__file__).resolve().parents[2]
    test_df = pd.read_csv(root / 'data/processed/test.csv')

    gen = NearDuplicatePairGenerator(test_df, n_pairs=50, seed=42)
    pairs = gen.generate_pairs()
    out_path = gen.save_pairs(root / 'results/tables/near_duplicate_pairs.csv')

    print('pairs_generated', len(pairs))
    if pairs:
        sims = [gen.compute_tfidf_similarity(a, b) for a, b, _ in pairs]
        print('similarity_min', min(sims))
        print('similarity_mean', sum(sims) / len(sims))
        print('sample_label', pairs[0][2])
    print('saved_to', out_path)
