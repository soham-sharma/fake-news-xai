"""Lexical debiasing utilities to reduce source-token leakage before training."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LexicalDebiaser:
    """Remove lexical shortcuts and audit leakage-prone source identifiers."""

    def __init__(self, seed: int = 42) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.seed = int(seed)
        self.tables_dir = PROJECT_ROOT / "results" / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        raw_patterns = [
            ("reuters_parenthetical", r"\(reuters\)"),
            ("reuters_dateline_dash", r"reuters\s*[-–]"),
            ("by_reuters_staff", r"by reuters staff"),
            ("ap_parenthetical", r"\(ap\)"),
            ("associated_press", r"associated press"),
            ("afp_parenthetical", r"\(afp\)"),
            ("bloomberg_parenthetical", r"\(bloomberg\)"),
            ("washington_post_parenthetical", r"\(washington post\)"),
            ("nyt_parenthetical", r"\(the new york times?\)"),
            ("leading_reuters_structural_dateline", r"^[\w\s]{2,30}\s*\(reuters\)\s*[-–]\s*"),
            # Extra shorthand forms frequently seen in wire-style dumps.
            ("dash_reuters", r"[-–]\s*reuters\b"),
            ("reuters_credit", r"photo\s*:\s*reuters"),
        ]

        self.pattern_catalogue = [
            {
                "name": name,
                "pattern": pattern,
                "compiled": re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE),
            }
            for name, pattern in raw_patterns
        ]

        self.known_leakage_tokens = {
            "reuters",
            "ap",
            "associated press",
            "afp",
            "bloomberg",
            "washington post",
            "new york times",
            "by reuters staff",
        }

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if text is None:
            return ""
        if isinstance(text, float) and np.isnan(text):
            return ""
        return str(text)

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    def strip(self, text: str) -> str:
        """Remove leakage patterns and collapse whitespace."""
        cleaned = self._normalize_text(text)
        for pattern_info in self.pattern_catalogue:
            cleaned = pattern_info["compiled"].sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def audit(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        """Audit leakage pattern prevalence and label skew by pattern."""
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")
        if "label" not in df.columns:
            raise ValueError("Input DataFrame must include 'label'.")

        work_df = df[[text_col, "label"]].copy()
        work_df[text_col] = work_df[text_col].map(self._normalize_text)
        work_df["label"] = work_df["label"].astype(int)

        rows: list[dict[str, Any]] = []
        for pattern_info in self.pattern_catalogue:
            compiled = pattern_info["compiled"]
            match_mask = work_df[text_col].apply(lambda txt: bool(compiled.search(txt)))
            matched = work_df[match_mask]
            count = int(len(matched))

            if count == 0:
                fake_count = 0
                real_count = 0
                fake_pct = 0.0
                real_pct = 0.0
            else:
                fake_count = int((matched["label"] == 1).sum())
                real_count = int((matched["label"] == 0).sum())
                fake_pct = 100.0 * fake_count / count
                real_pct = 100.0 * real_count / count

            max_class_pct = max(fake_pct, real_pct)
            if count > 50 and max_class_pct > 80.0:
                dominant = "fake" if fake_pct > real_pct else "real"
                self.logger.warning(
                    "Potential leakage detected: pattern=%s count=%s dominant_class=%s dominant_pct=%.2f",
                    pattern_info["name"],
                    count,
                    dominant,
                    max_class_pct,
                )

            rows.append(
                {
                    "pattern_name": pattern_info["name"],
                    "regex": pattern_info["pattern"],
                    "match_count": count,
                    "fake_count": fake_count,
                    "real_count": real_count,
                    "fake_pct": fake_pct,
                    "real_pct": real_pct,
                    "bias_gap_abs": abs(fake_pct - real_pct),
                }
            )

        audit_df = pd.DataFrame(rows).sort_values(
            ["bias_gap_abs", "match_count"], ascending=[False, False]
        ).reset_index(drop=True)

        out_path = self.tables_dir / "leakage_audit.csv"
        audit_df.to_csv(out_path, index=False)
        self.logger.info("Saved leakage audit to %s", out_path)
        return audit_df

    def apply(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        """Apply debiasing and add debiased_text while retaining original text."""
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        out_df = df.copy()
        original_texts = out_df[text_col].map(self._normalize_text)
        debiased_texts = original_texts.map(self.strip)

        original_tokens = original_texts.map(self._count_tokens).astype(float)
        debiased_tokens = debiased_texts.map(self._count_tokens).astype(float)
        tokens_removed = (original_tokens - debiased_tokens).clip(lower=0.0)

        out_df["debiased_text"] = debiased_texts

        self.logger.info(
            "Applied lexical debiasing to %s rows; average tokens removed/article=%.3f",
            len(out_df),
            float(tokens_removed.mean()) if len(tokens_removed) > 0 else 0.0,
        )
        return out_df

    def compare_label_distributions(
        self,
        original_df: pd.DataFrame,
        debiased_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compare top shortcut features before/after debiasing with shallow DT on TF-IDF."""
        if "label" not in original_df.columns or "label" not in debiased_df.columns:
            raise ValueError("Both DataFrames must include 'label'.")

        if "full_text" in original_df.columns:
            original_text_col = "full_text"
        elif "text" in original_df.columns:
            original_text_col = "text"
        else:
            raise ValueError("original_df must include 'full_text' or 'text'.")

        if "debiased_text" in debiased_df.columns:
            debiased_text_col = "debiased_text"
        elif "full_text" in debiased_df.columns:
            debiased_text_col = "full_text"
        else:
            raise ValueError("debiased_df must include 'debiased_text' or 'full_text'.")

        y = original_df["label"].astype(int).to_numpy()

        def _top_features(texts: pd.Series) -> tuple[list[str], list[float]]:
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            X = vectorizer.fit_transform(texts.map(self._normalize_text))
            clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=self.seed)
            clf.fit(X, y)
            importances = np.asarray(clf.feature_importances_, dtype=float)
            feature_names = np.asarray(vectorizer.get_feature_names_out())

            top_idx = np.argsort(importances)[::-1][:5]
            top_feats = [str(feature_names[i]) for i in top_idx]
            top_vals = [float(importances[i]) for i in top_idx]
            return top_feats, top_vals

        top_before, imp_before = _top_features(original_df[original_text_col])
        top_after, imp_after = _top_features(debiased_df[debiased_text_col])

        before_leakage_hits = [
            feat
            for feat in top_before
            if any(token in feat.lower() for token in self.known_leakage_tokens)
        ]
        confirmed_removed = [feat for feat in before_leakage_hits if feat.lower() not in {x.lower() for x in top_after}]

        rows: list[dict[str, Any]] = []
        for rank in range(5):
            b_feat = top_before[rank] if rank < len(top_before) else ""
            a_feat = top_after[rank] if rank < len(top_after) else ""
            rows.append(
                {
                    "rank": rank + 1,
                    "top_feature_before": b_feat,
                    "importance_before": imp_before[rank] if rank < len(imp_before) else np.nan,
                    "top_feature_after": a_feat,
                    "importance_after": imp_after[rank] if rank < len(imp_after) else np.nan,
                    "confirmed_leakage_removed": bool(b_feat in confirmed_removed),
                }
            )

        out_df = pd.DataFrame(rows)
        out_path = self.tables_dir / "debias_comparison.csv"
        out_df.to_csv(out_path, index=False)
        self.logger.info("Saved debias comparison to %s", out_path)

        return {
            "top_features_before": top_before,
            "top_features_after": top_after,
            "confirmed_leakage_removed": confirmed_removed,
            "output_path": str(out_path),
        }

    def save_debiased_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        path: str | Path = "data/processed/",
    ) -> dict[str, str]:
        """Save debiased train/test splits with debiased text replacing full_text."""
        output_dir = Path(path)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        def _materialize(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if "debiased_text" not in out.columns:
                text_col = "full_text" if "full_text" in out.columns else "text"
                out = self.apply(out, text_col=text_col)

            out["full_text"] = out["debiased_text"].map(self._normalize_text)
            if "debiased_text" in out.columns:
                out = out.drop(columns=["debiased_text"])
            return out

        train_out = _materialize(train_df)
        test_out = _materialize(test_df)

        train_path = output_dir / "debiased_train.csv"
        test_path = output_dir / "debiased_test.csv"
        train_out.to_csv(train_path, index=False)
        test_out.to_csv(test_path, index=False)

        self.logger.info("Saved debiased train split to %s", train_path)
        self.logger.info("Saved debiased test split to %s", test_path)
        return {
            "debiased_train_path": str(train_path),
            "debiased_test_path": str(test_path),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train/test CSV files in {processed_dir}. Run data loading first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    text_col = "full_text" if "full_text" in train_df.columns else "text"
    if text_col not in train_df.columns or text_col not in test_df.columns:
        raise ValueError("Expected 'full_text' or 'text' column in both train and test files.")

    debiaser = LexicalDebiaser(seed=42)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    audit_df = debiaser.audit(combined_df, text_col=text_col)

    print("\nTop 20 most biased patterns:")
    print(audit_df.head(20).to_string(index=False))

    train_debiased = debiaser.apply(train_df, text_col=text_col)
    test_debiased = debiaser.apply(test_df, text_col=text_col)
    split_paths = debiaser.save_debiased_splits(
        train_debiased,
        test_debiased,
        path=processed_dir,
    )

    combined_debiased = pd.concat([train_debiased, test_debiased], ignore_index=True)
    comparison = debiaser.compare_label_distributions(combined_df, combined_debiased)

    print("\nDebias comparison report:")
    print(comparison)
    print("\nDebiased split paths:")
    print(split_paths)