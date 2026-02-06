"""Data loading and preprocessing utilities for fake news experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)


class FakeNewsNetLoader:
    """Load, preprocess, and split fake news data for XAI experiments."""

    def __init__(
        self,
        config_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else self.project_root / "config.yaml"
        )
        self.config = self._load_config(self.config_path)

        raw_path = self.config.get("data", {}).get("raw_path", "data/raw")
        self.raw_dir = self.project_root / raw_path

        processed_path = self.config.get("data", {}).get("processed_path", "data/processed")
        self.processed_dir = self.project_root / processed_path

        split_ratio = self.config.get("data", {}).get("split_ratio", 0.8)
        self.split_ratio = float(split_ratio)
        self.seed = int(self.config.get("random_seed", 42))

        LOGGER.info(
            "Initialized FakeNewsNetLoader with split_ratio=%s, seed=%s",
            self.split_ratio,
            self.seed,
        )

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def load(self) -> pd.DataFrame:
        """Load dataset strictly from local ISOT raw files and return a normalized DataFrame."""
        if not self._isot_raw_files_exist():
            fake_path = self.raw_dir / "Fake.csv"
            true_path = self.raw_dir / "True.csv"
            raise FileNotFoundError(
                "ISOT-only mode requires both files to exist: "
                f"{fake_path} and {true_path}"
            )

        frame = self._load_from_isot_raw()
        source = "isot:local_raw"

        normalized = self._normalize_columns(frame, source=source)

        LOGGER.info("Loaded dataset with shape=%s from source=%s", normalized.shape, source)
        return normalized

    def _isot_raw_files_exist(self) -> bool:
        return (self.raw_dir / "Fake.csv").exists() and (self.raw_dir / "True.csv").exists()

    @staticmethod
    def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(csv_path)
        except UnicodeDecodeError:
            LOGGER.warning("UTF-8 decode failed for %s. Retrying with latin-1", csv_path)
            return pd.read_csv(csv_path, encoding="latin-1")

    def _load_from_isot_raw(self) -> pd.DataFrame:
        fake_path = self.raw_dir / "Fake.csv"
        true_path = self.raw_dir / "True.csv"

        if not fake_path.exists() or not true_path.exists():
            raise FileNotFoundError(
                "Expected ISOT files not found. "
                f"Required: {fake_path} and {true_path}"
            )

        LOGGER.info("Loading ISOT raw files from %s", self.raw_dir)
        fake_df = self._read_csv_with_fallback(fake_path)
        true_df = self._read_csv_with_fallback(true_path)

        required_cols = {"title", "text"}
        fake_missing = required_cols - set(fake_df.columns)
        true_missing = required_cols - set(true_df.columns)
        if fake_missing or true_missing:
            raise ValueError(
                "ISOT CSV files must include columns ['title', 'text']. "
                f"Fake missing: {sorted(fake_missing)} | True missing: {sorted(true_missing)}"
            )

        fake_df = fake_df.copy()
        true_df = true_df.copy()

        fake_df["id"] = [f"fake_{idx}" for idx in range(len(fake_df))]
        true_df["id"] = [f"true_{idx}" for idx in range(len(true_df))]
        fake_df["label"] = 1
        true_df["label"] = 0

        fake_df["source"] = (
            fake_df["subject"].fillna("ISOT-Fake") if "subject" in fake_df.columns else "ISOT-Fake"
        )
        true_df["source"] = (
            true_df["subject"].fillna("ISOT-True") if "subject" in true_df.columns else "ISOT-True"
        )

        combined = pd.concat([fake_df, true_df], ignore_index=True)
        LOGGER.info(
            "Loaded ISOT rows: fake=%d, true=%d, total=%d",
            len(fake_df),
            len(true_df),
            len(combined),
        )
        return combined

    @staticmethod
    def _first_available_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
        frame_cols = {col.lower(): col for col in frame.columns}
        for candidate in candidates:
            if candidate.lower() in frame_cols:
                return frame_cols[candidate.lower()]
        return None

    @staticmethod
    def _normalize_label(label: Any) -> int:
        if pd.isna(label):
            return 0

        if isinstance(label, (int, float)):
            return int(label)

        text = str(label).strip().lower()
        if text in {"fake", "false", "1", "rumor", "rumour"}:
            return 1
        if text in {"real", "true", "0", "non-fake", "genuine"}:
            return 0

        try:
            return int(float(text))
        except ValueError:
            LOGGER.warning("Unknown label value '%s'. Mapping to 0", label)
            return 0

    def _normalize_columns(self, frame: pd.DataFrame, source: str) -> pd.DataFrame:
        title_col = self._first_available_column(frame, ["title", "headline"])
        text_col = self._first_available_column(
            frame,
            ["text", "content", "article", "body", "news", "statement"],
        )
        label_col = self._first_available_column(frame, ["label", "class", "target"])
        id_col = self._first_available_column(frame, ["id", "news_id", "article_id"])
        source_col = self._first_available_column(frame, ["source", "publisher", "domain"])

        if text_col is None or label_col is None:
            raise ValueError(
                "Could not infer required text/label columns from dataset. "
                f"Columns found: {list(frame.columns)}"
            )

        normalized = pd.DataFrame()
        normalized["id"] = (
            frame[id_col].astype(str)
            if id_col is not None
            else pd.Series(range(len(frame)), dtype="int64").astype(str)
        )
        normalized["title"] = frame[title_col].fillna("").astype(str) if title_col else ""
        normalized["text"] = frame[text_col].fillna("").astype(str)
        normalized["label"] = frame[label_col].apply(self._normalize_label).astype(int)
        normalized["source"] = (
            frame[source_col].fillna(source).astype(str) if source_col is not None else source
        )
        normalized["full_text"] = (
            normalized["title"].str.strip() + " " + normalized["text"].str.strip()
        ).str.strip()

        required_order = ["id", "title", "text", "full_text", "label", "source"]
        return normalized.reindex(columns=required_order)

    def preprocess(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Clean text fields and add engineered columns for downstream steps."""
        LOGGER.info("Preprocessing dataset with %d rows", len(frame))
        processed = frame.copy()

        for col in ["title", "text", "full_text"]:
            processed[col] = processed[col].fillna("").astype(str)
            processed[col] = processed[col].str.lower()
            processed[col] = processed[col].str.replace(
                r"https?://\S+|www\.\S+",
                " ",
                regex=True,
            )
            processed[col] = processed[col].str.replace(r"<[^>]+>", " ", regex=True)
            processed[col] = processed[col].str.replace(r"\s+", " ", regex=True).str.strip()

        processed["full_text"] = (
            processed["title"].str.strip() + " " + processed["text"].str.strip()
        ).str.replace(r"\s+", " ", regex=True).str.strip()

        processed["text_length"] = processed["full_text"].str.split().str.len().fillna(0).astype(int)
        processed["has_author"] = (
            processed["full_text"].str.contains(
                r"(^|\n)\s*(by|author|written by)\s+[a-z][a-z\s\-']{1,80}",
                regex=True,
            )
        ).astype(int)

        label_dist = processed["label"].value_counts(dropna=False).to_dict()
        LOGGER.info("Class distribution: %s", label_dist)
        return processed

    def split(
        self,
        frame: pd.DataFrame,
        train_ratio: float | None = None,
        save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return stratified train/test split and optionally persist as CSV files."""
        ratio = float(train_ratio) if train_ratio is not None else self.split_ratio
        test_size = 1.0 - ratio

        LOGGER.info(
            "Splitting data with train_ratio=%.3f, test_ratio=%.3f, seed=%d",
            ratio,
            test_size,
            self.seed,
        )
        train_df, test_df = train_test_split(
            frame,
            test_size=test_size,
            stratify=frame["label"],
            random_state=self.seed,
        )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        LOGGER.info("Train shape=%s | Test shape=%s", train_df.shape, test_df.shape)

        if save:
            self.save_splits(train_df, test_df)

        return train_df, test_df

    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save train/test splits to the processed data directory."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.processed_dir / "train.csv"
        test_path = self.processed_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        LOGGER.info("Saved processed train split to %s", train_path)
        LOGGER.info("Saved processed test split to %s", test_path)


def _print_summary(processed_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== Data Summary ===")
    print(f"Processed shape: {processed_df.shape}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("\nLabel distribution (processed):")
    print(processed_df["label"].value_counts(normalize=False).sort_index())
    print("\nAverage text length (words):")
    print(f"{processed_df['text_length'].mean():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FakeNewsNet data loading pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    loader = FakeNewsNetLoader(config_path=args.config)
    loaded_df = loader.load()
    processed_df = loader.preprocess(loaded_df)
    train_df, test_df = loader.split(processed_df)
    _print_summary(processed_df, train_df, test_df)


if __name__ == "__main__":
    main()
