"""Fidelity evaluation for explanation faithfulness across tabular and text models."""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class FidelityEvaluator:
    """Evaluate explanation fidelity via sufficiency/necessity and random baselines."""

    def __init__(
        self,
        model_predict_fn,
        explainer,
        feature_type: str,
        approach_name: str | None = None,
        run_tag: str = "original",
        seed: int = 42,
    ) -> None:
        if feature_type not in {"tabular", "text"}:
            raise ValueError("feature_type must be 'tabular' or 'text'.")

        self.model_predict_fn = model_predict_fn
        self.explainer = explainer
        self.feature_type = feature_type
        self.approach_name = approach_name or ("dt" if feature_type == "tabular" else "bert")
        self.run_tag = str(run_tag).strip().lower() or "original"
        self.suffix = f"_{self.run_tag}" if self.run_tag != "original" else ""
        self.rng = random.Random(int(seed))
        self.logger = logging.getLogger(self.__class__.__name__)

        self.results_dir = PROJECT_ROOT / "results"
        self.tables_dir = self.results_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = getattr(self.explainer, "tokenizer", None)
        self.pad_token = "[PAD]"
        if tokenizer is not None:
            self.pad_token = (
                getattr(tokenizer, "pad_token", None)
                or getattr(tokenizer, "mask_token", None)
                or "[PAD]"
            )

    def _predict_labels(self, X) -> np.ndarray:
        raw = self.model_predict_fn(X)
        arr = np.asarray(raw)

        if arr.ndim == 0:
            return np.asarray([int(arr)], dtype=int)

        if arr.ndim == 1:
            if np.issubdtype(arr.dtype, np.integer):
                return arr.astype(int)
            return (arr >= 0.5).astype(int)

        if arr.ndim == 2:
            if self.feature_type == "text":
                if arr.shape[1] >= 2:
                    return (arr[:, 1] >= 0.5).astype(int)
                return (arr[:, 0] >= 0.5).astype(int)
            return np.argmax(arr, axis=1).astype(int)

        raise ValueError(f"Unsupported prediction output shape: {arr.shape}")

    @staticmethod
    def _to_2d_dense(X) -> np.ndarray:
        if sparse.issparse(X):
            return np.asarray(X.toarray(), dtype=float)
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _tabular_top_indices(self, shap_values: np.ndarray, sample_idx: int, k: int) -> np.ndarray:
        row = np.asarray(shap_values[sample_idx], dtype=float)
        k_eff = min(int(k), row.shape[0])
        if k_eff <= 0:
            return np.asarray([], dtype=int)
        idx = np.argsort(np.abs(row))[::-1][:k_eff]
        return np.asarray(idx, dtype=int)

    @staticmethod
    def _normalize_word(token: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(token).lower())

    def _extract_word_scores(self, shap_map: dict[str, float]) -> list[tuple[str, float]]:
        agg: dict[str, float] = {}
        for raw_token, score in shap_map.items():
            token = self._normalize_word(raw_token)
            if not token:
                continue
            agg[token] = agg.get(token, 0.0) + float(score)

        ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return ranked

    def _mask_text(self, text: str, chosen_words: set[str], keep_only: bool) -> str:
        parts = re.findall(r"\w+|\s+|[^\w\s]", str(text))
        out: list[str] = []
        for part in parts:
            if re.fullmatch(r"\w+", part):
                normalized = self._normalize_word(part)
                in_set = normalized in chosen_words
                if keep_only:
                    out.append(part if in_set else self.pad_token)
                else:
                    out.append(self.pad_token if in_set else part)
            else:
                out.append(part)
        return "".join(out)

    def sufficiency_score(self, X_samples, shap_values, top_k: int = 5) -> float:
        """Fraction of predictions preserved when only top-k explanation features are retained."""
        if self.feature_type == "tabular":
            X_arr = self._to_2d_dense(X_samples)
            shap_arr = np.asarray(shap_values, dtype=float)
            if shap_arr.ndim == 1:
                shap_arr = shap_arr.reshape(1, -1)

            orig_pred = self._predict_labels(X_arr)
            masked_rows = np.zeros_like(X_arr)
            for i in range(X_arr.shape[0]):
                top_idx = self._tabular_top_indices(shap_arr, i, top_k)
                masked_rows[i, top_idx] = X_arr[i, top_idx]

            masked_pred = self._predict_labels(masked_rows)
            return float(np.mean(masked_pred == orig_pred))

        texts = [str(t) for t in X_samples]
        shap_list = list(shap_values)
        orig_pred = self._predict_labels(texts)

        masked_texts: list[str] = []
        for text, shap_map in zip(texts, shap_list):
            ranked = self._extract_word_scores(dict(shap_map))
            keep = {tok for tok, _ in ranked[: max(0, int(top_k))]}
            masked_texts.append(self._mask_text(text, keep, keep_only=True))

        masked_pred = self._predict_labels(masked_texts)
        return float(np.mean(masked_pred == orig_pred))

    def necessity_score(self, X_samples, shap_values, top_k: int = 5) -> float:
        """Fraction of predictions changed when top-k explanation features are removed."""
        if self.feature_type == "tabular":
            X_arr = self._to_2d_dense(X_samples)
            shap_arr = np.asarray(shap_values, dtype=float)
            if shap_arr.ndim == 1:
                shap_arr = shap_arr.reshape(1, -1)

            orig_pred = self._predict_labels(X_arr)
            masked_rows = X_arr.copy()
            for i in range(X_arr.shape[0]):
                top_idx = self._tabular_top_indices(shap_arr, i, top_k)
                masked_rows[i, top_idx] = 0.0

            masked_pred = self._predict_labels(masked_rows)
            return float(np.mean(masked_pred != orig_pred))

        texts = [str(t) for t in X_samples]
        shap_list = list(shap_values)
        orig_pred = self._predict_labels(texts)

        masked_texts: list[str] = []
        for text, shap_map in zip(texts, shap_list):
            ranked = self._extract_word_scores(dict(shap_map))
            remove = {tok for tok, _ in ranked[: max(0, int(top_k))]}
            masked_texts.append(self._mask_text(text, remove, keep_only=False))

        masked_pred = self._predict_labels(masked_texts)
        return float(np.mean(masked_pred != orig_pred))

    def random_baseline(self, X_samples, k: int = 5, n_trials: int = 10) -> dict[str, float]:
        """Random feature/token masking baseline for sufficiency and necessity."""
        k = max(0, int(k))
        n_trials = max(1, int(n_trials))

        suff_scores: list[float] = []
        nec_scores: list[float] = []

        if self.feature_type == "tabular":
            X_arr = self._to_2d_dense(X_samples)
            orig_pred = self._predict_labels(X_arr)
            n_features = X_arr.shape[1]
            k_eff = min(k, n_features)

            for _ in range(n_trials):
                suff_rows = np.zeros_like(X_arr)
                nec_rows = X_arr.copy()
                for i in range(X_arr.shape[0]):
                    random_idx = self.rng.sample(range(n_features), k_eff) if k_eff > 0 else []
                    suff_rows[i, random_idx] = X_arr[i, random_idx]
                    nec_rows[i, random_idx] = 0.0

                suff_pred = self._predict_labels(suff_rows)
                nec_pred = self._predict_labels(nec_rows)
                suff_scores.append(float(np.mean(suff_pred == orig_pred)))
                nec_scores.append(float(np.mean(nec_pred != orig_pred)))

        else:
            texts = [str(t) for t in X_samples]
            orig_pred = self._predict_labels(texts)

            token_sets: list[list[str]] = []
            for text in texts:
                words = [self._normalize_word(w) for w in re.findall(r"\w+", text)]
                words = [w for w in words if w]
                token_sets.append(sorted(set(words)))

            for _ in range(n_trials):
                suff_texts: list[str] = []
                nec_texts: list[str] = []
                for text, vocab in zip(texts, token_sets):
                    k_eff = min(k, len(vocab))
                    chosen = set(self.rng.sample(vocab, k_eff)) if k_eff > 0 else set()
                    suff_texts.append(self._mask_text(text, chosen, keep_only=True))
                    nec_texts.append(self._mask_text(text, chosen, keep_only=False))

                suff_pred = self._predict_labels(suff_texts)
                nec_pred = self._predict_labels(nec_texts)
                suff_scores.append(float(np.mean(suff_pred == orig_pred)))
                nec_scores.append(float(np.mean(nec_pred != orig_pred)))

        suff_arr = np.asarray(suff_scores, dtype=float)
        nec_arr = np.asarray(nec_scores, dtype=float)
        return {
            "mean_sufficiency_random": float(np.mean(suff_arr)),
            "mean_necessity_random": float(np.mean(nec_arr)),
            "std": float(np.mean([np.std(suff_arr), np.std(nec_arr)])),
            "std_sufficiency_random": float(np.std(suff_arr)),
            "std_necessity_random": float(np.std(nec_arr)),
        }

    def fidelity_report(self, X_samples, shap_values, top_k: int = 5) -> dict[str, float | int]:
        """Compute and persist fidelity metrics for one top-k setting."""
        sufficiency = self.sufficiency_score(X_samples, shap_values, top_k=top_k)
        necessity = self.necessity_score(X_samples, shap_values, top_k=top_k)
        baseline = self.random_baseline(X_samples, k=top_k, n_trials=10)

        report: dict[str, float | int] = {
            "sufficiency": float(sufficiency),
            "necessity": float(necessity),
            "random_sufficiency": float(baseline["mean_sufficiency_random"]),
            "random_necessity": float(baseline["mean_necessity_random"]),
            "sufficiency_lift": float(sufficiency - baseline["mean_sufficiency_random"]),
            "necessity_lift": float(necessity - baseline["mean_necessity_random"]),
            "composite_fidelity": float((sufficiency + necessity) / 2.0),
            "top_k": int(top_k),
        }

        out_path = self.tables_dir / f"fidelity_{self.approach_name}{self.suffix}.csv"
        row_df = pd.DataFrame([report])
        if out_path.exists():
            old_df = pd.read_csv(out_path)
            if "top_k" in old_df.columns:
                old_df = old_df[old_df["top_k"].astype(int) != int(top_k)]
            merged = pd.concat([old_df, row_df], ignore_index=True)
        else:
            merged = row_df

        merged = merged.sort_values("top_k").reset_index(drop=True)
        merged.to_csv(out_path, index=False)
        self.logger.info("Saved fidelity report to %s", out_path)
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Run fidelity evaluation for both approaches.")
    parser.add_argument("--sample-size", type=int, default=128, help="Max test samples per run mode.")
    args = parser.parse_args()

    from src.approach_a.explain import TreeExplainer
    from src.approach_b.explain import BERTShapExplainer
    from src.approach_b.train import BERTClassifier
    from src.features import FeatureEngineer

    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_path", "data/processed")
    tables_dir = PROJECT_ROOT / "results" / "tables"

    run_tags = ["original", "debiased"]
    top_k_values = [3, 5, 10]

    for run_tag in run_tags:
        if run_tag == "debiased":
            train_path = processed_dir / "debiased_train.csv"
            test_path = processed_dir / "debiased_test.csv"
            dt_models_dir = PROJECT_ROOT / "results" / "models" / "approach_a" / "debiased"
            dt_cmp_path = tables_dir / "dt_depth_comparison_debiased.csv"
        else:
            train_path = processed_dir / "train.csv"
            test_path = processed_dir / "test.csv"
            dt_models_dir = PROJECT_ROOT / "results" / "models" / "approach_a"
            dt_cmp_path = tables_dir / "dt_depth_comparison.csv"

        if not train_path.exists() or not test_path.exists():
            logging.warning("Skipping %s fidelity: missing train/test split files.", run_tag)
            continue
        if not dt_cmp_path.exists():
            logging.warning("Skipping %s fidelity: missing DT depth comparison file.", run_tag)
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path).head(max(1, int(args.sample_size)))

        # Decision Tree fidelity
        dt_cmp = pd.read_csv(dt_cmp_path)
        if dt_cmp.empty:
            logging.warning("Skipping %s DT fidelity: empty depth comparison.", run_tag)
            continue
        best_row = dt_cmp.sort_values("cv_f1_mean", ascending=False).iloc[0]
        dt_model_path = dt_models_dir / f"decision_tree_depth_{int(best_row['depth'])}_{str(best_row['criterion'])}.joblib"
        if dt_model_path.exists():
            dt_model = joblib.load(dt_model_path)
            tfidf_max = int(max(config.get("models", {}).get("tfidf", {}).get("max_features", [5000])))
            fe = FeatureEngineer(max_tfidf_features=tfidf_max)
            fe.build_feature_matrix(train_df, fit_tfidf=True)
            X_test, feature_names = fe.build_feature_matrix(test_df, fit_tfidf=False)
            dt_explainer = TreeExplainer(dt_model, feature_names, run_tag=run_tag)
            dt_reports = dt_explainer.run_fidelity_test(X_test, top_k=top_k_values)
            logging.info("%s DT fidelity reports: %s", run_tag, dt_reports)
        else:
            logging.warning("Skipping %s DT fidelity: model not found at %s", run_tag, dt_model_path)

        # BERT fidelity
        text_col = "full_text" if "full_text" in test_df.columns else "text"
        try:
            bert_classifier = BERTClassifier(config, run_tag=run_tag)
            bert_classifier.load()
            bert_explainer = BERTShapExplainer(bert_classifier, bert_classifier.tokenizer, run_tag=run_tag)
            texts = test_df[text_col].fillna("").astype(str).tolist()
            bert_reports = bert_explainer.run_fidelity_test(texts, top_k=top_k_values)
            logging.info("%s BERT fidelity reports: %s", run_tag, bert_reports)
        except Exception as exc:
            logging.warning("Skipping %s BERT fidelity due to error: %s", run_tag, exc)
