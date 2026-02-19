"""SHAP explanations for Approach B (BERT classifier)."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import yaml
from scipy.stats import pearsonr
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.approach_b.train import BERTClassifier
from src.evaluation.experiments import NearDuplicatePairGenerator
from src.evaluation.fidelity import FidelityEvaluator


class BERTShapExplainer:
    """Local/global SHAP explanations and stability tests for BERT predictions."""

    def __init__(self, model: BERTClassifier, tokenizer, run_tag: str = "original") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.run_tag = str(run_tag).strip().lower() or "original"
        self.suffix = f"_{self.run_tag}" if self.run_tag != "original" else ""

        # Grab max_tokens from the model's config or default to 256
        self.max_tokens = getattr(self.model, "max_tokens", 256)

        self.predict_fn = self._predict_proba_wrapper
        self.masker = shap.maskers.Text(tokenizer)  # type: ignore[attr-defined]
        self.explainer = shap.Explainer(self.predict_fn, self.masker)

        self.figures_dir = PROJECT_ROOT / "results" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _as_text_list(texts: Iterable[str]) -> list[str]:
        return pd.Series(list(texts), dtype="object").fillna("").astype(str).tolist()

    def _truncate_for_shap(self, text: str) -> str:
        """Force text to be <= max_tokens BEFORE SHAP sees it to prevent indexing errors."""
        tokens = self.tokenizer.encode(str(text), max_length=self.max_tokens, truncation=True)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _predict_proba_wrapper(self, texts: Iterable[str]) -> np.ndarray:
        """High-speed inference wrapper bypassing Arrow Dataset overhead."""
        text_list = self._as_text_list(texts)
        
        # 1. Direct memory tokenization (Extremely fast for SHAP permutations)
        inputs = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors="pt"
        )
        
        device = self.model.device
        hf_model = self.model.model  # Access the underlying HuggingFace PyTorch model
        hf_model.eval()
        
        all_probs = []
        batch_size = 256  # Maximize GPU utilization
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                for i in range(0, len(text_list), batch_size):
                    batch_inputs = {k: v[i:i+batch_size].to(device) for k, v in inputs.items()}
                    outputs = hf_model(**batch_inputs)
                    probs = torch.softmax(outputs.logits.float(), dim=1).cpu().numpy()
                    all_probs.append(probs)
                    
        return np.vstack(all_probs)

    def _select_label_index(self, explanation, label: Optional[int], text: str) -> int:
        values = np.asarray(explanation.values)
        if values.ndim < 2:
            return 0

        if label is not None:
            return int(np.clip(label, 0, values.shape[1] - 1))

        pred = self.predict_fn([text])[0]
        return int(np.argmax(pred))

    @staticmethod
    def _safe_base_value(base_values, label_idx: int) -> float:
        base_arr = np.asarray(base_values)
        if base_arr.ndim == 0:
            return float(base_arr)
        flat = base_arr.reshape(-1)
        idx = int(np.clip(label_idx, 0, len(flat) - 1))
        return float(flat[idx])

    def _single_output_explanation(self, explanation, label_idx: int):
        values = np.asarray(explanation.values)
        if values.ndim == 2:
            token_values = values[:, label_idx]
        else:
            token_values = values

        base_value = self._safe_base_value(explanation.base_values, label_idx)
        return shap.Explanation(
            values=token_values,
            base_values=base_value,
            data=np.asarray(explanation.data),
            feature_names=getattr(explanation, "feature_names", None),
        )

    def _token_values_from_explanation(self, explanation, label_idx: int) -> tuple[list[str], np.ndarray]:
        exp = self._single_output_explanation(explanation, label_idx)
        tokens = [str(tok) for tok in np.asarray(exp.data).tolist()]
        values = np.asarray(exp.values, dtype=float)
        return tokens, values

    def token_to_word_shap(self, tokens: list[str], values: np.ndarray) -> dict[str, float]:
        """Convert token-level SHAP values to word-level values by merging subwords."""
        special_tokens = set(getattr(self.tokenizer, "all_special_tokens", []))
        word_values: dict[str, float] = defaultdict(float)

        current_word = ""
        current_value = 0.0

        def flush_current() -> None:
            nonlocal current_word, current_value
            if current_word:
                word_values[current_word] += float(current_value)
                current_word = ""
                current_value = 0.0

        for tok, val in zip(tokens, values):
            token = str(tok)
            value = float(val)

            if token in special_tokens or token.strip() == "":
                flush_current()
                continue

            if token.startswith("##"):
                piece = token[2:]
                if not current_word:
                    current_word = piece
                else:
                    current_word += piece
                current_value += value
                continue

            if token.startswith("Ġ") or token.startswith("▁"):
                flush_current()
                current_word = token[1:]
                current_value = value
                continue

            if token.startswith("</") or token.startswith("<"):
                flush_current()
                continue

            flush_current()
            current_word = token
            current_value = value

        flush_current()
        return dict(word_values)

    def _compute_shap_batch(self, batch_texts: list[str]):
        return self.explainer(batch_texts)

    @staticmethod
    def _as_explanation_list(explanations) -> list[Any]:
        if isinstance(explanations, list):
            return explanations
        try:
            size = len(explanations)
            return [explanations[i] for i in range(size)]
        except Exception:
            return [explanations]

    def _save_local_html(self, explanation, output_path: Path) -> None:
        """Save a SHAP text visualization as HTML, with fallback renderer."""
        try:
            html_obj = shap.plots.text(explanation, display=False)
            if hasattr(html_obj, "data"):
                html_content = str(html_obj.data)
            else:
                html_content = str(html_obj)
            output_path.write_text(html_content, encoding="utf-8")
            return
        except Exception as exc:
            self.logger.warning("Could not render SHAP text HTML directly (%s). Using fallback HTML.", exc)

        tokens = [str(t) for t in np.asarray(explanation.data).tolist()]
        values = np.asarray(explanation.values, dtype=float)
        rows = []
        for tok, val in zip(tokens, values):
            color = "#f8d7da" if val > 0 else "#d1ecf1"
            rows.append(
                f"<span style='display:inline-block;margin:2px;padding:3px 6px;background:{color};'>"
                f"{tok} ({val:.4f})</span>"
            )
        html_content = (
            "<html><head><meta charset='utf-8'><title>BERT SHAP Local</title></head><body>"
            "<h3>Token SHAP Contributions</h3>"
            + "\n".join(rows)
            + "</body></html>"
        )
        output_path.write_text(html_content, encoding="utf-8")

    def explain_local(self, text: str, label: Optional[int] = None) -> dict[str, float]:
        """Explain one text and return top-20 word-level SHAP contributions."""
        text = self._truncate_for_shap(text) # <--- Truncate BEFORE SHAP
        explanation = self._compute_shap_batch([text])[0]
        label_idx = self._select_label_index(explanation, label, text)

        tokens, values = self._token_values_from_explanation(explanation, label_idx)
        word_map = self.token_to_word_shap(tokens, values)

        sorted_items = sorted(word_map.items(), key=lambda item: abs(item[1]), reverse=True)[:20]
        top_dict = {token: float(score) for token, score in sorted_items}

        html_path = self.figures_dir / f"bert_shap_local_example{self.suffix}.html"
        exp_for_plot = self._single_output_explanation(explanation, label_idx)
        self._save_local_html(exp_for_plot, html_path)

        return top_dict

    def _sample_texts(self, texts: Iterable[str], sample_size: int) -> list[str]:
        all_texts = self._as_text_list(texts)
        if len(all_texts) <= sample_size:
            return all_texts
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_texts), size=sample_size, replace=False)
        return [all_texts[i] for i in idx]

    def explain_global(self, texts: Iterable[str], sample_size: int = 100) -> pd.DataFrame:
        """Compute global token importance from SHAP with batched inference."""
        sampled_texts = self._sample_texts(texts, sample_size)
        
        # <--- Truncate ALL texts BEFORE SHAP to prevent warnings and speed up processing
        sampled_texts = [self._truncate_for_shap(t) for t in sampled_texts]
        
        batch_size = 128

        abs_sum: dict[str, float] = defaultdict(float)
        count: dict[str, int] = defaultdict(int)

        total_batches = int(math.ceil(len(sampled_texts) / batch_size))
        for start in tqdm(range(0, len(sampled_texts), batch_size), total=total_batches, desc="Global SHAP"):
            batch = sampled_texts[start : start + batch_size]
            batch_explanations = self._as_explanation_list(self._compute_shap_batch(batch))

            for text, exp in zip(batch, batch_explanations):
                label_idx = self._select_label_index(exp, None, text)
                tokens, values = self._token_values_from_explanation(exp, label_idx)
                word_map = self.token_to_word_shap(tokens, values)
                for token, score in word_map.items():
                    abs_sum[token] += abs(float(score))
                    count[token] += 1

        rows = [
            {"token": token, "mean_abs_shap": abs_sum[token] / count[token]}
            for token in abs_sum
            if count[token] > 0
        ]
        global_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        top_plot = global_df.head(30).iloc[::-1]
        plt.figure(figsize=(10, 12))
        plt.barh(top_plot["token"], top_plot["mean_abs_shap"], color="#1f77b4")
        plt.title("BERT Global Token Importance (mean |SHAP|)")
        plt.xlabel("Mean |SHAP|")
        plt.ylabel("Token")
        plt.tight_layout()
        global_path = self.figures_dir / f"bert_shap_global{self.suffix}.png"
        plt.savefig(global_path, dpi=300, bbox_inches="tight")
        plt.close()

        return global_df

    def _word_importance_vector(self, text: str) -> dict[str, float]:
        text = self._truncate_for_shap(text) # <--- Truncate BEFORE SHAP
        explanation = self._compute_shap_batch([text])[0]
        label_idx = self._select_label_index(explanation, None, text)
        tokens, values = self._token_values_from_explanation(explanation, label_idx)
        word_map = self.token_to_word_shap(tokens, values)
        return {k: abs(v) for k, v in word_map.items()}

    def compute_stability(self, text_pairs: list[tuple[str, str]]) -> dict[str, Any]:
        """Compute mean/std Pearson r over near-duplicate text explanation vectors."""
        per_pair_scores: list[dict[str, float]] = []
        all_scores: list[float] = []

        for i, (text1, text2) in enumerate(tqdm(text_pairs, desc="Stability SHAP")):
            vec1 = self._word_importance_vector(str(text1))
            vec2 = self._word_importance_vector(str(text2))

            shared_tokens = sorted(set(vec1).intersection(vec2))
            if len(shared_tokens) < 2:
                r = float("nan")
            else:
                v1 = np.asarray([vec1[tok] for tok in shared_tokens], dtype=float)
                v2 = np.asarray([vec2[tok] for tok in shared_tokens], dtype=float)
                if np.std(v1) == 0 or np.std(v2) == 0:
                    r = float("nan")
                else:
                    pearson_result = pearsonr(v1, v2)
                    stat = getattr(pearson_result, "statistic", pearson_result[0])
                    r = float(np.asarray(stat, dtype=float).item())

            all_scores.append(r)
            per_pair_scores.append({"pair_index": float(i), "pearson_r": r})

        score_arr = np.asarray(all_scores, dtype=float)
        if np.isfinite(score_arr).any():
            mean_r = float(np.nanmean(score_arr))
            std_r = float(np.nanstd(score_arr))
        else:
            mean_r = float("nan")
            std_r = float("nan")

        return {
            "mean_pearson_r": mean_r,
            "std_pearson_r": std_r,
            "per_pair_scores": per_pair_scores,
        }

    def compute_stability_from_pairs_file(
        self,
        pair_csv_path: str | Path,
        source_df: pd.DataFrame,
        n_pairs: int = 50,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Load/generate reproducible near-duplicate text pairs and compute SHAP stability."""
        labeled_pairs = NearDuplicatePairGenerator.load_or_generate_pairs(
            df=source_df,
            path=pair_csv_path,
            n_pairs=n_pairs,
            seed=seed,
        )
        text_pairs = [(str(t1), str(t2)) for t1, t2, _ in labeled_pairs]
        return self.compute_stability(text_pairs)

    def get_top_tokens(self, shap_values, top_n: int = 20) -> list[str]:
        """Return ordered top tokens by mean |SHAP|."""
        if isinstance(shap_values, pd.DataFrame):
            if {"token", "mean_abs_shap"}.issubset(set(shap_values.columns)):
                return shap_values.sort_values("mean_abs_shap", ascending=False)["token"].head(top_n).tolist()

        if isinstance(shap_values, list) and shap_values and isinstance(shap_values[0], dict):
            agg_sum: dict[str, float] = defaultdict(float)
            agg_count: dict[str, int] = defaultdict(int)
            for sample_map in shap_values:
                for token, score in sample_map.items():
                    agg_sum[token] += abs(float(score))
                    agg_count[token] += 1
            rows = [
                (token, agg_sum[token] / agg_count[token])
                for token in agg_sum
                if agg_count[token] > 0
            ]
            rows.sort(key=lambda item: item[1], reverse=True)
            return [token for token, _ in rows[:top_n]]

        values = np.asarray(getattr(shap_values, "values", shap_values))
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
        if values.ndim == 1:
            values = values.reshape(1, -1)

        mean_abs = np.mean(np.abs(values), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        feature_names = getattr(shap_values, "feature_names", None)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(values.shape[1])]
        return [str(feature_names[i]) for i in top_idx]

    def get_shap_values(self, texts: Iterable[str]) -> list[dict[str, float]]:
        """Return per-sample word-level SHAP maps for a batch of texts."""
        texts_list = self._as_text_list(texts)
        all_maps: list[dict[str, float]] = []

        batch_size = 128
        for start in range(0, len(texts_list), batch_size):
            batch_texts = [self._truncate_for_shap(t) for t in texts_list[start : start + batch_size]]
            batch_explanations = self._as_explanation_list(self._compute_shap_batch(batch_texts))

            for text, exp in zip(batch_texts, batch_explanations):
                label_idx = self._select_label_index(exp, None, text)
                tokens, values = self._token_values_from_explanation(exp, label_idx)
                all_maps.append(self.token_to_word_shap(tokens, values))

        return all_maps

    def run_fidelity_test(self, X_samples, top_k: int | Iterable[int] = 5) -> dict[str, Any]:
        """Run fidelity evaluation for one or multiple top-k settings."""
        evaluator = FidelityEvaluator(
            model_predict_fn=self.predict_fn,
            explainer=self,
            feature_type="text",
            approach_name="bert",
            run_tag=self.run_tag,
        )
        texts = self._as_text_list(X_samples)
        shap_vals = self.get_shap_values(texts)

        if isinstance(top_k, int):
            return evaluator.fidelity_report(texts, shap_vals, top_k=int(top_k))

        reports: list[dict[str, Any]] = []
        for k in list(top_k):
            reports.append(evaluator.fidelity_report(texts, shap_vals, top_k=int(k)))
        return {"approach": "bert", "run_tag": self.run_tag, "reports": reports}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Explain Approach B BERT classifier")
    parser.add_argument(
        "--debiased",
        action="store_true",
        help="Use debiased models/data and append '_debiased' to output artifacts.",
    )
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_path", "data/processed")
    tables_dir = PROJECT_ROOT / "results" / "tables"

    if args.debiased:
        train_path = processed_dir / "debiased_train.csv"
        test_path = processed_dir / "debiased_test.csv"
        run_tag = "debiased"
    else:
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        run_tag = "original"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    test_df = pd.read_csv(test_path)
    text_col = "full_text" if "full_text" in test_df.columns else "text"
    if text_col not in test_df.columns:
        raise ValueError("Test data must include 'full_text' or 'text'.")

    # Load classifier using the run_tag so it grabs from the right directory
    classifier = BERTClassifier(config, run_tag=run_tag)
    try:
        classifier.load()
    except Exception as e:
        raise FileNotFoundError(f"Could not load BERT model for run_tag '{run_tag}'. Run training first.") from e

    explainer = BERTShapExplainer(model=classifier, tokenizer=classifier.tokenizer, run_tag=run_tag)

    sample_texts = test_df[text_col].fillna("").astype(str).tolist()
    
    global_sample_size = int(config.get("explainability", {}).get("shap", {}).get("global_sample_size", 15))
    
    local_top = explainer.explain_local(sample_texts[0])
    global_df = explainer.explain_global(sample_texts, sample_size=global_sample_size)

    pair_count = int(config.get("explainability", {}).get("shap", {}).get("stability_test_sample_size", 15))
    pair_csv_path = tables_dir / f"near_duplicate_pairs{explainer.suffix}.csv"
    
    stability = explainer.compute_stability_from_pairs_file(
        pair_csv_path=pair_csv_path,
        source_df=test_df,
        n_pairs=pair_count,
        seed=int(config.get("random_seed", 42)),
    )

    top_tokens = explainer.get_top_tokens(global_df, top_n=20)

    print(f"\n=== Run mode: {run_tag} ===")
    print("\nTop local tokens:")
    print(local_top)

    print("\nTop global tokens:")
    print(global_df.head(20).to_string(index=False))

    print("\nStability summary:")
    print({
        "mean_pearson_r": stability["mean_pearson_r"],
        "std_pearson_r": stability["std_pearson_r"],
        "pairs_evaluated": len(stability["per_pair_scores"]),
    })

    print("\nTop tokens from helper:")
    print(top_tokens)

    print("\nArtifacts Saved:")
    print(PROJECT_ROOT / "results" / "figures" / f"bert_shap_local_example{explainer.suffix}.html")
    print(PROJECT_ROOT / "results" / "figures" / f"bert_shap_global{explainer.suffix}.png")
    print(pair_csv_path)

    fidelity_sample_size = int(
        config.get("explainability", {})
        .get("shap", {})
        .get("fidelity_sample_size", 64)
    )
    fidelity_texts = sample_texts[: min(fidelity_sample_size, len(sample_texts))]
    fidelity_reports = explainer.run_fidelity_test(fidelity_texts, top_k=[3, 5, 10])

    print("\nFidelity reports:")
    print(fidelity_reports)