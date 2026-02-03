"""SHAP-based explanations for Approach A Decision Tree models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from scipy import sparse
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import FeatureEngineer
from src.evaluation.experiments import NearDuplicatePairGenerator
from src.evaluation.fidelity import FidelityEvaluator


class TreeExplainer:
    """Provides local/global SHAP explanations and rule extraction for a decision tree."""

    def __init__(self, model, feature_names: Iterable[str], run_tag: str = "original") -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.explainer = shap.TreeExplainer(model)
        self.predict_fn = self.model.predict
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.run_tag = str(run_tag).strip().lower() or "original"
        self.suffix = f"_{self.run_tag}" if self.run_tag != "original" else ""

        self.figures_dir = PROJECT_ROOT / "results" / "figures"
        self.tables_dir = PROJECT_ROOT / "results" / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_2d_input(X):
        if sparse.issparse(X):
            if X.ndim == 1:
                return X.reshape(1, -1)
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _to_dense_array(X) -> np.ndarray:
        if sparse.issparse(X):
            return np.asarray(X.toarray(), dtype=float)
        return np.asarray(X, dtype=float)

    @staticmethod
    def _coerce_shap_values(shap_output: Any) -> np.ndarray:
        """Normalize SHAP output formats to a 2D array of class-1 contributions."""
        values = shap_output.values if hasattr(shap_output, "values") else shap_output

        if isinstance(values, list):
            # Binary/multi-class older SHAP API.
            values = values[1] if len(values) > 1 else values[0]

        values = np.asarray(values)
        if values.ndim == 3:
            # Newer API can return (n_samples, n_features, n_classes).
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return values

    def _compute_shap_values(self, X) -> np.ndarray:
        """Compute SHAP values and handle sparse fallback when needed."""
        X_2d = self._ensure_2d_input(X)
        X_eval = self._to_dense_array(X_2d)
        try:
            output = self.explainer.shap_values(X_eval)
            return self._coerce_shap_values(output)
        except Exception as exc:
            self.logger.warning("SHAP computation on native input failed (%s). Retrying with dense array.", exc)
            dense = np.asarray(X_eval)
            output = self.explainer.shap_values(dense)
            return self._coerce_shap_values(output)

    def _sample_row_value(self, X_sample, feature_index: int) -> float:
        if sparse.issparse(X_sample):
            return float(X_sample[0, feature_index])
        return float(np.asarray(X_sample)[0, feature_index])

    def _decision_path_rules_for_sample(self, X_sample) -> list[str]:
        """Generate IF-THEN style rules for one sample's traversed decision path."""
        X_2d = self._ensure_2d_input(X_sample)
        tree = self.model.tree_
        node_indicator = self.model.decision_path(X_2d)
        leaf_id = int(self.model.apply(X_2d)[0])
        node_index = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]

        rules: list[str] = []
        for node_id in node_index:
            if tree.feature[node_id] == -2:
                continue

            feature_idx = int(tree.feature[node_id])
            feature_name = self.feature_names[feature_idx]
            threshold = float(tree.threshold[node_id])
            feature_value = self._sample_row_value(X_2d, feature_idx)
            operator = "<=" if feature_value <= threshold else ">"
            rules.append(
                f"IF {feature_name} {operator} {threshold:.6f} (value={feature_value:.6f})"
            )

        proba = self.model.predict_proba(X_2d)[0]
        pred_class = int(np.argmax(proba))
        rules.append(
            f"THEN leaf={leaf_id}, predict class={pred_class}, probability={float(np.max(proba)):.6f}"
        )
        return rules

    def explain_local(self, X_sample) -> dict[str, Any]:
        """Return sorted local SHAP contributions and sample-specific decision path."""
        X_2d = self._ensure_2d_input(X_sample)
        shap_values = self._compute_shap_values(X_2d)[0]

        sorted_items = sorted(
            zip(self.feature_names, shap_values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        local_map = {name: float(value) for name, value in sorted_items}
        rules = self._decision_path_rules_for_sample(X_2d)

        # Waterfall plot for one prediction.
        base_values = self.explainer.expected_value
        if isinstance(base_values, (list, np.ndarray)):
            base = float(np.asarray(base_values).ravel()[-1])
        else:
            base = float(base_values)

        sample_values_dense = self._to_dense_array(X_2d)[0]
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base,
            data=sample_values_dense,
            feature_names=self.feature_names,
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=20, show=False)
        waterfall_path = self.figures_dir / f"dt_shap_local_waterfall{self.suffix}.png"
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "local_shap": local_map,
            "decision_path": rules,
            "waterfall_plot_path": str(waterfall_path),
        }

    def explain_global(self, X_test, top_n: int = 20) -> pd.DataFrame:
        """Compute global SHAP importances and save SHAP summary plot."""
        X_2d = self._ensure_2d_input(X_test)
        shap_values = self._compute_shap_values(X_2d)
        mean_abs = np.mean(np.abs(shap_values), axis=0)

        global_df = pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        top_df = global_df.head(top_n).reset_index(drop=True)

        plot_input = self._to_dense_array(X_2d)
        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            shap_values,
            plot_input,
            feature_names=self.feature_names,
            show=False,
            max_display=top_n,
            plot_type="bar",
            color="#E07A5F"
        )
        summary_path = self.figures_dir / f"dt_shap_global{self.suffix}.png"
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            shap_values,
            plot_input,
            feature_names=self.feature_names,
            show=False,
            max_display=top_n,
            plot_type="dot",
        )
        beeswarm_path = self.figures_dir / f"dt_shap_global_beeswarm{self.suffix}.png"
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()

        return top_df

    def compute_stability(self, X_pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
        """Measure explanation stability via Pearson r across near-duplicate pairs."""
        scores: list[float] = []
        per_pair: list[dict[str, float]] = []

        for pair_idx, (x1, x2) in enumerate(X_pairs):
            v1 = self._compute_shap_values(x1)[0]
            v2 = self._compute_shap_values(x2)[0]

            if np.std(v1) == 0 or np.std(v2) == 0:
                r = np.nan
            else:
                pearson_result = pearsonr(v1, v2)
                r_stat = getattr(pearson_result, "statistic", pearson_result[0])
                r = float(np.asarray(r_stat, dtype=float).item())

            scores.append(r)
            per_pair.append({"pair_index": pair_idx, "pearson_r": r})

        arr = np.asarray(scores, dtype=float)
        return {
            "mean_pearson_r": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
            "std_pearson_r": float(np.nanstd(arr)) if np.isfinite(arr).any() else float("nan"),
            "per_pair_scores": per_pair,
        }

    def compute_stability_from_pairs_file(
        self,
        pair_csv_path: str | Path,
        feature_engineer: FeatureEngineer,
        source_df: pd.DataFrame,
        n_pairs: int = 50,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Load/generate reproducible near-duplicate text pairs and compute SHAP stability."""
        pairs = NearDuplicatePairGenerator.load_or_generate_pairs(
            df=source_df,
            path=pair_csv_path,
            n_pairs=n_pairs,
            seed=seed,
        )

        if not pairs:
            return {
                "mean_pearson_r": float("nan"),
                "std_pearson_r": float("nan"),
                "per_pair_scores": [],
            }

        flat_texts: list[str] = []
        for original_text, variant_text, _ in pairs:
            flat_texts.extend([original_text, variant_text])

        pair_df = pd.DataFrame({"full_text": flat_texts})
        X_flat, _ = feature_engineer.build_feature_matrix(
            pair_df,
            tfidf_vectorizer=feature_engineer.vectorizer,
            fit_tfidf=False,
        )

        X_pairs: list[tuple[Any, Any]] = []
        for idx in range(0, X_flat.shape[0], 2):
            if idx + 1 >= X_flat.shape[0]:
                break
            X_pairs.append((X_flat.getrow(idx), X_flat.getrow(idx + 1)))

        return self.compute_stability(X_pairs)

    def extract_decision_rules(
        self,
        model,
        feature_names: Iterable[str],
        max_depth: int = 10,
    ) -> list[str]:
        """Export tree into human-readable IF-THEN rules and save to disk."""
        feature_names = list(feature_names)
        tree = model.tree_
        rules: list[str] = []

        def _recurse(node_id: int, depth: int, path_conditions: list[str]) -> None:
            if tree.feature[node_id] == -2 or depth >= max_depth:
                class_counts = tree.value[node_id][0]
                pred_class = int(np.argmax(class_counts))
                total = float(np.sum(class_counts))
                prob = float(np.max(class_counts) / total) if total > 0 else 0.0
                prefix = " AND ".join(path_conditions) if path_conditions else "TRUE"
                rules.append(
                    f"IF {prefix} THEN class={pred_class} (node_samples={int(tree.n_node_samples[node_id])}, prob={prob:.6f})"
                )
                return

            f_idx = int(tree.feature[node_id])
            f_name = feature_names[f_idx]
            thr = float(tree.threshold[node_id])
            left_cond = f"{f_name} <= {thr:.6f}"
            right_cond = f"{f_name} > {thr:.6f}"

            _recurse(tree.children_left[node_id], depth + 1, path_conditions + [left_cond])
            _recurse(tree.children_right[node_id], depth + 1, path_conditions + [right_cond])

        _recurse(0, 0, [])

        rules_path = self.tables_dir / f"dt_rules{self.suffix}.txt"
        with rules_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(rules))
        self.logger.info("Saved decision rules to %s", rules_path)
        return rules

    def get_top_features(
        self,
        shap_values,
        feature_names: Iterable[str],
        top_n: int = 20,
    ) -> list[str]:
        """Return ordered top-N features by mean absolute SHAP importance."""
        shap_matrix = self._coerce_shap_values(shap_values)
        feature_names = list(feature_names)
        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        idx = np.argsort(mean_abs)[::-1][:top_n]
        return [feature_names[i] for i in idx]

    def get_shap_values(self, X_samples) -> np.ndarray:
        """Return SHAP value matrix for one or more tabular samples."""
        return self._compute_shap_values(X_samples)

    def run_fidelity_test(self, X_samples, top_k: int | Iterable[int] = 5) -> dict[str, Any]:
        """Run fidelity evaluation for one or multiple top-k settings."""
        evaluator = FidelityEvaluator(
            model_predict_fn=self.predict_fn,
            explainer=self,
            feature_type="tabular",
            approach_name="dt",
            run_tag=self.run_tag,
        )
        shap_vals = self.get_shap_values(X_samples)

        if isinstance(top_k, int):
            return evaluator.fidelity_report(X_samples, shap_vals, top_k=int(top_k))

        reports: list[dict[str, Any]] = []
        for k in list(top_k):
            reports.append(evaluator.fidelity_report(X_samples, shap_vals, top_k=int(k)))
        return {"approach": "dt", "run_tag": self.run_tag, "reports": reports}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Explain Approach A Decision Tree pipeline")
    parser.add_argument(
        "--debiased",
        action="store_true",
        help="Use debiased models/data and append '_debiased' to output artifacts.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_path", "data/processed")
    tables_dir = PROJECT_ROOT / "results" / "tables"

    if args.debiased:
        train_path = processed_dir / "debiased_train.csv"
        test_path = processed_dir / "debiased_test.csv"
        run_tag = "debiased"
        models_dir = PROJECT_ROOT / "results" / "models" / "approach_a" / "debiased"
        comparison_path = tables_dir / "dt_depth_comparison_debiased.csv"
    else:
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        run_tag = "original"
        models_dir = PROJECT_ROOT / "results" / "models" / "approach_a"
        comparison_path = tables_dir / "dt_depth_comparison.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test data at {processed_dir}")
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing depth comparison file at {comparison_path}. Run training first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    comparison_df = pd.read_csv(comparison_path)
    if comparison_df.empty:
        raise ValueError("Depth comparison table is empty.")

    best_row = comparison_df.sort_values("cv_f1_mean", ascending=False).iloc[0]
    best_depth = int(best_row["depth"])
    best_criterion = str(best_row["criterion"])
    model_path = models_dir / f"decision_tree_depth_{best_depth}_{best_criterion}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected model file at {model_path}")

    model = joblib.load(model_path)

    # Note: Fitting TF-IDF on the training set so test features align correctly
    tfidf_candidates = config.get("models", {}).get("tfidf", {}).get("max_features", [5000])
    tfidf_max_features = int(max(tfidf_candidates))
    feature_engineer = FeatureEngineer(max_tfidf_features=tfidf_max_features)
    feature_engineer.build_feature_matrix(train_df, fit_tfidf=True) 
    X_test, feature_names = feature_engineer.build_feature_matrix(test_df, fit_tfidf=False)

    explainer = TreeExplainer(model=model, feature_names=feature_names, run_tag=run_tag)
    global_top = explainer.explain_global(X_test, top_n=20)

    def _row_slice(matrix: Any, idx: int):
        if sparse.issparse(matrix):
            return matrix.getrow(idx)
        return np.asarray(matrix)[idx : idx + 1]

    first_sample = _row_slice(X_test, 0)
    local_info = explainer.explain_local(first_sample)
    rules = explainer.extract_decision_rules(model, feature_names, max_depth=10)

    stability_sample_size = int(
        config.get("explainability", {})
        .get("shap", {})
        .get("stability_test_sample_size", 50)
    )
    
    # Save the pairs file with the matching suffix to prevent mixing up data
    pair_csv_path = tables_dir / f"near_duplicate_pairs{explainer.suffix}.csv"
    stability = explainer.compute_stability_from_pairs_file(
        pair_csv_path=pair_csv_path,
        feature_engineer=feature_engineer,
        source_df=test_df,
        n_pairs=stability_sample_size,
        seed=int(config.get("random_seed", 42)),
    )

    top_feature_names = explainer.get_top_features(
        explainer._compute_shap_values(X_test),
        feature_names,
        top_n=20,
    )

    print(f"\n=== Run mode: {run_tag} ===")
    print("\nTop Global SHAP Features:")
    print(global_top.to_string(index=False))

    print("\nTop Local SHAP Contributions (first sample):")
    for i, (name, value) in enumerate(local_info["local_shap"].items()):
        if i >= 10:
            break
        print(f"{name}: {value:.6f}")

    print("\nDecision Path Rules (first sample):")
    for rule in local_info["decision_path"]:
        print(rule)

    print("\nStability (Pearson r on near-neighbor pairs):")
    print({
        "mean_pearson_r": stability["mean_pearson_r"],
        "std_pearson_r": stability["std_pearson_r"],
        "pairs_evaluated": len(stability["per_pair_scores"]),
    })

    print("\nTop features from get_top_features():")
    print(top_feature_names)

    print("\nArtifacts Saved:")
    print(PROJECT_ROOT / "results" / "figures" / f"dt_shap_global{explainer.suffix}.png")
    print(PROJECT_ROOT / "results" / "figures" / f"dt_shap_local_waterfall{explainer.suffix}.png")
    print(PROJECT_ROOT / "results" / "tables" / f"dt_rules{explainer.suffix}.txt")
    print(pair_csv_path)

    fidelity_sample_size = int(
        config.get("explainability", {})
        .get("shap", {})
        .get("fidelity_sample_size", 256)
    )
    fidelity_count = min(fidelity_sample_size, X_test.shape[0])
    if sparse.issparse(X_test):
        X_fidelity = sparse.vstack([X_test.getrow(i) for i in range(fidelity_count)])
    else:
        X_fidelity = np.asarray(X_test)[:fidelity_count]
    fidelity_reports = explainer.run_fidelity_test(X_fidelity, top_k=[3, 5, 10])

    print("\nTotal extracted rules:", len(rules))
    print("\nFidelity reports:")
    print(fidelity_reports)