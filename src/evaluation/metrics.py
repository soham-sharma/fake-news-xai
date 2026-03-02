"""Comparison metrics and reporting utilities across Approach A and Approach B."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ComparisonMetrics:
    """Compute comparative metrics and generate final experiment summaries."""

    def __init__(self, run_tag: str = "original") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.run_tag = str(run_tag).strip().lower() or "original"
        self.suffix = f"_{self.run_tag}" if self.run_tag != "original" else ""

        self.results_dir = PROJECT_ROOT / "results"
        self.tables_dir = self.results_dir / "tables"
        self.summary_path = self.results_dir / f"RESULTS_SUMMARY{self.suffix}.md"
        
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _normalize_result_metrics(result_obj: dict[str, Any]) -> dict[str, float]:
        """Accept either {'metrics': {...}} or a direct metric dict."""
        metrics = result_obj.get("metrics", result_obj)
        keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        missing = [k for k in keys if k not in metrics]
        if missing:
            raise ValueError(f"Missing required metrics: {missing}")
        return {k: float(metrics[k]) for k in keys}

    @staticmethod
    def _df_to_markdown_table(df: pd.DataFrame) -> str:
        """Render a DataFrame as markdown without optional tabulate dependency."""
        if df.empty:
            return "| (empty) |\n|---|"

        columns = [str(col) for col in df.columns]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"

        rows: list[str] = []
        for _, row in df.iterrows():
            values: list[str] = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.6f}")
                else:
                    values.append(str(val))
            rows.append("| " + " | ".join(values) + " |")

        return "\n".join([header, separator] + rows)

    def accuracy_comparison(self, dt_results: dict[str, Any], bert_results: dict[str, Any]) -> pd.DataFrame:
        """Compare core classification metrics and compute H1 accuracy gap."""
        dt_metrics = self._normalize_result_metrics(dt_results)
        bert_metrics = self._normalize_result_metrics(bert_results)

        gap = float(bert_metrics["accuracy"] - dt_metrics["accuracy"])
        table = pd.DataFrame(
            [
                {
                    "approach": "Decision Tree (Approach A)",
                    **dt_metrics,
                    "accuracy_gap_bert_minus_dt": gap,
                },
                {
                    "approach": "BERT (Approach B)",
                    **bert_metrics,
                    "accuracy_gap_bert_minus_dt": gap,
                },
            ]
        )

        out_path = self.tables_dir / f"accuracy_comparison{self.suffix}.csv"
        table.to_csv(out_path, index=False)
        self.logger.info("Saved accuracy comparison to %s", out_path)

        self._cache["accuracy_table"] = table
        self._cache["accuracy_gap"] = gap
        return table

    def stability_comparison(
        self,
        dt_stability: dict[str, Any],
        bert_stability: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare stability (H2) and run a paired t-test when pair-level scores are available."""
        dt_mean = float(dt_stability.get("mean_pearson_r", np.nan))
        bert_mean = float(bert_stability.get("mean_pearson_r", np.nan))

        dt_scores_raw = dt_stability.get("per_pair_scores", [])
        bert_scores_raw = bert_stability.get("per_pair_scores", [])

        dt_map: dict[int, float] = {}
        for row in dt_scores_raw:
            idx = int(row.get("pair_index", len(dt_map)))
            dt_map[idx] = float(row.get("pearson_r", np.nan))

        bert_map: dict[int, float] = {}
        for row in bert_scores_raw:
            idx = int(row.get("pair_index", len(bert_map)))
            bert_map[idx] = float(row.get("pearson_r", np.nan))

        common_idx = sorted(set(dt_map).intersection(bert_map))
        dt_vals = np.asarray([dt_map[i] for i in common_idx], dtype=float)
        bert_vals = np.asarray([bert_map[i] for i in common_idx], dtype=float)

        valid_mask = np.isfinite(dt_vals) & np.isfinite(bert_vals)
        dt_vals = dt_vals[valid_mask]
        bert_vals = bert_vals[valid_mask]

        if len(dt_vals) >= 2:
            t_stat, p_value = ttest_rel(dt_vals, bert_vals)
            t_stat = float(np.asarray(t_stat, dtype=float).item())
            p_value = float(np.asarray(p_value, dtype=float).item())
        else:
            t_stat = float("nan")
            p_value = float("nan")

        if np.isnan(p_value):
            interpretation = "Insufficient paired stability data for significance testing."
        elif p_value < 0.05 and bert_mean > dt_mean:
            interpretation = "BERT explanations are significantly more stable than Decision Tree explanations."
        elif p_value < 0.05 and bert_mean < dt_mean:
            interpretation = "Decision Tree explanations are significantly more stable than BERT explanations."
        else:
            interpretation = "No statistically significant stability difference between approaches."

        result = {
            "mean_r_dt": dt_mean,
            "mean_r_bert": bert_mean,
            "t_stat": t_stat,
            "p_value": p_value,
            "interpretation": interpretation,
        }
        self._cache["stability"] = result
        return result

    def feature_overlap(
        self,
        dt_top_features: list[str],
        bert_top_tokens: list[str],
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Compute Jaccard overlap and rank correlation on shared top features/tokens."""

        def _normalize(items: list[str]) -> list[str]:
            return [str(x).strip().lower() for x in items[:top_n] if str(x).strip()]

        dt_list = _normalize(dt_top_features)
        bert_list = _normalize(bert_top_tokens)

        dt_set = set(dt_list)
        bert_set = set(bert_list)
        union = dt_set.union(bert_set)
        overlap = sorted(dt_set.intersection(bert_set))

        jaccard = float(len(overlap) / len(union)) if union else 0.0

        if len(overlap) >= 2:
            dt_rank = {token: rank for rank, token in enumerate(dt_list, start=1)}
            bert_rank = {token: rank for rank, token in enumerate(bert_list, start=1)}
            dt_ranks = np.asarray([dt_rank[token] for token in overlap], dtype=float)
            bert_ranks = np.asarray([bert_rank[token] for token in overlap], dtype=float)
            rho, p_val = spearmanr(dt_ranks, bert_ranks)
            rho = float(np.asarray(rho, dtype=float).item())
            p_val = float(np.asarray(p_val, dtype=float).item())
        else:
            rho = float("nan")
            p_val = float("nan")

        result = {
            "jaccard_score": jaccard,
            "spearman_rho": rho,
            "p_value": p_val,
            "overlapping_features": overlap,
        }
        self._cache["overlap"] = result
        return result

    def depth_tradeoff_table(self, dt_depth_results: pd.DataFrame) -> pd.DataFrame:
        """Add interpretability score to depth-wise Decision Tree metrics."""
        if "depth" not in dt_depth_results.columns:
            raise ValueError("dt_depth_results must include a 'depth' column.")

        table = dt_depth_results.copy()
        if "num_leaves" not in table.columns:
            # Approximate leaves for balanced trees when true leaf count is unavailable.
            table["num_leaves"] = table["depth"].astype(int).apply(lambda d: int(2 ** d))

        table["interpretability_score"] = 1.0 / np.log(table["num_leaves"].astype(float) + 1.0)
        table = table.sort_values("depth").reset_index(drop=True)

        out_path = self.tables_dir / f"dt_depth_tradeoff{self.suffix}.csv"
        table.to_csv(out_path, index=False)
        self.logger.info("Saved depth tradeoff table to %s", out_path)

        self._cache["depth_tradeoff"] = table
        return table

    @staticmethod
    def _normalize_fidelity_input(report_obj: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(report_obj, list):
            return [dict(x) for x in report_obj]
        if isinstance(report_obj, dict) and "reports" in report_obj:
            return [dict(x) for x in report_obj.get("reports", [])]
        if isinstance(report_obj, dict) and "top_k" in report_obj:
            return [dict(report_obj)]
        raise ValueError("Unsupported fidelity report format.")

    def fidelity_comparison(
        self,
        dt_fidelity_report: dict[str, Any] | list[dict[str, Any]],
        bert_fidelity_report: dict[str, Any] | list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Compare fidelity metrics side-by-side and compute composite deltas."""
        dt_reports = self._normalize_fidelity_input(dt_fidelity_report)
        bert_reports = self._normalize_fidelity_input(bert_fidelity_report)

        dt_by_k = {int(r["top_k"]): r for r in dt_reports if "top_k" in r}
        bert_by_k = {int(r["top_k"]): r for r in bert_reports if "top_k" in r}
        all_k = sorted(set(dt_by_k).union(bert_by_k))

        rows: list[dict[str, Any]] = []
        for k in all_k:
            dt_row = dt_by_k.get(k, {})
            bert_row = bert_by_k.get(k, {})
            dt_comp = float(dt_row.get("composite_fidelity", np.nan))
            bert_comp = float(bert_row.get("composite_fidelity", np.nan))
            delta_comp = float(bert_comp - dt_comp) if np.isfinite(dt_comp) and np.isfinite(bert_comp) else float("nan")

            if np.isnan(delta_comp):
                verdict = "Insufficient data"
            elif dt_comp > bert_comp:
                verdict = "DT explanations more faithful"
            elif bert_comp > dt_comp:
                verdict = "BERT explanations more faithful"
            else:
                verdict = "Comparable fidelity"

            rows.append(
                {
                    "top_k": int(k),
                    "dt_sufficiency": float(dt_row.get("sufficiency", np.nan)),
                    "dt_necessity": float(dt_row.get("necessity", np.nan)),
                    "dt_random_sufficiency": float(dt_row.get("random_sufficiency", np.nan)),
                    "dt_random_necessity": float(dt_row.get("random_necessity", np.nan)),
                    "dt_composite_fidelity": dt_comp,
                    "bert_sufficiency": float(bert_row.get("sufficiency", np.nan)),
                    "bert_necessity": float(bert_row.get("necessity", np.nan)),
                    "bert_random_sufficiency": float(bert_row.get("random_sufficiency", np.nan)),
                    "bert_random_necessity": float(bert_row.get("random_necessity", np.nan)),
                    "bert_composite_fidelity": bert_comp,
                    "delta_composite": delta_comp,
                    "interpretation": verdict,
                }
            )

        table = pd.DataFrame(rows).sort_values("top_k").reset_index(drop=True)
        out_path = self.tables_dir / f"fidelity_comparison{self.suffix}.csv"
        table.to_csv(out_path, index=False)
        self.logger.info("Saved fidelity comparison to %s", out_path)

        self._cache["fidelity_table"] = table
        return table

    def generate_full_report(self) -> Path:
        """Generate markdown report from previously computed comparison artifacts."""
        required_keys = ["accuracy_table", "stability", "overlap", "depth_tradeoff"]
        missing = [key for key in required_keys if key not in self._cache]
        if missing:
            raise ValueError(
                "Missing cached results for report generation: "
                f"{missing}. Run comparison methods first."
            )

        accuracy_table: pd.DataFrame = self._cache["accuracy_table"]
        stability = self._cache["stability"]
        overlap = self._cache["overlap"]
        depth_tradeoff: pd.DataFrame = self._cache["depth_tradeoff"]
        accuracy_gap = float(self._cache.get("accuracy_gap", np.nan))

        if np.isnan(accuracy_gap):
            h1_verdict = "Partial"
        elif accuracy_gap > 0:
            h1_verdict = "Supported"
        elif abs(accuracy_gap) < 1e-4:
            h1_verdict = "Partial"
        else:
            h1_verdict = "Rejected"

        p_val_h2 = float(stability.get("p_value", np.nan))
        if np.isnan(p_val_h2):
            h2_verdict = "Partial"
        elif p_val_h2 < 0.05:
            h2_verdict = "Supported"
        else:
            h2_verdict = "Rejected"

        overlap_jaccard = float(overlap.get("jaccard_score", 0.0))
        if overlap_jaccard >= 0.30:
            h3_verdict = "Supported"
        elif overlap_jaccard >= 0.10:
            h3_verdict = "Partial"
        else:
            h3_verdict = "Rejected"

        lines: list[str] = []
        lines.append(f"# Results Summary ({self.run_tag.upper()})")
        lines.append("")
        lines.append("## Accuracy Comparison (H1)")
        lines.append(self._df_to_markdown_table(accuracy_table))
        lines.append("")
        lines.append(f"Accuracy gap (BERT - Decision Tree): **{accuracy_gap:.6f}**")
        lines.append("")
        lines.append("## Stability Comparison (H2)")
        lines.append(f"- mean_r_dt: {stability['mean_r_dt']:.6f}")
        lines.append(f"- mean_r_bert: {stability['mean_r_bert']:.6f}")
        lines.append(f"- t_stat: {stability['t_stat']:.6f}")
        lines.append(f"- p_value: {stability['p_value']:.6f}")
        lines.append(f"- Interpretation: {stability['interpretation']}")
        lines.append("")
        lines.append("## Feature Overlap")
        lines.append(f"- Jaccard similarity: {overlap['jaccard_score']:.6f}")
        lines.append(f"- Spearman rho: {overlap['spearman_rho']:.6f}")
        lines.append(f"- p_value: {overlap['p_value']:.6f}")
        lines.append(
            "- Overlapping features: "
            + (", ".join(overlap["overlapping_features"]) if overlap["overlapping_features"] else "None")
        )
        lines.append("")
        lines.append("## Decision Tree Depth Tradeoff")
        lines.append(self._df_to_markdown_table(depth_tradeoff))
        lines.append("")

        if "fidelity_table" in self._cache:
            fidelity_table: pd.DataFrame = self._cache["fidelity_table"]
            lines.append("## Fidelity Comparison")
            lines.append(self._df_to_markdown_table(fidelity_table))
            lines.append("")

            finite_deltas = pd.to_numeric(fidelity_table.get("delta_composite", pd.Series(dtype=float)), errors="coerce")
            finite_deltas = finite_deltas[np.isfinite(finite_deltas)]
            if len(finite_deltas) == 0:
                fidelity_verdict = "Fidelity verdict unavailable due to missing values."
            elif float(np.mean(finite_deltas)) < 0:
                fidelity_verdict = "Decision Tree SHAP appears more faithful than post-hoc BERT SHAP overall."
            elif float(np.mean(finite_deltas)) > 0:
                fidelity_verdict = "Post-hoc BERT SHAP appears at least as faithful as Decision Tree SHAP overall."
            else:
                fidelity_verdict = "Decision Tree and BERT SHAP have comparable fidelity overall."

            lines.append(f"Fidelity verdict: **{fidelity_verdict}**")
            lines.append("")

        lines.append("## Hypothesis Verdicts")
        lines.append(f"- H1 (Performance gap): **{h1_verdict}**")
        lines.append(f"- H2 (Stability difference): **{h2_verdict}**")
        lines.append(f"- H3 (Feature overlap): **{h3_verdict}**")
        lines.append("")
        lines.append("## Key Findings")
        lines.append("- BERT and Decision Tree metrics are reported side-by-side for transparent comparison.")
        lines.append("- Stability was tested using paired correlations on reproducible near-duplicate pairs.")
        lines.append("- Interpretability tradeoff shows how depth changes model complexity vs readability.")

        self.summary_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info("Saved full report to %s", self.summary_path)
        return self.summary_path