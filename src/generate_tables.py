"""Generate paper-ready LaTeX and HTML tables from experiment CSV artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PaperTableGenerator:
    """Builds all requested publication tables from results/tables artifacts."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.results_dir = self.project_root / "results"
        self.tables_dir = self.results_dir / "tables"
        self.experiments_dir = self.project_root / "experiments"
        self.logger = logging.getLogger(self.__class__.__name__)

        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def _read_csv(self, name: str) -> pd.DataFrame:
        path = self.tables_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating, int, np.integer)):
            if np.isnan(float(value)):
                return "-"
            return f"{float(value):.3f}"
        return str(value)

    def _to_display_tables(
        self,
        df: pd.DataFrame,
        id_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return LaTeX and HTML display frames with row-wise best numeric value bolded."""
        if df.empty:
            empty = pd.DataFrame({"Note": ["Data unavailable"]})
            return empty, empty

        latex_df = df.copy().astype(object)
        html_df = df.copy().astype(object)

        numeric_cols = [
            col for col in df.columns if col not in id_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        for idx in range(len(df)):
            row = df.iloc[idx]
            candidates = [float(row[col]) for col in numeric_cols if pd.notna(row[col])]
            best_val = max(candidates) if candidates else None

            for col in df.columns:
                val = row[col]
                formatted = self._fmt(val)
                if col in numeric_cols and best_val is not None and pd.notna(val) and float(val) == float(best_val):
                    latex_df.at[idx, col] = f"\\textbf{{{formatted}}}"
                    html_df.at[idx, col] = f"<b>{formatted}</b>"
                else:
                    latex_df.at[idx, col] = formatted
                    html_df.at[idx, col] = formatted

        return latex_df, html_df

    def _save_table(
        self,
        table_idx: int,
        df: pd.DataFrame,
        caption: str,
        id_cols: list[str],
    ) -> tuple[Path, Path]:
        latex_df, html_df = self._to_display_tables(df, id_cols=id_cols)

        tex_path = self.tables_dir / f"table_{table_idx}.tex"
        html_path = self.tables_dir / f"table_{table_idx}.html"

        latex_text = latex_df.to_latex(index=False, escape=False, caption=caption, label=f"tab:table_{table_idx}")
        tex_path.write_text(latex_text, encoding="utf-8")

        html_body = html_df.to_html(index=False, escape=False)
        html_text = (
            "<html><head><meta charset='utf-8'><style>"
            "body{font-family:Arial,sans-serif;margin:20px;}"
            "table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}"
            "th{background:#f2f2f2;}"
            "</style></head><body>"
            f"<h3>{caption}</h3>{html_body}</body></html>"
        )
        html_path.write_text(html_text, encoding="utf-8")

        return tex_path, html_path

    @staticmethod
    def _extract_metric(df: pd.DataFrame, approach_needle: str, metric: str) -> float:
        if df.empty or "approach" not in df.columns or metric not in df.columns:
            return float("nan")
        row = df[df["approach"].str.contains(approach_needle, case=False, na=False)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][metric])

    def _latest_run_summary(self) -> dict[str, Any]:
        run_dirs = sorted(
            [p for p in self.experiments_dir.glob("run_*_") if p.is_dir()],
            key=lambda p: p.name,
        )
        if not run_dirs:
            run_dirs = sorted([p for p in self.experiments_dir.glob("run_*") if p.is_dir()], key=lambda p: p.name)
        if not run_dirs:
            return {}

        summary_path = run_dirs[-1] / "run_summary.json"
        if not summary_path.exists():
            return {}
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _build_table_1(self) -> pd.DataFrame:
        """Accuracy comparison across approaches and splits."""
        orig = self._read_csv("accuracy_comparison.csv")
        deb = self._read_csv("accuracy_comparison_debiased.csv")

        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        rows: list[dict[str, Any]] = []
        for metric in metrics:
            rows.append(
                {
                    "metric": metric.upper(),
                    "dt_original": self._extract_metric(orig, "decision tree", metric),
                    "bert_original": self._extract_metric(orig, "bert", metric),
                    "dt_debiased": self._extract_metric(deb, "decision tree", metric),
                    "bert_debiased": self._extract_metric(deb, "bert", metric),
                }
            )
        return pd.DataFrame(rows)

    def _build_table_2(self) -> pd.DataFrame:
        """Decision Tree depth tradeoff with interpretability score."""
        orig = self._read_csv("dt_depth_tradeoff.csv")
        deb = self._read_csv("dt_depth_tradeoff_debiased.csv")

        rows: list[dict[str, Any]] = []
        for split, df in [("original", orig), ("debiased", deb)]:
            if df.empty:
                continue
            for _, row in df.sort_values("depth").iterrows():
                rows.append(
                    {
                        "split": split,
                        "depth": int(row.get("depth", np.nan)),
                        "f1": float(row.get("cv_f1_mean", np.nan)),
                        "interpretability_score": float(row.get("interpretability_score", np.nan)),
                    }
                )
        return pd.DataFrame(rows)

    def _build_table_3(self) -> pd.DataFrame:
        """Explanation stability summary for both approaches and splits."""
        orig = self._read_csv("stability_comparison.csv")
        deb = self._read_csv("stability_comparison_debiased.csv")

        rows: list[dict[str, Any]] = []

        def _append_from_df(split: str, df: pd.DataFrame) -> None:
            if df.empty:
                return
            first = df.iloc[0]
            rows.append(
                {
                    "split": split,
                    "approach": "Decision Tree",
                    "mean_r": float(first.get("dt_mean_r", np.nan)),
                    "std_r": float(first.get("dt_std_r", np.nan)),
                    "p_value": float(first.get("p_value", np.nan)),
                }
            )
            rows.append(
                {
                    "split": split,
                    "approach": "BERT",
                    "mean_r": float(first.get("bert_mean_r", np.nan)),
                    "std_r": float(first.get("bert_std_r", np.nan)),
                    "p_value": float(first.get("p_value", np.nan)),
                }
            )

        _append_from_df("original", orig)
        _append_from_df("debiased", deb)

        if rows:
            return pd.DataFrame(rows)

        summary = self._latest_run_summary()
        per_run = summary.get("per_run", {}) if isinstance(summary, dict) else {}
        for split in ["original", "debiased"]:
            item = per_run.get(split, {}) if isinstance(per_run, dict) else {}
            stability = item.get("stability", {}) if isinstance(item, dict) else {}
            if not stability:
                continue
            rows.append(
                {
                    "split": split,
                    "approach": "Decision Tree",
                    "mean_r": float(stability.get("mean_r_dt", np.nan)),
                    "std_r": float("nan"),
                    "p_value": float(stability.get("p_value", np.nan)),
                }
            )
            rows.append(
                {
                    "split": split,
                    "approach": "BERT",
                    "mean_r": float(stability.get("mean_r_bert", np.nan)),
                    "std_r": float("nan"),
                    "p_value": float(stability.get("p_value", np.nan)),
                }
            )

        return pd.DataFrame(rows)

    def _top_n_with_defaults(self, df: pd.DataFrame, name_col: str, value_col: str, top_n: int) -> pd.DataFrame:
        if df.empty or name_col not in df.columns or value_col not in df.columns:
            return pd.DataFrame({name_col: ["-"] * top_n, value_col: [float("nan")] * top_n})
        out = df.sort_values(value_col, ascending=False).head(top_n).copy()
        if len(out) < top_n:
            missing = top_n - len(out)
            filler = pd.DataFrame({name_col: ["-"] * missing, value_col: [float("nan")] * missing})
            out = pd.concat([out, filler], ignore_index=True)
        return out.reset_index(drop=True)

    def _build_table_4(self) -> pd.DataFrame:
        """Top-15 SHAP features/tokens by approach and split."""
        dt_orig = self._read_csv("dt_global_shap_cache.csv")
        dt_deb = self._read_csv("dt_global_shap_cache_debiased.csv")
        bert_orig = self._read_csv("bert_global_shap_cache.csv")
        bert_deb = self._read_csv("bert_global_shap_cache_debiased.csv")

        dt_orig_top = self._top_n_with_defaults(dt_orig, "feature", "mean_abs_shap", 15)
        dt_deb_top = self._top_n_with_defaults(dt_deb, "feature", "mean_abs_shap", 15)
        bert_orig_top = self._top_n_with_defaults(bert_orig, "token", "mean_abs_shap", 15)
        bert_deb_top = self._top_n_with_defaults(bert_deb, "token", "mean_abs_shap", 15)

        rows: list[dict[str, Any]] = []
        for i in range(15):
            rows.append(
                {
                    "rank": i + 1,
                    "dt_feature_original": dt_orig_top.iloc[i]["feature"],
                    "dt_shap_original": float(dt_orig_top.iloc[i]["mean_abs_shap"]),
                    "bert_token_original": bert_orig_top.iloc[i]["token"],
                    "bert_shap_original": float(bert_orig_top.iloc[i]["mean_abs_shap"]),
                    "dt_feature_debiased": dt_deb_top.iloc[i]["feature"],
                    "dt_shap_debiased": float(dt_deb_top.iloc[i]["mean_abs_shap"]),
                    "bert_token_debiased": bert_deb_top.iloc[i]["token"],
                    "bert_shap_debiased": float(bert_deb_top.iloc[i]["mean_abs_shap"]),
                }
            )

        return pd.DataFrame(rows)

    def _build_table_5(self) -> pd.DataFrame:
        """Feature overlap summary (Jaccard and Spearman)."""
        rows: list[dict[str, Any]] = []

        for split, csv_name in [("original", "overlap_comparison.csv"), ("debiased", "overlap_comparison_debiased.csv")]:
            df = self._read_csv(csv_name)
            if df.empty:
                continue
            first = df.iloc[0]
            rows.append(
                {
                    "split": split,
                    "jaccard": float(first.get("jaccard_score", np.nan)),
                    "spearman_rho": float(first.get("spearman_rho", np.nan)),
                    "p_value": float(first.get("p_value", np.nan)),
                }
            )

        if rows:
            return pd.DataFrame(rows)

        summary = self._latest_run_summary()
        per_run = summary.get("per_run", {}) if isinstance(summary, dict) else {}
        for split in ["original", "debiased"]:
            item = per_run.get(split, {}) if isinstance(per_run, dict) else {}
            overlap = item.get("overlap", {}) if isinstance(item, dict) else {}
            if overlap:
                rows.append(
                    {
                        "split": split,
                        "jaccard": float(overlap.get("jaccard_score", np.nan)),
                        "spearman_rho": float(overlap.get("spearman_rho", np.nan)),
                        "p_value": float(overlap.get("p_value", np.nan)),
                    }
                )
        return pd.DataFrame(rows)

    def _build_table_6(self) -> pd.DataFrame:
        """Top-10 leakage patterns by bias severity and frequency."""
        audit = self._read_csv("leakage_audit.csv")
        if audit.empty:
            return pd.DataFrame()
        cols = [
            "pattern_name",
            "match_count",
            "fake_pct",
            "real_pct",
            "bias_gap_abs",
        ]
        valid_cols = [c for c in cols if c in audit.columns]
        return audit.sort_values(["bias_gap_abs", "match_count"], ascending=[False, False]).head(10)[valid_cols]

    def _build_table_7(self) -> pd.DataFrame:
        """Fidelity comparison table for both splits and top_k values."""
        orig = self._read_csv("fidelity_comparison.csv")
        deb = self._read_csv("fidelity_comparison_debiased.csv")

        keep_cols = [
            "top_k",
            "dt_sufficiency",
            "dt_necessity",
            "dt_random_sufficiency",
            "dt_random_necessity",
            "dt_composite_fidelity",
            "bert_sufficiency",
            "bert_necessity",
            "bert_random_sufficiency",
            "bert_random_necessity",
            "bert_composite_fidelity",
            "delta_composite",
        ]

        rows: list[pd.DataFrame] = []
        for split, df in [("original", orig), ("debiased", deb)]:
            if df.empty:
                continue
            valid = [c for c in keep_cols if c in df.columns]
            out = df[valid].copy()
            out.insert(0, "split", split)
            rows.append(out)

        if not rows:
            return pd.DataFrame()
        merged = pd.concat(rows, ignore_index=True)
        return merged.sort_values(["split", "top_k"]).reset_index(drop=True)

    def generate_all(self) -> list[Path]:
        """Generate and save all seven requested tables in LaTeX and HTML."""
        outputs: list[Path] = []

        table_specs = [
            (1, self._build_table_1(), "Table 1: Accuracy comparison (DT vs BERT, original vs debiased splits)", ["metric"]),
            (2, self._build_table_2(), "Table 2: Decision Tree depth vs F1 vs interpretability", ["split", "depth"]),
            (3, self._build_table_3(), "Table 3: Explanation stability summary", ["split", "approach"]),
            (4, self._build_table_4(), "Table 4: Top-15 SHAP features/tokens by approach and split", ["rank", "dt_feature_original", "bert_token_original", "dt_feature_debiased", "bert_token_debiased"]),
            (5, self._build_table_5(), "Table 5: Feature overlap metrics", ["split"]),
            (6, self._build_table_6(), "Table 6: Leakage audit (top-10 most biased patterns)", ["pattern_name"]),
            (7, self._build_table_7(), "Table 7: Fidelity comparison across top_k for both approaches and splits", ["split", "top_k"]),
        ]

        for idx, df, caption, id_cols in table_specs:
            tex_path, html_path = self._save_table(idx, df, caption, id_cols)
            outputs.extend([tex_path, html_path])
            self.logger.info("Saved table %s to %s and %s", idx, tex_path, html_path)

        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready LaTeX and HTML tables")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Path to fake_news_xai project root",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    generator = PaperTableGenerator(project_root=Path(args.project_root))
    outputs = generator.generate_all()

    print("Generated tables:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
