"""Plotting utilities for experiment figures used in reports and slides."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from scipy.stats import ttest_ind, ttest_rel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ResultsVisualizer:
	"""Generates paper-style figures from experiment outputs."""

	def __init__(self, run_tag: str = "original") -> None:
		self.logger = logging.getLogger(self.__class__.__name__)
		self.run_tag = str(run_tag).strip().lower() or "original"
		self.suffix = "" if self.run_tag == "original" else f"_{self.run_tag}"

		self.results_dir = PROJECT_ROOT / "results"
		self.tables_dir = self.results_dir / "tables"
		self.figures_dir = self.results_dir / "figures"
		self.figures_dir.mkdir(parents=True, exist_ok=True)

		sns.set_theme(style="whitegrid", context="paper")
		plt.rcParams["figure.figsize"] = (10, 6)
		plt.rcParams["figure.dpi"] = 300

		self.dt_color = "#1f77b4"
		self.bert_color = "#ff7f0e"

	def _resolve_out_path(self, base_name: str, apply_suffix: bool = True) -> Path:
		stem = Path(base_name).stem
		ext = Path(base_name).suffix or ".png"
		filename = f"{stem}{self.suffix}{ext}" if apply_suffix and self.suffix else f"{stem}{ext}"
		return self.figures_dir / filename

	@staticmethod
	def _as_scores(values: Any) -> np.ndarray:
		if isinstance(values, dict) and "per_pair_scores" in values:
			vals = [float(x.get("pearson_r", np.nan)) for x in values.get("per_pair_scores", [])]
			arr = np.asarray(vals, dtype=float)
		elif isinstance(values, list):
			arr = np.asarray(values, dtype=float)
		else:
			arr = np.asarray([], dtype=float)
		return arr[np.isfinite(arr)]

	@staticmethod
	def _safe_numeric(series: pd.Series) -> pd.Series:
		return pd.to_numeric(series, errors="coerce")

	def plot_accuracy_tradeoff(self, depth_df: pd.DataFrame) -> Path:
		"""Depth vs F1 and interpretability with highlighted balanced zone."""
		df = depth_df.copy()
		if "depth" not in df.columns:
			raise ValueError("depth_df must include 'depth'.")

		if "cv_f1_mean" in df.columns:
			f1_col = "cv_f1_mean"
		elif "f1" in df.columns:
			f1_col = "f1"
		else:
			raise ValueError("depth_df must include 'cv_f1_mean' or 'f1'.")

		if "interpretability_score" not in df.columns:
			if "num_leaves" in df.columns:
				leaves = self._safe_numeric(df["num_leaves"]).fillna(1.0)
			else:
				leaves = (2 ** self._safe_numeric(df["depth"]).fillna(1.0)).astype(float)
			df["interpretability_score"] = 1.0 / np.log(leaves + 1.0)

		df = df.sort_values("depth")

		fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
		ax2 = ax1.twinx()

		ax1.axvspan(8, 10, color="#2a9d8f", alpha=0.15, label="Balanced zone (8-10)")
		ax1.plot(df["depth"], df[f1_col], marker="o", color=self.dt_color, linewidth=2, label="F1-score")
		ax2.plot(
			df["depth"],
			df["interpretability_score"],
			marker="s",
			linestyle="--",
			color=self.bert_color,
			linewidth=2,
			label="Interpretability score",
		)

		ax1.set_xlabel("Tree Depth")
		ax1.set_ylabel("F1-score", color=self.dt_color)
		ax2.set_ylabel("Interpretability Score", color=self.bert_color)
		ax1.set_xticks([4, 6, 8, 10, 12])
		ax1.set_title("Decision Tree Accuracy vs Interpretability Tradeoff")

		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1 + h2, l1 + l2, loc="best")

		out_path = self._resolve_out_path("dt_depth_tradeoff.png")
		fig.tight_layout()
		fig.savefig(out_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return out_path

	def plot_accuracy_comparison(self, comparison_df: pd.DataFrame) -> Path:
		"""Grouped bar chart comparing DT and BERT across metrics."""
		df = comparison_df.copy()
		metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc"]
		for col in metric_cols:
			if col not in df.columns:
				raise ValueError(f"comparison_df missing '{col}'.")

		approach_col = "approach" if "approach" in df.columns else None
		if approach_col is None:
			raise ValueError("comparison_df must include an 'approach' column.")

		dt_row = df[df[approach_col].str.contains("decision tree", case=False, na=False)].head(1)
		bert_row = df[df[approach_col].str.contains("bert", case=False, na=False)].head(1)
		if dt_row.empty or bert_row.empty:
			raise ValueError("comparison_df must include both Decision Tree and BERT rows.")

		dt_vals = dt_row.iloc[0][metric_cols].astype(float).to_numpy()
		bert_vals = bert_row.iloc[0][metric_cols].astype(float).to_numpy()

		x = np.arange(len(metric_cols), dtype=float)
		width = 0.36

		fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
		ax.bar(x - width / 2, dt_vals, width, label="Decision Tree", color=self.dt_color)
		ax.bar(x + width / 2, bert_vals, width, label="BERT", color=self.bert_color)

		ax.set_xticks(x)
		ax.set_xticklabels([m.upper() for m in metric_cols])
		ax.set_ylim(0.0, 1.05)
		ax.set_ylabel("Score")
		ax.set_title("Model Performance Comparison")
		ax.legend(loc="best")

		out_path = self._resolve_out_path("accuracy_comparison.png")
		fig.tight_layout()
		fig.savefig(out_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return out_path

	def plot_stability_comparison(
		self,
		dt_stability_scores: list[float] | dict[str, Any],
		bert_stability_scores: list[float] | dict[str, Any],
	) -> Path:
		"""Side-by-side boxplots with mean and p-value annotation."""
		dt_scores = self._as_scores(dt_stability_scores)
		bert_scores = self._as_scores(bert_stability_scores)
		if len(dt_scores) == 0 or len(bert_scores) == 0:
			raise ValueError("Stability inputs must contain at least one finite score per model.")

		n = min(len(dt_scores), len(bert_scores))
		if n >= 2:
			try:
				stat, p_value = ttest_rel(dt_scores[:n], bert_scores[:n])
			except Exception:
				stat, p_value = ttest_ind(dt_scores, bert_scores, equal_var=False)
		else:
			stat, p_value = np.nan, np.nan

		fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
		bp = ax.boxplot(
			[dt_scores, bert_scores],
			labels=["Decision Tree", "BERT"],
			patch_artist=True,
			widths=0.5,
		)
		bp["boxes"][0].set_facecolor(self.dt_color)
		bp["boxes"][0].set_alpha(0.6)
		bp["boxes"][1].set_facecolor(self.bert_color)
		bp["boxes"][1].set_alpha(0.6)

		dt_mean = float(np.mean(dt_scores))
		bert_mean = float(np.mean(bert_scores))
		annotation = (
			f"Mean DT={dt_mean:.3f}\n"
			f"Mean BERT={bert_mean:.3f}\n"
			f"p-value={float(p_value):.3g}" if np.isfinite(p_value) else "p-value=nan"
		)
		ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", fontsize=10)
		ax.set_ylabel("Pearson r")
		ax.set_title("Explanation Stability Comparison")

		out_path = self._resolve_out_path("stability_comparison.png")
		fig.tight_layout()
		fig.savefig(out_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return out_path

	def plot_feature_overlap(self, dt_features: list[str], bert_features: list[str], top_n: int = 20) -> Path:
		"""Venn-style overlap chart with top overlapping feature list."""
		dt_set = set(str(x).strip().lower() for x in dt_features[:top_n] if str(x).strip())
		bert_set = set(str(x).strip().lower() for x in bert_features[:top_n] if str(x).strip())

		overlap = sorted(dt_set.intersection(bert_set))
		only_dt = sorted(dt_set - bert_set)
		only_bert = sorted(bert_set - dt_set)

		fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
		ax.set_aspect("equal")
		ax.set_xlim(0, 10)
		ax.set_ylim(0, 6)
		ax.axis("off")

		left = Circle((4, 3), radius=2.2, color=self.dt_color, alpha=0.35)
		right = Circle((6, 3), radius=2.2, color=self.bert_color, alpha=0.35)
		ax.add_patch(left)
		ax.add_patch(right)

		ax.text(2.7, 5.3, "Decision Tree", ha="center", fontsize=11, fontweight="bold")
		ax.text(7.3, 5.3, "BERT", ha="center", fontsize=11, fontweight="bold")
		ax.text(2.9, 3.0, f"{len(only_dt)}\nunique", ha="center", va="center", fontsize=12)
		ax.text(5.0, 3.0, f"{len(overlap)}\nshared", ha="center", va="center", fontsize=12, fontweight="bold")
		ax.text(7.1, 3.0, f"{len(only_bert)}\nunique", ha="center", va="center", fontsize=12)

		top_overlap = overlap[:10]
		overlap_text = ", ".join(top_overlap) if top_overlap else "None"
		ax.text(
			0.2,
			0.25,
			f"Top overlapping features (up to 10): {overlap_text}",
			fontsize=9,
			ha="left",
			va="bottom",
			transform=ax.transAxes,
		)
		ax.set_title("Top-Feature Overlap (Venn Style)")

		out_path = self._resolve_out_path("feature_overlap.png")
		fig.tight_layout()
		fig.savefig(out_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return out_path

	def plot_shap_comparison(self, dt_shap_df: pd.DataFrame, bert_shap_df: pd.DataFrame, top_n: int = 15) -> Path:
		"""Side-by-side horizontal bar charts of top SHAP features."""
		for col in ["feature", "mean_abs_shap"]:
			if col not in dt_shap_df.columns:
				raise ValueError(f"dt_shap_df missing '{col}'.")
		for col in ["token", "mean_abs_shap"]:
			if col not in bert_shap_df.columns:
				raise ValueError(f"bert_shap_df missing '{col}'.")

		dt_top = dt_shap_df.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]
		bert_top = bert_shap_df.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]

		fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharex=False)

		axes[0].barh(dt_top["feature"], dt_top["mean_abs_shap"], color=self.dt_color)
		axes[0].set_title("Decision Tree Top Features")
		axes[0].set_xlabel("Mean |SHAP|")
		axes[0].set_ylabel("Feature")

		axes[1].barh(bert_top["token"], bert_top["mean_abs_shap"], color=self.bert_color)
		axes[1].set_title("BERT Top Tokens")
		axes[1].set_xlabel("Mean |SHAP|")
		axes[1].set_ylabel("Token")

		fig.suptitle("SHAP Importance Comparison", y=1.02)
		out_path = self._resolve_out_path("shap_comparison.png")
		fig.tight_layout()
		fig.savefig(out_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return out_path

	def plot_cheat_code_bar_chart(
		self,
		accuracy_original_df: pd.DataFrame,
		accuracy_debiased_df: pd.DataFrame,
		metric: str = "f1",
	) -> Path:
		"""Grouped bars showing drop from original to debiased for DT vs BERT."""
		metric = metric.lower()
		if metric not in {"accuracy", "precision", "recall", "f1", "auc_roc"}:
			raise ValueError("metric must be one of accuracy/precision/recall/f1/auc_roc")

		def _extract(df: pd.DataFrame, needle: str) -> float:
			row = df[df["approach"].str.contains(needle, case=False, na=False)]
			if row.empty:
				return float("nan")
			return float(row.iloc[0][metric])

		dt_orig = _extract(accuracy_original_df, "decision tree")
		bert_orig = _extract(accuracy_original_df, "bert")
		dt_deb = _extract(accuracy_debiased_df, "decision tree")
		bert_deb = _extract(accuracy_debiased_df, "bert")

		x = np.arange(2)
		width = 0.36
		fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
		ax.bar(x - width / 2, [dt_orig, dt_deb], width=width, color=self.dt_color, label="Decision Tree")
		ax.bar(x + width / 2, [bert_orig, bert_deb], width=width, color=self.bert_color, label="BERT")
		ax.set_xticks(x)
		ax.set_xticklabels(["Original", "Debiased"])
		ax.set_ylim(0.0, 1.05)
		ax.set_ylabel(metric.upper())
		ax.set_title("Cheat Code Bar Chart: Performance Drop After Debiasing")
		ax.legend(loc="best")

		out_path = self._resolve_out_path("cheat_code_bar_chart.png", apply_suffix=False)
		fig.tight_layout()
		fig.savefig(out_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return out_path

	def plot_fidelity_decay_curve(
		self,
		fidelity_original_df: pd.DataFrame,
		fidelity_debiased_df: pd.DataFrame,
	) -> Path:
		"""Line chart of composite fidelity vs top_k for both runs and models."""
		for col in ["top_k", "dt_composite_fidelity", "bert_composite_fidelity"]:
			if col not in fidelity_original_df.columns or col not in fidelity_debiased_df.columns:
				raise ValueError(f"Missing '{col}' in fidelity comparison DataFrame.")

		orig = fidelity_original_df.sort_values("top_k")
		deb = fidelity_debiased_df.sort_values("top_k")

		fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
		ax.plot(orig["top_k"], orig["dt_composite_fidelity"], color=self.dt_color, linewidth=2, marker="o", label="DT Original")
		ax.plot(deb["top_k"], deb["dt_composite_fidelity"], color=self.dt_color, linestyle="--", linewidth=2, marker="o", label="DT Debiased")
		ax.plot(orig["top_k"], orig["bert_composite_fidelity"], color=self.bert_color, linewidth=2, marker="s", label="BERT Original")
		ax.plot(deb["top_k"], deb["bert_composite_fidelity"], color=self.bert_color, linestyle="--", linewidth=2, marker="s", label="BERT Debiased")

		ax.set_xlabel("top_k")
		ax.set_ylabel("Composite Fidelity")
		ax.set_title("Fidelity Decay Curve")
		ax.legend(loc="best")

		out_path = self._resolve_out_path("fidelity_decay_curve.png", apply_suffix=False)
		fig.tight_layout()
		fig.savefig(out_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return out_path

	def plot_reuters_vanishing_act(
		self,
		dt_original_shap_df: pd.DataFrame,
		dt_debiased_shap_df: pd.DataFrame,
		top_n: int = 15,
	) -> Path:
		"""Side-by-side DT SHAP bars before and after debiasing."""
		for col in ["feature", "mean_abs_shap"]:
			if col not in dt_original_shap_df.columns or col not in dt_debiased_shap_df.columns:
				raise ValueError(f"Missing '{col}' in DT SHAP DataFrame.")

		left = dt_original_shap_df.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]
		right = dt_debiased_shap_df.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]

		fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
		axes[0].barh(left["feature"], left["mean_abs_shap"], color=self.dt_color)
		axes[0].set_title("DT Original")
		axes[0].set_xlabel("Mean |SHAP|")

		axes[1].barh(right["feature"], right["mean_abs_shap"], color="#2a9d8f")
		axes[1].set_title("DT Debiased")
		axes[1].set_xlabel("Mean |SHAP|")

		fig.suptitle("The Reuters Vanishing Act", y=1.02)
		out_path = self._resolve_out_path("reuters_vanishing_act.png", apply_suffix=False)
		fig.tight_layout()
		fig.savefig(out_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return out_path

	def generate_all(self, results_dict: dict[str, Any]) -> list[Path]:
		"""Generate all available figures from provided result objects."""
		saved: list[Path] = []

		depth_df = results_dict.get("depth_df")
		if isinstance(depth_df, pd.DataFrame) and not depth_df.empty:
			saved.append(self.plot_accuracy_tradeoff(depth_df))

		accuracy_df = results_dict.get("accuracy_df")
		if isinstance(accuracy_df, pd.DataFrame) and not accuracy_df.empty:
			saved.append(self.plot_accuracy_comparison(accuracy_df))

		dt_stability_scores = results_dict.get("dt_stability_scores")
		bert_stability_scores = results_dict.get("bert_stability_scores")
		if dt_stability_scores is not None and bert_stability_scores is not None:
			saved.append(self.plot_stability_comparison(dt_stability_scores, bert_stability_scores))

		dt_features = results_dict.get("dt_features")
		bert_features = results_dict.get("bert_features")
		if isinstance(dt_features, list) and isinstance(bert_features, list):
			saved.append(self.plot_feature_overlap(dt_features, bert_features, top_n=20))

		dt_shap_df = results_dict.get("dt_shap_df")
		bert_shap_df = results_dict.get("bert_shap_df")
		if isinstance(dt_shap_df, pd.DataFrame) and isinstance(bert_shap_df, pd.DataFrame):
			if not dt_shap_df.empty and not bert_shap_df.empty:
				saved.append(self.plot_shap_comparison(dt_shap_df, bert_shap_df, top_n=15))

		accuracy_original_df = results_dict.get("accuracy_original_df")
		accuracy_debiased_df = results_dict.get("accuracy_debiased_df")
		if isinstance(accuracy_original_df, pd.DataFrame) and isinstance(accuracy_debiased_df, pd.DataFrame):
			if not accuracy_original_df.empty and not accuracy_debiased_df.empty:
				saved.append(self.plot_cheat_code_bar_chart(accuracy_original_df, accuracy_debiased_df, metric="f1"))

		fidelity_original_df = results_dict.get("fidelity_original_df")
		fidelity_debiased_df = results_dict.get("fidelity_debiased_df")
		if isinstance(fidelity_original_df, pd.DataFrame) and isinstance(fidelity_debiased_df, pd.DataFrame):
			if not fidelity_original_df.empty and not fidelity_debiased_df.empty:
				saved.append(self.plot_fidelity_decay_curve(fidelity_original_df, fidelity_debiased_df))

		dt_original_shap_df = results_dict.get("dt_original_shap_df")
		dt_debiased_shap_df = results_dict.get("dt_debiased_shap_df")
		if isinstance(dt_original_shap_df, pd.DataFrame) and isinstance(dt_debiased_shap_df, pd.DataFrame):
			if not dt_original_shap_df.empty and not dt_debiased_shap_df.empty:
				saved.append(self.plot_reuters_vanishing_act(dt_original_shap_df, dt_debiased_shap_df, top_n=15))

		for path in saved:
			print(path)
		return saved
