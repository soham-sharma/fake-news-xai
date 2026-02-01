"""Decision Tree training pipeline for Approach A (white-box + SHAP-ready)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import FeatureEngineer


class DecisionTreeExplainer:
    """Train and evaluate Decision Tree models over configured depth values."""

    def __init__(self, config: str | Path | dict[str, Any], run_tag: str = "original") -> None:
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            with config_path.open("r", encoding="utf-8") as fp:
                self.config = yaml.safe_load(fp)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be a path or a dictionary")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.random_seed = int(self.config.get("random_seed", 42))
        self.run_tag = str(run_tag).strip().lower() or "original"
        self.depths = self.config.get("models", {}).get("decision_tree", {}).get(
            "depth_range", [4, 6, 8, 10, 12]
        )
        self.criteria = ["gini", "entropy"]

        self.tables_dir = PROJECT_ROOT / "results" / "tables"
        if self.run_tag == "original":
            self.models_dir = PROJECT_ROOT / "results" / "models" / "approach_a"
            self.depth_comparison_filename = "dt_depth_comparison.csv"
        else:
            self.models_dir = PROJECT_ROOT / "results" / "models" / "approach_a" / self.run_tag
            self.depth_comparison_filename = f"dt_depth_comparison_{self.run_tag}.csv"

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X_train,
        y_train,
        depth: int,
        criterion: str = "gini",
    ) -> dict[str, Any]:
        """Train one Decision Tree with cross-validation at a given depth."""
        self.logger.info("Training DecisionTree depth=%s criterion=%s", depth, criterion)
        model = DecisionTreeClassifier(
            max_depth=depth,
            criterion=criterion,
            class_weight="balanced",
            random_state=self.random_seed,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        model.fit(X_train, y_train)

        return {
            "model": model,
            "cv_scores": {
                "mean": float(np.mean(cv_scores)),
                "std": float(np.std(cv_scores)),
            },
            "best_params": {
                "max_depth": int(depth),
                "criterion": criterion,
                "class_weight": "balanced",
            },
        }

    def evaluate(self, model, X_test, y_test) -> dict[str, Any]:
        """Evaluate a trained model on test data."""
        y_pred = model.predict(X_test)
        y_prob: Optional[np.ndarray]
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else float("nan"),
        }

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": cm,
        }

    def grid_search(self, X_train, y_train) -> pd.DataFrame:
        """Search depth and criterion, then keep one best criterion per depth."""
        self.logger.info("Running grid search across depths=%s and criteria=%s", self.depths, self.criteria)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)

        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "roc_auc": "roc_auc",
        }

        all_rows: list[dict[str, Any]] = []
        for depth in self.depths:
            for criterion in self.criteria:
                model = DecisionTreeClassifier(
                    max_depth=depth,
                    criterion=criterion,
                    class_weight="balanced",
                    random_state=self.random_seed,
                )
                cv_result = cross_validate(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                )
                row = {
                    "depth": int(depth),
                    "criterion": criterion,
                    "cv_accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
                    "cv_accuracy_std": float(np.std(cv_result["test_accuracy"])),
                    "cv_precision_mean": float(np.mean(cv_result["test_precision"])),
                    "cv_recall_mean": float(np.mean(cv_result["test_recall"])),
                    "cv_f1_mean": float(np.mean(cv_result["test_f1"])),
                    "cv_f1_std": float(np.std(cv_result["test_f1"])),
                    "cv_auc_roc_mean": float(np.mean(cv_result["test_roc_auc"])),
                }
                all_rows.append(row)

        all_results = pd.DataFrame(all_rows)
        best_per_depth = (
            all_results.sort_values(["depth", "cv_f1_mean"], ascending=[True, False])
            .groupby("depth", as_index=False)
            .head(1)
            .sort_values("depth")
            .reset_index(drop=True)
        )

        output_path = self.tables_dir / self.depth_comparison_filename
        best_per_depth.to_csv(output_path, index=False)
        self.logger.info("Saved depth comparison table to %s", output_path)
        return best_per_depth

    def save_model(self, model, depth: int, criterion: str = "gini") -> Path:
        """Persist a trained model to disk with depth in filename."""
        model_path = self.models_dir / f"decision_tree_depth_{depth}_{criterion}.joblib"
        joblib.dump(model, model_path)
        self.logger.info("Saved model to %s", model_path)
        return model_path

    def run(self, X_train, y_train) -> pd.DataFrame:
        """Train models across all configured depths and return comparison table."""
        comparison_df = self.grid_search(X_train, y_train)
        for _, row in comparison_df.iterrows():
            result = self.train(
                X_train,
                y_train,
                depth=int(row["depth"]),
                criterion=str(row["criterion"]),
            )
            self.save_model(
                result["model"],
                depth=int(row["depth"]),
                criterion=str(row["criterion"]),
            )
        return comparison_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train Approach A Decision Tree pipeline")
    parser.add_argument(
        "--debiased",
        action="store_true",
        help="Use debiased_train.csv / debiased_test.csv and save artifacts in debiased-specific paths.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_path", "data/processed")
    if args.debiased:
        train_path = processed_dir / "debiased_train.csv"
        test_path = processed_dir / "debiased_test.csv"
        run_tag = "debiased"
    else:
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        run_tag = "original"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train/test CSV files in {processed_dir}. Run data loading first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Train/test files must include a 'label' column.")

    tfidf_candidates = config.get("models", {}).get("tfidf", {}).get("max_features", [5000])
    tfidf_max_features = int(max(tfidf_candidates))

    feature_engineer = FeatureEngineer(max_tfidf_features=tfidf_max_features)
    X_train, feature_names = feature_engineer.build_feature_matrix(train_df, fit_tfidf=True)
    X_test, _ = feature_engineer.build_feature_matrix(
        test_df,
        tfidf_vectorizer=feature_engineer.vectorizer,
        fit_tfidf=False,
    )

    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    trainer = DecisionTreeExplainer(config, run_tag=run_tag)
    comparison = trainer.run(X_train, y_train)

    print(f"\nRun mode: {run_tag}")
    print(f"Depth comparison file: {trainer.tables_dir / trainer.depth_comparison_filename}")
    print(f"Model directory: {trainer.models_dir}")

    best_row = comparison.sort_values("cv_f1_mean", ascending=False).iloc[0]
    best_result = trainer.train(
        X_train,
        y_train,
        depth=int(best_row["depth"]),
        criterion=str(best_row["criterion"]),
    )
    best_eval = trainer.evaluate(best_result["model"], X_test, y_test)

    print("\nDecision Tree Depth Comparison (best criterion per depth):")
    print(comparison.to_string(index=False))

    print("\nBest Params:")
    print(best_result["best_params"])
    print("CV Scores:")
    print(best_result["cv_scores"])

    print("\nTest Metrics:")
    print(best_eval["metrics"])
    print("Confusion Matrix:")
    print(best_eval["confusion_matrix"])

    report_df = pd.DataFrame(best_eval["classification_report"]).transpose()
    print("\nClassification Report:")
    print(report_df.to_string())

    print("\nFeature count:", len(feature_names))
