"""Single entry point to run all fake-news XAI experiments end-to-end.

Pipeline coverage:
1. Load processed data and config
2. Debiasing audit + debiased split generation
3. Approach A (DT) on original/debiased
4. Approach B (BERT) on original/debiased
5. Comparison metrics (accuracy, stability, fidelity, overlap)
6. DT depth-wise SHAP tradeoff comparison
7. Full markdown report generation
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import gc
import torch
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.model_selection import train_test_split

from src.approach_a.explain import TreeExplainer
from src.approach_a.train import DecisionTreeExplainer
from src.approach_b.explain import BERTShapExplainer
from src.approach_b.train import BERTClassifier
from src.evaluation.debias import LexicalDebiaser
from src.evaluation.metrics import ComparisonMetrics
from src.features import FeatureEngineer
from src.generate_tables import PaperTableGenerator
from src.visualization.plots import ResultsVisualizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentRunner:
    """Coordinates all experiment phases with resumable, cached artifacts."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = self._load_config(Path(args.config))
        self.seed = int(self.config.get("random_seed", 42))

        processed_rel = self.config.get("data", {}).get("processed_path", "data/processed")
        self.processed_dir = PROJECT_ROOT / processed_rel
        self.results_dir = PROJECT_ROOT / "results"
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir = self.results_dir / "figures"
        self.experiments_dir = PROJECT_ROOT / "experiments"

        self.run_dir = self._create_run_dir(self.experiments_dir)
        self.log_path = self.run_dir / "pipeline.log"
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.summary_path = self.run_dir / "run_summary.json"

        self._setup_logging(self.log_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.state: dict[str, Any] = {
            "run_dir": str(self.run_dir),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "completed": {},
        }
        self._save_checkpoint()

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)

    @staticmethod
    def _create_run_dir(experiments_dir: Path) -> Path:
        experiments_dir.mkdir(parents=True, exist_ok=True)

        pattern = re.compile(r"^run_(\d+)_")
        max_idx = 0
        for child in experiments_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

        run_idx = max_idx + 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = experiments_dir / f"run_{run_idx:03d}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @staticmethod
    def _setup_logging(log_path: Path) -> None:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        root_logger.handlers = []

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    @staticmethod
    def _suffix(run_tag: str) -> str:
        tag = str(run_tag).strip().lower() or "original"
        return "" if tag == "original" else f"_{tag}"

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _mark_stage(self, stage_name: str, payload: dict[str, Any]) -> None:
        self.state["completed"][stage_name] = payload
        self.state["last_updated"] = datetime.now().isoformat(timespec="seconds")
        self._save_checkpoint()

    @staticmethod
    def _model_file_exists(model_dir: Path) -> bool:
        if not model_dir.exists():
            return False
        has_model = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
        has_cfg = (model_dir / "config.json").exists()
        has_tok = (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists()
        return has_model and has_cfg and has_tok

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            return

        try:
            dst.symlink_to(src)
            return
        except Exception:
            pass

        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    def _snapshot_artifacts(self, run_tag: str, artifact_paths: list[Path]) -> None:
        target_root = self.run_dir / "artifacts" / run_tag

        unique_paths: list[Path] = []
        seen: set[str] = set()
        for path in artifact_paths:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(path)

        linked_count = 0
        for src in unique_paths:
            if not src.exists():
                continue

            try:
                rel = src.resolve().relative_to(PROJECT_ROOT.resolve())
                dst = target_root / rel
            except Exception:
                dst = target_root / src.name

            self._link_or_copy(src, dst)
            linked_count += 1

        self.logger.info("Captured %s artifacts for run_tag=%s into %s", linked_count, run_tag, target_root)

    def _resolve_run_tags(self) -> list[str]:
        if self.args.original and not self.args.debiased:
            return ["original"]
        if self.args.debiased and not self.args.original:
            return ["debiased"]
        return ["original", "debiased"]

    def _load_split(self, run_tag: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
        if run_tag == "debiased":
            train_path = self.processed_dir / "debiased_train.csv"
            test_path = self.processed_dir / "debiased_test.csv"
        else:
            train_path = self.processed_dir / "train.csv"
            test_path = self.processed_dir / "test.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing split files for run_tag='{run_tag}': {train_path}, {test_path}"
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df, train_path, test_path

    def run_debiasing(self) -> dict[str, Any]:
        self.logger.info("Debiasing stage started")
        train_path = self.processed_dir / "train.csv"
        test_path = self.processed_dir / "test.csv"
        out_train = self.processed_dir / "debiased_train.csv"
        out_test = self.processed_dir / "debiased_test.csv"
        audit_path = self.tables_dir / "leakage_audit.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Original processed splits missing: {train_path} and/or {test_path}"
            )

        should_run = (
            self.args.force_debias
            or not out_train.exists()
            or not out_test.exists()
            or not audit_path.exists()
        )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        text_col = "full_text" if "full_text" in train_df.columns else "text"

        debiaser = LexicalDebiaser(seed=self.seed)

        if should_run:
            self.logger.info("Running lexical debiasing (force_debias=%s)", self.args.force_debias)
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            audit_df = debiaser.audit(combined_df, text_col=text_col)

            train_debiased = debiaser.apply(train_df, text_col=text_col)
            test_debiased = debiaser.apply(test_df, text_col=text_col)
            split_paths = debiaser.save_debiased_splits(train_debiased, test_debiased, path=self.processed_dir)

            comparison = debiaser.compare_label_distributions(combined_df, pd.concat([train_debiased, test_debiased]))
            self.logger.info("Debias comparison: %s", comparison)
            self.logger.info("Debiased split paths: %s", split_paths)

            audit_count = int(len(audit_df))
        else:
            self.logger.info("Skipping debias computation; cached files found and force_debias=False")
            audit_count = int(pd.read_csv(audit_path).shape[0]) if audit_path.exists() else 0

        out = {
            "audit_path": str(audit_path),
            "debiased_train_path": str(out_train),
            "debiased_test_path": str(out_test),
            "audit_rows": audit_count,
        }
        self._mark_stage("debiasing", out)
        return out

    def _train_or_load_dt_models(
        self,
        trainer: DecisionTreeExplainer,
        X_train,
        y_train,
    ) -> pd.DataFrame:
        comparison_path = trainer.tables_dir / trainer.depth_comparison_filename

        need_train = self.args.force_train or not comparison_path.exists()
        comparison_df = pd.DataFrame()

        if not need_train:
            comparison_df = pd.read_csv(comparison_path)
            if comparison_df.empty:
                need_train = True
            else:
                for _, row in comparison_df.iterrows():
                    model_path = trainer.models_dir / (
                        f"decision_tree_depth_{int(row['depth'])}_{str(row['criterion'])}.joblib"
                    )
                    if not model_path.exists():
                        need_train = True
                        self.logger.info("Missing DT model artifact: %s", model_path)
                        break

        if need_train:
            self.logger.info("Training DT depth sweep (force_train=%s)", self.args.force_train)
            comparison_df = trainer.run(X_train, y_train)
        else:
            self.logger.info("Using existing DT depth sweep at %s", comparison_path)

        return comparison_df

    def _dt_depth_shap_comparison(
        self,
        run_tag: str,
        comparison_df: pd.DataFrame,
        feature_names: list[str],
        X_eval,
        models_dir: Path,
    ) -> tuple[pd.DataFrame, Path]:
        suffix = self._suffix(run_tag)
        out_path = self.tables_dir / f"dt_depth_shap_comparison{suffix}.csv"

        sample_size = int(
            self.config.get("explainability", {})
            .get("shap", {})
            .get("global_sample_size", 500)
        )
        if sparse.issparse(X_eval):
            X_sample = X_eval[: min(sample_size, X_eval.shape[0])]
        else:
            X_sample = np.asarray(X_eval)[: min(sample_size, X_eval.shape[0])]

        if comparison_df.empty:
            empty_df = pd.DataFrame(
                columns=[
                    "depth",
                    "criterion",
                    "cv_accuracy_mean",
                    "cv_f1_mean",
                    "top_features",
                    "jaccard_with_best_depth_top20",
                ]
            )
            empty_df.to_csv(out_path, index=False)
            return empty_df, out_path

        best_row = comparison_df.sort_values("cv_f1_mean", ascending=False).iloc[0]
        best_model_path = models_dir / (
            f"decision_tree_depth_{int(best_row['depth'])}_{str(best_row['criterion'])}.joblib"
        )
        best_model = joblib.load(best_model_path)
        best_explainer = TreeExplainer(best_model, feature_names, run_tag=run_tag)
        best_shap = best_explainer.get_shap_values(X_sample)
        best_top = best_explainer.get_top_features(best_shap, feature_names, top_n=20)
        best_set = set(str(x).lower() for x in best_top)

        rows: list[dict[str, Any]] = []
        for _, row in comparison_df.sort_values("depth").iterrows():
            depth = int(row["depth"])
            criterion = str(row["criterion"])
            model_path = models_dir / f"decision_tree_depth_{depth}_{criterion}.joblib"
            if not model_path.exists():
                continue

            model = joblib.load(model_path)
            explainer = TreeExplainer(model, feature_names, run_tag=run_tag)
            shap_values = explainer.get_shap_values(X_sample)
            top_features = explainer.get_top_features(shap_values, feature_names, top_n=20)
            depth_set = set(str(x).lower() for x in top_features)
            union = best_set.union(depth_set)
            jaccard = float(len(best_set.intersection(depth_set)) / len(union)) if union else float("nan")

            rows.append(
                {
                    "depth": depth,
                    "criterion": criterion,
                    "cv_accuracy_mean": float(row.get("cv_accuracy_mean", np.nan)),
                    "cv_f1_mean": float(row.get("cv_f1_mean", np.nan)),
                    "top_features": " | ".join(top_features),
                    "jaccard_with_best_depth_top20": jaccard,
                }
            )

        out_df = pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)
        out_df.to_csv(out_path, index=False)
        self.logger.info("Saved DT depth SHAP comparison to %s", out_path)
        return out_df, out_path

    def run_approach_a(self, run_tag: str) -> dict[str, Any]:
        self.logger.info("Approach A started for run_tag=%s", run_tag)
        train_df, test_df, train_path, test_path = self._load_split(run_tag)

        if "label" not in train_df.columns or "label" not in test_df.columns:
            raise ValueError("Train/test files must include 'label' column for Approach A.")

        tfidf_candidates = self.config.get("models", {}).get("tfidf", {}).get("max_features", [5000])
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

        trainer = DecisionTreeExplainer(self.config, run_tag=run_tag)
        comparison_df = self._train_or_load_dt_models(trainer, X_train, y_train)
        if comparison_df.empty:
            raise ValueError(f"DT depth comparison is empty for run_tag={run_tag}")

        best_row = comparison_df.sort_values("cv_f1_mean", ascending=False).iloc[0]
        best_depth = int(best_row["depth"])
        best_criterion = str(best_row["criterion"])
        best_model_path = trainer.models_dir / f"decision_tree_depth_{best_depth}_{best_criterion}.joblib"

        if self.args.force_train or not best_model_path.exists():
            best_fit = trainer.train(X_train, y_train, depth=best_depth, criterion=best_criterion)
            trainer.save_model(best_fit["model"], depth=best_depth, criterion=best_criterion)

        model = joblib.load(best_model_path)
        eval_result = trainer.evaluate(model, X_test, y_test)

        explainer = TreeExplainer(model=model, feature_names=feature_names, run_tag=run_tag)
        global_sample_size = int(
            self.config.get("explainability", {})
            .get("shap", {})
            .get("global_sample_size", 500)
        )
        if sparse.issparse(X_test):
            X_global = X_test[: min(global_sample_size, X_test.shape[0])]
        else:
            X_global = np.asarray(X_test)[: min(global_sample_size, len(X_test))]

        global_df = explainer.explain_global(X_global, top_n=20)

        dt_shap_values = explainer.get_shap_values(X_global)
        dt_mean_abs = np.mean(np.abs(dt_shap_values), axis=0)
        dt_global_cache_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": dt_mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        dt_cache_path = self.tables_dir / f"dt_global_shap_cache{self._suffix(run_tag)}.csv"
        dt_global_cache_df.to_csv(dt_cache_path, index=False)
        self.logger.info("Saved DT global SHAP cache to %s", dt_cache_path)
        explainer.extract_decision_rules(model, feature_names, max_depth=10)

        stability_n = int(
            self.config.get("explainability", {})
            .get("shap", {})
            .get("stability_test_sample_size", 50)
        )
        pair_csv_path = self.tables_dir / f"near_duplicate_pairs{self._suffix(run_tag)}.csv"
        stability = explainer.compute_stability_from_pairs_file(
            pair_csv_path=pair_csv_path,
            feature_engineer=feature_engineer,
            source_df=test_df,
            n_pairs=stability_n,
            seed=self.seed,
        )

        fidelity_n = int(
            self.config.get("explainability", {})
            .get("shap", {})
            .get("fidelity_sample_size", 256)
        )
        if sparse.issparse(X_test):
            X_fidelity = X_test[: min(fidelity_n, X_test.shape[0])]
        else:
            X_fidelity = np.asarray(X_test)[: min(fidelity_n, len(X_test))]
        fidelity_reports = explainer.run_fidelity_test(X_fidelity, top_k=[3, 5, 10])

        _, depth_shap_path = self._dt_depth_shap_comparison(
            run_tag=run_tag,
            comparison_df=comparison_df,
            feature_names=feature_names,
            X_eval=X_test,
            models_dir=trainer.models_dir,
        )

        top_features = (
            dt_global_cache_df["feature"].astype(str).head(20).tolist()
            if not dt_global_cache_df.empty
            else []
        )

        out = {
            "run_tag": run_tag,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "best_depth": best_depth,
            "best_criterion": best_criterion,
            "best_model_path": str(best_model_path),
            "depth_comparison": comparison_df,
            "evaluation": eval_result,
            "stability": stability,
            "fidelity": fidelity_reports,
            "top_features": top_features,
            "global_shap_df": dt_global_cache_df,
            "depth_shap_comparison_path": str(depth_shap_path),
            "artifact_paths": [
                trainer.tables_dir / trainer.depth_comparison_filename,
                self.tables_dir / f"dt_depth_tradeoff{self._suffix(run_tag)}.csv",
                self.tables_dir / f"dt_depth_shap_comparison{self._suffix(run_tag)}.csv",
                dt_cache_path,
                self.tables_dir / f"fidelity_dt{self._suffix(run_tag)}.csv",
                self.tables_dir / f"near_duplicate_pairs{self._suffix(run_tag)}.csv",
                self.tables_dir / f"dt_rules{self._suffix(run_tag)}.txt",
                self.figures_dir / f"dt_shap_global{self._suffix(run_tag)}.png",
                self.figures_dir / f"dt_shap_global_beeswarm{self._suffix(run_tag)}.png",
                trainer.models_dir,
            ],
        }

        self._mark_stage(
            f"approach_a_{run_tag}",
            {
                "best_depth": best_depth,
                "best_criterion": best_criterion,
                "best_model_path": str(best_model_path),
                "f1": float(eval_result["metrics"]["f1"]),
            },
        )
        return out

    def run_approach_b(self, run_tag: str) -> dict[str, Any]:
        self.logger.info("Approach B started for run_tag=%s", run_tag)
        train_df, test_df, train_path, test_path = self._load_split(run_tag)

        if "label" not in train_df.columns or "label" not in test_df.columns:
            raise ValueError("Train/test files must include 'label' column for Approach B.")

        text_col = "full_text" if "full_text" in train_df.columns else "text"
        if text_col not in train_df.columns or text_col not in test_df.columns:
            raise ValueError("Train/test data must include 'full_text' or 'text'.")

        classifier = BERTClassifier(self.config, run_tag=run_tag)
        model_ready = self._model_file_exists(classifier.output_dir)

        train_info: dict[str, Any] = {}
        if self.args.force_train or not model_ready:
            self.logger.info("Fine-tuning BERT model (force_train=%s)", self.args.force_train)
            tr_texts, val_texts, tr_labels, val_labels = train_test_split(
                train_df[text_col].fillna("").astype(str),
                train_df["label"].astype(int),
                test_size=0.1,
                random_state=self.seed,
                stratify=train_df["label"].astype(int),
            )
            train_info = classifier.fine_tune(tr_texts, tr_labels, val_texts, val_labels)
        else:
            self.logger.info("Loading existing BERT model from %s", classifier.output_dir)
            classifier.load()

        eval_result = classifier.evaluate(
            test_df[text_col].fillna("").astype(str),
            test_df["label"].astype(int),
        )

        explainer = BERTShapExplainer(model=classifier, tokenizer=classifier.tokenizer, run_tag=run_tag)

        global_sample_size = int(
            self.config.get("explainability", {})
            .get("shap", {})
            .get("global_sample_size", 500)
        )
        shap_cache_path = self.tables_dir / f"bert_global_shap_cache{self._suffix(run_tag)}.csv"

        if self.args.skip_shap_bert:
            if shap_cache_path.exists():
                self.logger.info("Using cached BERT global SHAP at %s", shap_cache_path)
                global_df = pd.read_csv(shap_cache_path)
            else:
                self.logger.warning(
                    "--skip-shap-bert set but cache missing (%s). Global BERT SHAP skipped.",
                    shap_cache_path,
                )
                global_df = pd.DataFrame(columns=["token", "mean_abs_shap"])
        else:
            global_df = explainer.explain_global(
                test_df[text_col].fillna("").astype(str).tolist(),
                sample_size=global_sample_size,
            )
            global_df.to_csv(shap_cache_path, index=False)
            self.logger.info("Saved BERT global SHAP cache to %s", shap_cache_path)

        pair_count = int(
            self.config.get("explainability", {})
            .get("shap", {})
            .get("stability_test_sample_size", 50)
        )
        pair_csv_path = self.tables_dir / f"near_duplicate_pairs{self._suffix(run_tag)}.csv"
        stability = explainer.compute_stability_from_pairs_file(
            pair_csv_path=pair_csv_path,
            source_df=test_df,
            n_pairs=pair_count,
            seed=self.seed,
        )

        fidelity_sample_size = int(
            self.config.get("explainability", {})
            .get("shap", {})
            .get("fidelity_sample_size", 256)
        )
        fidelity_texts = (
            test_df[text_col]
            .fillna("")
            .astype(str)
            .tolist()[: min(fidelity_sample_size, len(test_df))]
        )
        fidelity_reports = explainer.run_fidelity_test(fidelity_texts, top_k=[3, 5, 10])

        if not global_df.empty and {"token", "mean_abs_shap"}.issubset(global_df.columns):
            top_tokens = (
                global_df.sort_values("mean_abs_shap", ascending=False)["token"]
                .astype(str)
                .head(20)
                .tolist()
            )
        else:
            fallback_maps = explainer.get_shap_values(fidelity_texts[: min(64, len(fidelity_texts))])
            top_tokens = explainer.get_top_tokens(fallback_maps, top_n=20)

        out = {
            "run_tag": run_tag,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "model_dir": str(classifier.output_dir),
            "train_info": train_info,
            "evaluation": eval_result,
            "stability": stability,
            "fidelity": fidelity_reports,
            "top_tokens": top_tokens,
            "global_shap_df": global_df,
            "artifact_paths": [
                classifier.output_dir,
                shap_cache_path,
                self.tables_dir / f"fidelity_bert{self._suffix(run_tag)}.csv",
                self.tables_dir / f"near_duplicate_pairs{self._suffix(run_tag)}.csv",
                self.figures_dir / f"bert_shap_global{self._suffix(run_tag)}.png",
            ],
        }

        self._mark_stage(
            f"approach_b_{run_tag}",
            {
                "model_dir": str(classifier.output_dir),
                "f1": float(eval_result["metrics"]["f1"]),
                "used_cached_shap": bool(self.args.skip_shap_bert),
            },
        )
        return out

    def run_comparisons(
        self,
        run_tag: str,
        dt_out: dict[str, Any],
        bert_out: dict[str, Any],
    ) -> dict[str, Any]:
        self.logger.info("Comparison metrics started for run_tag=%s", run_tag)

        metrics = ComparisonMetrics(run_tag=run_tag)
        accuracy_table = metrics.accuracy_comparison(dt_out["evaluation"], bert_out["evaluation"])
        stability = metrics.stability_comparison(dt_out["stability"], bert_out["stability"])
        overlap = metrics.feature_overlap(dt_out["top_features"], bert_out["top_tokens"], top_n=20)
        depth_tradeoff = metrics.depth_tradeoff_table(dt_out["depth_comparison"])
        fidelity_table = metrics.fidelity_comparison(dt_out["fidelity"], bert_out["fidelity"])
        summary_path = metrics.generate_full_report()

        stability_df = pd.DataFrame(
            [
                {
                    "dt_mean_r": float(stability.get("mean_r_dt", np.nan)),
                    "dt_std_r": float(dt_out.get("stability", {}).get("std_pearson_r", np.nan)),
                    "bert_mean_r": float(stability.get("mean_r_bert", np.nan)),
                    "bert_std_r": float(bert_out.get("stability", {}).get("std_pearson_r", np.nan)),
                    "t_stat": float(stability.get("t_stat", np.nan)),
                    "p_value": float(stability.get("p_value", np.nan)),
                    "interpretation": str(stability.get("interpretation", "")),
                }
            ]
        )
        stability_path = self.tables_dir / f"stability_comparison{self._suffix(run_tag)}.csv"
        stability_df.to_csv(stability_path, index=False)

        overlap_df = pd.DataFrame(
            [
                {
                    "jaccard_score": float(overlap.get("jaccard_score", np.nan)),
                    "spearman_rho": float(overlap.get("spearman_rho", np.nan)),
                    "p_value": float(overlap.get("p_value", np.nan)),
                    "overlapping_features": " | ".join(overlap.get("overlapping_features", [])),
                }
            ]
        )
        overlap_path = self.tables_dir / f"overlap_comparison{self._suffix(run_tag)}.csv"
        overlap_df.to_csv(overlap_path, index=False)

        out = {
            "accuracy_table": accuracy_table,
            "stability": stability,
            "overlap": overlap,
            "depth_tradeoff": depth_tradeoff,
            "fidelity_table": fidelity_table,
            "stability_table": stability_df,
            "overlap_table": overlap_df,
            "summary_path": str(summary_path),
            "artifact_paths": [
                self.tables_dir / f"accuracy_comparison{self._suffix(run_tag)}.csv",
                self.tables_dir / f"dt_depth_tradeoff{self._suffix(run_tag)}.csv",
                self.tables_dir / f"fidelity_comparison{self._suffix(run_tag)}.csv",
                stability_path,
                overlap_path,
                self.results_dir / f"RESULTS_SUMMARY{self._suffix(run_tag)}.md",
            ],
        }

        self._mark_stage(
            f"comparisons_{run_tag}",
            {
                "summary_path": str(summary_path),
                "stability_p": float(stability.get("p_value", np.nan)),
                "overlap_jaccard": float(overlap.get("jaccard_score", np.nan)),
            },
        )
        return out

    def generate_visualizations_for_run(
        self,
        run_tag: str,
        dt_out: dict[str, Any],
        bert_out: dict[str, Any],
        cmp_out: dict[str, Any],
    ) -> list[Path]:
        self.logger.info("Visualization stage started for run_tag=%s", run_tag)
        visualizer = ResultsVisualizer(run_tag=run_tag)

        figure_paths = visualizer.generate_all(
            {
                "depth_df": cmp_out.get("depth_tradeoff"),
                "accuracy_df": cmp_out.get("accuracy_table"),
                "dt_stability_scores": dt_out.get("stability"),
                "bert_stability_scores": bert_out.get("stability"),
                "dt_features": dt_out.get("top_features", []),
                "bert_features": bert_out.get("top_tokens", []),
                "dt_shap_df": dt_out.get("global_shap_df"),
                "bert_shap_df": bert_out.get("global_shap_df"),
            }
        )

        self._mark_stage(
            f"visualization_{run_tag}",
            {
                "figures": [str(path) for path in figure_paths],
            },
        )
        return figure_paths

    def generate_cross_run_visualizations(self, run_tags: list[str]) -> list[Path]:
        if not ("original" in run_tags and "debiased" in run_tags):
            self.logger.info("Skipping cross-run figures: both original and debiased runs are required.")
            return []

        self.logger.info("Generating cross-run presentation figures")
        visualizer = ResultsVisualizer(run_tag="original")

        acc_orig = self._read_table_csv("accuracy_comparison.csv")
        acc_deb = self._read_table_csv("accuracy_comparison_debiased.csv")
        fidelity_orig = self._read_table_csv("fidelity_comparison.csv")
        fidelity_deb = self._read_table_csv("fidelity_comparison_debiased.csv")
        dt_shap_orig = self._read_table_csv("dt_global_shap_cache.csv")
        dt_shap_deb = self._read_table_csv("dt_global_shap_cache_debiased.csv")

        figure_paths = visualizer.generate_all(
            {
                "accuracy_original_df": acc_orig,
                "accuracy_debiased_df": acc_deb,
                "fidelity_original_df": fidelity_orig,
                "fidelity_debiased_df": fidelity_deb,
                "dt_original_shap_df": dt_shap_orig,
                "dt_debiased_shap_df": dt_shap_deb,
            }
        )

        self._mark_stage(
            "visualization_cross_run",
            {
                "figures": [str(path) for path in figure_paths],
            },
        )
        return figure_paths

    def _read_table_csv(self, filename: str) -> pd.DataFrame:
        path = self.tables_dir / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def generate_paper_tables(self) -> list[Path]:
        self.logger.info("Generating paper-ready tables")
        table_generator = PaperTableGenerator(project_root=PROJECT_ROOT)
        table_paths = table_generator.generate_all()

        self._mark_stage(
            "paper_tables",
            {
                "tables": [str(path) for path in table_paths],
            },
        )
        return table_paths

    def run(self) -> dict[str, Any]:
        self.logger.info("Run directory: %s", self.run_dir)
        self.logger.info("Loading config from %s", self.args.config)
        self.logger.info("Processed data directory: %s", self.processed_dir)

        run_tags = self._resolve_run_tags()
        self.logger.info("Selected run tags: %s", run_tags)

        # Debiasing is required for default all-runs and debiased-specific runs.
        if "debiased" in run_tags or self.args.force_debias:
            debias_out = self.run_debiasing()
        else:
            debias_out = {"skipped": True, "reason": "--original mode"}
            self._mark_stage("debiasing", debias_out)

        full_summary: dict[str, Any] = {
            "run_dir": str(self.run_dir),
            "run_tags": run_tags,
            "debiasing": debias_out,
            "per_run": {},
        }

        combined_artifacts: list[Path] = []

        for run_tag in run_tags:
            self.logger.info("====================")
            self.logger.info("Executing run_tag=%s", run_tag)
            self.logger.info("====================")

            dt_out = self.run_approach_a(run_tag)
            bert_out = self.run_approach_b(run_tag)
            cmp_out = self.run_comparisons(run_tag, dt_out, bert_out)
            vis_paths = self.generate_visualizations_for_run(run_tag, dt_out, bert_out, cmp_out)

            artifacts: list[Path] = []
            artifacts.extend([Path(p) for p in dt_out["artifact_paths"]])
            artifacts.extend([Path(p) for p in bert_out["artifact_paths"]])
            artifacts.extend([Path(p) for p in cmp_out["artifact_paths"]])
            artifacts.extend(vis_paths)

            if run_tag == "debiased":
                artifacts.extend(
                    [
                        self.processed_dir / "debiased_train.csv",
                        self.processed_dir / "debiased_test.csv",
                        self.tables_dir / "leakage_audit.csv",
                        self.tables_dir / "debias_comparison.csv",
                    ]
                )
            else:
                artifacts.extend([self.processed_dir / "train.csv", self.processed_dir / "test.csv"])

            self._snapshot_artifacts(run_tag=run_tag, artifact_paths=artifacts)
            combined_artifacts.extend(artifacts)

            full_summary["per_run"][run_tag] = {
                "dt_best_depth": dt_out["best_depth"],
                "dt_best_criterion": dt_out["best_criterion"],
                "dt_metrics": dt_out["evaluation"]["metrics"],
                "bert_metrics": bert_out["evaluation"]["metrics"],
                "stability": cmp_out["stability"],
                "overlap": cmp_out["overlap"],
                "summary_path": cmp_out["summary_path"],
                "figure_paths": [str(path) for path in vis_paths],
            }
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        cross_run_figures = self.generate_cross_run_visualizations(run_tags)
        table_paths = self.generate_paper_tables()
        combined_artifacts.extend(cross_run_figures)
        combined_artifacts.extend(table_paths)

        self._snapshot_artifacts(run_tag="combined", artifact_paths=combined_artifacts)

        full_summary["cross_run_figures"] = [str(path) for path in cross_run_figures]
        full_summary["paper_tables"] = [str(path) for path in table_paths]

        full_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.summary_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")

        self._mark_stage("pipeline_complete", {"summary_path": str(self.summary_path)})
        self.logger.info("Pipeline complete. Summary saved to %s", self.summary_path)
        return full_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full fake-news XAI experiment pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train models even if artifacts already exist. Default behavior loads cached models.",
    )
    parser.add_argument(
        "--force-debias",
        action="store_true",
        help="Re-run lexical debiasing even if debiased CSVs already exist.",
    )
    parser.add_argument(
        "--skip-shap-bert",
        action="store_true",
        help="Skip slow BERT SHAP generation and use cached global SHAP CSV when available.",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Run experiments only on original (biased) splits.",
    )
    parser.add_argument(
        "--debiased",
        action="store_true",
        help="Run experiments only on debiased splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(args)
    summary = runner.run()

    print("\n=== Pipeline Complete ===")
    print(f"Run directory: {summary['run_dir']}")
    print(f"Run tags: {summary['run_tags']}")
    print(f"Summary file: {runner.summary_path}")


if __name__ == "__main__":
    main()
