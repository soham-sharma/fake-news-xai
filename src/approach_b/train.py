"""BERT fine-tuning pipeline for Approach B (black-box + SHAP-ready)."""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class BERTClassifier:
    """Fine-tunes and evaluates a binary text classifier with HuggingFace Transformers."""

    def __init__(
        self, config: str | Path | dict[str, Any], run_tag: str = "original"
    ) -> None:
        if isinstance(config, (str, Path)):
            with Path(config).open("r", encoding="utf-8") as fp:
                self.config = yaml.safe_load(fp)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be a path or a dictionary")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.random_seed = int(self.config.get("random_seed", 42))
        self.run_tag = str(run_tag).strip().lower() or "original"
        self.bert_cfg = self.config.get("models", {}).get("bert", {})

        self.model_name = str(self.bert_cfg.get("model_name", "distilroberta-base"))
        self.fallback_model_name = "bert-base-uncased"
        self.max_tokens = int(self.bert_cfg.get("max_tokens", 256))
        self.epochs = int(self.bert_cfg.get("epochs", 3))
        self.learning_rate = float(self.bert_cfg.get("learning_rate", 2e-5))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Using device: %s", self.device)

        model_dir_name = (
            "bert_best" if self.run_tag == "original" else f"bert_best_{self.run_tag}"
        )
        self.output_dir = (
            PROJECT_ROOT / "results" / "models" / "approach_b" / model_dir_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer: Any = None
        self.model: Any = None
        self.trainer: Optional[Trainer] = None

        self._set_seed(self.random_seed)
        self._initialize_model_and_tokenizer()

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _initialize_model_and_tokenizer(self) -> None:
        """Load tokenizer/model with fallback to bert-base-uncased."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
            )
            self.logger.info("Loaded model/tokenizer: %s", self.model_name)
        except Exception as exc:
            self.logger.warning(
                "Failed loading %s (%s). Falling back to %s.",
                self.model_name,
                exc,
                self.fallback_model_name,
            )
            self.model_name = self.fallback_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
            )
            self.logger.info("Loaded fallback model/tokenizer: %s", self.model_name)

        self.model.to(self.device)

    def tokenize(self, texts, max_length: int = 256) -> Dataset:
        """Tokenize raw texts and return a HuggingFace Dataset."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")
        tokenizer = self.tokenizer

        text_list = (
            pd.Series(list(texts), dtype="object").fillna("").astype(str).tolist()
        )
        dataset = Dataset.from_dict({"text": text_list})

        def _tokenize_batch(batch):
            return tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=max_length,
            )

        tokenized = dataset.map(_tokenize_batch, batched=True)
        return tokenized.remove_columns(["text"])

    def _prepare_dataset(
        self, texts, labels=None, max_length: Optional[int] = None
    ) -> Dataset:
        tokenized = self.tokenize(texts, max_length=max_length or self.max_tokens)
        if labels is not None:
            y = pd.Series(labels).astype(int).tolist()
            tokenized = tokenized.add_column("labels", y)
        return tokenized

    @staticmethod
    def compute_metrics(eval_pred) -> dict[str, float]:
        """Custom Trainer metric function that includes F1."""
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        preds = np.argmax(probs, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            zero_division=0,
        )
        accuracy = accuracy_score(labels, preds)

        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            auc = float("nan")

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc),
        }

    def _build_training_arguments(self) -> TrainingArguments:
        """Create TrainingArguments with compatibility across transformers versions."""
        signature = inspect.signature(TrainingArguments.__init__).parameters

        kwargs: dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "seed": self.random_seed,
        }

        # Compatibility for versions that renamed this argument.
        if "evaluation_strategy" in signature:
            kwargs["evaluation_strategy"] = "epoch"
        elif "eval_strategy" in signature:
            kwargs["eval_strategy"] = "epoch"

        if "save_strategy" in signature:
            kwargs["save_strategy"] = "epoch"
        if "logging_strategy" in signature:
            kwargs["logging_strategy"] = "epoch"
        if "report_to" in signature:
            kwargs["report_to"] = []

        # Enable fp16 when requested in config and CUDA is available.
        requested_fp16 = bool(self.bert_cfg.get("fp16", False))
        if "fp16" in signature:
            kwargs["fp16"] = bool(requested_fp16 and torch.cuda.is_available())
            if requested_fp16 and not kwargs["fp16"]:
                self.logger.warning(
                    "fp16 requested in config but CUDA not available; disabling fp16."
                )

        return TrainingArguments(**kwargs)

    def fine_tune(
        self,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
    ) -> dict[str, Any]:
        """Fine-tune the model using HuggingFace Trainer and save best checkpoint."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not initialized.")
        tokenizer = self.tokenizer
        model = self.model

        train_ds = self._prepare_dataset(train_texts, train_labels)
        val_ds = self._prepare_dataset(val_texts, val_labels)

        training_args = self._build_training_arguments()
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_ds,
            "eval_dataset": val_ds,
            "data_collator": data_collator,
            "compute_metrics": self.compute_metrics,
        }
        trainer_sig = inspect.signature(Trainer.__init__).parameters
        if "tokenizer" in trainer_sig:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_sig:
            trainer_kwargs["processing_class"] = tokenizer

        self.trainer = Trainer(**trainer_kwargs)

        self.logger.info("Starting fine-tuning for %s epoch(s)", self.epochs)
        train_result = self.trainer.train()

        for log_entry in self.trainer.state.log_history:
            if "loss" in log_entry and "epoch" in log_entry:
                self.logger.info(
                    "Epoch %.0f - training loss: %.6f",
                    float(log_entry["epoch"]),
                    float(log_entry["loss"]),
                )

        self.save()
        self.logger.info("Best model saved to %s", self.output_dir)

        return {
            "train_metrics": train_result.metrics,
            "best_checkpoint": self.trainer.state.best_model_checkpoint,
        }

    def predict_proba(self, texts) -> np.ndarray:
        """Return class probabilities for a batch of texts."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not initialized.")
        model = self.model

        dataset = self._prepare_dataset(texts, labels=None)
        columns = ["input_ids", "attention_mask"]
        if "token_type_ids" in dataset.column_names:
            columns.append("token_type_ids")
        dataset.set_format(type="torch", columns=columns)

        loader = DataLoader(dataset, batch_size=32, shuffle=False)  # type: ignore[arg-type]
        model.eval()

        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def evaluate(self, test_texts, test_labels) -> dict[str, Any]:
        """Evaluate model on held-out test set."""
        y_true = pd.Series(test_labels).astype(int).to_numpy()
        probs = self.predict_proba(test_texts)
        y_pred = np.argmax(probs, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, probs[:, 1])
        except ValueError:
            auc = float("nan")

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc),
        }

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        return {"metrics": metrics, "classification_report": report}

    def save(self, save_dir: str | Path | None = None) -> Path:
        """Save model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not initialized.")
        model = self.model
        tokenizer = self.tokenizer

        out_dir = Path(save_dir) if save_dir is not None else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        return out_dir

    def load(self, load_dir: str | Path | None = None) -> None:
        """Load model and tokenizer from disk."""
        in_dir = Path(load_dir) if load_dir is not None else self.output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(in_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(in_dir)
        self.model.to(self.device)
        self.logger.info("Loaded model/tokenizer from %s", in_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train Approach B BERT classifier")
    parser.add_argument(
        "--debiased",
        action="store_true",
        help="Use debiased_train.csv / debiased_test.csv and save model to a debiased-specific directory.",
    )
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    processed_dir = PROJECT_ROOT / config.get("data", {}).get(
        "processed_path", "data/processed"
    )
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
        raise ValueError("Train/test data must include a 'label' column.")

    text_col = "full_text" if "full_text" in train_df.columns else "text"
    if text_col not in train_df.columns or text_col not in test_df.columns:
        raise ValueError("Train/test data must include 'full_text' or 'text' columns.")

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_df[text_col].fillna("").astype(str),
        train_df["label"].astype(int),
        test_size=0.1,
        random_state=int(config.get("random_seed", 42)),
        stratify=train_df["label"].astype(int),
    )

    classifier = BERTClassifier(config, run_tag=run_tag)
    tune_info = classifier.fine_tune(tr_texts, tr_labels, val_texts, val_labels)
    eval_out = classifier.evaluate(
        test_df[text_col].fillna("").astype(str),
        test_df["label"].astype(int),
    )

    print("\nFine-tuning summary:")
    print(tune_info)
    print("\nRun mode:", run_tag)
    print("Model output directory:", classifier.output_dir)
    print("\nTest metrics:")
    print(eval_out["metrics"])
    print("\nClassification report:")
    print(pd.DataFrame(eval_out["classification_report"]).transpose().to_string())
