"""LightGBM / XGBoost baselines for GL coding.

Training data is the exact domain split used by the QLoRA experiment:
152 synthetic ledgerloop instructions for training and the 30 held-out
unseen-merchant instructions in ``sft/data/val.jsonl`` for evaluation.

Features
--------
* TF-IDF word 1-2 grams over ``merchant + description``
* TF-IDF character 3-5 grams (handles merchant spelling variants)
* categorical: currency, amount bucket, weekday, month
* numeric: log amount, amount / 1000, day of month, is-weekend

No LLM output, embeddings, or validation labels are used as features.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "sft" / "data" / "train.jsonl"
VAL_FILE = ROOT / "sft" / "data" / "val.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALLOWED = [
    "TRAVEL",
    "MEALS",
    "OFFICE_SUPPLIES",
    "SOFTWARE",
    "HARDWARE",
    "MARKETING",
    "PROFESSIONAL_SERVICES",
    "UTILITIES",
    "RENT",
    "INSURANCE",
    "TRAINING",
    "ENTERTAINMENT",
    "OTHER",
]
AMOUNT_BUCKETS = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, float("inf"))]
SEED = 42


@dataclass
class CodingExample:
    id: str
    merchant: str
    description: str
    amount: float
    currency: str
    iso_date: str
    label: str

    @property
    def text(self) -> str:
        return f"{self.merchant} {self.description}".strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "merchant": self.merchant,
            "description": self.description,
            "amount": self.amount,
            "currency": self.currency,
            "date": self.iso_date,
        }


def _amount_bucket(amount: float) -> str:
    for low, high in AMOUNT_BUCKETS:
        if low <= amount < high:
            return f"{low}-{high}"
    return "1000+"


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


_INSTRUCTION_FALLBACK = re.compile(
    r"Merchant:\s*(?P<merchant>[^\n]+).*?Amount:\s*(?P<currency>[A-Z]{3})\s*"
    r"(?P<amount>[\d,.]+).*?Date:\s*(?P<date>[\d-]+).*?Description:\s*(?P<description>[^\n]+)",
    re.DOTALL,
)


def load_examples(path: Path, source: str = "ledgerloop") -> list[CodingExample]:
    examples: list[CodingExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != source:
                continue
            metadata = row.get("metadata") or {}
            if not metadata:
                match = _INSTRUCTION_FALLBACK.search(row.get("instruction", ""))
                if not match:
                    continue
                metadata = {
                    "merchant": match.group("merchant").strip(),
                    "description": match.group("description").strip(),
                    "amount": float(match.group("amount").replace(",", "")),
                    "currency": match.group("currency"),
                    "date": match.group("date"),
                }
            label = str(row.get("response", "")).strip()
            if label not in ALLOWED:
                continue
            examples.append(
                CodingExample(
                    id=str(row.get("id")),
                    merchant=str(metadata.get("merchant", "")),
                    description=str(metadata.get("description", "")),
                    amount=float(metadata.get("amount", 0.0) or 0.0),
                    currency=str(metadata.get("currency", "USD")),
                    iso_date=str(metadata.get("date", "")),
                    label=label,
                )
            )
    return examples


def _frame_rows(examples: Iterable[CodingExample]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        parsed = _parse_date(example.iso_date)
        rows.append(
            {
                "text": example.text.lower(),
                "currency": example.currency,
                "amount_bucket": _amount_bucket(example.amount),
                "weekday": parsed.weekday() if parsed else -1,
                "month": parsed.month if parsed else -1,
                "log_amount": float(np.log1p(max(0.0, example.amount))),
                "amount_k": example.amount / 1000.0,
                "day": parsed.day if parsed else -1,
                "is_weekend": int(parsed.weekday() >= 5) if parsed else 0,
            }
        )
    return rows


class GLFeaturizer:
    """Sparse feature builder that can be fit once and reused at inference."""

    def __init__(self, max_word_features: int = 6000, max_char_features: int = 6000):
        self.word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=max_word_features,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=max_char_features,
        )
        self.categorical = ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ["currency", "amount_bucket", "weekday", "month"],
                ),
                ("num", "passthrough", ["log_amount", "amount_k", "day", "is_weekend"]),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        self.text_pipeline = FeatureUnion(
            [("word", self.word), ("char", self.char)], n_jobs=1
        )

    def fit(self, examples: list[CodingExample]) -> GLFeaturizer:
        rows = _frame_rows(examples)
        self.text_pipeline.fit([row["text"] for row in rows])
        self.categorical.fit(pd.DataFrame(rows))
        return self

    def transform(self, examples: list[CodingExample]) -> sparse.csr_matrix:
        rows = _frame_rows(examples)
        text = self.text_pipeline.transform([row["text"] for row in rows])
        categorical = self.categorical.transform(pd.DataFrame(rows))
        return sparse.hstack([text, categorical], format="csr")

    def fit_transform(self, examples: list[CodingExample]) -> sparse.csr_matrix:
        return self.fit(examples).transform(examples)


def _labels(examples: list[CodingExample]) -> list[str]:
    return [example.label for example in examples]


def _lightgbm_params(**overrides) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": len(ALLOWED),
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 3,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    params.update(overrides)
    return params


def _fit_lightgbm(matrix, labels, **overrides):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**_lightgbm_params(**overrides))
    model.fit(matrix, labels)
    return model


def _fit_xgboost(matrix, labels):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(ALLOWED),
        n_estimators=400,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(matrix, labels)
    return model


def _metrics(y_true: list[str], y_pred: np.ndarray, name: str) -> dict[str, Any]:
    labels = ALLOWED
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "model": name,
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                         zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted",
                                            zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> int:
    global SEED
    parser = argparse.ArgumentParser(description="LightGBM/XGBoost GL-coding baseline")
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--val", type=Path, default=VAL_FILE)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "gl_metrics.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    SEED = args.seed
    train = load_examples(args.train)
    val = load_examples(args.val)
    if not train or not val:
        raise SystemExit("empty train or validation split")

    featurizer = GLFeaturizer()
    x_train = featurizer.fit_transform(train)
    x_val = featurizer.transform(val)
    y_train, y_val = _labels(train), _labels(val)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    import lightgbm as lgb

    cv_estimator = lgb.LGBMClassifier(**_lightgbm_params())
    cv_scores = cross_val_score(
        cv_estimator,
        x_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )

    lightgbm = _fit_lightgbm(x_train, y_train)
    lgb_pred = lightgbm.predict(x_val)
    lgb_metrics = _metrics(y_val, lgb_pred, "lightgbm")

    encoder = LabelEncoder().fit(ALLOWED)
    y_train_encoded = encoder.transform(y_train)
    xgboost = _fit_xgboost(x_train, y_train_encoded)
    xgb_pred = encoder.inverse_transform(xgboost.predict(x_val))
    xgb_metrics = _metrics(y_val, xgb_pred, "xgboost")

    report = {
        "task": "GL coding",
        "split": {
            "train_file": str(args.train.relative_to(ROOT)),
            "val_file": str(args.val.relative_to(ROOT)),
            "n_train": len(train),
            "n_val": len(val),
            "note": "same 152/30 domain split as the QLoRA adapter",
        },
        "features": {
            "word_tfidf_features": len(featurizer.word.vocabulary_),
            "char_tfidf_features": len(featurizer.char.vocabulary_),
            "total_design_matrix": int(x_train.shape[1]),
            "numeric": ["log_amount", "amount_k", "day", "is_weekend"],
            "categorical": ["currency", "amount_bucket", "weekday", "month"],
        },
        "seed": args.seed,
        "lightgbm_cv5_macro_f1": {
            "mean": round(float(np.mean(cv_scores)), 4),
            "std": round(float(np.std(cv_scores)), 4),
            "folds": [round(float(score), 4) for score in cv_scores],
        },
        "lightgbm": lgb_metrics,
        "xgboost": xgb_metrics,
        "labels": ALLOWED,
        "predictions": [
            {
                "id": example.id,
                "gold": gold,
                "lightgbm": str(lgb),
                "xgboost": str(xgb),
            }
            for example, gold, lgb, xgb in zip(val, y_val, lgb_pred, xgb_pred, strict=True)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_train": len(train),
        "n_val": len(val),
        "lightgbm": {k: lgb_metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
        "xgboost": {k: xgb_metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
        "cv5_macro_f1": report["lightgbm_cv5_macro_f1"],
    }, indent=2))
    print(f"results -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
