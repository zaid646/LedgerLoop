"""Tests for the BERT/RoBERTa GL-coding pipeline (nlp/train_bert.py).

Tests exercise data loading, transaction extraction, metric computation, and
(when transformers + torch are available) a tiny end-to-end CPU training run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "nlp"))
sys.path.insert(0, str(REPO_ROOT / "sft"))

from train_bert import (  # noqa: E402
    LABEL2ID,
    NUM_LABELS,
    build_dataset,
    extract_transaction,
    load_jsonl,
)


# ---------------------------------------------------------------------------
# Data loading / extraction tests (no transformers required)
# ---------------------------------------------------------------------------


def test_load_jsonl():
    path = REPO_ROOT / "sft" / "data" / "val.jsonl"
    records = load_jsonl(path)
    assert len(records) == 55
    assert records[0]["id"] == "ledgerloop-039"


def test_build_dataset_counts():
    records = load_jsonl(REPO_ROOT / "sft" / "data" / "train.jsonl")
    examples = build_dataset(records)
    assert len(examples) == 327

    domain_only = build_dataset(records, source_filter="ledgerloop")
    assert len(domain_only) == 152

    dolly_only = build_dataset(records, source_filter="dolly")
    assert len(dolly_only) == 175


def test_extract_transaction():
    record = {
        "id": "ledgerloop-045",
        "source": "ledgerloop",
        "category": "SOFTWARE",
        "instruction": (
            "Classify this business transaction into the correct GL account. "
            "Answer with one code from the allowed list and nothing else.\n\n"
            "Allowed categories: TRAVEL, MEALS, OFFICE_SUPPLIES, SOFTWARE, "
            "HARDWARE, MARKETING, PROFESSIONAL_SERVICES, UTILITIES, RENT, "
            "INSURANCE, TRAINING, ENTERTAINMENT, OTHER\n\n"
            "Transaction:\nMerchant: Microsoft\nAmount: USD 1312.96\n"
            "Date: 2026-04-07\nDescription: Microsoft 365 licenses for the finance team"
        ),
        "response": "SOFTWARE",
    }
    text = extract_transaction(record)
    assert "Merchant: Microsoft" in text
    assert "USD 1312.96" in text
    assert "Microsoft 365 licenses" in text
    assert "Allowed categories" not in text


def test_label2id_covers_all_categories():
    expected = {
        "TRAVEL", "MEALS", "OFFICE_SUPPLIES", "SOFTWARE", "HARDWARE",
        "MARKETING", "PROFESSIONAL_SERVICES", "UTILITIES", "RENT",
        "INSURANCE", "TRAINING", "ENTERTAINMENT", "OTHER",
    }
    assert set(LABEL2ID.keys()) == expected
    assert NUM_LABELS == 13


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------


def test_macro_f1_perfect():
    from train_bert import _macro_f1
    preds = [0, 1, 2, 0, 1]
    golds = [0, 1, 2, 0, 1]
    assert _macro_f1(preds, golds, 3) == 1.0


def test_macro_f1_worst():
    from train_bert import _macro_f1
    preds = [0, 0, 0]
    golds = [1, 2, 0]
    # label 0: tp=1 fp=2 fn=0 => f1 = 2/(2+2+0) = 0.5; label 1: tp=0 => 0; label 2: tp=0 => 0
    f1 = _macro_f1(preds, golds, 3)
    assert abs(f1 - 0.5 / 3) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end CPU smoke test (transformers required)
# ---------------------------------------------------------------------------


@pytest.mark.bert
def test_tiny_bert_cpu_train(tmp_path: Path):
    """Tiny BERT training run on CPU. Requires transformers."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")

    import torch

    records = load_jsonl(REPO_ROOT / "sft" / "data" / "val.jsonl")
    examples = build_dataset(records[:8])  # tiny subset

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModelForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=NUM_LABELS, id2label={i: c for c, i in LABEL2ID.items()}, label2id=LABEL2ID
    )
    model.to("cpu")

    from train_bert import GLDataset
    from torch.utils.data import DataLoader

    ds = GLDataset(examples, tokenizer, max_length=64)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            assert out.logits.shape == (len(batch["input_ids"]), NUM_LABELS)
            break  # one batch is enough


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
