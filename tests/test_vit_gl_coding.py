"""Tests for the ViT GL-classification pipeline (cv/train_vit.py).

Tests exercise metric computation and data preparation helpers without
requiring transformers or a GPU. End-to-end ViT training is gated behind the
``vit`` marker and requires transformers + torchvision + a CPU/GPU device.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "cv"))

from train_vit import (  # noqa: E402
    GL_CATEGORIES,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    _macro_f1,
    _per_category_metrics,
)


# ---------------------------------------------------------------------------
# Label mapping tests
# ---------------------------------------------------------------------------


def test_label2id_covers_all_categories():
    expected = {
        "TRAVEL", "MEALS", "OFFICE_SUPPLIES", "SOFTWARE", "HARDWARE",
        "MARKETING", "PROFESSIONAL_SERVICES", "UTILITIES", "RENT",
        "INSURANCE", "TRAINING", "ENTERTAINMENT", "OTHER",
    }
    assert set(LABEL2ID.keys()) == expected
    assert NUM_LABELS == 13


def test_id2label_roundtrip():
    for cat, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == cat


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------


def test_macro_f1_perfect():
    preds = [0, 1, 2, 0, 1]
    golds = [0, 1, 2, 0, 1]
    assert _macro_f1(preds, golds, 3) == 1.0


def test_macro_f1_worst():
    preds = [0, 0, 0]
    golds = [1, 2, 0]
    f1 = _macro_f1(preds, golds, 3)
    assert abs(f1 - 0.5 / 3) < 1e-6


def test_per_category_metrics():
    preds = [0, 0, 1, 1, 2]
    golds = [0, 1, 1, 2, 2]
    metrics = _per_category_metrics(preds, golds, 3, {0: "A", 1: "B", 2: "C"})
    assert set(metrics.keys()) == {"A", "B", "C"}
    # Label A(0): tp=1(fp at idx1), fp=1, fn=0 => precision=0.5, recall=1.0
    assert metrics["A"]["precision"] == 0.5
    assert metrics["A"]["recall"] == 1.0
    # Label B(1): tp=1(idx2), fp=1(idx3), fn=1(idx1) => precision=0.5, recall=0.5
    assert metrics["B"]["precision"] == 0.5
    assert metrics["B"]["recall"] == 0.5
    # Label C(2): tp=1(idx4), fp=0, fn=1(idx3: pred=1, gold=2) => precision=1.0, recall=0.5
    assert metrics["C"]["precision"] == 1.0
    assert metrics["C"]["recall"] == 0.5


# ---------------------------------------------------------------------------
# End-to-end ViT smoke test (transformers + torchvision required)
# ---------------------------------------------------------------------------


@pytest.mark.vit
def test_vit_model_loads_cpu(tmp_path: Path):
    """Verify a ViT model can be loaded on CPU. Requires transformers."""
    try:
        from transformers import AutoModelForImageClassification, AutoImageProcessor  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")

    model_id = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )
    assert model.config.num_labels == NUM_LABELS
    assert processor is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
