from __future__ import annotations

import numpy as np
import pytest
from runtime.common import (
    MAX_LENGTH,
    TOKEN_FIELDS,
    artifacts_ready,
    get_tokenizer,
    labels,
    load_metadata,
    load_prototypes,
    preprocess,
)

pytestmark = pytest.mark.skipif(
    not artifacts_ready(),
    reason="runtime artifacts missing; run `make runtime-export` first",
)


@pytest.fixture(scope="session")
def parity_result():
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tensorflow")
    from runtime.parity import compare

    return compare(threads=4)


def test_export_metadata_and_label_order() -> None:
    metadata = load_metadata()
    assert metadata["model_id"] == "BAAI/bge-small-en-v1.5"
    assert metadata["model_kind"] == "embedding"
    assert metadata["max_length"] == MAX_LENGTH
    assert metadata["labels"] == labels()
    assert metadata["preprocessing"]["padding"] == "max_length"
    assert metadata["preprocessing"]["truncation"] is True
    assert metadata["preprocessing"]["input_fields"] == list(TOKEN_FIELDS)
    assert metadata["classifier_export"]["attempted"] is False
    for name in ("bge-small-en-v1.5.fp32.onnx", "bge-small-en-v1.5.int8.onnx", "bge-small-en-v1.5.fp32.tflite"):
        assert metadata["files"][name]["bytes"] > 0


def test_preprocessing_is_identical_for_every_runtime() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tensorflow")
    from runtime.backends import OnnxEmbedder, TFLiteEmbedder

    text = "team lunch - Starbucks"
    encoded = preprocess([text])
    assert list(encoded) == list(TOKEN_FIELDS)
    assert encoded["input_ids"].shape == (1, MAX_LENGTH)
    assert encoded["attention_mask"].shape == (1, MAX_LENGTH)
    assert encoded["token_type_ids"].shape == (1, MAX_LENGTH)
    assert encoded["input_ids"][0, 0] == get_tokenizer().cls_token_id
    assert encoded["attention_mask"][0].sum() == len(get_tokenizer()(text)["input_ids"])

    onnx = OnnxEmbedder(int8=False)
    tflite = TFLiteEmbedder()
    assert onnx.input_names == list(TOKEN_FIELDS)
    assert tflite.input_names == list(TOKEN_FIELDS)


def test_torch_onnx_fp32_parity(parity_result) -> None:
    result = parity_result["runtimes"]["onnx_fp32"]
    assert result["cosine_min"] >= 0.99999
    assert result["max_abs_diff"] <= 1e-5
    assert result["label_agreement"] == 1.0


def test_torch_onnx_int8_parity(parity_result) -> None:
    result = parity_result["runtimes"]["onnx_int8"]
    assert result["cosine_min"] >= 0.97
    assert result["max_abs_diff"] <= 0.06
    assert result["label_agreement"] == 1.0


def test_torch_tflite_parity(parity_result) -> None:
    result = parity_result["runtimes"]["tflite_fp32"]
    assert result["cosine_min"] >= 0.999
    assert result["max_abs_diff"] <= 1e-4
    assert result["label_agreement"] == 1.0


def test_label_mapping_matches_expected_gl_categories(parity_result) -> None:
    assert parity_result["expected_accuracy"] >= 0.9
    for name, row in parity_result["runtimes"].items():
        assert row["label_agreement"] == 1.0, name
        assert row["accuracy"] >= 0.9, name


def test_prototypes_match_label_count() -> None:
    prototypes = load_prototypes()
    assert prototypes.shape == (len(labels()), 384)
    assert np.isfinite(prototypes).all()
