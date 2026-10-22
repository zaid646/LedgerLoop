"""Unit tests for the CV preprocessing pipeline and classical-ML baseline."""

from __future__ import annotations

import numpy as np
import pytest

from cv.ocr import parse_fields
from cv.preprocess import (
    adaptive_threshold,
    crop_receipt,
    denoise,
    deskew,
    find_receipt_contour,
    four_point_crop,
    preprocess,
    to_gray,
)
from cv.synthetic import CLASS_NAMES, load_golden, render_document


def _receipt_like() -> np.ndarray:
    image = np.full((480, 320, 3), 255, dtype=np.uint8)
    image[30:450, 25:295] = 245
    for row in range(60, 400, 28):
        image[row : row + 3, 45:270] = 20
    image[410:415, 45:270] = 0
    return image


def test_to_gray_and_threshold() -> None:
    image = _receipt_like()
    gray = to_gray(image)
    assert gray.shape == (480, 320)
    binary = adaptive_threshold(gray)
    assert binary.shape == gray.shape
    assert set(np.unique(binary)).issubset({0, 255})
    assert 0 in np.unique(binary)


def test_denoise_preserves_shape_and_dtype() -> None:
    gray = to_gray(_receipt_like())
    out = denoise(gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


def test_preprocess_returns_all_stages() -> None:
    result = preprocess(_receipt_like())
    assert result.gray.shape == (480, 320)
    assert result.denoised.shape == result.gray.shape
    assert result.binary.shape == result.gray.shape
    assert result.cropped.size > 0
    assert np.isfinite(result.angle)
    assert isinstance(result.crop_found, bool)


def test_four_point_crop_known_quad() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    quad = np.array([[10, 10], [190, 12], [188, 90], [12, 88]], dtype=np.float32)
    cropped = four_point_crop(image, quad)
    assert cropped.shape[0] > 50
    assert cropped.shape[1] > 150


def test_empty_document_falls_back_to_frame() -> None:
    blank = np.full((200, 120), 255, dtype=np.uint8)
    quad, found = find_receipt_contour(blank)
    assert found is False
    assert quad.shape == (4, 2)
    cropped, crop_found = crop_receipt(blank)
    assert crop_found is False
    assert cropped.shape == blank.shape


def test_deskew_recovers_rotation() -> None:
    canvas = np.full((400, 600), 0, dtype=np.uint8)
    for y in range(80, 320, 40):
        canvas[y : y + 4, 80:520] = 255
    import cv2

    rotated = cv2.warpAffine(
        canvas,
        cv2.getRotationMatrix2D((300, 200), 5.0, 1.0),
        (600, 400),
        borderValue=0,
    )
    corrected, angle = deskew(rotated)
    assert corrected.size > 0
    assert abs(angle) > 2.0


def test_parse_fields_heuristics() -> None:
    text = "BOOK TA .K SDN BHD\n123 MAIN STREET\nDATE: 25/12/2018 8:13:39 PM\n"
    text += "SUBTOTAL 10.00\nTOTAL: 9.00\nCASH 10.00\n"
    fields = parse_fields(text)
    assert fields["merchant"] == "BOOK TA .K SDN BHD"
    assert fields["transaction_date"] == "25/12/2018"
    assert fields["total_amount"] == 9.0


def test_renderer_is_deterministic_and_boxes_valid() -> None:
    record = load_golden()[0]
    first = render_document(record, variant=3, seed=7)
    second = render_document(record, variant=3, seed=7)
    assert np.array_equal(first.image, second.image)
    assert first.boxes == second.boxes
    height, width = first.image.shape[:2]
    assert set(first.boxes) == set(CLASS_NAMES)
    for left, top, right, bottom in first.boxes.values():
        assert 0 <= left < right < width
        assert 0 <= top < bottom < height


def test_domain_split_sizes() -> None:
    from ml.gl_lightgbm import load_examples

    train = load_examples(__import__("pathlib").Path("sft/data/train.jsonl"))
    val = load_examples(__import__("pathlib").Path("sft/data/val.jsonl"))
    assert len(train) == 152
    assert len(val) == 30
    assert {row.label for row in train} <= set(__import__("ml.gl_lightgbm", fromlist=["ALLOWED"]).ALLOWED)


def test_lightgbm_training_smoke() -> None:
    pytest.importorskip("lightgbm")
    from ml.gl_lightgbm import (
        ALLOWED,
        CodingExample,
        GLFeaturizer,
        _fit_lightgbm,
        _metrics,
    )

    examples = []
    for index in range(30):
        label = ALLOWED[index % 3]
        examples.append(
            CodingExample(
                id=f"t-{index}",
                merchant=f"Merchant {label}",
                description=f"purchase for {label.lower()}",
                amount=10.0 + index * 7,
                currency="USD",
                iso_date=f"2026-0{index % 9 + 1}-15",
                label=label,
            )
        )
    train, test = examples[:20], examples[20:]
    featurizer = GLFeaturizer()
    x_train = featurizer.fit_transform(train)
    x_test = featurizer.transform(test)
    assert x_train.shape[1] == x_test.shape[1]
    model = _fit_lightgbm(x_train, [row.label for row in train], n_estimators=40)
    pred = model.predict(x_test)
    metrics = _metrics([row.label for row in test], pred, "lightgbm")
    assert metrics["accuracy"] >= 0.5
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["confusion_matrix"]) == len(ALLOWED)
