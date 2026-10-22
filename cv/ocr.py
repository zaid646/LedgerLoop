"""Tesseract OCR + deterministic field parsing for receipts.

The OCR path is CPU-only and uses ``pytesseract``.  Parsing is deliberately
simple (regex + label heuristics) so errors are attributable to OCR rather than
to a learned post-processor.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv.preprocess import preprocess

_DATE_RE = re.compile(r"\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b")
_AMOUNT_RE = re.compile(r"-?\d[\d,]*\.\d{2}\b")
_GENERIC_HEADER = re.compile(r"receipt|invoice|tax invoice|cash bill", re.IGNORECASE)


@dataclass
class OCRResult:
    text: str
    confidence: float
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "fields": self.fields,
        }


def _require_tesseract() -> Any:
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pytesseract is required for OCR: pip install -r requirements-cv.txt") from exc
    return pytesseract


def ocr_image(image: np.ndarray, psm: int = 6, lang: str = "eng") -> OCRResult:
    """Run Tesseract on an image and return text + mean word confidence."""
    pytesseract = _require_tesseract()
    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        gray, config=config, lang=lang, output_type=pytesseract.Output.DICT
    )
    words: list[str] = []
    confidences: list[float] = []
    for word, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
        if str(word).strip():
            words.append(str(word).strip())
            try:
                value = float(conf)
            except (TypeError, ValueError):
                value = -1.0
            if value >= 0:
                confidences.append(value)
    text = " ".join(words)
    confidence = float(np.mean(confidences) / 100.0) if confidences else 0.0
    return OCRResult(text=text, confidence=confidence)


def ocr_text_lines(image: np.ndarray, psm: int = 6, lang: str = "eng") -> str:
    """Run Tesseract and preserve line breaks (needed for merchant parsing)."""
    pytesseract = _require_tesseract()
    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config=f"--oem 3 --psm {psm}", lang=lang)


def parse_date(text: str) -> str | None:
    match = _DATE_RE.search(text or "")
    return match.group(0) if match else None


def parse_amount(text: str) -> float | None:
    values = []
    for raw in _AMOUNT_RE.findall(text or ""):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values[-1] if values else None


def parse_merchant(text: str) -> str | None:
    for line in (text or "").splitlines():
        cleaned = " ".join(line.split())
        if len(cleaned) < 3:
            continue
        if not any(char.isalpha() for char in cleaned):
            continue
        if _GENERIC_HEADER.search(cleaned) and len(cleaned) < 16:
            continue
        return cleaned
    return None


def parse_total_from_text(text: str) -> float | None:
    total = None
    for line in (text or "").splitlines():
        lowered = line.lower()
        if "total" not in lowered or "subtotal" in lowered:
            continue
        amount = parse_amount(line)
        if amount is not None:
            total = amount
    if total is not None:
        return total
    values = [parse_amount(line) for line in (text or "").splitlines()]
    values = [value for value in values if value is not None]
    return values[-1] if values else None


def parse_fields(text: str) -> dict[str, Any]:
    return {
        "merchant": parse_merchant(text),
        "transaction_date": parse_date(text),
        "total_amount": parse_total_from_text(text),
    }


def extract_fields(image: np.ndarray, use_preprocess: bool = True) -> OCRResult:
    """Full-page OCR field extraction."""
    processed = preprocess(image).cropped if use_preprocess else image
    text = ocr_text_lines(processed)
    result = ocr_image(processed)
    fields = parse_fields(text)
    result.text = text
    result.fields = fields
    return result


def ocr_regions(
    image: np.ndarray,
    detector,
    *,
    conf: float = 0.25,
    use_preprocess: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect field regions with YOLO, then OCR each crop.

    Returns ``(fields, debug)`` where debug contains detections and per-crop
    text so the run is auditable.
    """
    processed = preprocess(image).cropped if use_preprocess else image
    best = detector.best_fields(processed, conf=conf)
    fields: dict[str, Any] = {
        "merchant": None,
        "transaction_date": None,
        "total_amount": None,
    }
    debug: dict[str, Any] = {"detections": {}, "crops": {}}
    for name, detection in best.items():
        crop = detector.crop(processed, detection, pad=8)
        text = ocr_text_lines(crop, psm=6)
        debug["detections"][name] = detection.as_dict()
        debug["crops"][name] = text.strip()
        if name == "merchant":
            fields["merchant"] = parse_merchant(text)
        elif name == "date":
            fields["transaction_date"] = parse_date(text)
        elif name == "total":
            fields["total_amount"] = parse_amount(text)
    return fields, debug


def fuse_ocr_llm(
    ocr_full: dict[str, Any],
    ocr_fields: dict[str, Any],
    llm: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic fusion used in the OCR+LLM row of the benchmark.

    Policy (fixed before seeing validation results):
    * merchant -> LLM when present (language model handles noisy merchant text),
      otherwise OCR;
    * total and date -> OCR region when present (numeric fields favour the
      detector+OCR path), otherwise OCR full page, otherwise LLM.
    """
    merchant = llm.get("merchant") or ocr_fields.get("merchant") or ocr_full.get("merchant")
    total = (
        ocr_fields.get("total_amount")
        if ocr_fields.get("total_amount") is not None
        else ocr_full.get("total_amount")
        if ocr_full.get("total_amount") is not None
        else llm.get("total_amount")
    )
    tx_date = (
        ocr_fields.get("transaction_date")
        or ocr_full.get("transaction_date")
        or llm.get("transaction_date")
    )
    return {
        "merchant": merchant,
        "transaction_date": tx_date,
        "total_amount": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR a receipt image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    result = extract_fields(image, use_preprocess=not args.no_preprocess)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
