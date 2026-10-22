"""Computer-vision utilities for receipt field extraction.

The module is intentionally dependency-light at import time so the pure
preprocessing helpers can be unit-tested on machines without Ultralytics or
Tesseract installed.
"""

from __future__ import annotations

__all__ = ["detect_fields", "ocr", "preprocess"]
