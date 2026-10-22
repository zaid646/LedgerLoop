# LedgerLoop CV: receipt preprocessing, field detection and OCR

This module adds a CPU-only classical-CV path to LedgerLoop and compares it
with the existing LLM extractor. It does **not** replace the LLM pipeline; it
documents where a detector + OCR combo helps and where it does not.

## What runs

| Stage | File | Output |
|---|---|---|
| OpenCV preprocess | `cv/preprocess.py` | grayscale → denoise → adaptive threshold → deskew → contour crop |
| Field detection | `cv/detect_fields.py` | fine-tuned YOLOv8n boxes for `merchant`, `date`, `total` |
| OCR | `cv/ocr.py` | Tesseract text + regex field parser |
| Comparison | `cv/evaluate.py` | `cv/results/ocr_vs_llm.json` |
| Synthetic data | `cv/synthetic.py` | deterministic 200-image YOLO set + 75-page OCR set |

## Setup

```bash
# Python 3.12; CPU is enough.
uv venv .venv-cv --python 3.12
uv pip install --python .venv-cv/bin/python -r requirements-cv.txt

# Tesseract binary (not a Python package)
brew install tesseract          # macOS
sudo apt-get install -y tesseract-ocr   # Ubuntu
```

## Reproduce

```bash
# 1. Render the document-disjoint YOLO set (150 train / 50 val images)
.venv-cv/bin/python -m cv.synthetic

# 2. Fine-tune YOLOv8n on CPU (~5 min on an Apple M5, 25 epochs, 416 px)
.venv-cv/bin/python -m cv.train_yolo > cv/results/yolo_train.log 2>&1

# 3. OCR / LLM comparison on all 75 golden documents
.venv-cv/bin/python -m cv.evaluate > cv/results/ocr_eval.log 2>&1

# 4. Classical-ML GL-coding baseline
.venv-cv/bin/python -m ml.gl_lightgbm
```

## Dataset: synthetic, deterministic, document-disjoint

The repository does not ship receipt photographs. We therefore render the 75
frozen texts in `evals/golden/v0.jsonl` (50 SROIE + 25 CORD) with a deterministic
Pillow/OpenCV renderer.

* 50 SROIE documents → 3 augmented images each = **150 training images**
* 25 CORD documents → 2 augmented images each = **50 held-out images**
* Augmentations: layout jitter, font-size jitter, ±1.8° rotation, light noise/blur
* The split is **document-disjoint and source-disjoint**: training is all SROIE,
  validation is all CORD. Augmented copies of a validation document never
  appear in training.
* Both halves use the same deterministic renderer, so the metrics measure
  cross-source generalisation of the synthetic layout — not phone-photo
  robustness.
* Seed 42; `cv/data/manifest.json` records the golden-file SHA-256 and doc ids.

The 75 OCR pages are a separate render (`variant=100`), one page per document.

## YOLO: fine-tuned, not zero-shot

A stock COCO YOLOv8n has no `merchant` / `date` / `total` classes, so zero-shot
detection of receipt fields is not meaningful. The committed checkpoint is
**fine-tuned** on the synthetic set above.

Held-out validation (50 images, 150 instances):

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| all | 0.9977 | 1.0000 | **0.9950** | 0.8066 |
| merchant | 0.9963 | 1.0000 | 0.9950 | 0.8121 |
| date | 0.9985 | 1.0000 | 0.9950 | 0.8260 |
| total | 0.9981 | 1.0000 | 0.9950 | 0.7817 |

Source: `cv/results/yolo_metrics.json`; weights:
`cv/weights/receipt_fields_yolov8n.pt` (6.5 MB). Training log:
`cv/results/yolo_train.log`.

**Caveat:** these images are rendered text with consistent layout. mAP is an
upper bound and is not evidence of phone-photo performance.

## OCR vs LLM vs OCR+LLM (75 documents)

Field accuracy over fields present in the golden labels (merchant 50 docs,
total 75, date 50). Exact-document match requires all present fields to match.

| Method | Merchant | Total | Date | Exact doc |
|---|---:|---:|---:|---:|
| OCR-only (full page + regex) | 1.0000 | 1.0000 | 0.8200 | 0.8800 |
| OCR+YOLO (crop per field) | 0.9400 | 1.0000 | 0.8200 | 0.8400 |
| LLM-only (cached Qwen2.5-7B) | 0.9600 | 0.6267 | 0.8400 | 0.4933 |
| **OCR+LLM fusion** | 0.9600 | **1.0000** | **0.8400** | 0.8667 |

Source: `cv/results/ocr_vs_llm.json`. LLM predictions are replayed from
`evals/results/llm-qwen7b-v0.json`; no new LLM calls were made.

Fusion policy (fixed before scoring): merchant from the LLM when present, total
and date from detector+OCR when present, falling back to full-page OCR and then
the LLM. On these clean renders OCR-only slightly beats fusion on exact-document
match because full-page OCR reads every merchant perfectly; fusion still
recovers 36 percentage points of exact-match over LLM-only and fixes LLM's
total-amount errors.

## Classical-ML GL coding

See [`ml/README.md`](../ml/README.md). LightGBM reaches 0.4000 accuracy /
0.4039 macro-F1 on the same 30 held-out GL instructions, versus **0.8889 ±
0.0192** exact match for the Qwen2.5-1.5B QLoRA adapter (committed multi-seed
result). The classical baseline is honest but far behind the LLM on this task.

## Caveats

* Synthetic text renders are cleaner than real receipts; OCR and YOLO numbers
  will drop on photos, skew, glare and crumpled paper.
* The cached LLM is Qwen2.5-7B, not the QLoRA 1.5B adapter (the adapter was
  trained for GL coding, not field extraction).
* No real SROIE/CORD images were downloaded; only their public text is used.
* Detection is CPU-only and trained for 25 epochs; no paid GPU or Vast.ai.
