"""Deterministic synthetic receipt renderer + YOLO dataset builder.

The repository does not ship receipt photographs, so the CV benchmark is built
from the 75 receipt texts in ``evals/golden/v0.jsonl``.  Rendering the *text*
keeps the experiment fully reproducible and license-free, but it also makes the
images cleaner than real phone photos -- see ``cv/README.md`` for caveats.

Each rendered page exposes ground-truth boxes for three fields:
``merchant``, ``date`` and ``total``.  Splits are made by *document id*, so a
validation document never appears in training, even though several augmented
images are rendered per document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evals" / "golden" / "v0.jsonl"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLASS_NAMES = ("merchant", "date", "total")

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
)


@dataclass
class Rendered:
    image: np.ndarray
    boxes: dict[str, tuple[int, int, int, int]]
    doc_id: str


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = list(_FONT_CANDIDATES)
    if bold:
        candidates.insert(0, "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        candidates.insert(1, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def load_golden(path: Path = GOLDEN) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty golden set: {path}")
    return rows


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _merge(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)
    return left, top, right, bottom


def _pad(box: tuple[int, int, int, int], pad: int, width: int, height: int):
    left, top, right, bottom = box
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width - 1, right + pad),
        min(height - 1, bottom + pad),
    )


def _rotate_with_boxes(image: np.ndarray, boxes: dict[str, tuple[int, int, int, int]], angle: float):
    if abs(angle) < 0.05:
        return image, boxes
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - w / 2.0
    matrix[1, 2] += new_h / 2.0 - h / 2.0
    rotated = cv2.warpAffine(
        image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    transformed: dict[str, tuple[int, int, int, int]] = {}
    for name, (left, top, right, bottom) in boxes.items():
        corners = np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32
        )
        ones = np.ones((4, 1), dtype=np.float32)
        points = np.hstack([corners, ones]) @ matrix.T
        xs, ys = points[:, 0], points[:, 1]
        transformed[name] = (
            int(max(0, round(float(xs.min())))),
            int(max(0, round(float(ys.min())))),
            int(min(new_w - 1, round(float(xs.max())))),
            int(min(new_h - 1, round(float(ys.max())))),
        )
    return rotated, transformed


def render_document(
    record: dict,
    *,
    variant: int = 0,
    seed: int = 42,
    size: tuple[int, int] = (900, 1300),
    noise: float = 4.0,
) -> Rendered:
    """Render one golden record to a synthetic receipt image.

    ``variant`` changes layout jitter, font scale, rotation and noise while
    keeping the same document text.  The same ``(doc_id, variant, seed)`` always
    produces byte-identical pixels on a given Pillow/OpenCV version.
    """
    width, height = size
    doc_id = str(record["id"])
    rng = np.random.default_rng(seed + variant * 1009 + sum(map(ord, doc_id)))
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    scale = 1.0 + float(rng.uniform(-0.06, 0.06))
    body = _font(max(14, int(24 * scale)))
    small = _font(max(12, int(20 * scale)))
    title = _font(max(20, int(38 * scale)), bold=True)
    total_font = _font(max(20, int(34 * scale)), bold=True)

    margin = int(width * 0.07) + int(rng.integers(-8, 9))
    max_text_width = width - 2 * margin
    y = int(height * 0.03) + int(rng.integers(0, 14))
    fields = record.get("fields") or {}
    merchant = str(fields.get("merchant") or "UNKNOWN MERCHANT")
    address = str(fields.get("address") or "123 MARKET STREET")
    tx_date = str(fields.get("transaction_date") or "2026-01-01")
    total = float(fields.get("total_amount") or 0.0)

    merchant_line = merchant.upper()
    title_size = max(20, int(38 * scale))
    while title_size > 16 and draw.textlength(merchant_line, font=title) > max_text_width:
        title_size -= 1
        title = _font(title_size, bold=True)
    merchant_boxes = []
    bbox = draw.textbbox((margin, y), merchant_line, font=title)
    draw.text((margin, y), merchant_line, font=title, fill=(15, 15, 15))
    merchant_boxes.append(bbox)
    y = bbox[3] + 6

    for line in _wrap(draw, address, small, max_text_width)[:2]:
        bbox = draw.textbbox((margin, y), line, font=small)
        draw.text((margin, y), line, font=small, fill=(55, 55, 55))
        y = bbox[3] + 3

    y += int(rng.integers(6, 14))
    draw.line((margin, y, width - margin, y), fill=(90, 90, 90), width=2)
    y += 12

    date_text = f"DATE: {tx_date}"
    date_bbox = draw.textbbox((margin, y), date_text, font=body)
    draw.text((margin, y), date_text, font=body, fill=(20, 20, 20))
    y = date_bbox[3] + 4
    receipt_no = sum(map(ord, doc_id)) % 100000
    draw.text((margin, y), f"RECEIPT NO: R-{receipt_no:05d}", font=small,
              fill=(70, 70, 70))
    y += int(rng.integers(12, 24))

    items = [
        ("PAPER TOWELS 12PK", 16.80),
        ("SPRING WATER 12PK", 9.49),
        ("KETTLE CHIPS 6PK", 12.75),
        ("GREEK YOGURT 1KG", 13.40),
        ("BATTERIES AA 4PK", 7.99),
        ("SHIPPING BOX", 4.25),
    ]
    n_items = int(rng.integers(3, 6))
    for name, price in items[:n_items]:
        line = f"{name}  {price:>6.2f}"
        draw.text((margin, y), line, font=body, fill=(25, 25, 25))
        y += int(28 * scale)
    y += int(rng.integers(6, 14))

    subtotal = round(sum(p for _, p in items[:n_items]), 2)
    draw.text((margin, y), f"SUBTOTAL  {subtotal:>8.2f}", font=small, fill=(60, 60, 60))
    y += int(26 * scale)
    draw.text((margin, y), "TAX  0.00", font=small, fill=(60, 60, 60))
    y += int(34 * scale)

    total_text = f"TOTAL: {total:,.2f}"
    total_bbox = draw.textbbox((margin, y), total_text, font=total_font)
    draw.text((margin, y), total_text, font=total_font, fill=(5, 5, 5))
    y = total_bbox[3] + int(24 * scale)

    draw.text((margin, y), "THANK YOU", font=small, fill=(70, 70, 70))
    y += int(24 * scale)
    draw.text((margin, y), "GOODS SOLD ARE NOT RETURNABLE", font=small, fill=(110, 110, 110))

    boxes = {
        "merchant": _merge(merchant_boxes),
        "date": date_bbox,
        "total": total_bbox,
    }
    boxes = {name: _pad(box, 6, width, height) for name, box in boxes.items()}

    image = np.array(canvas)
    if rng.random() < 0.5:
        background = int(rng.integers(243, 256))
        image = np.clip(image.astype(np.int16) + (background - 255), 0, 255).astype(np.uint8)

    angle = float(rng.uniform(-1.8, 1.8))
    image, boxes = _rotate_with_boxes(image, boxes, angle)

    if noise > 0:
        sigma = noise * float(rng.uniform(0.4, 1.0))
        image = np.clip(
            image.astype(np.float32) + rng.normal(0.0, sigma, image.shape), 0, 255
        ).astype(np.uint8)
    if rng.random() < 0.35:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return Rendered(image=image, boxes=boxes, doc_id=doc_id)


def _yolo_lines(boxes: dict[str, tuple[int, int, int, int]], width: int, height: int) -> list[str]:
    lines = []
    for class_id, name in enumerate(CLASS_NAMES):
        left, top, right, bottom = boxes[name]
        cx = (left + right) / 2.0 / width
        cy = (top + bottom) / 2.0 / height
        bw = max(1.0, right - left) / width
        bh = max(1.0, bottom - top) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def build_dataset(
    out_dir: Path = DATA_DIR,
    *,
    seed: int = 42,
    train_docs: int = 50,
    train_variants: int = 3,
    val_variants: int = 2,
) -> dict:
    records = load_golden()
    if train_docs <= 0 or train_docs >= len(records):
        raise ValueError("train_docs must leave at least one validation document")
    train_records = records[:train_docs]
    val_records = records[train_docs:]

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for split, docs, variants in (
        ("train", train_records, train_variants),
        ("val", val_records, val_variants),
    ):
        for doc_index, record in enumerate(docs):
            for variant in range(variants):
                rendered = render_document(
                    record,
                    variant=variant,
                    seed=seed + doc_index * 17 + (0 if split == "train" else 5000),
                )
                stem = f"{record['id']}-v{variant}"
                image_path = out_dir / "images" / split / f"{stem}.jpg"
                label_path = out_dir / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(image_path), rendered.image, [cv2.IMWRITE_JPEG_QUALITY, 94])
                h, w = rendered.image.shape[:2]
                label_path.write_text(
                    "\n".join(_yolo_lines(rendered.boxes, w, h)) + "\n", encoding="utf-8"
                )
                counts[split] += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "path: {}\ntrain: images/train\nval: images/val\nnames:\n{}\n".format(
            out_dir.resolve(),
            "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)),
        ),
        encoding="utf-8",
    )

    manifest = {
        "seed": seed,
        "renderer": "synthetic-receipt-v1",
        "source": "evals/golden/v0.jsonl",
        "golden_sha256": hashlib.sha256(GOLDEN.read_bytes()).hexdigest(),
        "train_docs": len(train_records),
        "val_docs": len(val_records),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "train_doc_ids": [r["id"] for r in train_records],
        "val_doc_ids": [r["id"] for r in val_records],
        "class_names": list(CLASS_NAMES),
        "image_size": [900, 1300],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def render_ocr_set(out_dir: Path = DATA_DIR / "ocr", seed: int = 42) -> list[dict]:
    """Render one image per golden document for the OCR comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(load_golden()):
        rendered = render_document(
            record,
            variant=100,
            seed=seed + index,
            size=(1000, 1500),
            noise=2.5,
        )
        path = out_dir / f"{record['id']}.png"
        cv2.imwrite(str(path), rendered.image)
        manifest.append({"id": record["id"], "path": str(path)})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic receipt YOLO dataset")
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_dataset(args.out, seed=args.seed)
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("doc_ids")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
