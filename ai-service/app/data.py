"""Dataset loaders for receipt extraction evaluation.

Primary: **SROIE** (`arvindrajan92/sroie_document_understanding`) — real OCR text
with per-line labels (company / date / address / total).
Secondary: **CORD v2** (`naver-clova-ix/cord-v2`) — menu items + total.

Both are streamed from the Hugging Face Hub so the full corpus is never
materialised. Records are normalised to :class:`Record` for evaluation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

SROIE_DATASET = "arvindrajan92/sroie_document_understanding"
CORD_DATASET = "naver-clova-ix/cord-v2"

_AMOUNT_RE = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class Record:
    id: str
    source: str
    text: str
    fields: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_golden(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "text": self.text, "fields": self.fields}


def parse_amount(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _AMOUNT_RE.search(str(value).replace(",", ""))
    return float(m.group()) if m else None


# --------------------------------------------------------------------------- SROIE


def _sroie_ocr_to_parts(ocr: Any) -> tuple[dict[str, Any], str]:
    """`ocr` is a list of {label, text, box}; returns (fields, text)."""
    fields: dict[str, Any] = {}
    lines: list[str] = []
    if not isinstance(ocr, list):
        return fields, ""
    for e in ocr:
        if not isinstance(e, dict):
            continue
        text = e.get("text")
        if text:
            lines.append(str(text))
        label = str(e.get("label", "")).lower()
        if label in ("company", "merchant", "store"):
            fields["merchant"] = text
        elif label in ("date", "transaction_date"):
            fields["transaction_date"] = text
        elif label in ("total", "total_amount"):
            fields["total_amount"] = parse_amount(text)
        elif label == "address":
            fields["address"] = text
    return fields, "\n".join(lines)


def load_sroie(limit: int = 200, split: str = "train") -> list[Record]:
    from datasets import load_dataset

    ds = load_dataset(SROIE_DATASET, split=split, streaming=True)
    records: list[Record] = []
    for i, ex in enumerate(ds):
        if len(records) >= limit:
            break
        fields, text = _sroie_ocr_to_parts(ex.get("ocr"))
        if not text:
            continue
        records.append(
            Record(id=f"sroie-{i}", source="sroie", text=text, fields=fields, raw={"n_ocr": len(ex.get("ocr", []))})
        )
    return records


# ---------------------------------------------------------------------------- CORD


def _cord_text_and_fields(gt: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            return "", {}
    if not isinstance(gt, dict):
        return "", {}
    parse = gt.get("gt_parse", gt)

    lines: list[str] = []
    merchant = None
    store = parse.get("store_info")
    if isinstance(store, dict):
        merchant = store.get("name") or store.get("store_name")
    if merchant:
        lines.append(str(merchant))

    menu = parse.get("menu", [])
    items = menu if isinstance(menu, list) else [menu]
    for item in items:
        if isinstance(item, dict):
            nm = item.get("nm") or item.get("name")
            price = item.get("price") or item.get("itemsubtotal")
            if nm:
                lines.append(f"{nm} {price if price is not None else ''}".strip())

    total = parse.get("total")
    total_amount = None
    if isinstance(total, dict):
        total_amount = parse_amount(
            total.get("total_price") or total.get("cashprice") or total.get("total")
        )
    sub = parse.get("sub_total")
    if total_amount is None and isinstance(sub, dict):
        total_amount = parse_amount(sub.get("subtotal_price"))

    return "\n".join(lines), {
        "merchant": merchant or "",
        "total_amount": total_amount,
        "transaction_date": None,
    }


def load_cord(limit: int = 200, split: str = "test") -> list[Record]:
    from datasets import load_dataset

    ds = load_dataset(CORD_DATASET, split=split, streaming=True)
    records: list[Record] = []
    for i, ex in enumerate(ds):
        if len(records) >= limit:
            break
        text, fields = _cord_text_and_fields(ex.get("ground_truth"))
        if not text:
            continue
        records.append(Record(id=f"cord-{i}", source="cord", text=text, fields=fields, raw={}))
    return records


def load(limit: int = 200, sources: tuple[str, ...] = ("sroie", "cord")) -> list[Record]:
    out: list[Record] = []
    if "sroie" in sources:
        out.extend(load_sroie(limit))
    if "cord" in sources:
        out.extend(load_cord(limit))
    return out
