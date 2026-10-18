"""Zero-cost heuristic receipt extractor — the baseline to beat."""
from __future__ import annotations

import re

_DATE_RE = re.compile(r"\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b")
_AMOUNT_RE = re.compile(r"-?\d[\d,]*\.\d{2}\b")


def _amounts(line: str) -> list[float]:
    out = []
    for m in _AMOUNT_RE.findall(line):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def heuristic_extract(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    merchant = lines[0] if lines else ""

    tx_date = None
    for ln in lines:
        m = _DATE_RE.search(ln)
        if m:
            tx_date = m.group(0)
            break

    total = None
    for ln in lines:
        if "total" in ln.lower():
            amts = _amounts(ln)
            if amts:
                total = amts[-1]
    if total is None:
        all_amts = [a for ln in lines for a in _amounts(ln)]
        if all_amts:
            total = max(all_amts)

    conf = {
        "merchant": 0.3 if merchant else 0.0,
        "total_amount": 0.5 if total is not None else 0.0,
        "transaction_date": 0.6 if tx_date else 0.0,
    }
    return {
        "merchant": merchant,
        "transaction_date": tx_date,
        "total_amount": total,
        "currency": None,
        "confidence": conf,
    }
