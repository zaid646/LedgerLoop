"""Field-level matching + normalization for receipt extraction evaluation."""
from __future__ import annotations

import re
from datetime import date, datetime

_WORD_RE = re.compile(r"[a-z0-9]+")
_DATE_FMTS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%Y",
)


def norm_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(_WORD_RE.findall(str(value).lower()))


def merchant_match(pred: str | None, gold: str | None) -> bool:
    p, g = norm_text(pred), norm_text(gold)
    if not g:
        return True
    if not p:
        return False
    if p == g or p in g or g in p:
        return True
    pt, gt = set(p.split()), set(g.split())
    if not gt:
        return True
    return len(pt & gt) / len(gt) >= 0.7


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    s = str(value).strip()
    m = re.search(r"(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})", s)
    if m:
        s = m.group(0)
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def date_match(pred: str | None, gold: str | None) -> bool:
    if not gold:
        return True
    g = parse_date(gold)
    if g is None:
        return norm_text(pred) == norm_text(gold)
    p = parse_date(pred)
    return p == g


def amount_match(pred: float | None, gold: float | None, tol: float = 0.01) -> bool:
    if gold is None:
        return True
    if pred is None:
        return False
    return abs(float(pred) - float(gold)) <= tol


def field_scores(pred: dict, gold: dict) -> dict[str, bool]:
    return {
        "merchant": merchant_match(pred.get("merchant"), gold.get("merchant")),
        "total_amount": amount_match(_as_float(pred.get("total_amount")), gold.get("total_amount")),
        "transaction_date": date_match(pred.get("transaction_date"), gold.get("transaction_date")),
    }


def _as_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
