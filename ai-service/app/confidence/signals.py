"""Independent signals used by the confidence scorer.

The signals are deliberately cheap and deterministic so they can run on CPU in
the request path:

* ``llm_confidence``      self-reported per-field confidence (fallback when the
                          serving stack does not expose token logprobs)
* ``logprob_confidence``  optional exp(mean token logprob) when available
* ``schema_ok``           the extraction passed Pydantic validation
* ``repair``              1 / (1 + self-heal attempts beyond the first)
* ``retrieval_agreement`` the extracted value is supported by an evidence
                          candidate retrieved from the source text
* ``policy``              deterministic policy-check outcome
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..catalog import infer_category
from ..policy import evaluate
from ..schemas import PolicyCheckRequest, ReceiptExtraction
from .config import ConfidenceConfig

_AMOUNT_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_DATE_RE = re.compile(r"(?<!\d)(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})(?!\d)")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class FieldSignals:
    llm_confidence: float
    logprob_confidence: float | None
    schema_ok: float
    repair: float
    retrieval_agreement: float
    policy: float
    present: bool


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(str(text).lower()))


def _normalise_date(value: str) -> tuple[int, int, int] | None:
    match = _DATE_RE.search(value)
    if not match:
        return None
    first, middle, last = match.groups()
    if len(first) == 4:
        year, month, day = first, middle, last
    else:
        day, month, year = first, middle, last
    if len(year) == 2:
        year = "20" + year if int(year) < 70 else "19" + year
    try:
        return int(year), int(month), int(day)
    except ValueError:
        return None


def _amount_candidates(text: str) -> list[float]:
    values: list[float] = []
    for raw in _AMOUNT_RE.findall(text.replace(",", "")):
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def retrieval_agreement(field: str, value: object, text: str, *, min_overlap: float = 0.60) -> float:
    """Return 0..1 support for ``value`` in the source text.

    Amounts are matched numerically, dates by calendar value and merchant names
    by token overlap. A missing value can never be grounded, which is what makes
    this signal useful for catching silent nulls.
    """
    if value in (None, "", []):
        return 0.0
    if field == "total_amount":
        try:
            amount = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if any(abs(candidate - amount) <= 0.011 for candidate in _amount_candidates(text)) else 0.0
    if field == "transaction_date":
        wanted = _normalise_date(str(value))
        if wanted is None:
            return 0.0
        return 1.0 if any(_normalise_date(m.group(0)) == wanted for m in _DATE_RE.finditer(text)) else 0.0
    wanted_tokens = _tokens(str(value))
    if not wanted_tokens:
        return 0.0
    overlap = len(wanted_tokens & _tokens(text)) / len(wanted_tokens)
    return 1.0 if overlap >= min_overlap else round(overlap, 4)


def policy_signal(route: str) -> float:
    return {"auto_approve": 1.0, "manager_review": 0.6, "human_queue": 0.2}.get(route, 0.2)


def infer_policy_route(extraction: ReceiptExtraction, *, source: str = "receipt") -> tuple[str, float]:
    """Evaluate the deterministic policy rules using the extracted fields."""
    amount = extraction.total_amount
    if amount is None:
        return "manager_review", policy_signal("manager_review")
    category = infer_category(extraction.merchant or "")
    try:
        result = evaluate(
            PolicyCheckRequest(
                amount=float(amount),
                category=category,
                merchant=extraction.merchant,
                has_receipt=source == "receipt",
            )
        )
    except Exception:  # noqa: BLE001 - confidence must never break extraction
        return "manager_review", policy_signal("manager_review")
    return result.route, policy_signal(result.route)


def build_field_signals(
    extraction: ReceiptExtraction,
    text: str,
    *,
    attempts: int,
    schema_ok: bool,
    policy_route: str,
    policy_score: float,
    config: ConfidenceConfig,
    logprobs: dict[str, float] | None = None,
) -> dict[str, FieldSignals]:
    repair = 1.0 / (1.0 + max(0, attempts - 1))
    reported = extraction.confidence or {}
    signals: dict[str, FieldSignals] = {}
    for field_name in config.fields:
        value = getattr(extraction, field_name, None)
        present = value not in (None, "", [])
        llm_conf = reported.get(field_name, 0.0)
        try:
            llm_conf = _clamp(float(llm_conf or 0.0))
        except (TypeError, ValueError):
            llm_conf = 0.0
        logprob = None
        if logprobs and field_name in logprobs:
            logprob = _clamp(math.exp(float(logprobs[field_name])))
        signals[field_name] = FieldSignals(
            llm_confidence=llm_conf,
            logprob_confidence=logprob,
            schema_ok=1.0 if schema_ok else 0.0,
            repair=repair,
            retrieval_agreement=retrieval_agreement(
                field_name,
                value,
                text,
                min_overlap=config.retrieval.merchant_token_overlap,
            ),
            policy=policy_score,
            present=present,
        )
    return signals
