"""PII / card-number redaction applied before every LLM call.

Adapted from the **Fortress** guardrail middleware (agent-fortress): a fast,
deterministic, regex-first layer plus a Luhn check for card numbers so we don't
redact arbitrary long digit strings. Runs in <1 ms and is fail-safe: callers
redact unconditionally.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PLACEHOLDER = {
    "card": "[REDACTED_CARD]",
    "ssn": "[REDACTED_SSN]",
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "iban": "[REDACTED_IBAN]",
}

_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)"
)
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _luhn(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_cards(text: str) -> tuple[str, int]:
    hits = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal hits
        digits = re.sub(r"\D", "", m.group(0))
        if _luhn(digits):
            hits += 1
            return PLACEHOLDER["card"]
        return m.group(0)

    return _CARD_RE.sub(repl, text), hits


def redact(text: str) -> RedactionResult:
    """Scrub card numbers, SSNs, emails, phones and IBANs from free text."""
    counts: dict[str, int] = {}

    text, counts["card"] = _redact_cards(text)

    for key, pattern in (
        ("ssn", _SSN_RE),
        ("email", _EMAIL_RE),
        ("iban", _IBAN_RE),
        ("phone", _PHONE_RE),
    ):
        text, n = pattern.subn(PLACEHOLDER[key], text)
        counts[key] = n

    counts = {k: v for k, v in counts.items() if v}
    return RedactionResult(text=text, counts=counts)
