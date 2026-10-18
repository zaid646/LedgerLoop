"""Pydantic contracts for the ExpenseIQ AI service."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- GL


class GLCategory(StrEnum):
    TRAVEL = "TRAVEL"
    MEALS = "MEALS"
    OFFICE_SUPPLIES = "OFFICE_SUPPLIES"
    SOFTWARE = "SOFTWARE"
    HARDWARE = "HARDWARE"
    MARKETING = "MARKETING"
    PROFESSIONAL_SERVICES = "PROFESSIONAL_SERVICES"
    UTILITIES = "UTILITIES"
    RENT = "RENT"
    INSURANCE = "INSURANCE"
    TRAINING = "TRAINING"
    ENTERTAINMENT = "ENTERTAINMENT"
    OTHER = "OTHER"


# ----------------------------------------------------------------- extraction


class LineItem(BaseModel):
    description: str = ""
    amount: float | None = None


class ReceiptExtraction(BaseModel):
    """Structured receipt fields. Used as the LLM's guided JSON schema."""

    merchant: str = Field(description="Merchant or vendor name")
    transaction_date: str | None = Field(
        default=None, description="Transaction date as printed on the receipt (e.g. 25/12/2018 or 2024-01-31)"
    )
    total_amount: float | None = Field(default=None, description="Grand total")
    tax_amount: float | None = Field(default=None, description="Tax/VAT amount if present")
    currency: str | None = Field(default=None, description="ISO-4217 currency code")
    payment_method: str | None = None
    receipt_number: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    confidence: dict[str, float] = Field(
        default_factory=dict, description="Per-field confidence 0..1"
    )


class ExtractRequest(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1, description="Redacted OCR / receipt text")
    source: Literal["receipt", "card_transaction"] = "receipt"


class ExtractResponse(BaseModel):
    document_id: str
    extraction: ReceiptExtraction
    prompt_version: str
    model: str
    attempts: int
    repaired: bool
    escalated: bool
    latency_ms: float
    cost_usd: float
    redactions: dict[str, int] = Field(default_factory=dict)


# ----------------------------------------------------------------------- coding


class CodeRequest(BaseModel):
    description: str = Field(min_length=1)
    merchant: str | None = None
    amount: float | None = None


class CodeResponse(BaseModel):
    category: GLCategory
    confidence: float
    rationale: str = ""
    candidates: list[GLCategory] = Field(default_factory=list)
    prompt_version: str
    model: str
    latency_ms: float
    cost_usd: float
    top_merchants: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------- policy


class PolicyRule(BaseModel):
    name: str
    applies_to: GLCategory | None = None
    max_amount: float | None = None
    requires_receipt: bool = False
    requires_manager_above: float | None = None


class PolicyCheckRequest(BaseModel):
    amount: float
    category: GLCategory
    merchant: str | None = None
    has_receipt: bool = True
    rules: list[PolicyRule] = Field(default_factory=list)


class PolicyViolation(BaseModel):
    rule: str
    detail: str


class PolicyCheckResponse(BaseModel):
    compliant: bool
    route: Literal["auto_approve", "manager_review", "human_queue"]
    violations: list[PolicyViolation] = Field(default_factory=list)


# ------------------------------------------------------------------------ misc


class HealthResponse(BaseModel):
    status: str
    llm_reachable: bool
    llm_model: str
    langfuse: bool
    version: str


class CorrectionRequest(BaseModel):
    document_id: str
    corrected: ReceiptExtraction
    operator: str = "unknown"


class CorrectionResponse(BaseModel):
    document_id: str
    appended_to_golden: bool
    golden_path: str


class ErrorResponse(BaseModel):
    detail: str
    context: dict[str, Any] | None = None
