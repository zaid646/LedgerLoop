"""Deterministic company-policy evaluation.

Kept free of FastAPI imports so the graph builder, catalog and offline ingestion
workers can use the same rules without pulling in the web stack.
"""

from __future__ import annotations

from .schemas import (
    GLCategory,
    PolicyCheckRequest,
    PolicyCheckResponse,
    PolicyRule,
    PolicyViolation,
)

DEFAULT_RULES: list[PolicyRule] = [
    PolicyRule(
        name="meals_limit", applies_to=GLCategory.MEALS, max_amount=75.0, requires_receipt=True
    ),
    PolicyRule(name="meals_manager", applies_to=GLCategory.MEALS, requires_manager_above=200.0),
    PolicyRule(name="travel_receipt", applies_to=GLCategory.TRAVEL, requires_receipt=True),
    PolicyRule(name="travel_manager", applies_to=GLCategory.TRAVEL, requires_manager_above=500.0),
    PolicyRule(name="entertainment_limit", applies_to=GLCategory.ENTERTAINMENT, max_amount=200.0),
    PolicyRule(
        name="entertainment_manager",
        applies_to=GLCategory.ENTERTAINMENT,
        requires_manager_above=100.0,
    ),
    PolicyRule(
        name="software_manager", applies_to=GLCategory.SOFTWARE, requires_manager_above=1000.0
    ),
    PolicyRule(name="global_manager", requires_manager_above=2500.0),
]


def _rules_for(category: GLCategory) -> list[PolicyRule]:
    return [r for r in DEFAULT_RULES if r.applies_to is None or r.applies_to == category]


def evaluate(req: PolicyCheckRequest) -> PolicyCheckResponse:
    rules = req.rules or _rules_for(req.category)
    violations: list[PolicyViolation] = []
    hard = False
    manager = False

    for r in rules:
        if r.applies_to is not None and r.applies_to != req.category:
            continue
        if r.requires_receipt and not req.has_receipt:
            violations.append(PolicyViolation(rule=r.name, detail="Receipt is required"))
            hard = True
        if r.max_amount is not None and req.amount > r.max_amount:
            violations.append(
                PolicyViolation(
                    rule=r.name, detail=f"Amount {req.amount:.2f} exceeds limit {r.max_amount:.2f}"
                )
            )
        if r.requires_manager_above is not None and req.amount > r.requires_manager_above:
            manager = True

    if hard:
        route = "human_queue"
    elif violations or manager:
        route = "manager_review"
    else:
        route = "auto_approve"

    return PolicyCheckResponse(compliant=not violations, route=route, violations=violations)
