from app.routers.policy import evaluate
from app.schemas import GLCategory, PolicyCheckRequest


def _req(amount, category, has_receipt=True):
    return PolicyCheckRequest(
        amount=amount, category=category, has_receipt=has_receipt, rules=[]
    )


def test_small_meal_auto_approves():
    res = evaluate(_req(20, GLCategory.MEALS))
    assert res.route == "auto_approve"
    assert res.compliant


def test_meal_over_limit_routes_to_manager():
    res = evaluate(_req(150, GLCategory.MEALS))
    assert res.route == "manager_review"
    assert not res.compliant


def test_missing_receipt_routes_to_human():
    res = evaluate(_req(20, GLCategory.TRAVEL, has_receipt=False))
    assert res.route == "human_queue"


def test_large_travel_routes_to_manager():
    res = evaluate(_req(900, GLCategory.TRAVEL))
    assert res.route == "manager_review"
