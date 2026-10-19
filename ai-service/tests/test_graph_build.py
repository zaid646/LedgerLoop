from app.graph.build import UNKNOWN_MERCHANT, build_graph
from app.graph.model import (
    CODED_AS,
    EXPENSE,
    MERCHANT,
    PAID_TO,
    category_id,
    date_bucket,
    expense_id,
    merchant_id,
)


def test_build_counts_and_shapes():
    graph = build_graph()
    assert graph.stats["entities"] == 175
    assert graph.stats["relationships"] == 462
    assert graph.stats["by_type"] == {
        "Category": 13,
        "Policy": 8,
        "Merchant": 63,
        "Source": 2,
        "Date": 11,
        "Route": 3,
        "Expense": 75,
    }


def test_expense_is_linked_to_merchant_category_and_route():
    graph = build_graph()
    ids = {e.id for e in graph.entities}
    assert expense_id("sroie-21") in ids
    assert merchant_id("TEO HENG STATIONERY & BOOKS") in ids

    rels = {(r.source, r.type, r.target) for r in graph.relationships}
    target = merchant_id("TEO HENG STATIONERY & BOOKS")
    assert (expense_id("sroie-21"), PAID_TO, target) in rels
    assert (
        expense_id("sroie-21"),
        CODED_AS,
        category_id("OFFICE_SUPPLIES"),
    ) in rels
    assert any(r.source == expense_id("sroie-21") and r.type == "ROUTED_TO" for r in graph.relationships)


def test_unnamed_cord_receipts_use_one_unknown_merchant():
    graph = build_graph()
    paid_to_unknown = [
        r
        for r in graph.relationships
        if r.type == PAID_TO and r.target == UNKNOWN_MERCHANT
    ]
    assert len(paid_to_unknown) == 25
    unknown = {e.id: e for e in graph.entities}[UNKNOWN_MERCHANT]
    assert unknown.type == MERCHANT
    assert unknown.properties["category"] == "OTHER"


def test_build_is_deterministic():
    first = build_graph()
    second = build_graph()
    assert [e.id for e in first.entities] == [e.id for e in second.entities]
    assert [(r.source, r.type, r.target) for r in first.relationships] == [
        (r.source, r.type, r.target) for r in second.relationships
    ]


def test_expense_nodes_carry_coding_fields():
    graph = build_graph()
    expense = {e.id: e for e in graph.entities}[expense_id("sroie-21")]
    assert expense.type == EXPENSE
    assert expense.properties["category"] == "OFFICE_SUPPLIES"
    assert expense.properties["total_amount"] == 4.9


def test_date_bucket_normalisation():
    assert date_bucket("25/12/2018 8:13:39 PM") == "2018-12"
    assert date_bucket("2019-01-31") == "2019-01"
    assert date_bucket(None) == "unknown"
    assert date_bucket("not a date") == "unknown"
