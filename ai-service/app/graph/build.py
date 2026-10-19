"""Deterministic graph construction from the golden set and the shared catalog.

The same function is used by the CLI (writes to Neo4j) and by tests (in-memory),
so entity/relationship counts and ids are reproducible.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..catalog import infer_category, merchant_docs
from ..config import REPO_ROOT
from ..policy import DEFAULT_RULES, evaluate
from ..schemas import GLCategory, PolicyCheckRequest
from .model import (
    CATEGORY,
    CODED_AS,
    DATE,
    EXPENSE,
    FROM_SOURCE,
    HAS_POLICY,
    MERCHANT,
    MERCHANT_CATEGORY,
    ON_DATE,
    PAID_TO,
    POLICY,
    ROUTE,
    ROUTED_TO,
    SOURCE,
    VIOLATES,
    Entity,
    Relationship,
    category_id,
    date_bucket,
    date_id,
    expense_id,
    merchant_id,
    policy_id,
    route_id,
    source_id,
)
from .repo import GraphRepo

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"
ROUTES = ("auto_approve", "manager_review", "human_queue")
UNKNOWN_MERCHANT = merchant_id("unknown")


@dataclass
class GraphBuild:
    entities: list[Entity]
    relationships: list[Relationship]
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats,
            "entities": [
                {"id": e.id, "type": e.type, "label": e.label, "properties": e.properties}
                for e in self.entities
            ],
            "relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.type,
                    "properties": r.properties,
                }
                for r in self.relationships
            ],
        }


def load_golden(version: str = "v0") -> list[dict]:
    path = GOLDEN_DIR / f"{version}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _amount(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_graph(
    records: list[dict] | None = None,
    merchants: list[dict] | None = None,
    version: str = "v0",
) -> GraphBuild:
    records = load_golden(version) if records is None else records
    merchants = merchant_docs() if merchants is None else merchants

    entities: dict[str, Entity] = {}
    relationships: dict[tuple[str, str, str], Relationship] = {}

    def add_entity(entity: Entity) -> None:
        entities.setdefault(entity.id, entity)

    def add_rel(source: str, target: str, rel_type: str, **props: Any) -> None:
        key = (source, rel_type, target)
        relationships.setdefault(
            key, Relationship(source=source, target=target, type=rel_type, properties=props)
        )

    # --- dimensions ---------------------------------------------------------
    for category in GLCategory:
        add_entity(
            Entity(
                id=category_id(category.value),
                type=CATEGORY,
                label=category.value,
                properties={"name": category.value},
            )
        )

    for rule in DEFAULT_RULES:
        add_entity(
            Entity(
                id=policy_id(rule.name),
                type=POLICY,
                label=rule.name,
                properties={
                    "rule": rule.name,
                    "applies_to": rule.applies_to.value if rule.applies_to else "ALL",
                    "max_amount": rule.max_amount,
                    "requires_receipt": rule.requires_receipt,
                    "requires_manager_above": rule.requires_manager_above,
                },
            )
        )
        categories = [rule.applies_to.value] if rule.applies_to else [c.value for c in GLCategory]
        for category in categories:
            add_rel(category_id(category), policy_id(rule.name), HAS_POLICY)

    merchant_category: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for merchant in merchants:
        name = str(merchant["name"]).strip()
        category = str(merchant["category"])
        eid = merchant_id(name)
        add_entity(
            Entity(
                id=eid,
                type=MERCHANT,
                label=name,
                properties={"name": name, "name_key": name.lower(), "category": category},
            )
        )
        add_rel(eid, category_id(category), MERCHANT_CATEGORY)
        merchant_category[eid] = category
        by_name[name.lower()] = eid

    add_entity(
        Entity(
            id=UNKNOWN_MERCHANT,
            type=MERCHANT,
            label="Unknown (no merchant)",
            properties={"name": "", "name_key": "", "category": "OTHER"},
        )
    )
    add_rel(UNKNOWN_MERCHANT, category_id("OTHER"), MERCHANT_CATEGORY)
    merchant_category[UNKNOWN_MERCHANT] = "OTHER"

    sources = sorted({str(rec.get("source", "unknown")) for rec in records})
    for source in sources:
        add_entity(
            Entity(
                id=source_id(source),
                type=SOURCE,
                label=source,
                properties={"name": source},
            )
        )

    buckets = sorted(
        {date_bucket((rec.get("fields") or {}).get("transaction_date")) for rec in records}
    )
    for bucket in buckets:
        add_entity(
            Entity(
                id=date_id(bucket),
                type=DATE,
                label=bucket,
                properties={"bucket": bucket},
            )
        )

    for route in ROUTES:
        add_entity(
            Entity(
                id=route_id(route),
                type=ROUTE,
                label=route.replace("_", " "),
                properties={"route": route},
            )
        )

    # --- expenses + edges ---------------------------------------------------
    for rec in records:
        fields = rec.get("fields") or {}
        name = str(fields.get("merchant") or "").strip()
        if name:
            key = name.lower()
            eid = by_name.get(key)
            if eid is None:
                category = infer_category(name)
                eid = merchant_id(name)
                add_entity(
                    Entity(
                        id=eid,
                        type=MERCHANT,
                        label=name,
                        properties={"name": name, "name_key": key, "category": category},
                    )
                )
                add_rel(eid, category_id(category), MERCHANT_CATEGORY)
                merchant_category[eid] = category
                by_name[key] = eid
        else:
            eid = UNKNOWN_MERCHANT

        category = merchant_category[eid]
        doc_id = str(rec["id"])
        xid = expense_id(doc_id)
        source = str(rec.get("source", "unknown"))
        bucket = date_bucket(fields.get("transaction_date"))
        amount = _amount(fields.get("total_amount"))

        add_entity(
            Entity(
                id=xid,
                type=EXPENSE,
                label=str(fields.get("merchant") or doc_id),
                properties={
                    "doc_id": doc_id,
                    "source": source,
                    "merchant": name,
                    "category": category,
                    "total_amount": amount,
                    "transaction_date": fields.get("transaction_date"),
                    "date_bucket": bucket,
                },
            )
        )
        add_rel(xid, eid, PAID_TO)
        add_rel(xid, category_id(category), CODED_AS)
        add_rel(xid, source_id(source), FROM_SOURCE)
        add_rel(xid, date_id(bucket), ON_DATE)

        policy = evaluate(
            PolicyCheckRequest(
                amount=amount,
                category=GLCategory(category),
                merchant=name or None,
                has_receipt=True,
            )
        )
        add_rel(xid, route_id(policy.route), ROUTED_TO)
        for violation in policy.violations:
            add_rel(xid, policy_id(violation.rule), VIOLATES, detail=violation.detail)

    entity_rows = list(entities.values())
    relation_rows = list(relationships.values())
    by_type = Counter(e.type for e in entity_rows)
    by_rel = Counter(r.type for r in relation_rows)
    stats = {
        "golden_version": version,
        "entities": len(entity_rows),
        "relationships": len(relation_rows),
        "by_type": dict(by_type),
        "by_relationship": dict(by_rel),
    }
    return GraphBuild(entities=entity_rows, relationships=relation_rows, stats=stats)


def write_graph(build: GraphBuild, repo: GraphRepo, reset: bool = False) -> dict[str, Any]:
    ensure = getattr(repo, "ensure_constraints", None)
    if ensure is not None:
        ensure()
    if reset:
        repo.clear()
    repo.upsert(build.entities, build.relationships)
    repo_stats = repo.stats()
    return {**build.stats, "neo4j": repo_stats}
