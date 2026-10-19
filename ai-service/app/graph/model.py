"""Graph node/relationship vocabulary and ids.

Every node carries a stable string id (``<type>:<key>``) plus a ``type`` and a
human-readable ``label``. We store a single ``:Entity`` label in Neo4j and filter
on ``type`` so the schema stays small; clusters are the exception (``:Cluster``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

EXPENSE = "Expense"
MERCHANT = "Merchant"
CATEGORY = "Category"
POLICY = "Policy"
SOURCE = "Source"
DATE = "Date"
ROUTE = "Route"
CLUSTER = "Cluster"

NODE_TYPES: tuple[str, ...] = (
    EXPENSE,
    MERCHANT,
    CATEGORY,
    POLICY,
    SOURCE,
    DATE,
    ROUTE,
    CLUSTER,
)

PAID_TO = "PAID_TO"
CODED_AS = "CODED_AS"
MERCHANT_CATEGORY = "MERCHANT_CATEGORY"
FROM_SOURCE = "FROM_SOURCE"
ON_DATE = "ON_DATE"
ROUTED_TO = "ROUTED_TO"
VIOLATES = "VIOLATES"
HAS_POLICY = "HAS_POLICY"
IN_COMMUNITY = "IN_COMMUNITY"

REL_TYPES: tuple[str, ...] = (
    PAID_TO,
    CODED_AS,
    MERCHANT_CATEGORY,
    FROM_SOURCE,
    ON_DATE,
    ROUTED_TO,
    VIOLATES,
    HAS_POLICY,
    IN_COMMUNITY,
)


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship:
    source: str
    target: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Neighbor:
    entity: Entity
    relationship: str
    direction: Literal["out", "in"]
    depth: int


@dataclass(frozen=True)
class Cluster:
    id: str
    label: str
    algorithm: str
    size: int
    metrics: dict[str, Any] = field(default_factory=dict)


def slug(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return out or "unknown"


def expense_id(doc_id: str) -> str:
    return f"expense:{doc_id}"


def merchant_id(name: str) -> str:
    return f"merchant:{slug(name)}"


def category_id(category: str) -> str:
    return f"category:{slug(category)}"


def policy_id(name: str) -> str:
    return f"policy:{slug(name)}"


def source_id(name: str) -> str:
    return f"source:{slug(name)}"


def date_id(bucket: str) -> str:
    return f"date:{slug(bucket)}"


def route_id(name: str) -> str:
    return f"route:{slug(name)}"


def cluster_id(algorithm: str, index: int) -> str:
    return f"cluster:{slug(algorithm)}:{index}"


_YMD_RE = re.compile(r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})")
_DMY_RE = re.compile(r"(?P<d>\d{1,2})[-/.](?P<m>\d{1,2})[-/.](?P<y>\d{4})")


def date_bucket(value: str | None) -> str:
    """Normalise printed dates to ``YYYY-MM`` (or ``unknown``)."""
    if not value:
        return "unknown"
    match = _DMY_RE.search(value)
    if match:
        return f"{match.group('y')}-{int(match.group('m')):02d}"
    match = _YMD_RE.search(value)
    if match:
        return f"{match.group('y')}-{int(match.group('m')):02d}"
    return "unknown"
