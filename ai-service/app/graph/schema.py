"""Strawberry GraphQL schema over the entity graph."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import strawberry
from strawberry.scalars import JSON

from .model import Cluster, Entity, Neighbor
from .repo import GraphRepo


class EntityKind(StrEnum):
    EXPENSE = "Expense"
    MERCHANT = "Merchant"
    CATEGORY = "Category"
    POLICY = "Policy"
    SOURCE = "Source"
    DATE = "Date"
    ROUTE = "Route"
    CLUSTER = "Cluster"


class RelKind(StrEnum):
    PAID_TO = "PAID_TO"
    CODED_AS = "CODED_AS"
    MERCHANT_CATEGORY = "MERCHANT_CATEGORY"
    FROM_SOURCE = "FROM_SOURCE"
    ON_DATE = "ON_DATE"
    ROUTED_TO = "ROUTED_TO"
    VIOLATES = "VIOLATES"
    HAS_POLICY = "HAS_POLICY"
    IN_COMMUNITY = "IN_COMMUNITY"


class ClusterAlgorithm(StrEnum):
    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    LOUVAIN = "louvain"


@strawberry.type
class EntityNode:
    id: strawberry.ID
    type: str
    label: str
    properties: JSON


@strawberry.type
class RelationshipEdge:
    source: strawberry.ID
    target: strawberry.ID
    type: str
    properties: JSON


@strawberry.type
class NeighborNode:
    entity: EntityNode
    relationship: str
    direction: str
    depth: int


@strawberry.type
class Count:
    key: str
    count: int


@strawberry.type
class GraphStats:
    entities: int
    relationships: int
    by_type: list[Count]
    by_relationship: list[Count]
    clusters: int


@strawberry.type
class ClusterInfo:
    id: strawberry.ID
    label: str
    algorithm: str
    size: int
    metrics: JSON


def _entity(entity: Entity) -> EntityNode:
    return EntityNode(
        id=entity.id,
        type=entity.type,
        label=entity.label,
        properties=entity.properties,
    )


def _neighbor(neighbor: Neighbor) -> NeighborNode:
    return NeighborNode(
        entity=_entity(neighbor.entity),
        relationship=neighbor.relationship,
        direction=neighbor.direction,
        depth=neighbor.depth,
    )


def _cluster(cluster: Cluster) -> ClusterInfo:
    return ClusterInfo(
        id=cluster.id,
        label=cluster.label,
        algorithm=cluster.algorithm,
        size=cluster.size,
        metrics=cast_metrics(cluster.metrics),
    )


def cast_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if v is not None}


def _repo(context: Any) -> GraphRepo:
    if isinstance(context, dict):
        return context["repo"]
    return context.repo


def build_schema() -> strawberry.Schema:
    @strawberry.type
    class Query:
        @strawberry.field(description="List entities, optionally filtered by node type.")
        def entities(
            self,
            info: strawberry.Info,
            type: EntityKind | None = None,
            limit: int = 50,
            offset: int = 0,
        ) -> list[EntityNode]:
            repo = _repo(info.context)
            return [
                _entity(entity)
                for entity in repo.entities(
                    type=type.value if type else None,
                    limit=max(1, min(limit, 200)),
                    offset=max(offset, 0),
                )
            ]

        @strawberry.field(description="Fetch a single entity by id.")
        def entity(self, info: strawberry.Info, id: strawberry.ID) -> EntityNode | None:
            entity = _repo(info.context).entity(str(id))
            return _entity(entity) if entity else None

        @strawberry.field(description="List relationships, optionally filtered by type.")
        def relationships(
            self,
            info: strawberry.Info,
            type: RelKind | None = None,
            limit: int = 100,
        ) -> list[RelationshipEdge]:
            rows = _repo(info.context).relationships(
                type=type.value if type else None, limit=max(1, min(limit, 500))
            )
            return [
                RelationshipEdge(
                    source=rel.source,
                    target=rel.target,
                    type=rel.type,
                    properties=rel.properties,
                )
                for rel in rows
            ]

        @strawberry.field(description="Traverse the graph around an entity (depth 1-3).")
        def neighbors(
            self,
            info: strawberry.Info,
            id: strawberry.ID,
            depth: int = 1,
            rel_type: RelKind | None = None,
        ) -> list[NeighborNode]:
            rows = _repo(info.context).neighbors(
                str(id),
                depth=depth,
                rel_type=rel_type.value if rel_type else None,
            )
            return [_neighbor(row) for row in rows]

        @strawberry.field(description="Case-insensitive label search.")
        def search_entities(
            self, info: strawberry.Info, q: str, limit: int = 10
        ) -> list[EntityNode]:
            return [
                _entity(entity)
                for entity in _repo(info.context).search(q, limit=max(1, min(limit, 50)))
            ]

        @strawberry.field(description="Node/relationship counts for the whole graph.")
        def graph_stats(self, info: strawberry.Info) -> GraphStats:
            stats = _repo(info.context).stats()
            return GraphStats(
                entities=stats.get("entities", 0),
                relationships=stats.get("relationships", 0),
                by_type=[
                    Count(key=str(key), count=int(value))
                    for key, value in sorted(stats.get("by_type", {}).items())
                ],
                by_relationship=[
                    Count(key=str(key), count=int(value))
                    for key, value in sorted(stats.get("by_relationship", {}).items())
                ],
                clusters=int(stats.get("clusters", 0)),
            )

        @strawberry.field(description="Persisted clustering / community results.")
        def clusters(
            self, info: strawberry.Info, algorithm: ClusterAlgorithm | None = None
        ) -> list[ClusterInfo]:
            return [
                _cluster(cluster)
                for cluster in _repo(info.context).clusters(algorithm.value if algorithm else None)
            ]

    return strawberry.Schema(query=Query)
