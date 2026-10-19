"""Graph repositories: Neo4j for production, an in-memory fake for tests."""

from __future__ import annotations

import json
import logging
from collections import deque
from functools import lru_cache
from typing import Any, Protocol

from ..config import get_settings
from .model import (
    REL_TYPES,
    Cluster,
    Entity,
    Neighbor,
    Relationship,
    cluster_id,
)

log = logging.getLogger(__name__)


class GraphRepo(Protocol):
    def entities(
        self, type: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Entity]: ...

    def entity(self, id: str) -> Entity | None: ...

    def relationships(self, type: str | None = None, limit: int = 100) -> list[Relationship]: ...

    def neighbors(self, id: str, depth: int = 1, rel_type: str | None = None) -> list[Neighbor]: ...

    def search(self, q: str, limit: int = 10) -> list[Entity]: ...

    def stats(self) -> dict[str, Any]: ...

    def clusters(self, algorithm: str | None = None) -> list[Cluster]: ...

    def upsert(self, entities: list[Entity], relationships: list[Relationship]) -> None: ...

    def upsert_clusters(self, result: dict[str, Any]) -> None: ...

    def clear(self) -> None: ...


class FakeGraphRepo:
    """Small deterministic repo used by unit tests and the offline UI fixture."""

    def __init__(
        self,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
    ) -> None:
        self._entities = {e.id: e for e in entities or []}
        self._relationships = list(relationships or [])
        self._clusters: dict[str, Cluster] = {}
        self._assignments: dict[str, dict[str, int]] = {}

    def entities(self, type: str | None = None, limit: int = 50, offset: int = 0) -> list[Entity]:
        rows = sorted(self._entities.values(), key=lambda e: e.id)
        if type:
            rows = [e for e in rows if e.type == type]
        return rows[max(offset, 0) : max(offset, 0) + max(limit, 0)]

    def entity(self, id: str) -> Entity | None:
        return self._entities.get(id)

    def relationships(self, type: str | None = None, limit: int = 100) -> list[Relationship]:
        rows = sorted(self._relationships, key=lambda r: (r.source, r.type, r.target))
        if type:
            rows = [r for r in rows if r.type == type]
        return rows[: max(limit, 0)]

    def _adjacent(self, id: str, rel_type: str | None) -> list[Neighbor]:
        out: list[Neighbor] = []
        for rel in self._relationships:
            if rel_type is not None and rel.type != rel_type:
                continue
            if rel.source == id:
                other, direction = rel.target, "out"
            elif rel.target == id:
                other, direction = rel.source, "in"
            else:
                continue
            entity = self._entities.get(other)
            if entity is not None:
                out.append(Neighbor(entity, rel.type, direction, 1))  # type: ignore[arg-type]
        return out

    def neighbors(self, id: str, depth: int = 1, rel_type: str | None = None) -> list[Neighbor]:
        depth = max(1, min(int(depth), 3))
        if id not in self._entities:
            return []
        seen = {id}
        out: list[Neighbor] = []
        queue: deque[tuple[str, int]] = deque([(id, 0)])
        while queue:
            current, level = queue.popleft()
            if level >= depth:
                continue
            for neighbor in self._adjacent(current, rel_type):
                if neighbor.entity.id in seen:
                    continue
                seen.add(neighbor.entity.id)
                out.append(
                    Neighbor(neighbor.entity, neighbor.relationship, neighbor.direction, level + 1)
                )
                queue.append((neighbor.entity.id, level + 1))
        return out

    def search(self, q: str, limit: int = 10) -> list[Entity]:
        needle = q.strip().lower()
        rows = [e for e in self._entities.values() if needle and needle in e.label.lower()]
        rows.sort(key=lambda e: (e.type, e.label))
        return rows[: max(limit, 0)]

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for entity in self._entities.values():
            by_type[entity.type] = by_type.get(entity.type, 0) + 1
        by_rel: dict[str, int] = {}
        for rel in self._relationships:
            by_rel[rel.type] = by_rel.get(rel.type, 0) + 1
        return {
            "entities": len(self._entities),
            "relationships": len(self._relationships),
            "by_type": by_type,
            "by_relationship": by_rel,
            "clusters": len(self._clusters),
        }

    def clusters(self, algorithm: str | None = None) -> list[Cluster]:
        rows = sorted(self._clusters.values(), key=lambda c: (c.algorithm, c.id))
        if algorithm:
            rows = [c for c in rows if c.algorithm == algorithm]
        return rows

    def upsert(self, entities: list[Entity], relationships: list[Relationship]) -> None:
        for entity in entities:
            self._entities[entity.id] = entity
        for rel in relationships:
            if rel.type not in REL_TYPES:
                raise ValueError(f"unknown relationship type: {rel.type}")
        self._relationships.extend(relationships)

    def upsert_clusters(self, result: dict[str, Any]) -> None:
        algorithms = result.get("algorithms", {})
        for algorithm in algorithms:
            for cid in [c.id for c in self._clusters.values() if c.algorithm == algorithm]:
                del self._clusters[cid]
        for algorithm, run in algorithms.items():
            for index in range(run["n_clusters"]):
                cid = cluster_id(algorithm, index)
                members = sum(1 for v in run["assignments"].values() if v == index)
                self._clusters[cid] = Cluster(
                    id=cid,
                    label=f"{algorithm} cluster {index}",
                    algorithm=algorithm,
                    size=members,
                    metrics={
                        "ari": run.get("ari"),
                        "nmi": run.get("nmi"),
                        "purity": run.get("purity"),
                        "modularity": run.get("modularity"),
                    },
                )

    def clear(self) -> None:
        self._entities.clear()
        self._relationships.clear()
        self._clusters.clear()


class Neo4jRepo:
    def __init__(
        self, uri: str | None = None, user: str | None = None, password: str | None = None
    ):
        from neo4j import GraphDatabase

        settings = get_settings()
        self._driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def _run(self, query: str, **params: Any) -> list[Any]:
        with self._driver.session() as session:
            return list(session.run(query, **params))

    def ensure_constraints(self) -> None:
        for statement in (
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
            "CREATE CONSTRAINT cluster_id IF NOT EXISTS FOR (c:Cluster) REQUIRE c.id IS UNIQUE",
        ):
            self._run(statement)

    @staticmethod
    def _entity(row: Any) -> Entity:
        props = dict(row["props"] or {})
        for key in ("id", "type", "label"):
            props.pop(key, None)
        return Entity(id=row["id"], type=row["type"], label=row["label"], properties=props)

    def entities(self, type: str | None = None, limit: int = 50, offset: int = 0) -> list[Entity]:
        rows = self._run(
            """
            MATCH (e:Entity)
            WHERE $type IS NULL OR e.type = $type
            RETURN e.id AS id, e.type AS type, e.label AS label, properties(e) AS props
            ORDER BY e.id
            SKIP $offset LIMIT $limit
            """,
            type=type,
            offset=max(offset, 0),
            limit=max(limit, 0),
        )
        return [self._entity(row) for row in rows]

    def entity(self, id: str) -> Entity | None:
        rows = self._run(
            """
            MATCH (e:Entity {id: $id})
            RETURN e.id AS id, e.type AS type, e.label AS label, properties(e) AS props
            LIMIT 1
            """,
            id=id,
        )
        return self._entity(rows[0]) if rows else None

    def relationships(self, type: str | None = None, limit: int = 100) -> list[Relationship]:
        rows = self._run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE $type IS NULL OR type(r) = $type
            RETURN a.id AS source, b.id AS target, type(r) AS type, properties(r) AS props
            ORDER BY source, type, target
            LIMIT $limit
            """,
            type=type,
            limit=max(limit, 0),
        )
        return [
            Relationship(
                source=row["source"],
                target=row["target"],
                type=row["type"],
                properties=dict(row["props"] or {}),
            )
            for row in rows
        ]

    def _neighbors_once(self, id: str, rel_type: str | None) -> list[Neighbor]:
        rows = self._run(
            """
            MATCH (e:Entity {id: $id})-[r]-(n:Entity)
            WHERE $rel_type IS NULL OR type(r) = $rel_type
            RETURN n.id AS id, n.type AS type, n.label AS label, properties(n) AS props,
                   type(r) AS relationship,
                   CASE WHEN startNode(r).id = $id THEN 'out' ELSE 'in' END AS direction
            ORDER BY relationship, id
            """,
            id=id,
            rel_type=rel_type,
        )
        return [
            Neighbor(
                entity=self._entity(row),
                relationship=row["relationship"],
                direction=row["direction"],
                depth=1,
            )
            for row in rows
        ]

    def neighbors(self, id: str, depth: int = 1, rel_type: str | None = None) -> list[Neighbor]:
        depth = max(1, min(int(depth), 3))
        if self.entity(id) is None:
            return []
        seen = {id}
        out: list[Neighbor] = []
        queue: deque[tuple[str, int]] = deque([(id, 0)])
        while queue:
            current, level = queue.popleft()
            if level >= depth:
                continue
            for neighbor in self._neighbors_once(current, rel_type):
                if neighbor.entity.id in seen:
                    continue
                seen.add(neighbor.entity.id)
                out.append(
                    Neighbor(neighbor.entity, neighbor.relationship, neighbor.direction, level + 1)
                )
                queue.append((neighbor.entity.id, level + 1))
        return out

    def search(self, q: str, limit: int = 10) -> list[Entity]:
        rows = self._run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.label) CONTAINS toLower($q)
            RETURN e.id AS id, e.type AS type, e.label AS label, properties(e) AS props
            ORDER BY e.type, e.label
            LIMIT $limit
            """,
            q=q,
            limit=max(limit, 0),
        )
        return [self._entity(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        type_rows = self._run("MATCH (e:Entity) RETURN e.type AS key, count(*) AS n ORDER BY key")
        rel_rows = self._run("MATCH ()-[r]->() RETURN type(r) AS key, count(*) AS n ORDER BY key")
        by_type = {row["key"]: row["n"] for row in type_rows}
        by_rel = {row["key"]: row["n"] for row in rel_rows}
        cluster_rows = self._run("MATCH (c:Cluster) RETURN count(*) AS n")
        return {
            "entities": sum(by_type.values()),
            "relationships": sum(by_rel.values()),
            "by_type": by_type,
            "by_relationship": by_rel,
            "clusters": cluster_rows[0]["n"] if cluster_rows else 0,
        }

    def clusters(self, algorithm: str | None = None) -> list[Cluster]:
        rows = self._run(
            """
            MATCH (c:Cluster)
            WHERE $algorithm IS NULL OR c.algorithm = $algorithm
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(m:Entity)
            RETURN c.id AS id, c.label AS label, c.algorithm AS algorithm,
                   c.size AS size, c.metrics AS metrics, count(m) AS members
            ORDER BY c.algorithm, c.id
            """,
            algorithm=algorithm,
        )
        out: list[Cluster] = []
        for row in rows:
            metrics = row["metrics"] or "{}"
            if isinstance(metrics, str):
                try:
                    metrics = json.loads(metrics)
                except json.JSONDecodeError:
                    metrics = {}
            out.append(
                Cluster(
                    id=row["id"],
                    label=row["label"],
                    algorithm=row["algorithm"],
                    size=row["members"] or row["size"] or 0,
                    metrics=dict(metrics),
                )
            )
        return out

    def upsert(self, entities: list[Entity], relationships: list[Relationship]) -> None:
        entity_rows = [
            {"id": e.id, "type": e.type, "label": e.label, "props": e.properties} for e in entities
        ]
        if entity_rows:
            self._run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {id: row.id})
                SET e.type = row.type, e.label = row.label
                SET e += row.props
                """,
                rows=entity_rows,
            )

        for rel_type in REL_TYPES:
            rows = [
                {"source": r.source, "target": r.target, "props": r.properties}
                for r in relationships
                if r.type == rel_type
            ]
            if not rows:
                continue
            # rel_type comes from the REL_TYPES whitelist above.
            self._run(
                f"""
                UNWIND $rows AS row
                MATCH (a:Entity {{id: row.source}}), (b:Entity {{id: row.target}})
                MERGE (a)-[r:`{rel_type}`]->(b)
                SET r += row.props
                """,
                rows=rows,
            )

    def upsert_clusters(self, result: dict[str, Any]) -> None:
        for algorithm, run in result.get("algorithms", {}).items():
            self._run(
                "MATCH (c:Cluster {algorithm: $algorithm}) DETACH DELETE c",
                algorithm=algorithm,
            )
            members: dict[int, int] = {}
            for _, index in run.get("assignments", {}).items():
                members[index] = members.get(index, 0) + 1
            rows = []
            for index in range(run["n_clusters"]):
                cid = cluster_id(algorithm, index)
                rows.append(
                    {
                        "id": cid,
                        "label": f"{algorithm} cluster {index}",
                        "algorithm": algorithm,
                        "size": members.get(index, 0),
                        "metrics": json.dumps(
                            {
                                "ari": run.get("ari"),
                                "nmi": run.get("nmi"),
                                "purity": run.get("purity"),
                                "modularity": run.get("modularity"),
                            }
                        ),
                    }
                )
            if rows:
                self._run(
                    """
                    UNWIND $rows AS row
                    MERGE (c:Cluster {id: row.id})
                    SET c.label = row.label, c.algorithm = row.algorithm,
                        c.size = row.size, c.metrics = row.metrics
                    """,
                    rows=rows,
                )
            links = [
                {"entity": entity_id, "cluster": cluster_id(algorithm, index)}
                for entity_id, index in run.get("assignments", {}).items()
            ]
            if links:
                self._run(
                    """
                    UNWIND $rows AS row
                    MATCH (e:Entity {id: row.entity}), (c:Cluster {id: row.cluster})
                    MERGE (e)-[r:IN_COMMUNITY]->(c)
                    SET r.algorithm = $algorithm
                    """,
                    rows=links,
                    algorithm=algorithm,
                )

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")


@lru_cache
def get_graph_repo() -> GraphRepo:
    return Neo4jRepo()
