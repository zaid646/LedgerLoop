from app.graph.build import build_graph
from app.graph.repo import FakeGraphRepo
from app.graph.schema import build_schema

SCHEMA = build_schema()


def _repo() -> FakeGraphRepo:
    graph = build_graph()
    return FakeGraphRepo(graph.entities, graph.relationships)


def _execute(query: str, repo: FakeGraphRepo):
    result = SCHEMA.execute_sync(query, context_value={"repo": repo})
    assert result.errors is None, result.errors
    return result.data


def test_schema_exposes_the_documented_queries():
    data = _execute(
        "{ __type(name: \"Query\") { fields { name } } }",
        _repo(),
    )
    names = {field["name"] for field in data["__type"]["fields"]}
    assert {
        "entities",
        "entity",
        "relationships",
        "neighbors",
        "searchEntities",
        "graphStats",
        "clusters",
    } <= names


def test_graph_stats():
    data = _execute(
        """
        { graphStats { entities relationships clusters byType { key count } } }
        """,
        _repo(),
    )
    stats = data["graphStats"]
    assert stats["entities"] == 175
    assert stats["relationships"] == 462
    assert {row["key"]: row["count"] for row in stats["byType"]}["Expense"] == 75


def test_entities_filter_and_pagination():
    repo = _repo()
    data = _execute(
        "{ entities(type: MERCHANT, limit: 3, offset: 0) { id type label } }",
        repo,
    )
    assert len(data["entities"]) == 3
    assert all(row["type"] == "Merchant" for row in data["entities"])

    second = _execute(
        "{ entities(type: MERCHANT, limit: 3, offset: 3) { id } }",
        repo,
    )
    assert {row["id"] for row in data["entities"]}.isdisjoint(
        {row["id"] for row in second["entities"]}
    )


def test_entity_by_id_and_unknown_id():
    data = _execute(
        """
        { entity(id: "expense:sroie-21") {
            id type label properties
        } }
        """,
        _repo(),
    )
    entity = data["entity"]
    assert entity["type"] == "Expense"
    assert entity["properties"]["category"] == "OFFICE_SUPPLIES"

    missing = _execute('{ entity(id: "expense:does-not-exist") { id } }', _repo())
    assert missing["entity"] is None


def test_relationships_filter():
    data = _execute(
        '{ relationships(type: PAID_TO, limit: 5) { source target type } }',
        _repo(),
    )
    assert len(data["relationships"]) == 5
    assert all(row["type"] == "PAID_TO" for row in data["relationships"])


def test_neighbors_depth_one_and_two():
    repo = _repo()
    one = _execute(
        """
        { neighbors(id: "expense:sroie-0", depth: 1) {
            relationship direction depth entity { id type }
        } }
        """,
        repo,
    )
    assert one["neighbors"]
    assert {row["relationship"] for row in one["neighbors"]} >= {"PAID_TO", "CODED_AS"}
    assert all(row["depth"] == 1 for row in one["neighbors"])

    two = _execute(
        """
        { neighbors(id: "expense:sroie-0", depth: 2) {
            relationship depth entity { id type }
        } }
        """,
        repo,
    )
    assert any(row["depth"] == 2 for row in two["neighbors"])

    unknown = _execute(
        '{ neighbors(id: "expense:nope", depth: 1) { relationship } }',
        repo,
    )
    assert unknown["neighbors"] == []


def test_neighbors_can_be_filtered_by_relationship():
    data = _execute(
        """
        { neighbors(id: "expense:sroie-0", depth: 1, relType: PAID_TO) {
            relationship
        } }
        """,
        _repo(),
    )
    assert {row["relationship"] for row in data["neighbors"]} == {"PAID_TO"}


def test_search_entities():
    data = _execute(
        '{ searchEntities(q: "ikea", limit: 5) { id type label } }',
        _repo(),
    )
    assert any(row["label"] == "IKEA" for row in data["searchEntities"])


def test_clusters_are_read_from_the_repo():
    repo = _repo()
    repo.upsert_clusters(
        {
            "algorithms": {
                "kmeans": {
                    "n_clusters": 2,
                    "ari": 0.42,
                    "nmi": 0.31,
                    "purity": 0.55,
                    "modularity": None,
                    "assignments": {},
                }
            }
        }
    )
    data = _execute(
        "{ clusters(algorithm: KMEANS) { id label algorithm size metrics } }",
        repo,
    )
    assert len(data["clusters"]) == 2
    assert data["clusters"][0]["algorithm"] == "kmeans"
    assert data["clusters"][0]["metrics"]["ari"] == 0.42
