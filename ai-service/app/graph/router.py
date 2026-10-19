"""FastAPI router exposing the graph at ``/graphql``."""

from __future__ import annotations

from typing import Any

from strawberry.fastapi import GraphQLRouter

from .repo import get_graph_repo
from .schema import build_schema


async def get_context() -> dict[str, Any]:
    return {"repo": get_graph_repo()}


router = GraphQLRouter(
    build_schema(),
    context_getter=get_context,
    graphql_ide="graphiql",
)
