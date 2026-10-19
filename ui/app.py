"""ExpenseIQ review console — Streamlit front end for the GraphQL graph."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8080").rstrip("/")
GRAPHQL_URL = os.getenv("GRAPHQL_URL", f"{API_URL}/graphql")

st.set_page_config(
    page_title="ExpenseIQ review console",
    page_icon="🧾",
    layout="wide",
)

st.markdown(
    """
    <style>
      .eq-muted { color: #4a4a4a; font-size: 0.95rem; }
      .eq-panel { border: 1px solid #d7dbe0; border-radius: 10px; padding: 0.9rem 1rem; }
      [data-testid="stCaptionContainer"] { opacity: 1 !important; }
      [data-testid="stCaptionContainer"] p { color: #4a4a4a !important; }
      [data-testid="stCaptionContainer"] code { color: #0b5394 !important; background: #eef3fb !important; }
      :focus-visible { outline: 3px solid #0b5394 !important; outline-offset: 2px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def gql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(
        GRAPHQL_URL,
        json={"query": query, "variables": variables or {}},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"][0].get("message", "GraphQL error"))
    return payload["data"]


@st.cache_data(ttl=20, show_spinner=False)
def load_dashboard() -> dict[str, Any]:
    return gql(
        """
        query Dashboard {
          graphStats { entities relationships clusters byType { key count } byRelationship { key count } }
          entities(type: EXPENSE, limit: 200) { id label properties }
          entitiesMerchants: entities(type: MERCHANT, limit: 200) { id label properties }
          clusters { id label algorithm size metrics }
          routes: relationships(type: ROUTED_TO, limit: 500) { source target }
          violations: relationships(type: VIOLATES, limit: 500) { source target }
        }
        """
    )


@st.cache_data(ttl=20, show_spinner=False)
def load_neighbors(entity_id: str) -> list[dict[str, Any]]:
    data = gql(
        """
        query Neighbors($id: ID!) {
          neighbors(id: $id, depth: 1) {
            relationship direction
            entity { id type label properties }
          }
        }
        """,
        {"id": entity_id},
    )
    return data["neighbors"]


def expense_frame(data: dict[str, Any]) -> pd.DataFrame:
    routes = {row["source"]: row["target"].removeprefix("route:") for row in data["routes"]}
    violations: dict[str, int] = {}
    for row in data["violations"]:
        violations[row["source"]] = violations.get(row["source"], 0) + 1

    rows = []
    for entity in data["entities"]:
        props = entity["properties"]
        rows.append(
            {
                "id": entity["id"],
                "doc": props.get("doc_id") or entity["label"],
                "merchant": props.get("merchant") or "Unknown (no merchant)",
                "category": props.get("category") or "OTHER",
                "amount": float(props.get("total_amount") or 0.0),
                "date": props.get("date_bucket") or "unknown",
                "source": props.get("source") or "",
                "route": routes.get(entity["id"], "unknown"),
                "violations": violations.get(entity["id"], 0),
            }
        )
    return pd.DataFrame(rows)


def cluster_frame(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cluster in data["clusters"]:
        metrics = cluster.get("metrics") or {}
        rows.append(
            {
                "algorithm": cluster["algorithm"],
                "cluster": cluster["label"],
                "merchants": cluster["size"],
                "ARI": metrics.get("ari"),
                "NMI": metrics.get("nmi"),
                "purity": metrics.get("purity"),
            }
        )
    return pd.DataFrame(rows)


st.title("ExpenseIQ review console")
st.markdown(
    '<p class="eq-muted">Human-in-the-loop review for coded expenses. '
    "The graph and cluster metrics refresh automatically every 20 seconds.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**Connection**")
    st.caption(f"GraphQL `{GRAPHQL_URL}`")
    st.caption(f"REST `{API_URL}`")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("**Filters**")
    categories: list[str] = []
    status_placeholder = st.empty()

try:
    dashboard = load_dashboard()
    expenses = expense_frame(dashboard)
    clusters = cluster_frame(dashboard)
    categories = sorted(expenses["category"].unique().tolist())
    status_placeholder.caption(f"Loaded {len(expenses)} expenses from the graph.")
except Exception as exc:  # pragma: no cover - operator feedback
    st.error(f"Could not reach the graph API at {GRAPHQL_URL}: {exc}")
    st.stop()

with st.sidebar:
    category_filter = st.selectbox("Category", ["All"] + categories)
    route_filter = st.selectbox(
        "Policy route",
        ["All", "auto_approve", "manager_review", "human_queue", "unknown"],
    )

search = st.text_input(
    "Search expenses",
    placeholder="Merchant or document id",
    help="Filters the review queue by merchant name or receipt document id.",
)

st.header("Review queue")

filtered = expenses.copy()
if category_filter != "All":
    filtered = filtered[filtered["category"] == category_filter]
if route_filter != "All":
    filtered = filtered[filtered["route"] == route_filter]
if search.strip():
    needle = search.strip().lower()
    filtered = filtered[
        filtered["merchant"].str.lower().str.contains(needle, regex=False)
        | filtered["doc"].str.lower().str.contains(needle, regex=False)
    ]

review_count = int((filtered["route"] != "auto_approve").sum())
kpi = st.columns(4)
kpi[0].metric("Expenses in view", f"{len(filtered):,}")
kpi[1].metric("Needs review", f"{review_count:,}")
kpi[2].metric("Merchants", f"{len(dashboard['entitiesMerchants']):,}")
kpi[3].metric("Policy violations", f"{int(filtered['violations'].sum()):,}")

st.dataframe(
    filtered[["doc", "merchant", "category", "amount", "date", "source", "route", "violations"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
        "violations": st.column_config.NumberColumn("Violations", format="%d"),
    },
)
st.markdown(
    '<p class="eq-muted">Amounts are shown in the receipt currency. '
    "Use the filters in the sidebar to narrow the queue.</p>",
    unsafe_allow_html=True,
)

st.header("Merchant clusters")
left, right = st.columns([3, 2])
with left:
    merchant_categories = pd.Series(
        [m["properties"].get("category", "OTHER") for m in dashboard["entitiesMerchants"]]
    ).value_counts()
    st.bar_chart(merchant_categories)
    top = ", ".join(f"{name}: {count}" for name, count in merchant_categories.head(5).items())
    st.caption(f"Merchant count by GL category. Top categories — {top}.")
with right:
    st.dataframe(
        clusters,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ARI": st.column_config.NumberColumn("ARI", format="%.3f"),
            "NMI": st.column_config.NumberColumn("NMI", format="%.3f"),
            "purity": st.column_config.NumberColumn("Purity", format="%.3f"),
        },
    )

st.header("Graph details")
st.dataframe(
    pd.DataFrame(
        {
            "entity type": [row["key"] for row in dashboard["graphStats"]["byType"]],
            "count": [row["count"] for row in dashboard["graphStats"]["byType"]],
        }
    ),
    use_container_width=True,
    hide_index=True,
)

options = {
    f"{row['doc']} · {row['merchant']} · ${row['amount']:,.2f}": row["id"]
    for _, row in filtered.head(50).iterrows()
}
if options:
    choice = st.selectbox("Inspect an expense", list(options))
    try:
        neighbors = load_neighbors(options[choice])
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "relationship": row["relationship"],
                        "direction": row["direction"],
                        "entity": row["entity"]["label"],
                        "type": row["entity"]["type"],
                    }
                    for row in neighbors
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:  # pragma: no cover - operator feedback
        st.warning(f"Could not load neighbors: {exc}")
else:
    st.info("No expenses match the current filters.")
