"""Human-in-the-loop review screen for confidence-routed documents."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8081").rstrip("/")

st.set_page_config(page_title="LedgerLoop confidence review", layout="wide")

st.markdown(
    """
    <style>
      .ll-muted { color: #3f4650; }
      :focus-visible { outline: 3px solid #0b5394 !important; outline-offset: 2px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api(path: str) -> Any:
    response = requests.get(f"{BACKEND_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def decide(queue_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}/review/{queue_id}/decision", json=payload, timeout=20
    )
    if response.status_code >= 400:
        raise RuntimeError(response.text)
    return response.json()


st.title("Confidence review queue")
st.markdown(
    '<p class="ll-muted">Approve, correct or reject documents routed by calibrated '
    "confidence. Every decision is written to the audit log and corrections are "
    "stored for evaluation.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Reviewer")
    reviewer = st.text_input("Reviewer id", value=os.getenv("REVIEWER", ""), help="Required for every decision.")
    st.header("Safety")
    try:
        switch = api("/admin/kill-switch")
        enabled = st.toggle(
            "Force all documents to human review",
            value=bool(switch.get("enabled")),
            help="Kill switch: bypass auto-approval for every document.",
        )
        if enabled != bool(switch.get("enabled")):
            requests.post(
                f"{BACKEND_URL}/admin/kill-switch",
                json={"enabled": enabled, "actor": reviewer or "ui"},
                timeout=10,
            ).raise_for_status()
            st.cache_data.clear()
            st.rerun()
    except Exception as exc:  # pragma: no cover - operator feedback
        st.error(f"Kill switch unavailable: {exc}")

try:
    stats = api("/review/stats")
    queue = api("/review?status=pending&limit=100")
except Exception as exc:  # pragma: no cover - operator feedback
    st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
    st.stop()

pending = queue.get("items", [])
by_status = {row["status"]: row["count"] for row in stats.get("by_status", [])}
decisions = {row["decision"]: row["count"] for row in stats.get("decisions", [])}

kpi = st.columns(4)
kpi[0].metric("Pending", stats.get("queue_depth", 0))
kpi[1].metric("Approved", by_status.get("approved", 0))
kpi[2].metric("Corrected", by_status.get("corrected", 0))
kpi[3].metric("Rejected", by_status.get("rejected", 0))
st.caption(
    f"Reviewer decisions: approve={decisions.get('approve', 0)}, "
    f"correct={decisions.get('correct', 0)}, reject={decisions.get('reject', 0)}. "
    f"Labeled corrections stored: {stats.get('labeled_corrections', 0)}."
)

if not pending:
    st.info("The review queue is empty.")
    st.stop()

frame = pd.DataFrame(
    [
        {
            "id": item["id"],
            "document": item["document_id"],
            "route": item["route"],
            "confidence": float(item["document_confidence"]),
            "reason": item.get("reason") or "",
            "created": item["created_at"],
        }
        for item in pending
    ]
)
st.subheader("Pending documents")
st.dataframe(
    frame,
    hide_index=True,
    use_container_width=True,
    column_config={
        "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
    },
)

labels = {
    f"#{row['id']} · {row['document']} · {float(row['confidence']):.3f} · {row['route']}": row["id"]
    for _, row in frame.iterrows()
}
selection = st.selectbox("Select a document to review", list(labels))
item = next(entry for entry in pending if entry["id"] == labels[selection])
original = item.get("original_values") or {}

st.subheader(f"Document {item['document_id']}")
left, right = st.columns([3, 2])
with left:
    st.write("Original extraction")
    st.dataframe(
        pd.DataFrame(
            [{"field": key, "value": "" if value is None else str(value)} for key, value in original.items()]
        ),
        hide_index=True,
        use_container_width=True,
    )
with right:
    st.write("Per-field confidence")
    field_confidence = item.get("field_confidence") or {}
    st.dataframe(
        pd.DataFrame(
            [
                {"field": key, "confidence": float(value)}
                for key, value in field_confidence.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
        column_config={"confidence": st.column_config.NumberColumn(format="%.3f")},
    )

st.caption(f"Routing reasons: {item.get('reason') or 'n/a'}")
st.caption(f"Created: {item['created_at']}")

with st.form("review_decision"):
    st.subheader("Decision")
    decision = st.radio(
        "Outcome",
        ["approve", "correct", "reject"],
        horizontal=True,
        captions=[
            "Accept the extraction exactly as shown.",
            "Edit fields, then save the correction as an approved document.",
            "Reject the document and escalate it.",
        ],
    )
    corrected: dict[str, Any] = {}
    if decision == "correct":
        st.markdown("**Corrected values**")
        merchant = st.text_input("Merchant", value=str(original.get("merchant") or ""))
        amount_text = st.text_input(
            "Total amount", value="" if original.get("total_amount") is None else str(original["total_amount"])
        )
        date_text = st.text_input(
            "Transaction date",
            value="" if original.get("transaction_date") is None else str(original["transaction_date"]),
        )
        currency = st.text_input("Currency", value=str(original.get("currency") or ""))
        if merchant != (original.get("merchant") or ""):
            corrected["merchant"] = merchant
        if amount_text != ("" if original.get("total_amount") is None else str(original["total_amount"])):
            try:
                corrected["total_amount"] = float(amount_text) if amount_text.strip() else None
            except ValueError:
                corrected["total_amount"] = amount_text
        if date_text != (original.get("transaction_date") or ""):
            corrected["transaction_date"] = date_text or None
        if currency != (original.get("currency") or ""):
            corrected["currency"] = currency or None
    note = st.text_area("Reviewer note", placeholder="Optional context for the audit log.")
    submitted = st.form_submit_button("Submit decision", type="primary")

if submitted:
    if not reviewer.strip():
        st.error("Enter a reviewer id before submitting.")
    else:
        payload = {
            "reviewer": reviewer.strip(),
            "decision": decision,
            "note": note or None,
        }
        if decision == "correct":
            payload["corrected"] = corrected
            if not corrected:
                st.error("Change at least one field for a correction.")
                st.stop()
        try:
            result = decide(int(item["id"]), payload)
            st.success(
                f"Recorded {result['decision']} for {result['document_id']} "
                f"({len(result.get('corrections', []))} field corrections)."
            )
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:  # pragma: no cover - operator feedback
            st.error(f"Decision failed: {exc}")
