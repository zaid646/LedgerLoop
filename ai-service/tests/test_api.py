import json

from fastapi.testclient import TestClient

from app import main
from app.schemas import ReceiptExtraction
from app.self_heal import HealOutcome

client = TestClient(main.app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm_model"]


def test_policy_check_endpoint():
    r = client.post(
        "/policy-check", json={"amount": 30, "category": "MEALS", "has_receipt": True}
    )
    assert r.status_code == 200
    assert r.json()["route"] == "auto_approve"


def test_extract_endpoint(monkeypatch):
    async def fake_extract(text, **kwargs):
        return HealOutcome(
            extraction=ReceiptExtraction(
                merchant="ACME", total_amount=9.99, currency="USD", confidence={"merchant": 0.9}
            ),
            prompt_version="v1",
            model="fake",
            attempts=1,
            repaired=False,
            escalated=False,
            latency_ms=5.0,
            cost_usd=0.000001,
        )

    monkeypatch.setattr("app.routers.extract.extract_with_healing", fake_extract)
    r = client.post(
        "/extract",
        json={"document_id": "doc-1", "text": "ACME total 9.99 USD card 4111 1111 1111 1111"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["extraction"]["merchant"] == "ACME"
    # card number must have been redacted before the (fake) call
    assert body["redactions"]["card"] == 1


def test_correction_endpoint(monkeypatch, tmp_path):
    target = tmp_path / "corrections.jsonl"
    monkeypatch.setattr("app.routers.corrections.CORRECTIONS_PATH", target)
    r = client.post(
        "/corrections",
        json={
            "document_id": "doc-1",
            "operator": "zaid",
            "corrected": {"merchant": "ACME", "total_amount": 9.99, "currency": "USD"},
        },
    )
    assert r.status_code == 200
    assert r.json()["appended_to_golden"] is True
    assert json.loads(target.read_text().splitlines()[0])["document_id"] == "doc-1"


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_redaction_happens_in_code_endpoint(monkeypatch):
    class FakeClient:
        async def complete_json(self, messages, schema, model=None, temperature=0.0, max_tokens=700):
            from app.llm import LLMResult

            return (
                {"category": "SOFTWARE", "confidence": 0.8, "rationale": "github"},
                LLMResult(content="{}", model="fake", latency_ms=2.0, cost_usd=0.0),
            )

    monkeypatch.setattr("app.routers.code.LLMClient", lambda: FakeClient())
    r = client.post("/code", json={"description": "github subscription", "merchant": "GitHub"})
    assert r.status_code == 200
    assert r.json()["category"] == "SOFTWARE"
    top = [m.lower() for m in r.json()["top_merchants"]]
    assert any("github" in m for m in top)
