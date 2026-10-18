"""POST /code — assign a GL category to a transaction (LLM + retrieval hints)."""
from __future__ import annotations

from fastapi import APIRouter

from .. import search
from ..llm import LLMClient
from ..metrics import STORE
from ..redaction import redact
from ..registry import PromptRegistry
from ..schemas import CodeRequest, CodeResponse, GLCategory
from ..tracing import get_tracer

router = APIRouter(tags=["coding"])

# Seed merchant -> category hints. Used as a fallback when OpenSearch hybrid
# retrieval is unavailable (e.g. CI); the `expenseiq-merchants` index is the
# primary source of retrieval hints.
MERCHANT_HINTS: dict[str, GLCategory] = {
    "uber": GLCategory.TRAVEL,
    "lyft": GLCategory.TRAVEL,
    "delta": GLCategory.TRAVEL,
    "airbnb": GLCategory.TRAVEL,
    "starbucks": GLCategory.MEALS,
    "restaurant": GLCategory.MEALS,
    "doordash": GLCategory.MEALS,
    "github": GLCategory.SOFTWARE,
    "aws": GLCategory.SOFTWARE,
    "slack": GLCategory.SOFTWARE,
    "staples": GLCategory.OFFICE_SUPPLIES,
    "apple": GLCategory.HARDWARE,
    "dell": GLCategory.HARDWARE,
    "google ads": GLCategory.MARKETING,
}

_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": [c.value for c in GLCategory]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["category", "confidence"],
}


def _retrieve(merchant: str | None, description: str) -> tuple[list[str], list[GLCategory]]:
    """Hybrid-retrieve neighbouring merchants; return (names, candidate categories).

    Falls back to the seeded ``MERCHANT_HINTS`` when OpenSearch has no hits.
    """
    query = f"{merchant or ''} {description}".strip()
    hits = search.search_merchants(query) if query else []
    m = (merchant or "").lower()

    names: list[str] = []
    candidates: list[GLCategory] = []
    for h in hits:
        if h.get("name"):
            names.append(h["name"])
        try:
            cat = GLCategory(h.get("category", "OTHER"))
        except ValueError:
            continue
        if cat not in candidates:
            candidates.append(cat)

    for key, cat in MERCHANT_HINTS.items():
        if key in m:
            if cat not in candidates:
                candidates.append(cat)
            if not names:
                names.append(key)

    return names, candidates


@router.post("/code", response_model=CodeResponse)
async def code(req: CodeRequest) -> CodeResponse:
    tracer = get_tracer()
    registry = PromptRegistry()
    prompt = registry.resolve("coding", "prod")
    client = LLMClient()

    desc = redact(req.description).text
    top, candidates = _retrieve(req.merchant, desc)
    categories = ", ".join(c.value for c in GLCategory)

    messages = [
        {"role": "system", "content": prompt.system},
        {
            "role": "user",
            "content": prompt.render(
                merchant=req.merchant or "",
                description=desc,
                amount=req.amount if req.amount is not None else "",
                categories=categories,
            ),
        },
    ]
    data, result = await client.complete_json(messages, schema=_CODE_SCHEMA)

    try:
        category = GLCategory(data.get("category", "OTHER"))
    except ValueError:
        category = GLCategory.OTHER

    STORE.observe("/code", result.latency_ms, result.cost_usd)
    STORE.incr("code_requests")

    tracer.generation(
        name="code",
        model=result.model,
        input={"description": desc, "merchant": req.merchant},
        output=data,
        usage=result.usage,
        metadata={"prompt_version": prompt.version},
    )

    return CodeResponse(
        category=category,
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rationale=str(data.get("rationale", "")),
        candidates=candidates,
        prompt_version=prompt.version,
        model=result.model,
        latency_ms=round(result.latency_ms, 2),
        cost_usd=round(result.cost_usd, 8),
        top_merchants=top,
    )
