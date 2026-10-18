"""POST /corrections — user corrections flow back into the golden eval set."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter

from ..config import REPO_ROOT
from ..metrics import STORE
from ..schemas import CorrectionRequest, CorrectionResponse

router = APIRouter(tags=["feedback"])

CORRECTIONS_PATH = REPO_ROOT / "evals" / "golden" / "corrections.jsonl"


@router.post("/corrections", response_model=CorrectionResponse)
async def submit_correction(req: CorrectionRequest) -> CorrectionResponse:
    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "document_id": req.document_id,
        "operator": req.operator,
        "corrected": req.corrected.model_dump(),
        "source": "user_correction",
    }
    with CORRECTIONS_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    STORE.incr("corrections_captured")
    try:
        golden_path = str(CORRECTIONS_PATH.relative_to(REPO_ROOT))
    except ValueError:
        golden_path = str(CORRECTIONS_PATH)
    return CorrectionResponse(
        document_id=req.document_id,
        appended_to_golden=True,
        golden_path=golden_path,
    )
