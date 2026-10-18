#!/usr/bin/env python3
"""Answer-correctness eval for the Phase 5 spend assistant.

Asks each question in ``evals/assistant_eval.jsonl`` against a running AI
service and scores five checks per question:

* ``sql_valid``       — the returned SQL passes the guard (single read-only SELECT)
* ``relation``        — the SQL targets the expected relation
* ``rows``            — the query returned at least one row
* ``answer_present``  — a prose answer was produced (ANSWER_SUMMARY on)
* ``answer_grounded`` — the answer contains a number / label present in the rows

    cd ai-service && uv run python ../evals/run_assistant_eval.py --url http://localhost:8080
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-service"))

from app.sql_guard import guard_sql  # noqa: E402

EVAL_FILE = REPO_ROOT / "evals" / "assistant_eval.jsonl"
RESULTS_DIR = REPO_ROOT / "evals" / "results"

CATEGORIES = {
    "TRAVEL", "MEALS", "OFFICE_SUPPLIES", "SOFTWARE", "HARDWARE", "MARKETING",
    "PROFESSIONAL_SERVICES", "UTILITIES", "RENT", "INSURANCE", "TRAINING",
    "ENTERTAINMENT", "OTHER",
}


def _num_variants(x: float) -> set[str]:
    return {f"{x:g}", f"{x:.2f}"}


def _grounding_tokens(columns: list[str], rows: list[list]) -> set[str]:
    toks: set[str] = set()
    numbers: list[float] = []
    for row in rows:
        for cell in row:
            if isinstance(cell, bool) or cell is None:
                continue
            if isinstance(cell, (int, float)):
                numbers.append(float(cell))
                toks |= _num_variants(float(cell))
            else:
                s = str(cell)
                toks.add(s.lower())
                if s.upper() in CATEGORIES:
                    toks.add(s.upper())
                try:
                    numbers.append(float(s))
                    toks |= _num_variants(float(s))
                except ValueError:
                    pass
    # The model legitimately aggregates: accept the sum of the returned values.
    if numbers:
        toks |= _num_variants(sum(numbers))
    return {t for t in toks if t}


def score_question(row: dict, resp: dict) -> dict:
    sql = resp.get("sql", "") or ""
    columns = resp.get("columns", []) or []
    rows = resp.get("rows", []) or []
    answer = (resp.get("answer", "") or "").strip()

    try:
        guard_sql(sql)
        sql_valid = True
    except Exception:  # noqa: BLE001
        sql_valid = False

    relation = bool(sql) and row["expect_relation"] in sql.lower()
    nonempty = len(rows) >= 1

    toks = _grounding_tokens(columns, rows)
    answer_present = bool(answer)
    norm_answer = answer.replace(",", "").lower()
    grounded = answer_present and any(t in norm_answer for t in toks)

    checks = {
        "sql_valid": sql_valid,
        "relation": relation,
        "rows": nonempty,
        "answer_present": answer_present,
        "answer_grounded": grounded,
    }
    return {
        "question": row["question"],
        "checks": checks,
        "score": sum(checks.values()) / len(checks),
        "sql": sql,
        "answer": answer,
        "row_count": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not EVAL_FILE.exists():
        print(f"missing {EVAL_FILE}", file=sys.stderr)
        return 2
    questions = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]

    results = []
    with httpx.Client(timeout=args.timeout) as client:
        for row in questions:
            r = client.post(f"{args.url.rstrip('/')}/assistant/ask", json={"question": row["question"]})
            if r.status_code != 200:
                results.append({"question": row["question"], "error": f"{r.status_code}: {r.text[:200]}", "score": 0.0})
                continue
            results.append(score_question(row, r.json()))

    overall = sum(r["score"] for r in results) / max(1, len(results))
    passed = sum(1 for r in results if r["score"] == 1.0)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": args.url,
        "n": len(results),
        "mean_score": round(overall, 4),
        "fully_passed": passed,
        "results": results,
    }

    out = Path(args.out) if args.out else RESULTS_DIR / f"assistant-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"assistant answer-correctness: mean={overall:.2%} fully_passed={passed}/{len(results)} -> {out}")
    for r in results:
        flag = "ok " if r["score"] == 1.0 else "!! "
        print(f"  {flag}{r['question'][:48]:50s} score={r['score']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
