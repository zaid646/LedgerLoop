#!/usr/bin/env python3
"""Verify the self-hosted vLLM endpoint and log a GPU session row.

Usage:
    python gpu/check_gpu.py                 # uses .env
    python gpu/check_gpu.py --base-url http://ip:port/v1 --api-key KEY --model M
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_LOG = REPO_ROOT / "runs" / "gpu_sessions.csv"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (does not override already-set env vars)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        os.environ.setdefault(key, val)


def _req(url: str, key: str, payload: dict | None = None, timeout: int = 30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--rate", type=float, default=float(os.getenv("GPU_RATE_USD_HR", "0") or 0))
    ap.add_argument("--note", default="check")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    ok = True

    # 1. /models
    try:
        models = _req(f"{base}/models", args.api_key)
        ids = [m.get("id") for m in models.get("data", [])]
        print(f"[ok] /v1/models -> {ids}")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"[FAIL] /v1/models unreachable: {e}")
        return 2

    # 2. schema-guided completion
    schema = {
        "type": "object",
        "properties": {
            "merchant": {"type": "string"},
            "total": {"type": "number"},
            "currency": {"type": "string"},
        },
        "required": ["merchant", "total", "currency"],
    }
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Extract receipt fields. Reply ONLY with JSON."},
            {"role": "user", "content": "Receipt: 'ACME STORE', total 12.50 USD."},
        ],
        "temperature": 0,
        "max_tokens": 128,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "receipt", "schema": schema},
        },
    }
    try:
        out = _req(f"{base}/chat/completions", args.api_key, payload, timeout=60)
        content = out["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        print(f"[ok] guided JSON completion -> {parsed}")
        if not {"merchant", "total", "currency"} <= parsed.keys():
            print("[warn] completion missing expected keys")
    except Exception as e:  # noqa: BLE001 - report and fail loudly
        print(f"[FAIL] chat completion failed: {e}")
        ok = False

    # 3. session log
    SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not SESSION_LOG.exists()
    with SESSION_LOG.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc", "base_url", "model", "rate_usd_hr", "status", "note"])
        w.writerow(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                base,
                args.model,
                args.rate,
                "ok" if ok else "fail",
                args.note,
            ]
        )
    print(f"[ok] logged session -> {SESSION_LOG.relative_to(REPO_ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
