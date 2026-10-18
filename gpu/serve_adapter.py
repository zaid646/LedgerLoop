"""Phase 6 — serve the fine-tuned GL-coding adapter over HTTP.

Minimal classifier endpoint that loads the base Qwen model plus the QLoRA
adapter and greedy-decodes a GL category for a receipt description. Demonstrates
serving the fine-tuned model (the vLLM image ships a transformers build that is
incompatible with peft, so we serve via transformers+peft here).

    python gpu/serve_adapter.py --adapter /workspace/expenseiq-qlora --port 8001
"""
from __future__ import annotations

import argparse
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from coding_data import CATEGORIES, PROMPT


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    text: str
    category: str
    latency_ms: float


def load(model_id: str, adapter: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def make_app(model, tok) -> FastAPI:
    app = FastAPI(title="ExpenseIQ GL-coding adapter")

    @app.get("/health")
    def health():
        return {"status": "ok", "categories": CATEGORIES}

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest):
        t0 = time.perf_counter()
        prompt = PROMPT.format(text=req.text)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).upper()
        category = next((c for c in CATEGORIES if c in gen), gen.strip().split("\n")[0][:24] or "OTHER")
        return ClassifyResponse(text=req.text, category=category, latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    model, tok = load(args.model, args.adapter)
    uvicorn.run(make_app(model, tok), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
