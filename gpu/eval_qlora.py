"""Phase 6 — evaluate the QLoRA classifier against its base model.

Runs the held-out coding pairs through (a) the base model and (b) the LoRA
adapter, greedy-decodes the category, and reports accuracy per variant.

    python gpu/eval_qlora.py --adapter /workspace/expenseiq-qlora
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coding_data import CATEGORIES, build, build_unseen


def _predict(model, tok, prompt: str, max_new: int = 8) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    up = gen.upper()
    for cat in CATEGORIES:
        if cat in up:
            return cat
    return up.strip().split("\n")[0][:24] or "OTHER"


def _score(model, tok, val: list[dict]) -> tuple[float, list[dict]]:
    correct, detail = 0, []
    for row in val:
        pred = _predict(model, tok, row["prompt"])
        ok = pred == row["label"]
        correct += ok
        detail.append({"text": row["text"], "label": row["label"], "pred": pred, "ok": ok})
    return correct / max(1, len(val)), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--split", choices=["random", "unseen"], default="random")
    ap.add_argument("--out", default="runs/qlora_eval.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _, val = (build_unseen() if args.split == "unseen" else build())

    def load(adapter: str):
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        if adapter:
            from peft import PeftModel

            m = PeftModel.from_pretrained(m, adapter)
        m.eval()
        return m

    base = load("")
    base_acc, _ = _score(base, tok, val)
    del base
    torch.cuda.empty_cache()

    result = {"base_model": args.model, "split": args.split, "val_rows": len(val), "base_accuracy": round(base_acc, 4)}
    if args.adapter:
        tuned = load(args.adapter)
        tuned_acc, detail = _score(tuned, tok, val)
        result["adapter"] = args.adapter
        result["adapter_accuracy"] = round(tuned_acc, 4)
        result["delta"] = round(tuned_acc - base_acc, 4)
        result["detail"] = detail

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
