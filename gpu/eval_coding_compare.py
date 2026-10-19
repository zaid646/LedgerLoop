"""Phase 6 — four-way GL-coding comparison on one held-out set.

Scores the *same* held-out pairs (default: unseen-merchant split) with four
variants so the QLoRA result can be read honestly against a prompted 7B:

    base 0.5B          Qwen2.5-0.5B-Instruct, greedy, no adapter
    tuned 0.5B         + the QLoRA adapter
    prompted 7B        Qwen2.5-7B-Instruct zero-shot (same plain prompt)
    prompted 7B + RAG  same 7B, prompt augmented with lexical retrieval
                       candidates from the merchant knowledge base

Writes ``runs/coding_compare.json`` and (with ``--render``) a bar-chart figure.

Run on the GPU box (7B served by vLLM):

    python gpu/eval_coding_compare.py --split unseen \
        --adapter /workspace/expenseiq-qlora-unseen --out runs/coding_compare.json

Render locally:

    uv run --with matplotlib python gpu/eval_coding_compare.py --render \
        --out runs/coding_compare.json --png evidence/png/15-coding-compare.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

from coding_data import CATEGORIES, DESCRIPTIONS, MERCHANTS, build, build_unseen

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# retrieval (lexical, over the merchant knowledge base)                        #
# --------------------------------------------------------------------------- #
def _corpus() -> list[dict[str, str]]:
    """One entry per (merchant, description) in the seed corpus."""
    out: list[dict[str, str]] = []
    for merchant, category in MERCHANTS.items():
        out.append({"merchant": merchant, "desc": "", "label": category})
        for desc in DESCRIPTIONS.get(category, []):
            out.append({"merchant": merchant, "desc": desc, "label": category})
    return out


def retrieve(text: str, k: int = 3) -> list[dict[str, str]]:
    """Return up to ``k`` knowledge-base entries ranked by token overlap."""
    lower = text.lower()
    qtok = set(lower.replace("-", " ").split())
    scored = []
    for entry in _corpus():
        hay = f"{entry['desc']} {entry['merchant']}".lower()
        htok = set(hay.replace("-", " ").split())
        overlap = len(qtok & htok)
        exact = 1 if entry["merchant"].lower() in lower else 0
        score = exact * 100 + overlap
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda s: s[0], reverse=True)
    picked, seen = [], set()
    for _, entry in scored:
        if entry["label"] in seen:
            continue
        seen.add(entry["label"])
        picked.append(entry)
        if len(picked) == k:
            break
    return picked


ZS_PROMPT = (
    "Classify this company expense line into exactly one of these GL categories: "
    "{cats}.\nDescription - Merchant: {text}\nAnswer with only the category name."
)

RAG_PROMPT = (
    "You are a corporate expense coder. Choose exactly one GL category for the "
    "expense line from this list: {cats}.\n"
    "Retrieved candidates from the merchant knowledge base (may hint the answer):\n"
    "{hints}\n"
    "Expense line: {text}\n"
    "Answer with only the category name."
)


def _zs_prompt(text: str) -> str:
    return ZS_PROMPT.format(cats=", ".join(CATEGORIES), text=text)


def _rag_prompt(text: str, k: int = 3) -> str:
    hints = "\n".join(f"- {e['merchant']} ({e['desc']}) -> {e['label']}" for e in retrieve(text, k))
    return RAG_PROMPT.format(cats=", ".join(CATEGORIES), hints=hints or "- none", text=text)


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", s.upper()).strip("_")


def _extract(pred: str) -> str:
    """Map a free-text answer onto the fixed label set, tolerating spaces vs underscores
    (a prompted 7B writes ``OFFICE SUPPLIES``; our labels are ``OFFICE_SUPPLIES``)."""
    up = _norm(pred)
    for cat in CATEGORIES:
        if cat in up:
            return cat
    return pred.strip().split("\n")[0][:24] or "OTHER"


# --------------------------------------------------------------------------- #
# local 0.5B (base + adapter)                                                  #
# --------------------------------------------------------------------------- #
def _local_scores(model_name: str, adapter: str, val: list[dict]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(path: str):
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        if path:
            from peft import PeftModel

            m = PeftModel.from_pretrained(m, path)
        m.eval()
        return m

    def generate(model, prompt: str, max_new: int = 12) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    out = {}
    base = load("")
    out["base_0.5b"] = _score_local(base, tok, val, generate)
    del base
    torch.cuda.empty_cache()
    if adapter:
        tuned = load(adapter)
        out["tuned_0.5b"] = _score_local(tuned, tok, val, generate)
    return out


def _score_local(model, tok, val: list[dict], generate) -> dict:
    detail, correct = [], 0
    for row in val:
        pred = _extract(generate(model, row["prompt"]))
        ok = pred == row["label"]
        correct += ok
        detail.append({"text": row["text"], "label": row["label"], "pred": pred, "ok": ok})
    return {"accuracy": round(correct / max(1, len(val)), 4), "detail": detail}


# --------------------------------------------------------------------------- #
# hosted 7B via vLLM (zero-shot + RAG)                                         #
# --------------------------------------------------------------------------- #
def _chat(base_url: str, api_key: str, model: str, prompt: str, timeout: int = 60) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 12,
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode())
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def _score_remote(base_url, api_key, model, val, prompt_fn) -> dict:
    detail, correct = [], 0
    for row in val:
        pred = _extract(_chat(base_url, api_key, model, prompt_fn(row)))
        ok = pred == row["label"]
        correct += ok
        detail.append({"text": row["text"], "label": row["label"], "pred": pred, "ok": ok})
    return {"accuracy": round(correct / max(1, len(val)), 4), "detail": detail}


def _load_env() -> None:
    p = REPO_ROOT / ".env"
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


# --------------------------------------------------------------------------- #
def _render(json_path: Path, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(json_path.read_text())
    variants = [("base_0.5b", "base 0.5B"), ("tuned_0.5b", "tuned 0.5B"),
                ("prompted_7b", "prompted 7B"), ("prompted_7b_rag", "7B + retrieval")]
    labels = [lbl for key, lbl in variants if key in data["variants"]]
    vals = [data["variants"][key]["accuracy"] * 100 for key, lbl in variants if key in data["variants"]]
    colors = ["#c0392b", "#27ae60", "#2980b9", "#8e44ad"][: len(vals)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("accuracy %")
    ax.set_title(f"GL coding — {data.get('split', 'unseen')} split ({data['val_rows']} held-out pairs)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "unseen"], default="unseen")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--vllm-base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--vllm-model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--vllm-api-key", default=os.getenv("VLLM_API_KEY", "changeme"))
    ap.add_argument("--out", default="runs/coding_compare.json")
    ap.add_argument("--render", action="store_true", help="render the figure from --out instead of scoring")
    ap.add_argument("--png", default="evidence/png/15-coding-compare.png")
    args = ap.parse_args()

    if args.render:
        _render(Path(args.out), Path(args.png))
        return 0

    _, val = (build_unseen() if args.split == "unseen" else build())
    variants: dict[str, dict] = _local_scores(args.base_model, args.adapter, val)
    variants["prompted_7b"] = _score_remote(
        args.vllm_base_url, args.vllm_api_key, args.vllm_model, val, lambda r: _zs_prompt(r["text"]))
    variants["prompted_7b_rag"] = _score_remote(
        args.vllm_base_url, args.vllm_api_key, args.vllm_model, val, lambda r: _rag_prompt(r["text"]))

    result = {
        "split": args.split,
        "val_rows": len(val),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "hosted_model": args.vllm_model,
        "variants": variants,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["accuracy"] for k, v in variants.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
