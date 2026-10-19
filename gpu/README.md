# GPU jobs (Phase 6 — fine-tuning & serving)

Scripts here run **on the rented GPU box** (see `RUNBOOK.md`), not in the repo's
local `ai-service` environment. They use the instance's PyTorch venv.

The Vast vLLM image ships a `transformers` build that is incompatible with the
`peft`/`trl`/`datasets` release train, so run the fine-tune in an **isolated venv**
that is separate from the serving stack:

```bash
# on the Vast instance — isolated training env (does not touch /venv/main)
/venv/main/bin/python -m venv --system-site-packages /workspace/train-venv
/workspace/train-venv/bin/pip install -U pip
/workspace/train-venv/bin/pip install \
    "transformers==4.57.6" "peft==0.21.2" "datasets==4.8.5" \
    "bitsandbytes==0.50.2" "trl==1.16.0" accelerate \
    "huggingface_hub==0.36.2" hf_transfer
```

`--system-site-packages` reuses the image's CUDA-enabled `torch`, so nothing in
`/venv/main` (the serving stack) is touched. Verified on the Vast vLLM 0.30.0
image (torch 2.13.0+cu130).

## Files

| file | purpose |
|---|---|
| `coding_data.py` | builds the GL-coding dataset (merchant × description → category); pure-python, runs anywhere |
| `finetune_qlora.py` | QLoRA-fine-tunes a 0.5–1.5B model on the pairs → LoRA adapter |
| `eval_qlora.py` | evaluates base vs adapter accuracy on the held-out split |
| `eval_coding_compare.py` | four-way comparison (base/tuned 0.5B, prompted 7B, 7B+retrieval) on one held-out set → `runs/coding_compare.json` + figure |
| `serve_adapter.py` | serves the adapter over HTTP (`transformers` + `peft`, FastAPI) |
| `check_gpu.py` | `make gpu-check` — verifies the served vLLM endpoint |
| `RUNBOOK.md` | how to rent + launch vLLM |

## Phase 6 — fine-tune a small GL-coding classifier

Task: map `"<description> - <merchant>"` → one of the 13 GL categories, the same
job `/code` does. A tiny model (default `Qwen/Qwen2.5-0.5B-Instruct`) QLoRA-tuned
on in-domain pairs is compared against its own base model.

```bash
# 1. dataset (also runs locally, no GPU)
python gpu/coding_data.py                 # -> gpu/data/coding_{train,val}.jsonl

# 2. train (4-bit NF4 QLoRA, ~minutes on a 4090)
python gpu/finetune_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 3 --out /workspace/expenseiq-qlora

# 3. evaluate base vs adapter
python gpu/eval_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter /workspace/expenseiq-qlora --out runs/qlora_eval.json

# 4. serve the adapter over HTTP
python gpu/serve_adapter.py --adapter /workspace/expenseiq-qlora --port 8001
```

`finetune_qlora.py` writes `train_meta.json` (loss, rows, minutes) next to the
adapter; `eval_qlora.py` writes base vs adapter accuracy and the `delta`.

Result on the rented 4090 (Qwen2.5-0.5B-Instruct, 4-bit NF4, 3 epochs): base
**10.9%** → adapter **98.4%** on 64 held-out coding pairs (**+87.5 pts**),
train_loss 0.36 in ~40 s.

> **Leakage caveat.** In the default random split each merchant maps to exactly
> one category, so the merchant token alone reveals the answer — a trivial
> merchant→category lookup scores **100%**, and the 10.9%→98.4% number is really
> a *same-model ablation* (adapter on/off), not a generalisation claim. Use the
> unseen-merchant split below for the honest comparison.

### Four-way comparison — unseen-merchant split

`coding_data.py` holds out whole merchants (`UNSEEN_MERCHANTS`); the held-out
merchants never appear in training, so the merchant-lookup baseline drops to
**0%**. `eval_coding_compare.py` scores the **same 75 held-out pairs** with four
variants, timing every call and turning throughput into cost-per-call
(`GPU $/hr ÷ throughput`), and writes `runs/coding_compare.json` (+ a
three-panel accuracy/latency/cost figure):

```bash
python gpu/finetune_qlora.py --split unseen --out /workspace/expenseiq-qlora-unseen
python gpu/eval_coding_compare.py --split unseen --adapter /workspace/expenseiq-qlora-unseen \
    --vllm-base-url http://127.0.0.1:18000/v1 --rate 0.35 --out runs/coding_compare.json
```

| variant | accuracy | mean latency | cost / call | notes |
|---|---|---|---|---|
| base 0.5B | 10.7% | 260 ms | $0.000025 | off-the-shelf, no adapter, plain prompt |
| tuned 0.5B | 60.0% | 178 ms | $0.000017 | + QLoRA adapter; no retrieval, single small model |
| prompted 7B | 93.3% | 93 ms | $0.000009 | zero-shot, given the 13-label list, no retrieval |
| prompted 7B + retrieval | 100.0% | 89 ms | $0.000009 | + lexical candidates from the merchant KB |

Honest read: on genuinely unseen merchants the 0.5B adapter (**60%**) does *not*
match a prompted 7B (**93%**); retrieval is what makes the 7B near-perfect. The
remaining 0.5B errors are merchant-bound (e.g. `laptop - IKEA → IKEA`), i.e. it
still latches onto the merchant rather than the description.

Latency/cost caveat: the vLLM-served 7B is *faster* per call than the naive
`transformers.generate()` 0.5B path here (continuous batching + paged attention
on the same GPU), so **serving stack dominates latency, not parameter count**. In
this harness the 0.5B's real edge is memory/co-tenancy and needing no retrieval
index, not throughput.

### Serve the adapter

`serve_adapter.py` loads the base model + LoRA adapter and exposes
`POST /classify {text}` (plus `GET /health`) — see `evidence/logs/qlora.txt`.

vLLM can also hot-load the LoRA via `--enable-lora --lora-modules coding=...`,
**but** the image's `transformers` build must first be made `peft`-compatible;
until then `serve_adapter.py` (transformers + peft) is the working path.

