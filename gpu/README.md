# GPU jobs (Phase 6 — fine-tuning & serving)

Scripts here run **on the rented GPU box** (see `RUNBOOK.md`), not in the repo's
local `ai-service` environment. They use the instance's PyTorch venv.

The Vast vLLM image ships a `transformers` build that is incompatible with the
`peft`/`trl`/`datasets` release train, so run the fine-tune in an **isolated venv**
that is separate from the serving stack:

```bash
# on the Vast instance — isolated training env (does not touch /venv/main)
/venv/main/bin/python -m venv --system-site-packages /workspace/train-venv
uv pip install --python /workspace/train-venv/bin/python \
    "transformers>=4.46,<5" "datasets>=3,<5" "huggingface_hub>=0.34,<1.0" \
    peft bitsandbytes accelerate hf_transfer
# the image's torch/torchvision pair is mismatched; align a matched pair
uv pip install --python /workspace/train-venv/bin/python --upgrade torch torchvision
```

## Files

| file | purpose |
|---|---|
| `coding_data.py` | builds the GL-coding dataset (merchant × description → category); pure-python, runs anywhere |
| `finetune_qlora.py` | QLoRA-fine-tunes a 0.5–1.5B model on the pairs → LoRA adapter |
| `eval_qlora.py` | evaluates base vs adapter accuracy on the held-out split |
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

### Serve the adapter

`serve_adapter.py` loads the base model + LoRA adapter and exposes
`POST /classify {text}` (plus `GET /health`) — see `evidence/logs/qlora.txt`.

vLLM can also hot-load the LoRA via `--enable-lora --lora-modules coding=...`,
**but** the image's `transformers` build must first be made `peft`-compatible;
until then `serve_adapter.py` (transformers + peft) is the working path.

