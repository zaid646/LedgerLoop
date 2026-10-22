# Supervised Fine-Tuning (SFT) — instruction tuning for LedgerLoop

A self-contained QLoRA pipeline that instruction-tunes **Qwen2.5-0.5B-Instruct** on a
mix of public instructions and LedgerLoop's expense-coding domain. It runs end-to-end
on a **free Google Colab T4 (16 GB)** — no rented GPU.

This complements `gpu/finetune_qlora.py` (a classification-style QLoRA job aimed at a
rented GPU). `sft/` uses HuggingFace `trl.SFTTrainer`, a public instruction corpus, and
measures instruction-following rather than only label accuracy.

## Results (checked in under `results/`)

Evaluated on **30 held-out LedgerLoop transactions** (unseen merchants) and **25 held-out
Dolly instructions**:

| Metric | Base Qwen2.5-0.5B-Instruct | + LedgerLoop LoRA | Delta |
|---|---:|---:|---:|
| Domain exact match | 6.7% | **63.3%** | **+56.7 pts** |
| Domain token F1 | 6.7% | **63.3%** | +56.7 pts |
| Format adherence (only one code) | 6.7% | **63.3%** | +56.7 pts |
| Domain completion loss | 1.232 | **0.266** | **−78.4%** |
| Dolly completion loss | 2.333 | **1.906** | −18.3% |
| Overall completion loss | 2.197 | **1.997** | −9.1% |
| Overall perplexity | 9.00 | **7.37** | −18.1% |

Training run: **63 optimizer steps**, **5m23s** on a free Colab T4, train loss **1.132**,
SFTTrainer validation loss **0.803** (full-sequence; epoch 1: 0.846, epoch 2: 0.805,
epoch 3: 0.803). The loss curve is `results/loss_curve.png`; the raw step log is
`results/trainer_log_history.json` and `results/loss_log.csv`.

> Evaluation loss is higher than the Trainer's `eval_loss` because `evaluate_sft.py`
> masks the prompt and scores only completion tokens.

The base model's predictions collapse onto a few frequent labels (`MEALS`, `TRAVEL`,
`OTHER`). The adapter learns the 13-code taxonomy and the merchant/description mapping,
while the lower Dolly loss shows that general instruction-following did not collapse.

## Dataset

| Source | Train | Held out | Notes |
|---|---:|---:|---|
| LedgerLoop synthetic GL coding | 152 | 30 | 182 unique instruction-response pairs across 13 GL categories |
| Databricks Dolly 15k | 175 | 25 | 200 used from the 320-row checked-in sample; no-context examples |
| **Total** | **327** | **55** | |

- **Domain corpus**: generated deterministically by `domain_data.py` from the same GL
  taxonomy as `ai-service/app/schemas.py` (`TRAVEL`, `MEALS`, `OFFICE_SUPPLIES`,
  `SOFTWARE`, `HARDWARE`, `MARKETING`, `PROFESSIONAL_SERVICES`, `UTILITIES`, `RENT`,
  `INSURANCE`, `TRAINING`, `ENTERTAINMENT`, `OTHER`). Each merchant appears exactly once,
  so the held-out merchants are unseen by construction — this is a genuine
  merchant-generalization test, not a row-random leak.
- **Public corpus**: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
  (CC BY-SA 3.0). `data/public_dolly_sample.jsonl` is a checked-in 320-row sample;
  `format_dataset.py` uses the first 200 by default. If the file is absent, the script
  downloads the dataset and recreates it.
- **Formatting**: Qwen ChatML. Each record stores `prompt`, `completion` and the combined
  `text`. The training text ends with `<|im_end|>`:

```text
<|im_start|>system
You are LedgerLoop's expense-coding assistant. ...<|im_end|>
<|im_start|>user
Classify this business transaction ... Allowed categories: ...<|im_end|>
<|im_start|>assistant
MEALS<|im_end|>
```

Training uses the combined text with full-sequence language-model loss. The evaluator
uses the same prompt and masks it when computing completion loss.

## Hyperparameters

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Quantization | 4-bit NF4, double quantization, fp16 compute (QLoRA) |
| LoRA | r=16, alpha=32, dropout=0.05, no bias, all linear projections |
| Learning rate | 2e-4, cosine schedule, 3% warmup (2 steps) |
| Epochs / batch | 3 epochs; batch 2 × grad-accum 8 = effective batch 16 |
| Sequence length | 512, no packing |
| Optimizer | `paged_adamw_8bit`, max grad norm 0.3, weight decay 0.01 |
| Precision | 4-bit base + fp32 LoRA adapters (AMP scaler disabled; see caveats) |
| Seed | 42 |
| Runtime | ~5–6 min training on a free T4; adapter saved to `outputs/.../final` |

## Run on a free Colab T4

1. Open <https://colab.research.google.com>, choose **Runtime → Change runtime type →
   T4 GPU**, and verify with `!nvidia-smi`.
2. Clone the repo and install the pinned environment:

```bash
git clone https://github.com/zaid646/ledgerloop.git
cd ledgerloop
git checkout feature/sft-instruction-tuning
pip install -q -r sft/requirements.txt
```

3. Build the data (uses the checked-in Dolly sample; downloads only if missing):

```bash
python sft/format_dataset.py
```

4. Fine-tune with QLoRA:

```bash
python sft/train_sft.py
```

5. Evaluate the base model and the adapter:

```bash
python sft/evaluate_sft.py --output sft/results/eval_base.json
python sft/evaluate_sft.py --adapter sft/outputs/qwen2.5-0.5b-ledgerloop-lora/final
```

The adapter, tokenizer and `adapter_config.json` are written to
`sft/outputs/qwen2.5-0.5b-ledgerloop-lora/final/`. Training metrics and the loss curve
are written to `sft/results/`.

### CPU / local smoke test

```bash
# Some Colab images preinstall torchao 0.10, which peft 0.21 rejects for
# non-quantized LoRA. Remove it for the CPU path only:
pip uninstall -y torchao

python sft/format_dataset.py --no-public
python sft/train_sft.py --no-4bit --device cpu --dry-run
python -m unittest discover -s sft/tests
```

`--dry-run` runs 10 steps with a 256-token limit. 4-bit is automatically disabled when
CUDA is unavailable. The CPU path was smoke-tested with one step (train loss 3.515,
`/tmp` output); the T4 path is the supported full run.

## Files

```
sft/
├── format_dataset.py    # public + domain data → train.jsonl / val.jsonl
├── domain_data.py       # deterministic 182-pair LedgerLoop GL-coding corpus
├── train_sft.py         # trl.SFTTrainer + peft QLoRA (free T4 defaults)
├── evaluate_sft.py      # loss, perplexity, exact match, token F1, format adherence
├── requirements.txt     # pinned Colab T4 environment
├── data/                # checked-in Dolly sample + generated train/val JSONL
├── results/             # metrics, loss curve, logs, Colab runtime evidence
└── tests/               # stdlib data/formatting tests
```

## Why this demonstrates the capability

- **Instruction fine-tuning**: the model is trained on prompt→response pairs with a
  chat template, not just a classification head. It learns to obey "return exactly one
  category code" and its Dolly completion loss falls by 18%.
- **Domain adaptation**: a generic 0.5B instruct model scores 6.7% on unseen
  merchant/expense pairs; QLoRA on 152 in-domain examples raises it to 63.3% while
  using only a free T4.
- **Measured, reproducible**: fixed seed, pinned versions, checked-in data sample,
  metrics JSON, training log, loss curve and runtime evidence.

## Caveats and next steps

- The domain labels are synthetic and some are inherently ambiguous (e.g. "customer
  dinner and show" could be `MEALS` or `ENTERTAINMENT`). Strict exact match understates
  quality on those cases.
- The held-out domain set has only 30 examples, so one example is 3.3 points. Treat the
  number as a smoke-level estimate, not a benchmark; run multiple seeds for confidence
  intervals.
- The general-instruction check is a 25-example loss measurement, not an accuracy
  benchmark.
- Mixed-precision autocast was disabled because Colab's PyTorch 2.11/CUDA 13 stack hit a
  `GradScaler`/bfloat16 incompatibility on T4. The base is still 4-bit NF4 and only LoRA
  adapters train, so this remains QLoRA.
- Completion-only loss was tried and underperformed (46.7% vs 63.3%), so full-sequence
  loss is used.
- Next: grow the domain set from human-reviewed corrections and the existing golden
  evals, hold out whole categories as well as merchants, compare 1.5B/7B and retrieval
  baselines, merge the adapter, and serve it through `gpu/serve_adapter.py`.

## Reproduce the checked-in numbers

The checked-in results came from a free Colab T4 session (`results/colab_runtime.txt`):
Tesla T4 16 GB, Python 3.13.15, torch 2.11.0+cu130, transformers 5.20.1, trl 1.16.0,
peft 0.21.2, datasets 4.8.5, bitsandbytes 0.50.2. No paid or rented GPU was used.
