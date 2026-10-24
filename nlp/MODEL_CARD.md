# Model Card — BERT/RoBERTa GL-Coding Classifier

Encoder-only text classifier for LedgerLoop general-ledger (GL) coding. Maps
merchant / amount / date / description text to one of 13 GL category codes.

## Quick facts

| Property | Value |
|---|---|
| Task | Multi-class text classification (13 GL categories) |
| Base models | `bert-base-uncased`, `roberta-base` |
| Fine-tune data | LedgerLoop GL corpus (152 train) + Dolly (175 train) = 327 |
| Eval data | 55 held-out (30 domain / unseen-merchant + 25 Dolly) |
| Framework | PyTorch + HuggingFace Transformers |
| Input max length | 128 tokens |

## GL categories (13)

TRAVEL, MEALS, OFFICE_SUPPLIES, SOFTWARE, HARDWARE, MARKETING,
PROFESSIONAL_SERVICES, UTILITIES, RENT, INSURANCE, TRAINING,
ENTERTAINMENT, OTHER.

## Training

See [`nlp/train_bert.py`](nlp/train_bert.py). Defaults: 3 epochs, batch 16,
lr 2e-5, linear warmup (3%), AdamW. Trains on CPU or CUDA.

## Results

Evaluation on the 30 unseen-merchant LedgerLoop rows (domain exact match):

| Model | Domain exact match | Domain macro F1 | Overall accuracy | Overall macro F1 |
|---|---:|---:|---:|---:|
| BERT base (seed 42) | TBD | TBD | TBD | TBD |
| RoBERTa base (seed 42) | TBD | TBD | TBD | TBD |

Comparison with the existing decoder-only QLoRA Qwen2.5-1.5B baseline
(from `sft/results/`):

| Model | Domain exact match | Domain F1 | Notes |
|---|---:|---:|---|
| Qwen2.5-1.5B base | 26.7% | 26.7% | Zero-shot prompted |
| Qwen2.5-1.5B + QLoRA (seed 42) | 86.7% | 86.7% | 152 domain examples, 4-bit NF4 |
| Qwen2.5-1.5B + QLoRA (3-seed mean) | 88.9% ± 1.9 | 88.9% ± 1.9 | Seeds 42/1/2 |
| BERT/RoBERTa (this work) | — | — | Encoder-only classification head |

Fill in TBD after the remote GPU training run completes.

## Inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./nlp/outputs/final")
tokenizer = AutoTokenizer.from_pretrained("./nlp/outputs/final")

text = "Merchant: Microsoft\nAmount: USD 1312.96\nDate: 2026-04-07\nDescription: Microsoft 365 licenses"
enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
with torch.inference_mode():
    logits = model(**enc).logits
pred = logits.argmax(dim=-1).item()
print(model.config.id2label[pred])  # → SOFTWARE
```

## Files

```
nlp/
├── train_bert.py     # fine-tuning script (CPU + CUDA)
├── evaluate_bert.py  # evaluation script
├── MODEL_CARD.md     # this file
├── outputs/          # fine-tuned model checkpoints (gitignored)
├── results/          # eval JSON + metrics (gitignored)
└── tests/            # unit tests
```

## Citation

If you use this model card or training script, please reference the LedgerLoop
repository.
