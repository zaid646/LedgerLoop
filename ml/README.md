# LedgerLoop classical-ML baseline: GL coding

`ml/gl_lightgbm.py` trains LightGBM and XGBoost classifiers on the same domain
split as the QLoRA adapter:

* train: **152** synthetic LedgerLoop instruction pairs (`sft/data/train.jsonl`)
* held-out: **30** unseen-merchant pairs (`sft/data/val.jsonl`)
* labels: 13 GL codes

## Features

* TF-IDF word 1–2 grams over `merchant + description`
* TF-IDF character 3–5 grams (merchant spelling variants)
* categorical: currency, amount bucket (`0-50`, `50-200`, `200-500`,
  `500-1000`, `1000+`), weekday, month
* numeric: `log1p(amount)`, `amount/1000`, day-of-month, is-weekend

No validation labels, LLM outputs or embeddings are used as features. Seeds are
fixed at 42; LightGBM runs with `deterministic=True`, `force_row_wise=True` and
one thread.

## Results (30 held-out instructions)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| LightGBM | **0.4000** | **0.4039** | 0.4131 |
| XGBoost | 0.3667 | 0.3284 | 0.3424 |
| Qwen2.5-1.5B QLoRA (committed) | 0.8889 ± 0.0192 | 0.8889 ± 0.0192 | — |

Source: `ml/results/gl_metrics.json` and
`sft/results/seeds/multiseed_summary.json`. The QLoRA numbers are the committed
3-seed mean ± standard deviation; training was **not** re-run. QLoRA "token F1"
is the same 0/1 exact-match score in the committed evaluation, so it is reported
as F1.

5-fold CV on the 152 training rows: macro-F1 **0.3358 ± 0.1223** (folds
0.1784 / 0.2538 / 0.2886 / 0.4833 / 0.4747).

## Reproduce

```bash
.venv-cv/bin/python -m ml.gl_lightgbm
```

The script writes `ml/results/gl_metrics.json`, including per-class precision /
recall / F1, the 13×13 confusion matrix and per-document predictions.

## Caveat

The validation merchants are deliberately unseen, so the merchant tokens give
the classical model little signal; description n-grams generalise only
partially. 40% accuracy is a real, measured result — not a tuned number — and
it is roughly half the QLoRA adapter's accuracy. The gap is the point of the
comparison: the LLM is doing semantic generalisation that bag-of-ngrams cannot.
