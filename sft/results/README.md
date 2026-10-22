# Checked-in SFT results

Headline model: `Qwen/Qwen2.5-1.5B-Instruct`, QLoRA, free Colab Tesla T4.

| File | Contents |
|---|---|
| `train_metrics.json` | Seed-42 train loss, last-step loss, validation loss, runtime, hyperparameters, versions, GPU. |
| `trainer_log_history.json` | Seed-42 per-step / per-epoch training and validation loss. |
| `loss_log.csv` | Seed-42 step, loss, learning rate and epoch for plotting. |
| `loss_curve.png` | Seed-42 training-loss curve. |
| `eval_base.json` | Base 1.5B metrics and 30 domain predictions. |
| `eval_adapter.json` | Seed-42 LoRA metrics and 30 domain predictions. |
| `train_full.log` | Seed-42 full training console log. |
| `colab_runtime.txt` | Free-Colab T4 runtime and pinned-version evidence. |
| `seeds/<seed>/` | Per-seed metrics, loss logs/curves, train logs and adapter evals. |
| `seeds/base_eval.json` | Base-model evaluation shared by all seeds. |
| `seeds/multiseed_summary.json` | Mean ± std across seeds 42, 1 and 2. |
| `archive-0.5b/` | Earlier 0.5B pilot, retained for provenance; not the headline result. |

Headline numbers (seed 42): base 26.7% → adapter **86.7%** exact match on 30
unseen-merchant LedgerLoop transactions; domain completion loss 0.811 → **0.060**; Dolly
completion loss 2.517 → 1.544; overall perplexity 7.74 → **5.60**. Train loss 1.025 in
63 steps / 6m39s on a Tesla T4.

Multi-seed (42, 1, 2): exact match **88.9% ± 1.9 points**, token F1 **88.9% ± 1.9
points**. See `../README.md` for the full table and caveats.
