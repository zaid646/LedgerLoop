# Checked-in SFT results

| File | Contents |
|---|---|
| `train_metrics.json` | Final train loss, runtime, hyperparameters, library versions, GPU. |
| `trainer_log_history.json` | Per-step / per-epoch training and validation loss. |
| `loss_log.csv` | Step, loss, learning rate and epoch for plotting. |
| `loss_curve.png` | Training-loss curve. |
| `eval_base.json` | Base-model metrics and 30 domain predictions. |
| `eval_adapter.json` | LoRA-adapter metrics and 30 domain predictions. |
| `train_full.log` | Full training/eval console log. |
| `colab_runtime.txt` | Free-Colab T4 runtime and pinned-version evidence. |

Headline numbers: base 6.7% → adapter **63.3%** exact match on 30 unseen-merchant
LedgerLoop transactions; domain completion loss 1.232 → **0.266**; Dolly completion loss
2.333 → 1.906. Train loss 1.132 in 63 steps / 5m23s on a Tesla T4. See
`../README.md` for the full table and caveats.
