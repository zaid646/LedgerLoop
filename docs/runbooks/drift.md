# Drift runbook

## Alerts

| Alert | Meaning | First action |
|---|---|---|
| `DriftDetected` | Any detector fired for 5 minutes | Open Grafana, capture `/drift/status`, check recent deploys and input sources |
| `AmountDriftScoreHigh` | Amount PSI above its calibrated threshold | Compare a sample of new receipts with the baseline; check currency/units |
| `ReviewQueueBacklog` | More than 100 pending review items for 15 minutes | Staff reviewers or temporarily raise the auto threshold only with ML approval |
| `AutoApprovalRateLow` | Auto-approval below 30% for 30 minutes | Check drift, provider errors, prompt/model version and calibration version |
| `ExtractionLatencyP95High` | Extraction p95 above 5 s | Check LLM provider, retry/self-heal counts and queue depth |

## Triage

1. Read the latest report: `curl -s localhost:8000/drift/status | jq`.
2. Inspect scores in Prometheus/Grafana: `expenseiq_drift_score`, `expenseiq_drift_metric`,
   `expenseiq_drift_fired`, `expenseiq_drift_window_docs`.
3. Identify the detector:
   - `psi_amount` or `ks_amount`: changed amounts, currency, taxes, discounts or OCR digits.
   - `psi_date_quarter` or `ks_date`: new date formats, timezone changes or bad parsing.
   - `psi_merchant_category`: new merchant mix or a category-catalog change.
   - `embedding_centroid_distance`: new receipt template, language, scan quality or a new
     upstream source. This detector is usually first to fire for format changes.
4. Compare 5-10 documents from the new window with golden examples. Do not retrain on the
   first alert; confirm the change is real and not a logging or sampling bug.

## Immediate mitigation

- If auto-approved quality is in doubt, enable the kill switch:
  `curl -X POST localhost:8081/admin/kill-switch -H 'content-type: application/json' -d '{"enabled":true,"actor":"oncall"}'`.
  This forces every new document to human review. The AI service is also told to force
  review; `ai_synced:false` means the AI service was unreachable and the backend still
  routes to review.
- Inform the finance operations lead of the queue impact and expected duration.
- Roll back the most recent prompt, model or parser change if drift started at the same
  time. Keep the old calibration file and container image available.

## Recalibrate or accept a new baseline

Only after the change is understood:

1. Label a representative sample of the new window and add it to the golden set with a new
   version (do not overwrite `v0`).
2. Re-run `uv run python evals/replay_confidence.py` and the quality gate.
3. Re-run `uv run python evals/drift_sim.py --backend auto --replicates 30 --null-windows 150`
   against a baseline that includes the accepted new documents.
4. Update `ai-service/config/drift.yaml` and `ai-service/config/calibration.json`, open a PR,
   and require the quality gate to pass.
5. Record the incident: detector, window, threshold, decision, reviewer and evidence paths.

## False alarms

If repeated null windows exceed the false-alarm budget, raise the threshold using
calibration rather than hand-editing a single number. The simulator records the 99.5th
percentile of null windows in `evals/results/drift-sim.json`; use it as the starting point.

## Escalation

- Page the ML platform on-call for `DriftDetected` with suspected auto-approval errors.
- Page the backend on-call if `/drift/status` or the kill switch is unavailable.
- Notify security if drift is caused by unexpected or malicious input.
