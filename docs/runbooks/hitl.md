# Human-in-the-loop review runbook

## Queue contract

- Every `human_review` and `reject_escalate` document appears in `/review` with its route,
  reason, calibrated document score and per-field scores.
- Target first response: same business day. The queue item carries `sla_due_at` (24 hours
  after enqueue). Escalate items older than one business day to the finance operations lead.
- Reviewers must use their own id. Decisions are append-only and cannot be overwritten.

## Decisions

| Decision | Effect |
|---|---|
| `approve` | Expense becomes `approved`; no fields change |
| `correct` | Edited fields are applied to the expense, each changed field is stored in `labeled_corrections`, status becomes `corrected` |
| `reject` | Expense becomes `rejected`, queue status becomes `rejected`; use for unreadable or out-of-policy documents |

Do not use `approve` to silently accept a wrong value. If any field is wrong, use `correct`;
the labelled correction is what improves calibration and the golden set.

## Daily checks

1. Open the Streamlit page `ui/pages/2_Review_Queue.py` (or `BACKEND_URL` + `/review`).
2. Confirm the pending count, oldest item age and route mix in `/review/stats`.
3. Work oldest first; add a short note for corrections and rejects.
4. If the queue exceeds 100 items, notify the finance operations lead and the ML platform
   owner. Do not raise auto-approval thresholds without ML approval and a quality-gate run.
5. If the model is producing unsafe approvals, enable the kill switch immediately (see
   below) and page the ML platform on-call.

## Kill switch

```bash
curl -X POST "$BACKEND_URL/admin/kill-switch" \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"actor":"your-name"}'
```

Verify with `GET /admin/kill-switch`. To restore normal routing, set `enabled:false` only
after the ML owner confirms the cause is resolved. `ai_synced:false` in the response means
the backend flag is set but the AI service could not be reached; the backend still forces
review, so do not assume auto-approval has resumed.

## Correction data and privacy

- Corrections are stored in `labeled_corrections` with reviewer, field, original value and
  corrected value. They are used for evaluation and future calibration.
- Do not paste full card numbers, credentials or unrelated personal data into notes; the
  extraction text is already redacted, but notes are free text.
- For deletion requests, export rows by `document_id` from `review_queue`,
  `review_decisions` and `labeled_corrections`, then follow the data-retention procedure.

## Useful queries

```sql
-- pending by route
SELECT route, count(*) FROM review_queue WHERE status='pending' GROUP BY route;
-- reviewer throughput for today
SELECT reviewer, decision, count(*) FROM review_decisions
WHERE created_at >= now() - interval '1 day' GROUP BY reviewer, decision;
-- corrections by field
SELECT field, count(*) FROM labeled_corrections GROUP BY field ORDER BY 2 DESC;
```
