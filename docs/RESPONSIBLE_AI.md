# Responsible AI notes

## Intended use

ExpenseIQ assists finance teams with receipt/invoice field extraction, GL coding and policy
routing. Outputs are recommendations. A qualified human remains accountable for approvals,
payments, tax treatment and audit conclusions.

## Out-of-scope use

Do not use the system for fully automated payment, credit, employment, insurance, housing or
audit decisions; for surveillance; for documents outside the approved receipt/invoice
classes; or with secrets, credentials or unapproved personal data.

## Human oversight

- Auto-approval requires a valid schema, all required fields above their calibrated
  thresholds and a document score of at least 0.75.
- Missing required fields fail closed to `reject_escalate`.
- Reviewers can approve, correct or reject; every action is logged with reviewer and time.
- The kill switch forces all new documents to human review.
- Corrections are retained as labelled data and reviewed before being added to a new golden
  set version.

## Measurement

- Calibration: isotonic regression, five-fold cross-fitted document ECE. Document ECE fell
  from 0.1493 to 0.0368 on the 75-document golden set.
- Routing: at 0.75, 44.0% of documents auto-approve at 96.97% macro field accuracy and
  90.91% exact-document accuracy; 92.1% of measured errors reach a human.
- Drift: PSI/KS on fields and bge-small embedding-centroid distance, calibrated on null
  windows. The synthetic replay detects all five shift scenarios at 100% with zero false
  alarms in 30 measured replicates; CI reruns a deterministic hashing variant.
- CI blocks regressions in ECE, auto-approval accuracy, catch rate, latency, false alarms
  and drift detection.

## Limitations

- Small public golden set (SROIE/CORD), not production traffic; subgroup performance is not
  yet measured at scale.
- Cached LLM outputs provide self-reported confidence, not token logprobs; the logprob
  signal is disabled. Calibration cannot correct a model that is uniformly overconfident.
- Reject rate is 34.7%, which is conservative and increases manual work.
- Drift scenarios are synthetic; embeddings are CPU-only and slower than field detectors.
- Handwriting, severe blur, non-receipt images and adversarial inputs are out of scope.
- No guarantee of correctness for currencies, tax rules or date formats outside the golden
  set.

## Privacy and security

- PII and card numbers are redacted before any model call.
- Secrets live in `.env` and are not committed.
- Review notes are free text; reviewers must not paste sensitive data.
- Retention and deletion requests cover review decisions, corrections and extracted text.

## Monitoring, ownership and escalation

- Prometheus metrics and Grafana dashboards cover confidence, routing, review depth, error
  rate, latency and drift.
- ML platform owns calibration, drift thresholds and model/prompt changes. Finance
  operations owns queue staffing and SLA. Backend on-call owns service availability.
- `DriftDetected` pages on-call; quality regressions and backlog alerts create tickets.
- Runbooks: [`drift.md`](runbooks/drift.md), [`hitl.md`](runbooks/hitl.md).

## Change control

Any prompt, model, calibration or threshold change requires a PR, the deterministic quality
gate and a reviewer from ML platform. Never edit production thresholds directly. Keep the
previous calibration file and image available for rollback.
