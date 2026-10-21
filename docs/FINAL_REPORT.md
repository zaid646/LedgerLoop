# LedgerLoop confidence, HITL and drift report

**Summary.** Added calibrated confidence routing with a fail-closed human-review path, a
reviewer UI backed by an append-only Postgres audit trail, and PSI/KS/embedding drift
detection with Grafana alerts. All numbers below come from the checked-in replay artifacts
and run on CPU; no paid API, GPU or live inference was used. Existing tests stayed green.

| Measure | Before | After |
|---|---|---|
| Document ECE (75 docs) | 0.1493 | **0.0368** (−75.4%) |
| Merchant / amount / date ECE | 0.0332 / 0.0183 / 0.1070 | **0.0232 / 0.0142 / 0.0533** |
| Auto-approved documents | n/a | 33/75 (44.0%) |
| Auto-approved accuracy | n/a | 96.97% macro field, 90.91% exact document |
| Review / reject rates | n/a | 21.3% / 34.7% |
| Errors reaching a human | n/a | 35/38 (92.1%); 3 auto-approved |
| Scoring + routing p95 | n/a | 0.10 ms/document |
| Drift detection (5 shift scenarios) | n/a | 100% each; 0/30 null false alarms |
| Drift time-to-detect | n/a | 25 documents (first window) |
| Drift detector / embedding latency | n/a | 2.3 ms p50 / 6.54 s per 50 docs (CPU) |
| AI service tests | 58 passed, 1 skipped | **88 passed** (87 + 1 skipped in CI without the optional embedder) |
| Backend | no review code | typecheck clean, 5 logic tests passed; 3 live decisions and 2 corrections verified against Postgres |

**Why it matters.** Calibration cut document ECE by 75% and date ECE by 50%, and the
operating point auto-approves 44% of documents at 97% macro field accuracy while sending
92% of measured errors to a human. The router is conservative: 34.7% rejects create manual
work, and 3 of 38 errors still auto-approved, so the kill switch and review queue are part
of the control, not an afterthought. Drift runs asynchronously (CPU bge-small embedding is
slow), alerts on new merchants, formats, OCR noise, amount shifts and date shifts, and CI
blocks changes that worsen calibration, catch rate, latency or drift detection.

| Commit | Scope |
|---|---|
| `335de59` | Calibrated ARS scoring, threshold routing, replay metrics and reliability plot |
| `e970799` | Review queue DDL, audit/correction tables, kill switch, reviewer UI and tests |
| `fe09afc` | Drift detectors, simulator, Prometheus metrics, Grafana dashboard and alerts |
| `e0ef2db` | Responsible AI notes, runbooks, CI quality gate and this report |

**Evidence.** [`confidence-replay.json`](../evals/results/confidence-replay.json) ·
[`drift-sim.json`](../evals/results/drift-sim.json) ·
[`confidence-reliability.png`](../evidence/png/confidence-reliability.png) ·
[`drift-detection.png`](../evidence/png/drift-detection.png) ·
[`RESPONSIBLE_AI.md`](RESPONSIBLE_AI.md) · [`runbooks`](runbooks/)
