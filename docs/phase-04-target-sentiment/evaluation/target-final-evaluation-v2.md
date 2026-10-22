# Phase 4 final evaluation — v2

Protocol `phase4-sentfin-multi-target-v2`, authorized on 2026-10-18 before any
Phase 4 test predictions. The authoritative
[plan](../../../configs/target-evaluation-plan-v2.json) and
[lock](../../../configs/target-evaluation-lock-v2.json) supersede v1 for execution.
The [v1 contract](target-final-evaluation.md), its lock, the original frozen
benchmark and Phase 3 results remain preserved.

The [overlap review](../../../reports/phase-04-target-sentiment/evaluation/overlap-review.md)
identified one multi-target headline whose near-duplicate was already evaluated
in Phase 3. V2 excludes only this group's reserved member: **262 headlines and
592 annotations**, with 118 negative, 287 neutral and 187 positive labels. Mixed
sentiment remains 110 headlines / 253 annotations. Exclusion is based on prior
exposure, with no model predictions consulted.

The plan records the group ID, both normalized text hashes, expected excluded
support and reason. The runner requires exactly one consumed and one reserved
member matching this registry; duplicate entries, changed/missing members,
development overlap and every other consumed-group overlap abort before scoring.
Original records are validated and never edited. Exclusion occurs only when
building the evaluation view. Expansion without an explicit registry still
rejects the original overlap.

Everything else follows v1: all 18 saved transformer checkpoints, both saved
TF-IDF baselines and the training prior; seeds 42/43/44; validation-selected
RoBERTa candidate; raw probabilities; five cohorts; seven paired contrasts; and
1,000 paired group-bootstrap replicates. No training, calibration, seed search or
selection change is permitted. Reports carry the revised cohort contract.

Execution requires a new clean committed source snapshot and new content-addressed
receipt. The failed v1 receipt/logs remain intact. Use the same
[runner commands](target-evaluation-runner.md), which now select v2 by default.
Previously evaluated single-target performance and unknown checkpoint exposure
remain limitations; correcting this pair does not make all research decisions
independent of prior test evaluation.

## Verified execution

All 21 fixed variants completed on 2026-10-18 under clean source commit
`ce1207b75e450695def45db3853fc7afe16331a0`. Both machines passed 583 tests and
Ruff; transferred result hashes and primary metrics verify. The
[final report](../../../reports/experiments/phase4-target-final.md) contains the
five cohorts and seven paired contrasts. This test cohort is now consumed.
