# Reports

Read reports by phase and task. Methods and reproduction commands are in the [documentation index](../docs/README.md).

| Phase | Reading guide |
| --- | --- |
| [Phase 1 — Data curation and EDA](phase-01-data/README.md) | Dataset selection, rights, quality review, leakage-aware splits and benchmark freeze. |
| [Phase 2 — Traditional sentiment baselines](phase-02-baselines/README.md) | VADER, sparse classifiers, class weighting, tuning and qualitative errors. |
| [Phase 3 — Sentence-level transformers](phase-03-transformers/README.md) | Financial/base transformer comparisons, seed studies, calibration, ensembles and locked evaluation. |
| [Phase 4 — Company-targeted sentiment](phase-04-target-sentiment/README.md) | Native target conditioning, matched baselines, error analysis and reserved evaluation. |

## Artifact policy

Published report files, JSON aggregates, figures and per-run archives retain their original paths and bytes. Frozen plans and receipts reference those paths/checksums. The phase indexes provide organized navigation without altering experimental evidence.

Phase 3 sentence-test results, Phase 4 development validation results and future target-test results have different scopes; their tables must not be combined as one independent benchmark.

New narrative summaries belong under the appropriate phase. New immutable experiment outputs should be indexed from that phase after their scope/privacy is reviewed. Do not move an existing artifact referenced by a lock.
