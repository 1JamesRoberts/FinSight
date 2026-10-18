# Documentation

Start with the current phase, then open its task-specific guide. For measured outcomes, use the [report index](../reports/README.md).

| Phase | Guides |
| --- | --- |
| [Phase 0 — Foundation](phase-00-foundation/README.md) | Project layout, environment and architectural boundaries. |
| [Phase 1 — Data curation and EDA](phase-01-data/README.md) | Dataset selection, rights, quality review, leakage-aware splits and benchmark freeze. |
| [Phase 2 — Traditional sentiment baselines](phase-02-baselines/README.md) | VADER, sparse classifiers, class weighting, tuning and qualitative errors. |
| [Phase 3 — Sentence-level transformers](phase-03-transformers/README.md) | Financial/base transformer comparisons, seed studies, calibration, ensembles and locked evaluation. |
| [Phase 4 — Company-targeted sentiment](phase-04-target-sentiment/README.md) | Native target conditioning, matched baselines, error analysis and reserved evaluation. |

## Shared references

- [Architecture decisions](architecture.md)
- [Dataset/model rights and usage ledger](datasets.md)
- [Code and machine setup guides](guides/README.md)
- [Roadmap and phase status](../ROADMAP.md)

## Organization rules

Put new guides under their phase and task, and add them to that phase index. Shared policies stay here. A few old filenames remain short compatibility links because immutable reports and generated report templates reference them. Edit the full guide at its new location.

Original experiment JSON, run receipts, figures and published report bytes retain their historical paths. Report indexes organize those records without rewriting their checksums or duplicating results.
