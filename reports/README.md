# Final phase reports

Start here for the outcome of each phase. Every canonical summary uses the same
`final-report.md` filename and section structure. Use the evidence indexes for
individual experiments, JSON metrics, figures and receipts, and the
[documentation index](../docs/README.md) for methods/setup.

| Phase | Canonical final report | Scope/status | Detailed evidence |
|---|---|---|---|
| 0 — Foundation | [Final report](phase-00-foundation/final-report.md) | Implementation complete; hosted-CI acceptance unverified | [Index](phase-00-foundation/README.md) |
| 1 — Data and EDA | [Final report](phase-01-data/final-report.md) | Complete within local research/curation scope | [Index](phase-01-data/README.md) |
| 2 — Traditional baselines | [Final report](phase-02-baselines/final-report.md) | Validation-only research complete | [Index](phase-02-baselines/README.md) |
| 3 — Sentence transformers | [Final report](phase-03-transformers/final-report.md) | Locked sentence evaluation complete; tracking/ablation items deferred | [Index](phase-03-transformers/README.md) |
| 4 — Company-targeted sentiment | [Final report](phase-04-target-sentiment/final-report.md) | Verified research/inference synthesis; independent company-test/per-company criteria open | [Index](phase-04-target-sentiment/README.md) |

**Attributes:** Phase is the roadmap unit, not a dataset or an experiment.
Canonical final report is the single entry point for that phase's conclusions.
Scope/status qualifies which work is complete and which acceptance criteria are
still open; a report named “final” does not override the roadmap. Detailed evidence
links the original task reports and immutable outputs. Rows are sequential work
areas, not independent benchmarks; phases can reuse data under different contracts.
No completion report is created for unimplemented Phases 5–10.

## Common report structure

Each report states its snapshot date/status and uses these sections:

1. Scope and status.
2. Data and method.
3. Results, with units and table attributes explained directly below tables.
4. Decisions.
5. Limitations and remaining work.
6. Evidence and reproduction.

These summaries are maintained editorial reading guides. Exact metrics, artifact
identities and historical execution conditions remain in their cited source
reports/JSON. Phase indexes contain the longer reading lists.

## Reading the evidence correctly

Phase 2 validation selection, Phase 3 single-target frozen testing, Phase 4
native-target frozen testing, mention development reviews and invented-input
GPU smoke checks have different purposes and denominators. Do not combine them
as one independent benchmark. Both evaluated SEntFiN test cohorts are consumed;
remaining reserved views and independent-company data are separate questions.
Mean scores across seeds are not the performance of one deployed checkpoint,
seed SD is not a confidence interval, and issuer linking is not sentiment accuracy.

## Artifact policy

Canonical summaries use `reports/phase-XX-name/final-report.md`; keep remaining
acceptance criteria explicit. Phase 4 supporting reports are grouped into
`experiments/`, `evaluation/` and `verification/`, with a
[migration record](phase-04-target-sentiment/verification/report-layout-migration.json)
for former paths. All moved JSON bytes and frozen configs remain unchanged;
Markdown links were updated. Historical receipts retain original paths and hashes,
so use that mapping to locate their evidence. Archived source releases remain
unchanged; their source-pinned checks must run with their matching historical code.
Shared comparison outputs under `reports/experiments/` retain their existing paths.

## Documentation verification

The 2026-10-22 consolidation checked local links, the shared section structure,
reported metric cells and unchanged existing evidence bytes. All 811 project
tests pass. Full Ruff lint/format checks still flag only the pre-existing scratch
`notebooks/experiment.ipynb`; checks excluding it pass. No model/data inference,
training, GPU change or new test-set access was performed for these summaries.
[Verification record](final-report-verification.json).

The subsequent Phase 4 layout migration verified 53 moves, unchanged JSON evidence
and frozen config bytes, 898 local links and 813 passing tests. Ruff passes outside
the same existing scratch notebook. The
[migration receipt](phase-04-target-sentiment/verification/report-layout-migration.json)
records old/current locations and hashes; the earlier consolidation receipt remains
a historical snapshot.
