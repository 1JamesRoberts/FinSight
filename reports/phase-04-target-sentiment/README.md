# Phase 4 — Company-targeted sentiment

**Start here: [Final report](final-report.md).** It contains the phase conclusions,
results, decisions and remaining work. Supporting evidence stays available below.

[All phase reports](../README.md) · [Methods and setup](../../docs/phase-04-target-sentiment/README.md)

```text
phase-04-target-sentiment/
├── README.md              # Reading guide
├── final-report.md        # Canonical phase summary
├── experiments/           # Development studies and comparisons
├── evaluation/            # Audits, reviewed benchmarks and serving selection
└── verification/          # Receipts, smoke checks and GPU execution evidence
```

## Experiments

[All experiment evidence](experiments/README.md)

- [Target-format comparison and decision](experiments/target-format-comparison.md)
- [Company-name sensitivity](experiments/target-name-stress.md)
- [Company mention coverage](experiments/company-mention-coverage.md)
- [Pretrained NER comparison](experiments/organization-detection.md)
- [Company type and issuer routing](experiments/company-mention-routing.md)
- [US-listed issuer directory](experiments/us-listed-directory.md)

## Evaluation

[All evaluation evidence](evaluation/README.md)

- [Overlap review](evaluation/overlap-review.md)
- [Target-span alignment](evaluation/target-span-audit.md) and [tokenizer audit](evaluation/target-tokenizer-audit.md)
- [Reviewed organization benchmark](evaluation/organization-benchmark-reviewed.md)
- [Approved US-company review](evaluation/us-company-review-reviewed.md)
- [Approved issuer-alias review](evaluation/issuer-alias-review-reviewed.md)
- [RoBERTa prediction selection](evaluation/target-prediction.md)

The earlier provisional benchmarks are retained for provenance; use the reviewed
versions for accepted development-review results. These reviews are distinct from
the frozen sentiment test.

## Verification

[All receipts and execution checks](verification/README.md)

- [Reviewed-alias GPU release](verification/issuer-alias-gpu.md)
- [Company-routing GPU check](verification/company-mention-routing-gpu.md)
- [Report layout migration record](verification/report-layout-migration.json)

These are artifact and runtime checks. Invented-input smoke checks do not measure
sentiment accuracy or supply a new independent company benchmark.

## Evaluation execution status

The v1 execution stopped before inference after detecting a duplicate group shared
with the consumed Phase 3 sentence test. The authorized v2 evaluation excluded
that group and completed all 21 fixed variants on the RTX 4070 SUPER. The original
freeze and v1 contract remain unchanged. The v2 cohort is consumed and must not
be used for tuning. See the [overlap review](evaluation/overlap-review.md) and
[v2 protocol](../../docs/phase-04-target-sentiment/evaluation/target-final-evaluation-v2.md).

## Final test results

Use the [final report](final-report.md#results) for the explained result tables,
[full frozen evaluation](../experiments/phase4-target-final.md) for detailed
contrasts and [aggregate JSON](../experiments/phase4-target-final.json) for exact
metrics and provenance. Shared model-comparison artifacts remain under
`reports/experiments/`, where their existing protocols expect them.

## Evidence preservation

This layout migration moved supporting files; it did not discard experiments.
All moved JSON evidence remains byte-identical. Markdown links were rebased;
the migration record stores both original and current hashes and locations.
Historical receipts retain their original path strings and describe their original
execution. Resolve those paths through the migration record when reviewing them.
The frozen target-format plan still pins its original audit path and hash; the
runner locates the moved audit without changing that plan. Source-pinned historical
execution checks still require the matching archived source release: current
runner output paths have changed, and old source checksums were not relaxed.

Independent company-labeled test collection and per-company-target acceptance
criteria remain open; organizing these reports does not close Phase 4.
