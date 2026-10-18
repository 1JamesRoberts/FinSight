# Reserved-test overlap review — 2026-10-18

The blocked evaluation identified a genuine near-duplicate pair, not training
leakage or a conflicting sentiment annotation. No model inference was performed
for this review. The original freeze, Phase 3 results and Phase 4 v1 contract/lock
remain unchanged; the v1 runner still correctly refuses to score this cohort.

## What overlaps

One group contains two versions of the same regulatory-event headline. The
longer version names a tribunal in full; the shorter version uses only its final
word. The remaining wording is identical. The single-target version has one
neutral annotation; the multi-target version has two neutral annotations. Their
shared target has the same label. These are different texts and different target
sets, so they were retained as separate records in the same duplicate group.

Re-running the existing screen reproduces word-trigram Jaccard similarity
**0.80**, exactly the inclusive threshold fixed in Phase 1. Of the longer title's
10 unique trigrams, eight also occur in the shorter title. The original private
Phase 1 annotation-review artifact already contains this edge and reports no
shared-target gold conflict. Grouping was therefore intentional, not a new
similarity rule introduced after evaluation.

Full source text, record/group identifiers and the original edge are retained in
ignored `data/interim/target-overlap-review/20261018/review.json`. This public
report contains only descriptive findings and aggregate counts under the existing
[local-research data scope](../../docs/datasets.md).

## Why it was allowed before but blocks this run

Both records were correctly assigned to the same frozen **test** partition.
No multi-target test group or normalized exact text overlaps training/validation.
The original group-safe train/validation/test split remains intact.

Later, splitting test records by target count placed one member in the Phase 3
single-target evaluation and the other in the reserved Phase 4 multi-target
evaluation. These task cohorts are disjoint by record but not entirely by group.
The single-target cohort was already evaluated and used in subsequent model
development decisions, so the related multi-target headline cannot be described
as fully independent of that earlier evaluation. This does not demonstrate
weights were trained on either test record, nor does it invalidate all prior
single-target scores.

## Recommendation for a separate protocol revision

Keep the frozen benchmark and historical results intact. Before scoring, declare
a new Phase 4 evaluation version that excludes multi-target members of any group
already consumed in Phase 3. Record the specific exclusion and its provenance;
retain the same models, seeds, selection rule and metrics. This is a correction
based on prior exposure, with no test predictions consulted.

The following counts describe that possible revision, **not an applied change**:

| Cohort | Current frozen cohort | After proposed group exclusion |
| --- | ---: | ---: |
| Multi-target headlines | 263 | 262 |
| Multi-target annotations | 594 | 592 |
| Negative annotations | 118 | 118 |
| Neutral annotations | 289 | 287 |
| Positive annotations | 187 | 187 |
| Mixed-sentiment headlines | 110 | 110 |
| Mixed-sentiment annotations | 253 | 253 |

Excluding this member would not erase prior test-driven research decisions or
establish unknown checkpoint pretraining exposure. Report the revised cohort's
exact scope and prior single-target evaluation alongside its eventual results.
No protocol revision, exclusion, receipt replacement or evaluation restart has
been performed as part of this review.
