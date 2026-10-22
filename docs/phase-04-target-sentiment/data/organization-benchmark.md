# Company/organization mention development benchmark

The fixed sample now has a user-accepted reference with all 50 rows approved
and three company-type adjudications resolved. Rows 1–10 were individually
approved; the remainder were accepted in bulk with subsequent authorization.
See the [reviewed results](../../../reports/phase-04-target-sentiment/evaluation/organization-benchmark-reviewed.md).
This is an assistant-drafted, user-accepted development reference, not independent
double annotation or a final test benchmark. Historical provisional artifacts remain.

[Annotation policy](organization-annotation-policy.md) ·
[Provisional results](../../../reports/phase-04-target-sentiment/evaluation/organization-benchmark.md) ·
[Sampling/comparison plan](../../../configs/organization-annotation-plan.json).

## Sample and annotation workflow

1. Verify the existing native development files and group validation records by
   headline ID. Strip native sentiment targets/labels from the annotation view.
2. Rank IDs by SHA-256 of `organization-mentions-dev-v1|42|<record_id>` and select
   the smallest 50. No class balancing, company allowlist, NER output, error flag
   or confidence is used. Retain entity-free headlines rather than replacing them.
3. Freeze `sample.jsonl` and its manifest. The private packet ID is
   `dec8836961e3891902bd6134fdc3dd942cc4ce920c3ec9440693f272922f8b03`.
4. Draft every explicit company and other organization from original text using
   the policy. Include complete names and repeated occurrences. Preserve uncertainty
   in type notes. The assistant drafted all 50 rows before new sample predictions
   were examined; native sentiment targets were not displayed for annotation.
5. Freeze draft annotation bytes and the policy digest before the comparison.
   Keep `review.status: pending` and `review.reviewer: null` until a person checks
   each record. Original drafts have 34 mentions: 27 company and seven other
   organizations, including three initially uncertain company types. The accepted
   reference resolves those types while retaining the same spans. There are 23 empty rows.
6. Score the same verified NER checkpoint with `simple` and word-aware `first`
   aggregation; retain unchanged resolved/all-candidate registry controls. No model
   is trained, threshold selected or default behavior changed.

NER had already scored all validation headlines in the preceding task. Two sampled
records were in that task's assistant review. Although selection is prediction-free,
this is not a new unseen corpus or a completely blinded annotation exercise.
Neither these drafts nor later human revisions become a final independent test set.

## Review the actual private annotations

Open the generated sheet at:

```text
data/interim/organization-annotations/dec8836961e3891902bd6134fdc3dd942cc4ce920c3ec9440693f272922f8b03/review.md
```

It contains all 50 original headlines, draft name spans with offsets/types, empty
cases and uncertainty notes. It deliberately omits model proposals and native
sentiment annotations to keep review focused on the source text.

Edit `annotations.provisional.jsonl` in that same directory and save a separate
`annotations.reviewed.jsonl`. Correct omissions, full-name boundaries and company
versus other-organization types. Resolve every `uncertain` flag. After checking a
row, set its review object to:

```json
{"status": "approved", "reviewer": "Your name"}
```

The row's `annotation_author` continues to record the assistant's draft origin.
Check every headline, including the 23 currently empty ones; approval is an actual
reviewer assertion. The software validates that assertion and cannot establish
reviewer identity or independently guarantee completeness.

## Commands

Use the repository root with the existing local development data. Preparation
reproduces the immutable source-only sample and does not overwrite annotations:

```bash
uv run --locked python -m finsight.sentiment.experiments.organization_benchmark prepare
```

Set a task-specific variable to the packet directory printed above:

```bash
packet=data/interim/organization-annotations/dec8836961e3891902bd6134fdc3dd942cc4ce920c3ec9440693f272922f8b03
```

Validate drafts and regenerate their readable sheet (no model needed):

```bash
uv run --locked python -m finsight.sentiment.experiments.organization_benchmark check \
  --packet "$packet" --annotations "$packet/annotations.provisional.jsonl" --provisional
```

Reproduce exploratory metrics against the assistant draft:

```bash
uv run --locked python -m finsight.sentiment.experiments.organization_benchmark evaluate \
  --packet "$packet" --annotations "$packet/annotations.provisional.jsonl" --provisional
```

After actual human review, validate and evaluate the separate reviewed file:

```bash
uv run --locked python -m finsight.sentiment.experiments.organization_benchmark check \
  --packet "$packet" --annotations "$packet/annotations.reviewed.jsonl"
uv run --locked python -m finsight.sentiment.experiments.organization_benchmark evaluate \
  --packet "$packet" --annotations "$packet/annotations.reviewed.jsonl"
```

Without `--provisional`, pending reviews and unresolved uncertainty abort before
model loading. Source text/membership changes, missing rows, invalid offsets,
unsupported entity types, and duplicate/overlapping spans are rejected. Reference
bytes are hashed and copied before inference. Reviewed and provisional public
results have separate filenames; evaluation packets preserve reference hashes.
Use the existing transformer optional dependency and locally verified NER files
from the [detector setup](../organization-detection.md); no new installation is needed.

## Evaluation meaning

Exact-span precision is correct complete spans / all proposed spans; recall is
correct complete spans / all reference spans. F1 is their harmonic mean. Counts
are pooled across headlines (micro averaging), and a wrong boundary counts as one
extra prediction and one missed reference. Repeated occurrences are separate;
empty-reference headlines still contribute false positives. Zero-denominator
precision or recall is null, not an invented perfect score.

All-organization metrics ignore company/other type and evaluate raw ORG boundaries.
The separate company-candidate diagnostic compares every ORG proposal with company
references only; it exposes the cost of treating institutions as companies. It is
not a learned company-type classifier. No partial-overlap credit, ticker linking,
sentiment evaluation, training, or deployment selection is included.
