# US-company mention and identity review

This is a 20-headline corporate-news development diagnostic from English
Wikinews, separate from SEntFiN and the frozen sentiment benchmark. The sample
was selected and assistant annotations were drafted before running the detector.
A company category is only a sampling aid, never a mention/identity reference.

## Review status

The user bulk-approved all 20 drafts. `annotations.reviewed.jsonl` and
`review-acceptance.json` beside the original review sheet record acceptance and
current-identity finalization. The original sheet remains marked as a draft for
provenance; it is not the authoritative review-status record. See the
[reviewed results](../../../reports/phase-04-target-sentiment/evaluation/us-company-review-reviewed.md).
The instructions below describe the original review workflow.

## What to review

Open the local [20-row review sheet](../../../data/interim/us-company-review/0712af66f547a820e4b6a1ac039d613e7239bea4a51ef45edb3cab251cf44fbc/review.md).
The sheet contains source links, complete headlines, draft spans, CIK links,
uncertainties and notes, with no model predictions. It is ignored local data and
will not exist in a fresh clone until the packet is copied or recreated.

Check every named company and other organization, including repeated names and
rows with no annotations. A name implied by a person or product is not a written
company mention. Use the complete organization surface and non-overlapping spans;
do not add a nested parent company inside a separately annotated lab name.
A brand counts as a company mention when the headline uses it to refer to the
business; a vehicle/product name alone does not. Label institutions/labs as
`other_organization` when they are not company references. Flag doubtful cases.

For each entity, `start` is its zero-based character start and `end` is exclusive;
`headline[start:end]` must equal `text`. `type` is `company` or
`other_organization`. `uncertain` records unresolved mention/type decisions.

Identity is a separate decision:

- `linked`: a company with a ten-digit `cik` and `identity_evidence` supporting the
  association. Evidence should name/link an authoritative identity source.
- `not_in_directory`: an explicitly reviewed company with no accepted current
  directory identity; `cik` must be null. State why in evidence.
- `not_applicable`: another organization, such as a university; no CIK.
- `uncertain`: an unadjudicated identity; no CIK. This blocks reference scoring.

The draft CIK cross-check uses the same SEC source as the resolver, so it is not
an independent identity master. Current CIKs must not be treated as proof of
historical legal continuity. In particular, review Dell/GM and the historical
Chrysler, Saturn and Yahoo identities; do not map brands automatically to current
parents. Review the Microsoft Research Asia organization type. Seven empty
headlines also need checking. At preparation time all 20 rows were pending, not just flagged rows.

You can reply with numbered approvals/corrections, as in the previous annotation
review. Alternatively copy `annotations.provisional.jsonl` beside the review
sheet to `annotations.reviewed.jsonl`, correct spans/types/identity decisions,
resolve every uncertainty and set each row's `review.status` to `approved` and
`review.reviewer` to your name. No assistant draft is auto-approved. The tool
validates a reviewer assertion; it cannot verify who actually performed review.
Keep the original draft and sample unchanged.

## Reproduction

Existing saved category responses reproduce the pinned packet:

```bash
uv run --locked python scripts/prepare_us_company_review.py prepare \
  --raw-dir data/raw/wikinews-us-review
```

A fresh, empty output directory can acquire eight bounded category responses:

```bash
uv run --locked python scripts/prepare_us_company_review.py acquire \
  --raw-dir data/raw/wikinews-us-review-new
```

No keys are required. Review the source-use entry in `docs/datasets.md`. New
category membership may differ from the original snapshot; a fresh acquisition
is not guaranteed to reproduce this sample. Selection fails if a category cannot
meet the fixed quota; do not silently adjust it after seeing model predictions.

To regenerate prediction-only counts locally, using the existing pinned NER and
SEC snapshots:

```bash
uv run --locked python -m finsight.sentiment.experiments.us_company_review \
  --packet data/interim/us-company-review/0712af66f547a820e4b6a1ac039d613e7239bea4a51ef45edb3cab251cf44fbc
```

Add `--annotations <packet>/annotations.reviewed.jsonl` after review to calculate
accuracy metrics. Pending or uncertain references are rejected before loading
models. Predictions and source/plan/model hashes go into a content-addressed
`runs` directory inside the packet. Source text, per-record predictions and
references stay private. No sentiment model inference or training is performed.

## Metrics after review

Company-span precision/recall/F1 treats all proposed ORG spans as company
candidates and compares them to reference company spans. A detected university
therefore counts as an extra company candidate in that view. It is not general
ORG extraction accuracy. Wrong boundaries count as an extra and a miss.

Issuer-link precision/recall/F1 requires an exact span and the correct CIK. A
wrong CIK counts as an extra and a missed link; abstention still misses any
reference link. Companies with reviewed `not_in_directory` decisions remain in
company extraction recall, but are not counted as reference CIK links. Any link
to such a company is an extra link. Undefined precision/recall denominators are
null, not zero. Neither score validates historical listing eligibility.

[Current preparation/prediction report](../../../reports/phase-04-target-sentiment/evaluation/us-company-review.md).
