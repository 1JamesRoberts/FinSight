# Approved fresh alias-development review

The user approved the new 20-headline draft with “Approved” after reviewing the
workflow. Record this as bulk acceptance of assistant-authored labels, including
eight flagged rows and two empty rows; it does not establish independent or
individual double annotation. Original sample/draft and pre-approval prediction
results remain intact. All entity spans, company types, CIKs, identity statuses
and notes were preserved; only review metadata and uncertainty flags changed.

The fixed comparison was rerun on this Mac with actual CPU BERT NER. All
per-record prediction bytes are identical to the pre-approval run. No aliases,
normalization rules, grouping, model weights or reference values were changed
in response to these results. The original first 20-headline results also remain
unchanged. This is Phase 4 development evaluation, not sentiment-model testing.

## Accepted reference and policy

The reference has 26 company occurrences and one other-organization occurrence
across 20 headlines; two headlines have no annotated organizations. Sixteen
company occurrences have an accepted current issuer CIK under the default
no-parent-roll-up policy. The remaining ten company occurrences have no accepted
standalone current issuer link; they are still companies and remain in company
classification/extraction recall. Current identities do not prove historical
legal continuity. In particular Dell is a company but receives no parent CIK by
default. The explicit parent-roll-up arm changes that policy and remains a
counts-only diagnostic, not accuracy against this no-roll-up reference.

Reference CIKs use the same SEC snapshot as the resolver. Shared-source identity
evidence, bulk acceptance of assistant drafts, targeted category sampling, small
sample size and possible pretraining exposure limit independent/generalization
claims. No claim of population-level perfect precision is made.

## End-to-end company classification

| Resolver | Classified company spans | Correct | Extra | Missed | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Archived v3 | 8 | 8 | 0 | 18 | 100.00% | 30.77% | 47.06% |
| Suffix fixes only | 12 | 12 | 0 | 14 | 100.00% | 46.15% | 63.16% |
| Reviewed aliases, no parent roll-up | 18 | 18 | 0 | 8 | 100.00% | 69.23% | 81.82% |

**Attributes:** Resolver changes the identity/type rules after the same BERT
proposals. Classified company spans counts occurrences the rules accept as
companies. Correct requires exact start/end agreement with a reference company;
extra means an accepted company span has no exact company reference; missed
counts reference companies not accepted, including NER misses and unresolved
types. Precision is correct/classified, recall is correct/26 reference company
occurrences, and F1 is their harmonic mean. Counts are micro occurrence totals,
not unique companies or macro averages. All rows overlap in source headlines
and cannot be added. Null denominators would be undefined, not zero.

The aliases recover ten more company occurrences than the archived resolver,
including three Dell mentions classified as companies without parent assignment.
Eight reference company occurrences still fail end-to-end classification: six
detected names remain uncertain and two reference spans are not extracted exactly.

## Issuer links under the default policy

| Resolver | Accepted links | Correct | Wrong | Missed | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Archived v3 | 8 | 8 | 0 | 8 | 100.00% | 50.00% | 66.67% |
| Suffix fixes only | 12 | 12 | 0 | 4 | 100.00% | 75.00% | 85.71% |
| Reviewed aliases, no parent roll-up | 15 | 15 | 0 | 1 | 100.00% | 93.75% | 96.77% |

**Attributes:** Accepted links counts predicted span/CIK pairs; correct requires
both an exact company span and the approved CIK. Wrong means an accepted pair
is absent from the reference, and missed means a reference pair is not produced.
A wrong issuer would count once as wrong and once as missed. Precision is
correct/accepted, recall is correct/16 reference links, and F1 is their harmonic
mean. The ten company occurrences without accepted current CIKs are excluded
from the link recall denominator but remain in classification recall above.
Rows reuse identical headlines/proposals. These are issuer-link metrics, not
sentiment accuracy, stock mapping validity, historical eligibility, or evidence
that the whole pipeline recognizes 93.75% of all companies.

Seven additional correct links are made relative to archived v3, with no wrong
links observed on this small reference. The one remaining reference link is a
Boeing occurrence inside an aircraft-model reference that BERT did not extract;
adding more issuer aliases cannot resolve a span that is absent.

## Raw organization proposals as company candidates

Before filtering, BERT proposes 26 organization spans: 24 exactly match company
references, two are extra company candidates, and two company spans are missed.
Raw company-candidate precision/recall/F1 are each 92.31% (24/26 for precision
and recall). This is identical across all resolver arms because extraction is
shared. The extra candidates are NASA (a genuine other organization, not an NER
error under its organization taxonomy) and the unmatched expanded span
“Nortel VoIP Team.” The missed reference spans are Boeing and Nortel.

Among exactly detected reference companies, six remain type-uncertain: SpaceX,
China Airlines, Chrysler, Waymo, Alcatel and Sybari Software. These are unresolved
coverage cases, not labels that they are non-companies. Do not introduce new
aliases from this review and then call the same data an untouched test.

## Parent roll-up diagnostic

Enabling explicit parent roll-up produces 18 company-classified spans and 18 CIK
links, including three Dell-to-Dell-Technologies mappings. The default arm has
18 classified spans, 15 links and three unlinked Dell occurrences. The extra
parent links are not scored against a reference that deliberately withholds
parent assignments. A parent-policy accuracy comparison needs separately
reviewed expected parent identities and a historical-ownership scope.

## Provenance, reproduction and limits

[Full reviewed metrics/provenance](issuer-alias-review-reviewed.json).
Sample manifest:
`ad6bb8ddb3553db5412f480a27f0839a3d8254a3df684274a1dbd0e4fd04794b`.
[Preparation, baseline source snapshot and source-use decisions](issuer-alias-review.md).
[Usage and review fields](../../../docs/phase-04-target-sentiment/issuer-aliases.md).
The frozen plan, archived v3 source, alias catalog, directory and detector version
are unchanged; source/reference/prediction hashes are recorded. Reproduce with:

```bash
uv run --locked python -m finsight.sentiment.experiments.issuer_alias_review \
  --packet data/interim/issuer-alias-review/ad6bb8ddb3553db5412f480a27f0839a3d8254a3df684274a1dbd0e4fd04794b \
  --annotations data/interim/issuer-alias-review/ad6bb8ddb3553db5412f480a27f0839a3d8254a3df684274a1dbd0e4fd04794b/annotations.reviewed.jsonl
```

This closes the bounded reference acceptance and fixed comparison task. It does
not create the independent entity test collection, train a company classifier,
change sentiment calibration, update the GPU release or start Phase 5. Source
headlines, per-record predictions, drafts/references and approval metadata remain
in ignored local data. No frozen sentiment records were opened.

## Verification

All 811 tests pass. An independent set-based audit recomputes classification and
issuer-link confusion counts and precision/recall/F1 for all three arms. Prediction
bytes match the pre-approval run exactly. Reference spans/types/CIKs/link statuses/
notes match the original draft; only approval metadata and uncertainty flags changed.
The first approved sample and protected registry/freeze/deployment/final-target
result hashes are unchanged. Full Ruff checks still flag only the existing
scratch notebook (nine lint errors and formatting); checks excluding it pass.
`git diff --check` passes.
[Verification receipt](../verification/issuer-alias-review-reviewed-verification.json).
