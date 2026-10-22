# Reviewed US-company extraction and issuer-linking comparison

The user bulk-approved all 20 assistant-drafted references with “Approved for
all.” This is a user-accepted development reference, not independent double
annotation. Original sample/draft/prediction-only report remain intact. No model,
resolver rule, alias, threshold or directory change followed inspection of this
sample. Both fixed pipelines were rerun against the approved reference on CPU.

## Reference decisions and identity scope

The reference has 15 company mentions and two other organizations across 20
headlines; seven headlines have no named organizations. Twelve company mentions
have accepted current SEC CIK associations. Chrysler, Saturn and Yahoo remain
company mentions with no accepted standalone current directory link; they were
not rolled up to current parents or treated as non-companies. The draft's null
links for these brands were finalized as `not_in_directory` under the bulk
acceptance; this is a no-accepted-link decision, not proof that no historical CIK
exists. Dell/GM links refer only to current issuer associations and make no
historical legal-continuity claim. Microsoft Research Asia remains an
other-organization lab span, without a nested Microsoft span or issuer link.
The five flagged rows and all other rows were accepted in bulk, not individually
adjudicated by the user. These finalization details are retained privately.

A **CIK (Central Index Key)** identifies an SEC filer. It is different from a
security's ticker: one issuer may have multiple ticker/exchange associations.
For example, Apple's CIK in our pinned source is `0000320193`, while its ticker
is AAPL. The SEC source establishes filer/listing associations, not a historical
security master or independent proof of every brand's identity. Reference CIKs
were cross-checked with the same SEC snapshot as the resolver, limiting the
independence of this linking diagnostic.

## Company candidate extraction

| Method | Reference company spans | Proposed ORG spans | Correct company spans | Extra candidates | Missed companies | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NER + five-company registry | 15 | 21 | 15 | 6 | 0 | 71.43% | 100.00% | 83.33% |
| NER + registry + SEC | 15 | 21 | 15 | 6 | 0 | 71.43% | 100.00% | 83.33% |

**Attributes:** Method changes only identity resolution, so both rows share the
same raw NER spans. Reference company spans counts annotated company occurrences,
not distinct issuers. Proposed ORG spans counts every NER organization proposal,
including actual institutions and errors. Correct company spans requires exact
start/end agreement with a reference company. Extra candidates are proposed
spans without an exact company reference; missed companies are reference company
spans not proposed. Precision is correct/proposed; recall is correct/reference;
F1 is their harmonic mean. Counts are across 20 headlines and rows overlap.
This evaluates ORG proposals as company candidates, not general ORG extraction
or post-resolution company accuracy. Legitimate non-company ORGs count as extras
in this company-specific view.

The six extras are a product name, an unnamed-business common noun, a vehicle
model fragment, a university, a corporate lab and a person. Actual university/lab
organizations are not NER extraction errors, but still need company filtering.
No correction to the NER or filtering policy was made during this evaluation.

## Exact-span plus issuer linking

| Method | Reference links | Accepted links | Correct links | Wrong links | Missed links | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NER + five-company registry | 12 | 5 | 5 | 0 | 7 | 100.00% | 41.67% | 58.82% |
| NER + registry + SEC | 12 | 8 | 8 | 0 | 4 | 100.00% | 66.67% | 80.00% |

**Attributes:** Reference links counts company occurrences with approved current
CIKs; the three no-current-link brands are excluded from this denominator but
remain in company extraction recall above. Accepted links counts spans assigned
a CIK; correct links requires both exact boundaries and matching CIK. Wrong
links are accepted span/CIK pairs absent from the reference; missed links are
reference pairs not produced, including abstentions. A wrong CIK would count
once as wrong and once as missed. Precision is correct/accepted; recall is
correct/reference; F1 is their harmonic mean. Rows share the same sample and
cannot be summed. All figures are occurrence-level micro metrics, not per-company
macro scores. Observed 100% precision on five/eight accepted links is not a
population guarantee.

The directory adds three correct links: one GM mention and two Tesla mentions.
Four current-identity reference occurrences remain unlinked: Dell, Boeing twice,
and Toyota. These names are not covered by the conservative conformed-name and
suffix-only matching policy in the pinned snapshot. All their company spans were
detected, so these are linking coverage failures, not missed extraction. No aliases
were introduced from this review.

## Verification and limits

[Aggregate metrics/provenance](us-company-review-reviewed.json). The reviewed
reference SHA, detector revision, SEC/registry hashes, sample/plan/source hashes,
CPU device and private per-record predictions are recorded. Original drafts and
reference decisions are retained; source text and per-record results remain
ignored. No sentiment model inference/training or frozen sentiment record access
occurred. Historical category sampling, small size, technology concentration,
shared SEC identity source, possible pretraining exposure and bulk acceptance of
assistant drafts prevent independent-test or representative US-news claims.

This completes the bounded review/comparison task. A separately collected,
independently annotated entity test collection and any resolver improvements are
separate tasks. [Source/sample methodology](us-company-review.md).

All 784 tests pass. Independent set-based recomputation confirms both issuer-link
confusion counts; pre/post-approval prediction bytes are identical, as are all
reference span/type/CIK values from the draft. Only review metadata, uncertainty
flags and null-link statuses were finalized. Protected registry/freeze/deployment/
target-test hashes remain unchanged. Ruff passes excluding the pre-existing
scratch notebook; full checks still report its nine lint errors and formatting.
`git diff --check` passes.
[Verification receipt](../verification/us-company-review-reviewed-verification.json).
