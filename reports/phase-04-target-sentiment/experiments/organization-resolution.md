# Organization resolution and remaining extraction errors

Reviewed 2026-10-21. [Method and CLI behavior](../../../docs/phase-04-target-sentiment/company-resolution.md) · [Aggregate results](organization-resolution.json) · [Verification](../verification/organization-resolution-verification.json) · [Accepted reference evaluation](../evaluation/organization-benchmark-reviewed.md).

A conservative resolution step now separates reviewed-registry company matches, explicit institutional titles and unresolved proposals. Default NER-based sentiment scores only verified company matches. The reviewed SEntFiN sample yields **zero resolution coverage**: all 30 proposals remain unresolved. This is a limitation of identity/rule coverage, not evidence of improved company extraction accuracy. The existing five-company registry is unchanged, and no extra benchmark names were added as aliases.

## Scope and policy

Reuse the 50-headline assistant-drafted, user-accepted development reference and its checksummed word-aware first predictions. The reference has 27 company and seven other-organization spans. Earlier review included individual approvals for rows 1–10 and bulk acceptance of the remainder with sourced adjudication; it is not independent expert double annotation or a final test set. No new corpus inference, training, sentiment scoring on SEntFiN or frozen-record access is needed for this cached evaluation.

Verified company requires a whole detected surface to resolve through the existing registry. Bare ambiguous names and parent rollups retain explicit context/permission requirements; ORG confidence does not satisfy them. Full-span institution title rules recognize departments/ministries, courts, securities regulatory titles and exchanges. Unknown names and acronyms remain unresolved. These lexical type rules do not verify legal identity and are deliberately incomplete; even unknown financial names cannot be assumed non-company.

The resolver does not trim quarter markers, split merged exchange names, drop financial terms by a newly learned exclusion list, or copy reference labels into decisions. It leaves extraction faults and absent spans visible. Detailed private assistant review covers all 12 original error instances in eight headlines.

## Remaining word-aware extraction errors

| Error category | Error instances |
| --- | ---: |
| Predicted name extended into a quarter marker | 2 |
| Multiple names merged into one proposal | 1 |
| Financial term proposed as an organization | 1 |
| Reference mentions lost to incorrect boundaries | 4 |
| Reference mentions completely omitted | 4 |

**Table attributes:** The first three rows count the four extra predicted spans: two overextensions, one merge, and one spurious term. The last two rows partition eight missed reference spans. A bad prediction can also cause one or more missed references, so categories overlap in headline/context and must not be interpreted as independent examples. The merged proposal covers two separate reference organizations, causing two misses. There are 12 span-level error instances across eight distinct headlines. The public report paraphrases findings; original contexts, offsets and assistant notes remain private.

The overextensions include quarter designations attached to issuer names. One proposal merges two exchanges. Another is a financial-event term rather than an organization. Complete omissions include three company names and one regulator abbreviation. Resolution cannot recover names that NER never proposed. The verified counts are four complete omissions and four boundary-lost reference mentions; the earlier conversational estimate of five complete omissions was incorrect.

## Resolution decisions against reference types

| Resolver outcome | Exact company matches | Exact other-organization matches | Unmatched spans | Total proposals |
| --- | ---: | ---: | ---: | ---: |
| Verified company | 0 | 0 | 0 | 0 |
| Other organization | 0 | 0 | 0 | 0 |
| Unresolved | 22 | 4 | 4 | 30 |

**Table attributes:** Each cell counts predicted occurrences, not unique companies or headlines. Exact matches require identical offsets in the same headline; unmatched spans have no exact reference match. These three columns are disjoint and sum to each row's total, and the three outcome rows partition all 30 proposals. Reference company/other types are joined only after the resolver has made its decision; they are never inputs to it. Unresolved company matches are abstentions on real reference companies, not correct non-company classifications.

None of the 22 exact company proposals matches the five-company registry. The four exact institution proposals are short forms outside the full-title rules, and the four extraction mistakes also remain unresolved. Decision coverage is therefore 0/30 = 0%. Conditional typed exact precision is undefined (null), because no company/other decision was emitted. The benchmark supplies no real-data validation of the institution-title rules or verified-company branch; invented behavior tests and functional smoke checks cover those code paths separately. No high-precision claim follows from emitting nothing.

## Company extraction consequence

| Output policy | Company TP | Extra company candidates | Missed company references | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Every raw ORG treated as a company candidate | 22 | 8 | 5 | 73.33 | 81.48 | 77.19 |
| Verified registry companies only | 0 | 0 | 27 | undefined | 0.00 | 0.00 |

**Table attributes:** TP counts exact matches to the 27 company reference spans. Extra candidates are unmatched predictions, including correctly detected non-company institutions. Misses include absent, badly bounded and unresolved company references. Precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2TP/(2TP+FP+FN), pooled across all 50 headlines and displayed as percentages. Precision is undefined when no candidates are emitted. The policies reuse the same underlying NER predictions and reference set; the filtered row explicitly includes abstention cost rather than hiding unknown companies from evaluation.

Verified-only scoring prevents unsupported identity assertions but has no useful company coverage on this Indian-finance sample. Do not present this as a performance gain or as proof that unknown company names are unimportant. Broader company recognition needs a separately reviewed company/alias directory or entity-linking source; this task neither acquires one nor trains a classifier from the 50 annotations.

## Inference behavior and verification

`organizations --with-sentiment` now returns `company_sentiments` for verified company identities only. All original NER mentions, original spans, resolution status and reason, registry checksum, and any matched identity/ticker remain visible. Repeated verified aliases share one company-level prediction using the first original surface; no text masking or ticker substitution is introduced. Institutional/unresolved-only requests skip RoBERTa loading/forward inference and receive no neutral placeholder.

`--score-all-candidates` with `--with-sentiment` explicitly preserves the old raw-ORG diagnostic and returns `candidate_sentiments` marked unverified. Existing direct-target prediction and standalone registry detection are unchanged. Rule behavior is typed/documented with version `organization-resolution-v1`; no NER or sentiment weights change.

The evaluator verifies the accepted reference (`9d465f9aa12f9b2d0109af86e0763a1eb5dd118085b8e0fc8b23b1a2c6ea1a67`), benchmark/cache bytes and exact execution packet before joining reference types. All 30 decision cells and 12 error categories were recomputed after writing; all eight error contexts were assistant-reviewed privately. Registry checksum remains `3414bd464620472b37f23944f0c6d8011f77fc8460491c4001010d2aa820d285`; reference spans and prior metrics are unchanged. This completes only the bounded review/resolution baseline, not broad company identification or Phase 4 as a whole.

Twenty-four new behavior cases cover known/unknown/ambiguous names, whole-title rules, source offsets, identity grouping, verified-only versus explicit raw scoring, no-weight unresolved requests, typed metrics and merged-span accounting. All 763 pytest tests pass. Required full Ruff runs still encounter only the unchanged scratch notebook's nine lint errors and formatting; excluding it passes. No new dependency is introduced.

## Full GPU CLI smoke

The [CUDA smoke receipt](../verification/organization-resolution-smoke.json) verifies the actual
updated CLI on invented Microsoft/Tesla text. Microsoft resolves to the existing
registry identity and is the only default sentiment target; its saved RoBERTa
output exactly matches a direct predictor call. Tesla remains unresolved, and a
Tesla-only request returns no company sentiment. An explicit all-candidate request
still scores both original targets. This checks routing and artifact compatibility,
not real-data classification accuracy or the institution rules' generalization.

The isolated 131-file source/evidence snapshot is
`0defc077d350215bd60175d647b962e00e8b0ce35e5fbc182dc73c06ee565107`,
verified before import under the existing project's ignored data directory. The
PC's primary source checkout and saved model artifacts were not replaced. There
was no new model acquisition or dataset sentiment run. Detection-only plus
unresolved-only `--with-sentiment` also passed on the Mac without RoBERTa weights.
