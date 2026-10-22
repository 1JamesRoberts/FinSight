# Reviewed organization benchmark and word-aware default

Reviewed 2026-10-20. [Metrics and provenance](organization-benchmark-reviewed.json) · [Independent recomputation](../verification/organization-benchmark-reviewed-verification.json) · [Review status](../verification/organization-annotation-review-status.json) · [Original provisional report](organization-benchmark.md) · [Method and commands](../../../docs/phase-04-target-sentiment/data/organization-benchmark.md).

Use word-aware `first` grouping as the default for the **experimental organization detector**. On the accepted development reference it yields 81.25% exact organization-span F1 versus 60.87% with simple grouping. This selects a postprocessing method for further development; it does not establish production company-identification accuracy. Registry-based company identity policy, ticker linking and the selected RoBERTa sentiment checkpoint are separate concerns.

## What was reviewed

The fixed prediction-free sample remains 50 validation headlines. Its assistant-drafted annotations were accepted by the user: rows 1–10 individually, remaining rows in bulk with later authorization to finalize after evidence adjudication. The user proposed identities for three uncertain names; contemporary reporting and company disclosures support company types for UTI Asset Management, Financial Technologies (India), and National Buildings Construction Corporation/NBCC. In particular, Fin Tech in this headline denotes the issuer rather than the generic industry term. The adjudication retains all names exactly as written; no canonical identity or ticker replaces a span. Supporting sources and discussion notes remain in the private review packet.

All 50 rows now record user approval and no type-uncertainty flags remain. This is a user-accepted assistant draft, not independent expert double annotation. The sample was previously seen during development and two headlines were in an earlier assistant review. Do not describe it as an independent final test set or treat its small size as representative of US-company news.

| Reference attribute | Count |
| --- | ---: |
| Validation headlines | 50 |
| Approved headlines | 50 |
| Empty organization-reference headlines | 23 |
| Company mention spans | 27 |
| Other-organization mention spans | 7 |
| Unresolved uncertain mentions | 0 |

**Table attributes:** Headlines counts distinct source records. Approved means recorded user acceptance under the scope above, not an independently audited annotation process. Empty headlines are included in the 50 and can contribute extra detections. Company and other-organization spans are disjoint, counting each written occurrence separately and summing to 34. Uncertainty counts unresolved annotation flags, not all possible real-world ambiguity. Neither native sentiment labels nor the sampled source text changed.

## Exact organization extraction

| Grouping | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple BIO grouping | 21 | 14 | 13 | 60.00 | 61.76 | 60.87 |
| Word-aware first grouping | 26 | 4 | 8 | 86.67 | 76.47 | 81.25 |

**Table attributes:** True positives (TP) are predicted spans with exact start/end offsets matching a reference organization in the same headline. False positives (FP) are unmatched predictions; false negatives (FN) are missed references. A partial name counts as both FP and FN, with no overlap credit. Precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2TP/(2TP+FP+FN); percentages pool occurrences across all 50 headlines (micro averaging). Both methods use the same 34-reference-span set, model weights and headlines, so these are paired development results, not independent test sets.

Word-aware grouping adds five exact matches and removes ten extra proposals/fragments, improving F1 by 20.38 percentage points. It still leaves four unmatched proposals and eight missed reference spans. The five-company registry controls emit zero candidates on this sample and have no supported reference names here; their full counts and undefined precision are preserved in the JSON, without interpreting restricted scope as a general detector failure.

The reviewed metrics equal the original provisional metrics because adjudication changed approval/uncertainty metadata but not the annotated offsets or company/other types. The new run reloaded the pinned NER checkpoint and independently generated all four methods' outputs. Membership, prediction hashes and every scope's metrics were independently checked after execution. Historical draft, predictions and provisional reports remain intact.

## Company-candidate limitation

| Grouping | Company TP | Extra candidates | Missed companies | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple | 17 | 18 | 10 | 48.57 | 62.96 | 54.84 |
| Word-aware first | 22 | 8 | 5 | 73.33 | 81.48 | 77.19 |

**Table attributes:** All raw ORG proposals are treated as company candidates and compared with the 27 company reference spans. Exact company matches are TP; unmatched proposals, including correctly detected non-company institutions, are extra candidates; missed references are FN. Percentages use the preceding formulas and same headlines. This diagnostic overlaps the all-organization evaluation; it does not add a trained company-type classifier or verify ticker identity.

Even the improved grouping cannot make every organization a company. Keep `company_identity_verified: false` in candidate sentiment output. No confidence threshold, registry expansion, entity linking, fine-tuning or stock-prediction functionality is added by this change.

## Implementation and verification

`OrganizationDetector.load()` and the organization CLI now default to `first`, recorded in each CLI response with detector version `organization-ner-v2`. `--aggregation-strategy simple` preserves the original behavior when requested. The historical full-validation audit and benchmark runner explicitly request simple for their simple arm, so changing the interactive default does not relabel their baseline. Historical source bytes were preserved in the private annotation packet before editing.

Reference SHA-256: `9d465f9aa12f9b2d0109af86e0763a1eb5dd118085b8e0fc8b23b1a2c6ea1a67`. Sample manifest: `dec8836961e3891902bd6134fdc3dd942cc4ce920c3ec9440693f272922f8b03`. The reviewed run uses the same 107,726,601-parameter BERT checkpoint, with 12 layers, hidden size 768 and nine BIO classes. Architecture, weights and sentiment pair formulation are unchanged; only ORG postprocessing is selected.

Four new behavior cases verify the first default, explicit simple/first overrides and rejection of unsupported grouping. All 739 pytest tests pass. Required full Ruff checks fail only on the unchanged scratch notebook's existing nine lint issues and formatting; checks excluding that notebook pass. No frozen records were read or models trained. This closes the bounded development review and grouping-selection task, not the independent entity-labeled test collection or all of Phase 4.

## Full GPU detection-to-sentiment smoke

The actual updated CLI completed on the RTX 4070 SUPER using the word-aware BERT
detector and the existing saved RoBERTa model on invented text about Infosys and
Microsoft. It returned complete name spans; both sentiment results exactly match
direct predictor calls with those targets. A second invented headline with no
organization returned empty mentions and empty candidate sentiments. No dataset
sentiment inference or training was performed. The [smoke receipt](../verification/organization-word-aware-smoke.json)
contains outputs, revision and source identities; this verifies function, not accuracy.

All seven NER model/tokenizer/card files verify against pinned sizes and hashes.
The initial PC transfer was interrupted at 201,845,760 weight bytes; direct
immutable-revision acquisition on the PC completed and verified the file before
loading. No partial checkpoint was used. An isolated 128-file source/evidence bundle
has manifest identity `54e60fb9c64b621f47f3cd1eb2a81d48219ac7c066ce06e2a816ec512aec90c5`;
its bytes were checked before importing. Source and evidence stay under
`data/interim/organization-release/` inside the existing project. The PC's primary
source checkout was not updated, no extra project directory was created, and
source news text was not transferred. NER files are stored once in the shared
ignored model directory; the original RoBERTa artifact remains in place.
