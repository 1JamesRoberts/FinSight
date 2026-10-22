# Organization mention development benchmark: provisional comparison

Prepared and run 2026-10-20. **All reference annotations are assistant drafts; human review is pending for all 50 headlines. These are exploratory development metrics, not human-gold test results.**

[Annotation and review guide](../../../docs/phase-04-target-sentiment/data/organization-benchmark.md) · [Annotation policy](../../../docs/phase-04-target-sentiment/data/organization-annotation-policy.md) · [Sampling plan](../../../configs/organization-annotation-plan.json) · [Provisional aggregate JSON](organization-benchmark-provisional.json) · [Independent metric verification](../verification/organization-benchmark-verification.json).

## Current annotation review

The user has now accepted all 50 reference rows and authorized finalization after
sourced adjudication of the three company names. See the [reviewed run and
word-aware default decision](organization-benchmark-reviewed.md) and [current
review status](../verification/organization-annotation-review-status.json). Rows 1–10 were approved
individually, with the remainder accepted in bulk. The historical provisional
snapshot and zero-approval counts below remain unchanged as provenance; they are
not the current review state.

## Sampling and reference status

A fixed hash rule samples 50 of the 1,053 native validation headlines, independent of predictions, error flags, native target labels and the earlier twelve-company allowlist. Rank IDs by SHA-256 of `organization-mentions-dev-v1|42|<record_id>` and take the smallest 50. No negative/empty headline is replaced with a company-containing one. The private sample manifest and exact source bytes were frozen before the annotation draft.

The assistant drafted exhaustive explicit company/other-organization mentions from the source-only text, without displaying native sentiment targets or current sampled NER predictions. Draft bytes and policy identity were then locked before running this comparison. This is not a fully blind or unseen study: the earlier NER task already scored all validation headlines, and two sampled headlines were in the earlier assistant review. This exposure is recorded rather than hidden by resampling.

| Draft support/status | Count |
| --- | ---: |
| Sampled validation headlines | 50 |
| Headlines with no drafted organization | 23 |
| Draft company mention spans | 27 |
| Draft other-organization mention spans | 7 |
| Draft organization mention spans, total | 34 |
| Mentions with uncertain company type | 3 |
| Human-approved headlines | 0 |

**Table attributes:** Headlines counts source records; spans counts every explicit occurrence, including repeats. Company and other-organization spans are disjoint and sum to 34. Empty headlines are included in the 50 and must also be reviewed. Uncertain company types are a subset of company spans, not additional entities; the provisional calculations include their draft type without silently excluding them. Human-approved headlines counts rows with recorded approval; currently all rows are pending. Every support count depends on the unreviewed draft and may change after human corrections.

The policy includes full written names and organizational units and excludes people, locations, indices, commodities and generic sectors. It treats exchanges as other organizations for this operational task, irrespective of their legal incorporation. Three issuer/acronym-style names require explicit company-type adjudication. Original SEntFiN sentiment annotations remain unchanged.

## Exact organization-span comparison against assistant drafts

The same pinned `dslim/bert-base-NER` checkpoint is scored twice, changing only token grouping. `simple` keeps the existing BIO grouping. Word-aware `first` assigns a word's subwords the first token's label before grouping, reducing within-word label disagreement; it cannot guarantee the right entity type or complete multiword boundary. This behavior is documented in the [Transformers pipeline API](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline). There is no new model, training or tuned threshold.

| Method | Correct spans (TP) | Extra spans (FP) | Missed spans (FN) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Five-company registry: resolved | 0 | 0 | 34 | undefined | 0.00 | 0.00 |
| Five-company registry: resolved + ambiguous | 0 | 0 | 34 | undefined | 0.00 | 0.00 |
| BERT NER: simple grouping | 21 | 14 | 13 | 60.00 | 61.76 | 60.87 |
| BERT NER: word-aware first grouping | 26 | 4 | 8 | 86.67 | 76.47 | 81.25 |

**Table attributes:** A correct span (true positive, TP) has exactly the same headline ID and start/end character offsets as a draft organization mention. Extra spans (false positives, FP) lack an exact draft match; missed spans (false negatives, FN) are unmatched draft mentions. A partial boundary is one FP and one FN, with no overlap credit. Counts pool occurrences across all 50 headlines (micro averaging); repeated occurrences count separately. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2TP/(2TP+FP+FN), displayed as percentages. Undefined precision means no proposals were emitted, not perfect precision. All organizations are evaluated together without company-type credit. All TP/FP/FN labels are relative to assistant drafts, so they do not yet establish actual error rates.

Word-aware grouping increases draft-reference F1 from 60.87% to 81.25% (+20.38 percentage points), with five more exact matches and ten fewer extra fragments/proposals. The provisional direction supports investigating word-aware grouping, but the size of the gain is not a verified generalization estimate. The unchanged five-company registry emits no candidates on this sample: its names have no draft support here, so the zeroes illustrate restricted scope rather than a fair test of general organization extraction quality. No confidence intervals or significance claims are made for this small, unreviewed development sample.

## Treating all ORG proposals as company candidates

| NER grouping | Company matches (TP) | Extra company candidates (FP) | Missed draft companies (FN) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple | 17 | 18 | 10 | 48.57 | 62.96 | 54.84 |
| first | 22 | 8 | 5 | 73.33 | 81.48 | 77.19 |

**Table attributes:** The same unfiltered ORG proposals are compared with only the 27 provisional company spans; no company-type classifier or ticker mapping is added. Company matches require an exact company-draft span; extra candidates include both incorrectly extracted names and correctly extracted non-company organizations. Precision, recall and F1 use the formulas above with company-only references. The three uncertain draft company types remain included, so these numbers can change after adjudication. This table overlaps the organization comparison and is not a separate test set or proof that NER itself predicts company status.

This diagnostic explains why broader automatic company sentiment still needs a type/identity step. Even correctly extracting a regulator or central bank does not make it a company candidate for ticker-level aggregation. No detector or sentiment deployment behavior is changed here.

## Human review and remaining work

The private `review.md` sheet contains all 50 source headlines, drafted spans with offsets and types, empty cases and notes. It does not show model outputs or native sentiment labels. Open the sheet using the path and commands in the annotation guide. Correct/add/delete spans and types in a separate `annotations.reviewed.jsonl`, resolve uncertainty and record actual per-row human approval. Keep the assistant draft unchanged for provenance.

The default evaluation command rejects pending reviews or uncertain types **before loading models**. An explicit `--provisional` flag is required for the exploratory comparison above. The software validates review metadata as a reviewer assertion; it cannot prove reviewer identity or independently guarantee exhaustiveness. Human corrections receive their own reference checksum and separate reviewed result file. No human-reviewed result exists yet.

Completed: prediction-free development sampling, annotation policy, exhaustive assistant draft, private review sheet, exact-span metric implementation, explicit review gating, and the two-grouping provisional comparison with unchanged registry controls. Pending: actual human review/adjudication, approved-reference evaluation, and any subsequent detector selection. This does not complete the roadmap's independent entity-labeled test collection or close Phase 4.

## Reproducibility and checks

The sample manifest is `dec8836961e3891902bd6134fdc3dd942cc4ce920c3ec9440693f272922f8b03`. Draft annotation SHA-256 is `1a8d697985b7a39841419afefc552bfbe24e00aa67e6b45e26a5bbeb0fe0c036`; it was fixed before inference. Model revision remains `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`, with 107,726,601 parameters, 12 layers, hidden size 768, 12 attention heads and nine BIO token classes. Source, plan, policy, lockfile, model-plan and reference digests are recorded. Neither NER nor RoBERTa was trained, and RoBERTa was not scored in this task.

All 200 private headline/method outputs (50 × four methods) have verified membership and prediction hashes. Exact-span TP/FP/FN and P/R/F1 were independently recomputed from occurrence sets for both scopes and all methods. The simple-grouping spans exactly reproduce those in the prior validation cache on the same headlines. Original sample, draft and each evaluated reference snapshot remain private; public artifacts have no source headlines, source IDs or row-level detections. No frozen records were read.

Fourteen new behavior tests cover sampling independence, input identity, annotation completeness/offsets/types, review gates before loading, exact boundary errors, repeated occurrences, empty cases and undefined denominators. All **735 pytest tests pass**. Required full Ruff commands encounter only the unchanged scratch notebook's nine existing lint errors and formatting issues; excluding that notebook, lint and formatting pass. No new dependency was introduced.
