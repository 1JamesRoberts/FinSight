# Phase 4 final report — Company-targeted sentiment

**As of:** 2026-10-22. **Status:** consolidated report for verified target-sentiment,
company-resolution and GPU-inference milestones. Broader Phase 4 acceptance is
still open for an independent entity-labeled company test collection and
per-company-target evaluation. This filename identifies the canonical phase
summary; it does not silently close those roadmap items.

[All final reports](../README.md) · [Phase evidence index](README.md)

## Scope and status

Determine whether conditioning on a target improves sentiment for multi-target
headlines, select/serve a saved model, and build a conservative organization →
company-type → issuer identity → target-sentiment path. Verify traditional
baselines, model variants, target formats, extraction/linking and real GPU
routing. No daily aggregation, market prediction or new model training is
introduced by the final report itself.

## Data and method

Native SEntFiN targets/labels remain attached to complete headlines. Development
backbone comparisons train on 11,547 target annotations and validate on 1,410
annotations from 1,053 headlines; these include non-company native entities.
Three-seed matched headline-only/target-pair training compares original DeBERTa
Base, RoBERTa Base and FinBERT, alongside TF-IDF Logistic Regression controls.
Training and inference paths remain separate with immutable artifact manifests.

The frozen v1 membership audit found one near-duplicate group shared with the
already consumed Phase 3 sentence test and stopped before inference. Authorized
v2 excludes that group only from the evaluation view, preserving original source
freeze bytes. The evaluated view has 262 headlines/592 target annotations;
110 mixed-label headlines contribute 253 annotations. All fixed variants run
under the predeclared protocol. The consumed v2 view must not guide new tuning.
[Overlap review](evaluation/overlap-review.md) and [v2 protocol](../../docs/phase-04-target-sentiment/evaluation/target-final-evaluation-v2.md).

Company extraction/linking is evaluated separately from sentiment, on small
assistant-drafted, user-accepted development references. Native target labels are
not exhaustive company spans. No company confidence is inferred from NER's ORG
score, and no sentiment label determines issuer identity.

## Results

### Locked target-sentiment evaluation

| Model family | Headline-only macro F1 (%) | Target-aware macro F1 (%) | Conditioning gain (pp) |
|---|---:|---:|---:|
| DeBERTa Base | 74.53 | 88.75 | +14.23 |
| RoBERTa Base | 72.82 | 88.34 | +15.52 |
| FinBERT | 73.19 | 86.84 | +13.65 |
| TF-IDF Logistic Regression | 68.47 | 76.15 | +7.68 |

**Attributes:** Model family identifies a matched model/input comparison.
Headline-only ignores the selected native target; target-aware uses the target
as part of input. Macro F1 equally averages the three sentiment classes on the
same 592 target annotations; transformer values average seeds 42/43/44, while
TF-IDF is a fixed pipeline. Conditioning gain is the unrounded paired difference
in percentage points, so rounded displayed values may not subtract exactly.
Rows share the same cohort; these are neither per-ticker scores nor sentence-test
scores. An annotation counts once per native target, not once per headline.

RoBERTa's primary conditioning gain has paired 95% interval [12.16, 19.10] pp.
On mixed-label headlines it improves from 49.45% to 85.96% F1 (+36.51 pp), while
same-label gain is only +0.16 pp with an interval spanning zero. RoBERTa minus
DeBERTa is -0.41 pp, interval [-1.62, 0.87]; this does not establish a backbone
winner. The validation-selected RoBERTa candidate remains fixed. Full class,
subgroup, seed, model-size and timing evidence is in the
[locked target report](../experiments/phase4-target-final.md) and [JSON](../experiments/phase4-target-final.json).

### Input-format and serving decision

Nine subset-matched RoBERTa trials compare original target/headline pairs,
markers around literal targets and target replacement. Shared membership contains
11,395 training and 1,396 validation annotations, so its scores are not the
full-development or frozen-test scores above. Mean all-target validation F1 is
89.70% original pair, 89.48% markers and 88.83% replacement. Replacement decreases
mean F1 by 0.86 pp; the experiment supports retaining the original pair, not an
independent superiority claim. Seen/unseen cohorts use exact native target
strings, not verified entity-disjoint companies.
[Format comparison](experiments/target-format-comparison.md) and [name sensitivity](experiments/target-name-stress.md).

Deployment selects RoBERTa Base seed 43/epoch 2 using the highest saved all-target
validation F1 (89.7683%), not test results. It is one selected checkpoint; the
three-seed test average is not its measured single-checkpoint performance.
Architecture: 12 layers, hidden size 768, 12 attention heads, 124,647,939 parameters.
Input is target first, full headline second, maximum 128 tokens with headline-only
truncation. Raw probabilities remain uncalibrated. Phase 3 sentence calibrators
are not silently reused for this target model. [Selection/verification](evaluation/target-prediction.md).

### Organization extraction and issuer linking

The unchanged `dslim/bert-base-NER` checkpoint supplies ORG proposals; its weights
were not trained here. It has 12 BERT layers, hidden size 768 and nine BIO labels
(107,726,601 parameters). On a user-accepted 50-headline development reference,
word-aware `first` grouping improves exact organization-span F1 from 60.87% to
81.25%. The 34 reference spans comprise 27 company and seven other organizations.
This is postprocessing selection on previously seen development data, not an
independent company benchmark. [Reviewed extraction study](evaluation/organization-benchmark-reviewed.md).

An opt-in current SEC snapshot (2026-10-21) contains 6,071 CIK issuers/7,699
listing associations; 531 obvious ETF/Fund/Trust issuers require type review.
US-listed includes foreign issuers. Preserve all listings and never guess a
primary security. Conformed-name/short-name/ticker matches retain collisions and
require appropriate caller context. The file is not a historical security master.
[Source/snapshot scope](experiments/us-listed-directory.md).

A first 20-headline US-company development review exposed short-name gaps.
Reviewed Boeing/Toyota issuer aliases, Co/Company suffix handling and Dell
subsidiary evidence were then fixed before selecting a disjoint second sample.
The second sample's approved default reference contains 26 company spans, of
which 16 have accepted current CIKs without implicit parent roll-up.

| Fresh-sample resolver | Correct company spans / 26 | Company F1 (%) | Correct CIK links / 16 | Link F1 (%) |
|---|---:|---:|---:|---:|
| Archived resolver v3 | 8 | 47.06 | 8 | 66.67 |
| Legal suffix fixes only | 12 | 63.16 | 12 | 85.71 |
| Reviewed aliases, no parent roll-up | 18 | 81.82 | 15 | 96.77 |

**Attributes:** Resolver identifies fixed rules applied to the same 26 NER
proposals across 20 headlines. Correct company spans requires exact offsets and
company type; its denominator includes all 26 reference company occurrences,
including unlinked companies. Company F1 is occurrence-level micro F1 for
company-classified outputs. Correct CIK links requires exact offsets plus the
approved issuer; its 16-reference denominator excludes company mentions with no
accepted current issuer. Link F1 is micro exact-span/CIK F1. All three arms have
zero observed false positives on this small accepted draft, not a population
precision guarantee. Rows overlap, and company/link scores use different
reference universes; neither is sentiment accuracy.

Aliases classify Dell as a company without assigning Dell Technologies by
default. Explicit `--allow-parent-rollup` permits the reviewed parent mapping;
that arm is a coverage diagnostic, not accuracy against no-roll-up reference
CIKs. Current identity/ownership evidence does not prove historical continuity.
The remaining default link miss is a Boeing span BERT did not extract. Six
other exactly detected company names remain type-uncertain; no aliases were
added after this second review. User bulk acceptance of assistant drafts,
shared SEC identity evidence, targeted company categories and unknown pretraining
exposure qualify all linking results. [Reviewed fresh sample](evaluation/issuer-alias-review-reviewed.md).

### Real GPU pipeline verification

On the RTX 4070 SUPER, actual CUDA BERT → resolver → saved RoBERTa checks verify:
Toyota/Boeing link under explicit company context without per-name confirmation;
Dell receives sentiment with null issuer fields by default; parent roll-up
changes its identity/eligibility only; unknown, context-missing and empty cases
skip sentiment loading. Strict, parent-roll-up and direct `--target` outputs have
maximum probability difference 0.0 for identical invented text/target batches.
This verifies integration, not news accuracy or calibrated probabilities.

The latest 142-file source release is manifest-pinned and both project-root
launchers are tested. Previous source/launcher bytes are retained for rollback;
the tracked checkout is untouched. Models, contacts and news/reference records
were not transferred. [GPU report](verification/issuer-alias-gpu.md) and [run/rollback instructions](../../docs/phase-04-target-sentiment/issuer-alias-gpu.md).

## Decisions

Keep target-aware RoBERTa and the original target/headline pair. Treat company
type, issuer identity and sentiment as separate outputs: confirmed companies can
receive sentiment with unresolved CIKs, while issuer grouping requires a link.
An ORG tag alone is insufficient. Use source-backed aliases/context guards and
explicit subsidiary roll-up; preserve unknown names rather than invent neutral
sentiment or merge similarly named businesses. Maintain checksum-verified source,
checkpoint, configuration and receipt lineage; preserve historical baselines.

## Limitations and remaining work

The independent entity-labeled company test collection and per-company-target
macro F1 remain open in the roadmap. Native SEntFiN entities and small accepted
mention samples do not satisfy those criteria. This phase has no US-company
population guarantee, full entity/coreference resolution, trained company-type
classifier, historical listing universe or calibrated target deployment.
The sentence and multi-target tests are consumed; changes inspired by results
need new untouched confirmatory data. Hosted CI verification remains separate
from local checks. Phase 5 aggregation and Phase 6 time-respecting prediction
are separate, not automatically implemented by company-link eligibility.

## Evidence and reproduction

- [Final target evaluation](../experiments/phase4-target-final.md), [artifact/metric JSON](../experiments/phase4-target-final.json), [phase evidence index](README.md).
- [Development backbone comparison](../experiments/target-backbones.md), [target error analysis](../experiments/target-error-analysis.md), [format comparison](experiments/target-format-comparison.md).
- [Reviewed NER](evaluation/organization-benchmark-reviewed.md), [first US-company review](evaluation/us-company-review-reviewed.md), [fresh alias review](evaluation/issuer-alias-review-reviewed.md).
- [GPU integration receipt](verification/issuer-alias-gpu.json), [verification](verification/issuer-alias-gpu-verification.json), [current commands](../../docs/phase-04-target-sentiment/issuer-alias-gpu.md).
- [Current deployment config](../../configs/target-roberta-inference.json), [alias catalog](../../configs/us-company-aliases.json), [source/terms ledger](../../docs/datasets.md), [methods](../../docs/phase-04-target-sentiment/README.md).

Review recorded aggregates/receipts before considering any rerun. Private source,
model weights and per-record outputs remain ignored; a fresh clone alone cannot
execute artifact-dependent inference. Use the verified GPU launchers for new
explicit targets and the existing cache/receipt tools for historical metrics.
No data/model inference was run to write this summary.
