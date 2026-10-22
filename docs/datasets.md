# Dataset, model, and API review checklist

Phase 4 evaluation revision (2026-10-18): the user authorized excluding one
previously evaluated near-duplicate group from the multi-target evaluation view.
The original frozen SEntFiN records remain unchanged; the v2 contract records both
member text hashes and the exclusion reason before any Phase 4 test prediction.
No new acquisition, label revision, training or redistribution scope is involved.
See the [v2 evaluation protocol](phase-04-target-sentiment/evaluation/target-final-evaluation-v2.md).

PhraseBank, SEntFiN, and an original-format FiQA participant copy have been acquired for reviewed local noncommercial
research; all source files remain in ignored storage. Local TF-IDF models are
stored only in ignored directories; no pretrained weights or provider integration
are present. See [dataset-loading.md](phase-01-data/acquisition/dataset-loading.md) for checksums,
loading instructions, and measured integrity results.
Phase 1's public-source candidate comparison is recorded in
[dataset-comparison.md](phase-01-data/acquisition/dataset-comparison.md), reviewed on 2026-10-10.
An independent-evaluation shortlist is recorded in
[evaluation-dataset-shortlist.md](phase-01-data/acquisition/evaluation-dataset-shortlist.md).
SEntFiN's scoped local-research assessment and immutable revision are recorded in
[sentfin-asset-review.md](phase-01-data/acquisition/sentfin-asset-review.md) and
[asset-revisions.json](../configs/asset-revisions.json).
The subsequent [evaluation contract](phase-01-data/benchmark/evaluation-contract.md) records the curated
SEntFiN freeze for local offline research; membership and gold digests are pinned
in `configs/benchmark-freeze.json`. Known accessible-corpus overlap is excluded;
full checkpoint pretraining exposure remains unknown.
Completed cards summarize the selected artifacts: [PhraseBank](phase-01-data/dataset-cards/phrasebank.md),
[SEntFiN](phase-01-data/dataset-cards/sentfin.md), and [FiQA copy](phase-01-data/dataset-cards/fiqa.md).
Development-only figures and completion evidence are in
[Phase 1 findings](phase-01-data/eda/phase1-findings.md).
**Other asset approvals and public-text rights remain pending.** This checklist
is not a legal conclusion or authorization to acquire or redistribute an asset.

VADER package/lexicon review (2026-10-11): standalone `vaderSentiment==3.3.2`
is approved for local inference/research under its MIT grant, with attribution
and notices retained. See [VADER asset decision and baseline](phase-02-baselines/models/vader.md). Packaged
code/lexicon/emoji digests are logged separately; the frozen source-data manifest
is unchanged. No upstream evaluation datasets or model weights are acquired.

Complete one record per exact dataset revision, model checkpoint, or provider
product before downloading, persisting, training on, or republishing content.

TF-IDF + Logistic Regression review (2026-10-11): use scikit-learn 1.7.2
under its [BSD-3-Clause license](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING).
Retain installed notices. Existing PhraseBank noncommercial research and SEntFiN
local-development training assessments support fitting on their verified training
partitions and storing fitted pipelines locally. Vocabulary/IDF and coefficients
remain ignored model artifacts; their public redistribution is not cleared.
Validation is development evaluation only. FiQA and all held-out records are
excluded; no new source assets or pretrained weights are acquired.

The subsequent PhraseBank class-weight comparison uses the same reviewed local
training/storage scope and the same frozen parent split. No terms or acquisition
decision changes. See [comparison and input-flow guide](phase-02-baselines/models/phrasebank-class-weights.md).

Linear SVM review (2026-10-11): `LinearSVC` uses the already reviewed and locked
scikit-learn 1.7.2 BSD-3-Clause distribution. Existing PhraseBank noncommercial
and SEntFiN local development-training assessments support weighted/unweighted
fits on their training partitions, local fitted-model storage, and aggregate
validation reports. No new assets or dependencies are acquired. Vocabularies,
coefficients, and per-record outputs remain ignored; public weights/text rights
remain unresolved. Reserved evaluation groups remain withheld.

## GPU preparation asset review (2026-10-11)

The existing approved PhraseBank, SEntFiN, and FiQA copies may be transferred
to the user's own Windows research computer under the same documented local
noncommercial scope. Transfer preserves source bytes, notices, acquisition
receipt, parent split, and frozen benchmark; it does not authorize publication.

Optional runtime libraries are pinned to `torch==2.11.0` and
`transformers==4.57.6`. PyTorch's [BSD-style license](https://github.com/pytorch/pytorch/blob/v2.11.0/LICENSE)
and Transformers' [Apache-2.0 license](https://github.com/huggingface/transformers/blob/v4.57.6/LICENSE)
support local installation and research use; installed third-party notices
remain intact. Windows uses the official
[CUDA 12.8 wheel index](https://pytorch.org/get-started/previous-versions/#v2110).
The [CUDA 12.8 EULA](https://docs.nvidia.com/cuda/archive/12.8.0/eula/index.html)
permits installation/use subject to its conditions. Bundled components retain
their own accompanying terms and notices. Scope is local development on the
user's NVIDIA GPU; no binary redistribution is part of this task.

Rechecked the [FinBERT API metadata](https://huggingface.co/api/models/ProsusAI/finbert)
and [pinned model card](https://huggingface.co/ProsusAI/finbert/blob/4556d13015211d73dccd3fdd39d39232506f3e43/README.md).
The revision remains `4556d13015211d73dccd3fdd39d39232506f3e43`.
Its card has no license field, and its repository tree has no LICENSE file.
The README SHA-256 still matches the existing asset record
(`23bf85400efe1e88cbe2bf79cec8c5ff81fe9806c599f3ca85e18288ab871008`).
The separate source-code Apache license does not establish hosted-weight scope.
Weight acquisition therefore remains `pending_explicit_license_scope`;
no checkpoint weights were downloaded. CUDA preparation uses synthetic tensors
and library imports only. PhraseBank remains known FinBERT supervision overlap.

Subsequent user decision (2026-10-11): proceed with this pinned checkpoint for
personal, local noncommercial portfolio research despite the unresolved hosted
weight-license declaration. This authorizes acquisition, local storage,
inference, and research fine-tuning within the project; it is a user decision,
not new evidence of a license grant. Public weight redistribution and commercial
deployment are outside this decision. Explicit license clarification is no
longer a prerequisite for the authorized local work. Keep the uncertainty in
experiment documentation. The Phase 1 asset manifest remains unchanged because
existing acquisition/split receipts pin its digest; its pending decision records
the original review, superseded for local use by this paragraph. No weights
were downloaded while recording this decision.

Pretrained FinBERT validation acquisition (2026-10-11): use only the pinned
revision above and download its README, config, PyTorch weights, vocabulary,
tokenizer config, and special-token map into ignored project storage on Windows.
The [revision's file metadata](https://huggingface.co/api/models/ProsusAI/finbert/tree/4556d13015211d73dccd3fdd39d39232506f3e43)
identifies `pytorch_model.bin` as 437,992,753 bytes, with Git LFS SHA-256
`e15a7b5738df7f17553399b6d94c6e2ff69c89245d066e8e5d183f5803a554e3`.
The loader verifies this digest plus the previously reviewed README/config
digests before loading. Retain SHA-256 and size for all six files in the local
acquisition receipt and aggregate run metadata. Use trusted library BERT code,
local-only loading after download, and `weights_only=True`; no remote code or
randomly initialized sentiment head is accepted. Scope remains the user's
local research decision with unresolved hosted-weight license declaration.
Acquisition and validation succeeded on Windows with all pinned digests matched.
Six retained files total 438,226,859 bytes; no weights were changed or published.
The per-run acquisition receipt is in ignored storage and the public aggregate
[report](../reports/experiments/finbert-validation.md)/JSON records its checksums.
The historical Phase 1 manifest still records the earlier pre-acquisition state;
the new receipt supersedes that state without invalidating dataset split digests.

## Asset record template

- Name, asset type, canonical source, owner/provider:
- Exact revision/checkpoint/product tier and retrieval date:
- License and API terms URLs; version/date; retained evidence location:
- Intended use (research, training, inference, storage, portfolio demo):
- Permitted and prohibited uses; review rationale:
- Attribution/citation requirements:
- Commercial-use, redistribution, derivative-work, and model-output terms:
- Full-text vs metadata rights; retention/deletion requirements:
- API rate limits, access restrictions, and display/link requirements:
- Provenance of underlying material; third-party rights or privacy constraints:
- Storage location, checksum/version, and reproducible acquisition notes:
- Reviewer, review date, decision (pending/approved/rejected), unresolved questions:

## Before acquisition

- [ ] Verify source authenticity and pin the exact asset revision.
- [ ] Review the license and provider terms for the intended usage.
- [ ] Distinguish dataset packaging rights from rights in underlying news text.
- [ ] Record training, local storage, public display, and redistribution rights separately.
- [ ] Record attribution, retention, rate limits, and other obligations.
- [ ] Decide whether only metadata/source links may be retained or displayed.
- [ ] Resolve missing or conflicting terms; record the review decision.
- [ ] Keep credentials out of Git; choose an ignored artifact location.

## Before experiments or publication (later phases)

- [ ] Document labels, annotation quality, coverage, selection bias, and limitations.
- [ ] Record dataset versions, splits, duplicate checks, seeds, and configurations.
- [ ] Audit overlap with pretrained-model training data. ProsusAI/finbert's
      Financial PhraseBank overlap must be addressed before claims of independent
      generalization; create an independent benchmark or state overlap explicitly.
- [ ] Freeze the final test set; do not train or tune against it.
- [ ] For price research, record news availability/ingestion times, market cutoff,
      exchange calendar, and chronological evaluation design.
- [ ] Recheck redistribution/display rights before publishing samples or artifacts.
- [ ] Publish limitations and reproducibility metadata alongside results.

## Candidate review status

| Candidate | Purpose in later phases | Status |
| --- | --- | --- |
| Financial PhraseBank | Three-class baseline development | Reviewed for local noncommercial research; pinned, acquired, validated |
| FiQA Task 1 | Target/aspect-aware research candidate | Original-format participant copy reviewed, pinned, acquired, validated for noncommercial research; test inputs unscored; mirror terms unresolved |
| SEntFiN 1.0 | Frozen external/target-aware holdout | Curated local release frozen; public text rights and full pretraining overlap unresolved |
| SemEval-2017 Task 5 headlines | Backup target-aware news test | Shortlisted; release agreement unresolved |
| Twitter Financial News Sentiment | Optional social-media robustness test | Shortlisted; annotation provenance/counts/rights pending |
| FNSPID | News/price corpus | Excluded from human gold benchmark; generated sentiment |
| ProsusAI/finbert | Financial transformer candidate | Revision pinned; weight license scope unresolved; no weights acquired |
| News provider | Article metadata/text access | Not selected |
| Market data provider | Daily prices | Not selected |

See ROADMAP.md for candidate references and the linked comparison for evidence.
Selection, terms verification, and artifact pinning are separate assessments;
benchmark freezing does not extend the existing local research permissions.

FiQA release selection, original noncommercial-use terms, participant-copy
provenance, unscored test inputs, and mirror limitations are recorded in
[FiQA asset review](phase-01-data/acquisition/fiqa-asset-review.md). Acquisition of that copy does not
resolve the gated mirror’s broader MIT claim.

Bounded TF-IDF tuning review (2026-10-11): the C/class-weight grid uses the
same reviewed scikit-learn dependency, pinned development inputs, and local
noncommercial training/storage scope as the earlier LR/SVM fits. No assets or
rights decision changes; no corpus pooling or final-test scoring. Candidate
models and predictions remain ignored. See [tuning guide](phase-02-baselines/experiments/tfidf-tuning.md).

Phase 2 qualitative review (2026-10-11): the existing local-research scope
supports joining saved validation predictions to reviewed local inputs. Only
aggregate categories/counts and digests are published; source text, identifiers,
targets, and individual notes remain ignored. No acquisition, training, label
correction, or source-text redistribution occurs. Frozen records remain withheld.
See [qualitative review](../reports/experiments/phase2-error-analysis.md).

FinBERT adaptation decision (2026-10-11): the user's existing authorization for
local research use and explicit fine-tuning request cover SEntFiN development
training and private derivative storage. Hosted checkpoint weight-license scope
remains unresolved; this is not a redistribution clearance. No new dataset or
checkpoint is acquired: reuse the checksum-pinned cached source. Train only on
reviewed single-target SEntFiN training records; keep validation for checkpoint
selection and withhold the frozen test. Safetensors/tokenizer artifacts and
record-level predictions remain ignored. Historical asset/freeze manifests stay
unchanged. See [training guide](phase-03-transformers/models/finbert-finetuning.md).

RoBERTa/DeBERTa review (2026-10-11): the user authorized unchanged inference on
financial sentiment checkpoints and SEntFiN training of two base models. Exact
revisions, file sizes and upstream SHA-256 digests are pinned in
`configs/transformer-assets.json`. Financial RoBERTa Large's model card declares
Apache-2.0 and describes further financial pretraining/fine-tuning, without
complete supervised sample provenance; SEntFiN exposure cannot be ruled out.
Financial DeBERTa-v3 Small declares MIT and describes 4,840 financial sentences
with 5–8 annotators, matching PhraseBank; its split provenance is incomplete.
Neither model's published score establishes independent performance here.
FacebookAI/RoBERTa Base and Microsoft/DeBERTa-v3 Base declare MIT and have no
trained sentiment head. Full raw-text pretraining exposure is unknown for all
four. Reviewed model cards support private local acquisition/inference and,
for the bases, fine-tuning. Keep all downloaded/derived weights and predictions
ignored; publish aggregate metadata only. No source text or weights are cleared
for redistribution in this task. The reviewed SEntFiN research terms and frozen
split remain unchanged. SentencePiece 0.2.1 (Apache-2.0) is added only to the
optional transformer extra for the base DeBERTa tokenizer.

Sources: [Financial RoBERTa](https://huggingface.co/soleimanian/financial-roberta-large-sentiment),
[Financial DeBERTa](https://huggingface.co/mrm8488/deberta-v3-ft-financial-news-sentiment-analysis),
[RoBERTa Base](https://huggingface.co/FacebookAI/roberta-base),
[DeBERTa Base](https://huggingface.co/microsoft/deberta-v3-base),
[SentencePiece license](https://github.com/google/sentencepiece/blob/master/LICENSE).

Transformer follow-up review (2026-10-11): the user's explicit request extends
the already-reviewed private research use to SEntFiN adaptation of the financial
RoBERTa/DeBERTa sentiment checkpoints, and separate PhraseBank-only training of
the two original base checkpoints. Reuse exact source revisions and immutable
asset manifest; no new acquisition or dependency. Preserve Financial PhraseBank's
local research terms and known FinBERT supervision caveat. PhraseBank training
uses 3,785 training/518 validation records; SEntFiN adaptation retains 6,299/781.
No pooled or sequential PhraseBank-to-SEntFiN training. PhraseBank-selected base
models may score SEntFiN validation afterward as a transfer diagnostic, without
using it for epoch selection. Neither corpus's reserved tests are used. Derived
weights and individual predictions remain ignored. No redistribution clearance.

PhraseBank inference extension (2026-10-12): reuse the already-reviewed, pinned
financial RoBERTa/DeBERTa source checkpoints for unchanged inference on the 518
PhraseBank validation records under the existing private research scope. No new
acquisition, training, selection, test scoring or redistribution. Financial
DeBERTa has apparent upstream PhraseBank supervision; report this as an exposure
diagnostic. Financial RoBERTa's supervised provenance is incomplete, so no
independent-evaluation claim. Only aggregate reports are published.

Bounded transformer tuning review (2026-10-12): the user authorized a three-rate
SEntFiN comparison of original RoBERTa Base and ProsusAI FinBERT, then three-seed
reporting at each selected rate. Reuse reviewed source revisions, licenses and
existing local research permissions (including the documented unresolved FinBERT
hosted-weight scope). No new datasets/models, pooled training, PhraseBank tuning
or test scoring. Selection uses the existing 6,299/781 single-target development
split. Derivative weights and per-record predictions remain private; only
aggregate configurations, metrics and lineage enter Git.

Transformer analysis review (2026-10-12): reuse the previously reviewed private
SEntFiN development headlines and six saved selected-rate RoBERTa/FinBERT runs
for probability diagnostics and exploratory qualitative review. No acquisition,
training, probability adjustment, relabeling or reserved-test access. Existing
local-research terms and FinBERT weight-scope limitation remain unchanged.
Keep source headlines, targets, IDs, predictions and reviewer notes ignored;
publish numeric aggregates, protocol, hashes and plots only. Suspected ambiguity
is not corrected annotation or redistribution clearance.

DeBERTa Base extension review (2026-10-12): the user explicitly authorized the
same learning-rate/three-seed experiment and probability/error analysis for
original Microsoft DeBERTa-v3 Base. Reuse its already-reviewed MIT checkpoint,
pinned revision and verified local source bytes, and the unchanged single-target
SEntFiN development split. No new acquisition or change to rights scope. Five
fresh source-initialized runs use a new sentiment head; weights/predictions and
headline review notes remain ignored. No final-test scoring, relabeling or
probability adjustment; public output contains aggregates and hashes only.

Temperature-scaling review (2026-10-12): the user authorized calibration first,
then a locked final evaluation and company-targeted sentiment. This bounded task
reuses nine reviewed saved SEntFiN validation runs for RoBERTa Base, FinBERT and
DeBERTa-v3 Base under existing local-research permissions. Fit scalar temperatures
with group-disjoint validation cross-fitting; save separate full-validation
calibrators for later evaluation. No acquisition, weight training, relabeling,
reserved-test scoring or new redistribution clearance. FinBERT's previously
documented hosted-weight limitation remains. Keep fold memberships, per-record
adjusted probabilities and calibrator artifacts ignored; publish aggregates,
protocol and source hashes only. This reused validation is not an independent
model-selection assessment even when calibration fits exclude assessed rows.

Final-evaluation protocol review (2026-10-13): lock the existing single-target
SEntFiN frozen cohort, selected traditional models, unchanged FinBERT and nine
selected transformer checkpoints with their existing temperatures. Reuse the
reviewed local-research assets and previously authorized FinBERT weight scope;
no new acquisition, training, redistribution or final-test inference in this
task. Hash frozen data as opaque bytes only. Copy existing private baseline
artifacts to the authorized Windows research checkout for verification when
needed. Keep model bytes, source labels/text, validation predictions and future
test predictions ignored; publish only settings, lineage, checksums and later
aggregate metrics. Multi-target and PhraseBank reserved tests stay unscored.

Ensemble-voting review (2026-10-13): user authorized equal-weight soft and
democratic/hard voting of the existing FinBERT/RoBERTa Base/DeBERTa Base runs.
Reuse their reviewed private raw and cross-fitted calibrated validation outputs;
no acquisition, new weight training, calibration fitting, test decoding/scoring
or rights-scope change. Vote recipes and supplementary final-test comparisons
are declared before development analysis. Preserve the original primary lock;
keep per-record votes/probabilities private and publish aggregates/hashes only.

Final-runner implementation review (2026-10-13): implement the previously
authorized locked evaluation pipeline using the existing reviewed local weights
and dataset artifacts. This task verifies with synthetic and development-only
inference; do not decode or score frozen records during implementation checks.
The future explicit final run uses the same private noncommercial research scope,
including the documented FinBERT weight limitation. No acquisition, training,
calibration fit, relabeling, membership change or new redistribution permission.
Keep execution receipts, gold/IDs, predictions, bootstrap draws and weights
private; export verified aggregate metrics/protocol lineage only after a final
evaluation is actually performed. Multi-target/PhraseBank tests remain reserved.

Final execution review (2026-10-13): the user explicitly authorized the locked
766-record single-target SEntFiN final evaluation. Reuse the same reviewed
local noncommercial assets and previously authorized FinBERT weight scope,
without acquisition, training, calibration fitting or relabeling. Original
protocol/model/calibrator/data identities remain unchanged. Publish aggregate
scores, intervals, profiles, timing and lineage only; native source records,
IDs/gold, per-record predictions, bootstrap draws and model weights stay private.
The single-target benchmark is now scored/consumed for this locked comparison;
test-inspired changes need new untouched data for confirmatory claims. Multi-
target and PhraseBank reserved tests remain unscored. No redistribution expansion.

## ProPrak01 financial DeBERTa review (2026-10-14)

[Model card](https://huggingface.co/ProPrak01/deberta-v3-financial-sentiment)
at revision `f008a38828d87fbf80bea4c841333d9c493972e7` declares MIT.
`configs/proprak-deberta-asset.json` pins all six files and the upstream LFS
weight SHA-256. Local noncommercial inference and caching are within the existing
research scope; no weights, dataset text or per-record outputs are redistributed.
The declared model license does not resolve upstream PhraseBank/FiQA text rights.

This is a custom DeBERTa-v3 Base multitask model: masked token mean pooling,
dropout 0.1, a three-class head (negative/neutral/positive), and a separate
regression head. The author reports joint PhraseBank, Twitter Financial News
and FiQA training plus a continuation not included in the public notebook.
Exact upstream memberships and SEntFiN exposure are not established. PhraseBank
evaluation would have apparent upstream supervision exposure; FiQA independence
cannot be claimed. Hosted performance is not directly comparable with our splits.

First project experiment: unchanged classification-head inference on the existing
781 single-target SEntFiN validation headlines, batch 32, length 128, seed 42,
FP32. No training, calibration fitting, regression thresholds, final-test scoring
or changes to locked final-evaluation protocols. This is a post-final exploratory
experiment. Fresh untouched data would be needed for confirmatory selection claims.
The standard sequence-classification loader must not be used: reconstruct the
reviewed architecture locally, load the entire tensor state strictly with
`weights_only=True`, and execute no downloaded Python code. The published
configuration was created with Transformers 5.13.1; verify compatibility with our
locked 4.57.6 stack before reporting a successful run.

Verified outcome: strict weight loading succeeds with the locked stack. The v5
auto-tokenizer metadata is incompatible; the complete serialized pipeline loads
via `PreTrainedTokenizerFast` with existing special-token roles and explicit
inference padding/truncation, checked against its original vocabulary/probe IDs.
On 781 SEntFiN validation records: accuracy 0.7900, macro F1 0.7915; no project
training, calibration fitting or test inference. See
`reports/experiments/proprak-deberta-validation.md`. No source bytes were modified.

## ProPrak01 SEntFiN adaptation review (2026-10-14)

User-authorized follow-up: adapt the same pinned MIT-declared checkpoint on the
reviewed local noncommercial SEntFiN training view only (6,299 single-target
headlines). Select epochs on the existing 781 validation headlines. Prior model
and SEntFiN terms reviews apply; no new corpus or rights expansion. Preserve
upstream attribution and both heads, freeze the regression head, and keep derived
weights, text and per-record outputs private. No weight redistribution is cleared
by this experiment. `configs/proprak-deberta-training-plan.json` fixes the source,
seed list, matched five-epoch budget, prior selected 1e-5 rate, patience and
validation-only scope before training. No final-test inference, fitting of
temperatures or further model selection against the consumed final benchmark.
This is post-final exploratory research; new confirmatory claims need new data.

Adaptation verified: three saved/reloaded selected models retain identical
regression-head tensors and original tokenizer IDs on every training/validation
text. Encoder/classification tensors changed; the shared encoder update means
retained regression outputs no longer inherit validated upstream FiQA scores.
Mean SEntFiN validation macro F1 is 0.8815 (sample SD 0.0062), accuracy 0.8818.
Weights and per-record outputs remain private; the consumed final benchmark and
original final report are unchanged. See
`reports/experiments/proprak-deberta-adaptation.md`.

## Phase 4 company identity reference review (2026-10-14)

`configs/companies.json` manually records company-name/ticker facts for the five
existing project companies, with issuer source URLs and a review date. We store
no investor-page prose, market prices, news articles or new third-party alias
dataset. Company names and ticker identities are factual reference metadata;
this adds no dataset acquisition or expansion of the research text/weight scope.
The curated alias policies and explicit Google-to-Alphabet grouping option are
documented in `docs/phase-04-target-sentiment/data/company-targets.md`; this is not a licensed security master.
No frozen coverage rules, dataset bytes, labels, predictions or models change.
Future provider data or additional corpora still require their own terms review.

## Phase 4 development coverage audit review (2026-10-14)

Reuse the previously reviewed local noncommercial SEntFiN development view only.
Audit its provided targets/labels against `companies-v1`; publish aggregate
counts and configuration/data/code identities. No dataset acquisition, provider
API, model, relabeling or redistribution expansion. Keep source IDs, target
strings and annotation-level review flags private in ignored `data/interim/`.
A separate explicitly labeled optional parent-rollup view preserves Google
versus Alphabet distinctions. Literal name screening is diagnostic and does not
establish entity truth or create sentiment labels. Freeze checks may hash
reserved bytes for integrity; only returned train/validation records enter
analysis. No reserved-test annotation/text analysis, training or inference.

Coverage audit verified: 9,651 development headlines, 12,957 source annotations;
95 nonliteral-target diagnostics retained without label/eligibility changes.
Publish only `reports/company-coverage.md`/JSON and aggregate interpretation in
`docs/phase-04-target-sentiment/data/company-coverage.md`. Source IDs/targets/labels and the manual identity review
remain ignored/private. The five-company matches are too sparse for credible
per-company validation; no test collection, corpus acquisition or redistribution
was added. Frozen receipts/data and final evaluation aggregates are unchanged.

## Phase 4 target-pair preparation review (2026-10-14)

User-authorized input/protocol preparation reuses the reviewed noncommercial
SEntFiN development view and locally cached MIT-declared Microsoft DeBERTa-v3
Base tokenizer. No acquisition or weight loading is needed. Keep derivative
target/headline/label JSONL and row membership private under ignored storage;
publish counts, hashes and aggregate token-length diagnostics only. Preserve
native entities, review flags and fixed train/validation groups, including
nonliteral targets and development-only training rows. No masking, alias
substitution, label adjudication, five-company restriction or row exclusion.
Pin the comparison to original Base initialization and matched three-seed
training budgets before any new training. Reserved test records are not analyzed
or scored; integrity-only freeze checks remain. This task adds no training,
inference, independent-test claims, model redistribution or new collection.

Verified target preparation retains all 12,957 provided development annotations.
All original group boundaries and labels remain; 95 nonliteral targets and 17
examples from development-only training headlines are not silently removed.
The private view is checksum-identical across reconstructed Mac/Windows inputs,
and all pair target tokens are preserved (no truncation or target UNKs). Only
aggregate token/count diagnostics, configuration and readiness lock are public.
No weights were loaded, no models trained/scored, and no reserved test annotations
were analyzed. Native target strings are financial-entity supervision, not a
verified legal-company taxonomy; five-company coverage limitations still apply.

## Phase 4 saved headline-control inference review (2026-10-14)

The authorized validation run reuses the three existing privately stored,
MIT-source DeBERTa Base adaptations and the reviewed noncommercial SEntFiN
native-target development view. No new acquisition, training, calibration fitting
or model redistribution is needed. Verify full saved artifact file hashes before
loading local safe weights. Reconstruct source development membership and compare
both prepared JSONL files before inference. Keep target-level predictions,
source IDs, headline text and native entity names private; publish only aggregate
cohort metrics, runtime/architecture details and provenance hashes. Existing
single-target validation prediction bytes/gold/membership are validated before
checking the reloaded models. The original final aggregate must remain unchanged;
reserved test annotations are not analyzed or scored. These reused development
validation results cannot establish independent test performance.

Saved-control inference completed for seeds 42/43/44 on validation only. All
1,410 targets were retained, with no adjudication or cohort-specific prediction
selection. Mac reconstructs the exact private prediction hashes and cohort
metrics using already-held source data plus exported IDs/probabilities; no new
text/target acquisition was necessary. Public output contains aggregate metrics
and architecture/provenance only. No training, temperature fitting, test scoring,
model redistribution or change to frozen source/Phase 3 final artifacts occurred.

## Phase 4 target TF-IDF training review (2026-10-14)

The user authorized matched headline-only and additive headline/native-target
TF-IDF Logistic Regression training. Reuse only the already-reviewed SEntFiN
noncommercial development view. There is no new dataset/model acquisition or
external service. Both arms fit the same original train annotations, without
PhraseBank combination, filtering, masking, relabeling or new partitions. Keep
fitted vocabularies, coefficients and target-level predictions in ignored local
storage: derivative artifacts may encode restricted source text/entity names.
Public artifacts contain settings, hashes, aggregate metrics and model sizes
only. No fitted weights/vocabulary redistribution is authorized. Validate the
existing preparation identities and unchanged original final aggregate before
publishing. No reserved-test use, calibration fitting or hyperparameter search.

The fixed linear comparison is complete: each arm uses all 11,547 training
annotations and evaluates 1,410 validation annotations. Both saved models exactly
reproduce probabilities after reload; headline vocabulary/IDF match between arms.
Independent all-cohort metric checks pass. The additive target block helps mixed-
label examples but does not improve overall macro F1. This is exploratory evidence
of entity label tendencies, not verified contextual target understanding or five-
company performance. No source/test membership, label, final result or original
transformer plan/lock was changed. Fitted artifacts and row results remain private.

## Phase 4 matched DeBERTa training review (2026-10-15)

The user authorized fresh matched headline-only and target-pair training under
the already-fixed native-target plan. Reuse the reviewed noncommercial SEntFiN
development view and the locally cached MIT-declared original Microsoft
DeBERTa-v3 Base checkpoint at revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3.
Verify every cached source file before local-only weight loading; no acquisition,
hub call, previous fine-tuned initialization or remote model code is needed.
Use all existing training annotations, seeds 42/43/44 and the fixed optimization
budget; validate only on the existing development validation view. Bind saved
artifacts to input mode, preparation/data identities and source ancestry. Keep
fitted weights, source rows and target predictions in ignored private storage;
publish aggregate metrics, model sizes and provenance only. No test scoring,
label repairs, class weighting, hyperparameter grid, calibration fitting or
weight redistribution is authorized by this training task.

The matched six-run training comparison completed on the authorized Windows GPU.
Every trial used all original 11,547 training annotations and the same 1,410
validation annotations, with matched starts and epoch row orders. Six selected
checkpoints and intermediate improved checkpoints remain in ignored private
storage. Aggregate/public trial reports contain metrics, hashes and architecture,
not native examples. Mac reconstructs exact private prediction bytes using its
already-held view plus IDs/probabilities; full weights stay on Windows. All 36
cohort metric sets independently verify. Reported gains are exploratory validation
results, not independent test or five-company generalization evidence. Probability
ECE did not improve; no calibration was fitted. Original source/freeze/final/plan/
lock bytes remain unchanged, and no reserved test was analyzed or scored.

Target-backbone comparison decision (2026-10-15): the user requested matched
RoBERTa Base and FinBERT comparisons before moving on. Reuse the already reviewed,
pinned local sources and native SEntFiN train/validation target view under the
existing noncommercial research scope. FinBERT's documented user authorization
covers private adaptation and derivative storage despite unresolved hosted-weight
license scope; it does not grant public redistribution. No new source acquisition,
text publication, weight redistribution or test inference is authorized by this
experiment. `configs/target-backbones-plan.json` records original source file
identities, retained FinBERT head semantics, fresh RoBERTa head, prior selected
learning rates and the unchanged original DeBERTa/data lineage. Known FinBERT
PhraseBank supervision and unknown checkpoint pretraining exposure remain caveats.

## Phase 4 target-error review (2026-10-17)

The user authorized qualitative analysis of the saved DeBERTa Base, RoBERTa Base
and FinBERT target-conditioning comparisons. Reuse only the already reviewed
native SEntFiN development validation view and its 18 saved prediction receipts.
No acquisition, weight loading, new inference, training, threshold/temperature
fitting or reserved-test analysis is needed. Verify receipt hashes, native target
membership/gold, saved cohort metrics, source context and artifact manifests.
Keep native headlines, target strings, source IDs, row predictions and reviewer
notes in ignored `data/interim/target-error-review`. Publish only aggregate counts,
methods, hashes and paraphrased findings. Multi-label review categories and
suspected gold ambiguity are exploratory Codex judgments, not independent human
adjudication or permission to modify the original labels or frozen benchmark.
Existing noncommercial research and FinBERT weight-scope decisions are unchanged.

## Educational training notebook (2026-10-17)

`notebooks/02_simple_transformer_training.ipynb` reuses the existing reviewed
local noncommercial SEntFiN single-target development partitions and already
reviewed pinned RoBERTa Base / DeBERTa Base checkpoints (or the previously
authorized FinBERT scope). No new acquisition, redistribution permission,
final-test evaluation, or independent-generalization claim is introduced.
The notebook verifies local checkpoint hashes and loads only verified development
records. Source texts, outputs and trained weights remain private; artifacts go
under ignored `data/interim/notebook-training/`. This teaching run does not alter
locked experiments or their reports. Existing source/model limitations apply.

## Target evaluation runner verification (2026-10-18)

The runner's development smoke reuses the existing private native SEntFiN validation
view and hash-verified saved transformer predictions. Both saved sparse baselines
are loaded from trusted checksummed local pickles and predicted on development
inputs only. This remains within the reviewed local noncommercial research scope;
no new corpus/model acquisition, training, public text display or weight/prediction
redistribution occurs. Published verification contains aggregate counts/digests only.
Reserved scoring remains a separate receipt-bound action; original terms and
pretraining-exposure limitations remain in force.

## Local RoBERTa target inference (2026-10-18)

The user authorized a reusable interface using the existing locally trained
RoBERTa Base target-pair checkpoint. Reuse the reviewed MIT base-model and
SEntFiN local-research derivative-storage scope above. Select seed 43 from saved
validation metadata only; no weights or dataset are acquired, relabeled, retrained
or redistributed. Smoke inputs are invented or existing development-validation
records. Weights, per-record outputs and runtime receipts stay ignored. This task
adds local Python/CLI inference, not public serving or a new redistribution grant.
See the [prediction guide](phase-04-target-sentiment/prediction.md).

## Registry mention baseline (2026-10-19)

Reuse the existing reviewed five-company names/tickers/aliases and their existing
identity policies. Matching and integration tests use invented headlines only;
no new data source, provider, tokenizer, model or dependency is acquired. The
optional local RoBERTa smoke uses invented inputs and the already approved private
checkpoint. No frozen record is read/scored and no labels, weights or existing
benchmark results change. Publish implementation/rules and synthetic verification,
not private news examples. This remains the existing local-research scope; no
new public serving or redistribution permission is implied. See the
[mention baseline guide](phase-04-target-sentiment/company-mentions.md).

## SEntFiN development mention audit (2026-10-19)

Reuse the existing hash-verified native train/validation target pairs under the
reviewed local-research terms. No acquisition, training, new labels or public
source-text redistribution occurs. Full matched headlines, native annotations
and assistant review notes remain in ignored local data; the public report
contains aggregate coverage counts and descriptions of issues only. Frozen
records are not opened. Native targets are not exhaustive company mention gold;
this task grants no new redistribution or deployment rights. See the
[development detection review](../reports/phase-04-target-sentiment/experiments/company-mention-coverage.md).

## Native target-format span preparation (2026-10-19)

The user authorized investigating target markers/replacement before aggregation,
with company detection deferred. Reuse only the existing locked SEntFiN native
train/validation pairs under the reviewed local-research scope. Dry-run literal
span transformations and save eligible IDs/full review text in ignored private
storage; publish aggregate counts, policies and digests only. No new corpus,
tokenizer or weights are acquired, no native labels are repaired, and no frozen
record is opened. No training or public derivative-text redistribution occurs in
this preparation task. Existing limitations and redistribution restrictions apply.
See the [span audit](../reports/phase-04-target-sentiment/evaluation/target-span-audit.md).

## RoBERTa target-format tokenizer preparation (2026-10-20)

Reuse the reviewed MIT-declared `FacebookAI/roberta-base` snapshot at
`e2da8e2f811d1448a5b465c236feacd80ffbac7b` and the existing local-research SEntFiN
development views. Copy only six existing tokenizer/config files from the user's
authorized PC cache to ignored Mac storage; verify original asset hashes and load
locally with remote code disabled. No new model/corpus is acquired and no weights
are read. Literal marker/replacement views and native labels remain in ignored,
content-addressed local storage. Public outputs contain aggregate tokenizer
counts, the nine-run protocol and hashes only. No frozen record, training,
inference or public derivative-text/weight redistribution occurs. Existing data
limitations and redistribution restrictions remain in force. See the
[tokenizer audit and lock](../reports/phase-04-target-sentiment/evaluation/target-tokenizer-audit.md).

## Detached RoBERTa target-format training (2026-10-20)

The user authorized the locked nine-trial comparison as a detached job on the
authorized RTX 4070 SUPER PC. Reuse the original pinned MIT RoBERTa Base snapshot
and the existing SEntFiN local-research derivative-storage scope. Transfer only
the public source snapshot and the already-prepared private target-format views
to that same research computer; verify hashes before reuse. No new corpus/model
acquisition or frozen-test access is authorized by this training task. Native
labels remain unchanged. Saved checkpoints, row predictions and execution logs
stay ignored/private; public exports contain aggregate metrics/configuration only.
Company detection and production model selection remain deferred.

## Target-format result verification and review (2026-10-20)

Reuse only the completed nine-run study's local artifacts and native development
prediction caches under the same reviewed research scope. Verify checkpoint and
cache hashes and recompute existing metrics on the PC without model inference,
training or frozen-record reads. Retrieve aggregate reports and preserve the exact
public execution source in private Git bundles on both machines; weights and row
predictions remain private. Publish only aggregate metrics, source/configuration
provenance and verification digests. No raw text, source example IDs or weights
are added to public reports, and no new redistribution rights are implied. See
the [reviewed comparison](../reports/phase-04-target-sentiment/experiments/target-format-comparison.md).

## Synthetic company-name sensitivity diagnostic (2026-10-20)

The user authorized a development-only name-substitution check with saved models.
Reuse the reviewed local SEntFiN derivative-storage and MIT RoBERTa checkpoint
scope. A fixed native-company-target allowlist selects 67 existing validation
annotations; two fixed name strings create private synthetic views. No new corpus,
entity detector, tokenizer or model is acquired or trained. Counterfactual labels
are not created or assumed; measure prediction/probability changes only. Native
headlines, synthetic text and per-record outputs stay ignored on the user's two
research machines. Public reports contain aggregate counts/diagnostics and hashes,
with no source examples or weights. No frozen-record access or new public-serving
or redistribution grant is implied. See the
[diagnostic protocol](phase-04-target-sentiment/analysis/target-name-stress.md).

## Pretrained organization detector review (2026-10-20)

For the bounded Phase 4 comparison, use `dslim/bert-base-NER` at immutable revision
`d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`. The publisher's
[model card](https://huggingface.co/dslim/bert-base-NER/blob/d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc/README.md)
declares MIT licensing and identifies BERT-base-cased fine-tuned on English
CoNLL-2003 Reuters news. Local acquisition, storage and research inference of the
published checkpoint are supported under that declaration. Keep weights/card
under ignored data; no weight or source-news redistribution is performed.
This does not grant rights to acquire or republish Reuters/CoNLL training text;
that corpus is not downloaded. The repository card declares the model license;
an independent provenance/rights audit of upstream training text is unavailable.

`configs/organization-ner-plan.json` pins all acquired file SHA-256 values and
sizes before weight acquisition (LFS identity for weights; small blobs checked
against repository Git identities). No remote code or pickle weights are used.
Reuse the already reviewed local SEntFiN validation derivative only; no new data
acquisition, training, frozen-test access or hosted inference is involved.
Headlines and per-row detections remain private; publish aggregates and review
findings only. Native sentiment targets are incomplete financial-entity annotations,
not exhaustive company/organization mention gold. Model-card CoNLL accuracy is not
FinSight accuracy. `ORG` predictions include institutions and other non-company
organizations; they do not establish ticker identity or authorize parent rollup.

## Organization mention development annotations (2026-10-20)

Reuse only the previously reviewed local SEntFiN validation derivative and pinned
MIT-declared BERT NER files. No new dataset/model acquisition, training, external
annotation service, or frozen-test access is needed. The user authorized a small
fully annotated development collection and grouping comparison. A prediction-free
hash sample fixes 50 validation headlines. Source text, full organization-span
drafts, reviewer notes and per-record detections remain under ignored local data;
public artifacts contain only policy, code, aggregate metrics and checksums.
The existing local derivative-storage scope supports this research annotation
work; it does not clear public news-text redistribution.

Assistant drafts attempt exhaustive company/other-organization spans independently
of displayed native sentiment targets and current sampled model outputs. They
remain provisional, with pending human review and explicit uncertain types.
Earlier NER inference already covered the entire validation set; two selected
headlines overlap the prior assistant review. This is development analysis, not
an unseen, independent or human-gold test collection. Review and source provenance
must remain explicit. Do not replace SEntFiN sentiment labels or the frozen test.
See the [annotation policy](phase-04-target-sentiment/data/organization-annotation-policy.md).

## Shortlist: company-linked news and market outcomes (2026-10-20)

The user requested shortlisting only for
[News Sentiment Analysis for Stock Data by Company](https://www.kaggle.com/datasets/sidarcidiacono/news-sentiment-analysis-for-stock-data-by-company)
by Sid Arcidiacono. Status: **candidate for a later Phase 6 provenance/timing audit;
not acquired, approved for use, or selected**. Only the public dataset card was
reviewed; actual file schema, dates, rows and five-company support are unverified.

The publisher describes r/WorldNews headlines associated with US stock symbols
and Yahoo Finance market data. Its labels encode stock direction by market close
on the publication day: 0 down, 1 up, 2 unchanged. These are market-outcome labels,
not human financial-sentiment annotations, so do not use them as sentiment or
company-mention gold. The card does not establish the exact price baseline.

Before acquisition/use, verify company-linking provenance, source rights, precise
publication/availability timestamps, target construction, company/date coverage,
duplicate story groups and chronological evaluation eligibility. Same-day labels
need a cutoff audit to avoid including news unavailable before the predicted move.
The card lists "Data files © Original Authors"; local storage, derivative use and
redistribution permissions remain unresolved. No download, experiment, integration
or Phase 5/6 implementation is authorized by this shortlist entry.

## Reviewed mention reference and word-aware NER verification (2026-10-20)

The user accepted the assistant-drafted 50-headline development annotations and
authorized finalization after sourced adjudication of UTI, Fin Tech and National
Buildings. Record the exact review scope: rows 1–10 were individually approved;
remaining rows were accepted in bulk and finalized after the follow-up. This is
an assistant-drafted, user-accepted development reference, not independent double
annotation or an unseen test set. Keep original drafts, discussion/evidence notes,
approved reference snapshots and per-record predictions private. Public results
contain aggregates and hashes only. Type uncertainty is resolved without changing
written source spans or native SEntFiN sentiment labels.

The same reviewed BERT checkpoint is reused with word-aware `first` aggregation;
no training, new corpus, remote annotation service or frozen-record access is
involved. Existing local-use scope also covers a copy of the pinned seven-file
NER checkpoint on the user's GPU PC, under ignored project data, for invented-text
end-to-end verification with the existing RoBERTa artifact. No weights or news
text are redistributed publicly; no new upstream rights are inferred. Preserve
historical `simple` experiments and their source snapshots separately from the
new experimental default.

## Conservative organization resolution (2026-10-21)

Reuse the accepted local 50-headline development reference and checksummed saved
word-aware predictions. The task adds handwritten whole-title institution rules
and exact resolution through the unchanged five-company registry; it acquires no
company directory, entity-linking dataset, news, model or external service. No
new model is trained, and native sentiment labels/reference spans are unchanged.
The resolver never receives gold types, sentiment predictions or model confidence
as identity evidence. Unknown names/acronyms stay unresolved rather than being
labeled non-company or added to the registry from the benchmark.

Full error contexts and per-span resolution decisions remain under ignored local
data. Public artifacts contain only aggregate counts, implementation policy,
provenance and invented-input smoke results. Existing local-use terms cover cache
analysis and invented-text inference with the already verified GPU checkpoints.
No frozen records are accessed or additional redistribution rights inferred.

## US-listed issuer identity source review (2026-10-21)

The user authorized broader US-listed identity coverage within Phase 4. Select the
SEC's official `company_tickers_exchange.json` association file:
https://www.sec.gov/files/company_tickers_exchange.json . The SEC's
[access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
identifies it as conformed company-name/CIK/ticker/exchange associations, permits
public downloads, asks clients to declare a User-Agent and limits automated
requests (currently ten/second). Use one bulk request with a truthful FinSight
identity and the user-supplied contact address, configured locally only. Never
commit the address or copy it into snapshots/reports; do not impersonate browsers,
crawl filings or use retry loops to evade access controls.

The SEC's [website dissemination policy](https://www.sec.gov/about/privacy-information#website)
states its website information may be copied and further distributed without
permission, with SEC attribution requested. This supports local acquisition,
storage and derived identity indexes of this factual association file. Do not
use SEC logos or imply endorsement; this decision does not grant rights to
third-party news or treat company trademarks as licensed branding. Original bytes
and source/retrieval hashes will remain in ignored data; public outputs use
aggregate counts and provenance.

Scope: Nasdaq, NYSE, CBOE and TXSE association rows, excluding OTC/null/other exchange
rows. US-listed does not mean US-domiciled: foreign issuers/ADRs may remain. CIK is
the issuer identity; preserve all listings and do not invent a primary share class.
The file is not a guaranteed complete/accurate universe or a security-type master.
Obvious ETF/fund/trust names require separate type review rather than automatic
company verification. Remaining matches establish an issuer association, not
verified operating-company status. Current snapshots are not historical listing
membership and must not be applied as point-in-time financial features. No news,
price ingestion, stock prediction or new model training belongs to this task.

Acquired one bulk snapshot on 2026-10-21 (523,851 bytes; SHA-256
`6c29e2e3db973d4479875164603c2434f81ad22e14c9f1694999aca9cca246cb`).
The corrected exchange policy retains 6,071 CIK issuers and 7,699 listing
associations; 531 issuers have a conservative entity-type hold. Source bytes and
contact configuration stay ignored locally. The public pin is
`configs/us-listed-directory.json`; see the
[coverage and verification report](../reports/phase-04-target-sentiment/experiments/us-listed-directory.md).

## US-news mention/linking development review source (2026-10-21)

For the user-authorized small US-company news review, use English Wikinews
headline titles with attribution, source-category URLs, retrieval time and saved
response checksums. The [official copyright policy](https://en.wikinews.org/wiki/Wikinews:Copyright)
permits text reuse under CC BY 4.0 for material published after December 16, 2024,
CC BY 2.5 for September 25, 2005–December 15, 2024, and public domain earlier.
Retain both applicable license links and Wikinews attribution when publication
version is not established; do not claim all headlines use the newest license.
Only headline titles/category metadata are acquired, not articles, third-party
quotations, images or linked publisher material. No Kaggle acquisition is implied.

Select a small, deterministic, category-balanced corporate-news convenience sample
without inspecting NER/resolver outputs. Company categories intentionally enrich
US-company mentions; category membership is not a gold mention or CIK label.
Historical headlines are allowed for extraction/linking diagnostics, but today's
SEC directory cannot establish historical listings or security types. Unknown
model pretraining exposure, older text, search/category selection, and limited
sectors prevent representative or independent-test claims. Human review remains
required: assistant drafts are not approved reference labels. Keep raw text and
predictions in ignored local data and publish aggregate/provenance outputs only.
This is bounded Phase 4 development work, with no sentiment training, model tuning,
price features, frozen sentiment records or historical investment simulation.

## US-company review acceptance (2026-10-21)

The user bulk-approved all 20 assistant-drafted Wikinews references. Record this
as user-accepted development data, not independent human annotation. Original
drafts and source/selection bytes are preserved. The current-identity policy
finalizes Chrysler/Saturn/Yahoo as company mentions without accepted standalone
current directory links, retaining null CIKs without inferring historical absence
or rolling up to parent issuers. Dell/GM refer to current identities only;
Microsoft Research Asia remains an organizational unit without a company link.
The user's bulk approval is not evidence of individual historical identity checks.
Per-record drafts, reviewed references, acceptance metadata and predictions stay
private; public metrics/provenance and limitations are in the
[reviewed report](../reports/phase-04-target-sentiment/evaluation/us-company-review-reviewed.md).

## Company-type/identity routing separation (2026-10-21)

The user authorized separating company-mention sentiment from issuer linking.
This bounded Phase 4 change uses already reviewed local NER/SEC sources and
caller-supplied exact company-name confirmations. No new source, alias dictionary,
training data, checkpoint acquisition or redistribution is introduced. The
original deployment/input encoding remains unchanged; target masking is not an
identity source. Invented-text CPU NER plus stub sentiment-routing verification
uses the existing local research scope. Prior approved references and frozen
records remain unchanged. See the
[routing report](../reports/phase-04-target-sentiment/experiments/company-mention-routing.md).

## GPU routing verification scope (2026-10-21)

The user explicitly authorized updating their GPU PC and verifying the complete
company pipeline with real sentiment inference. Reuse the already reviewed NER
and RoBERTa checkpoint files on that user-controlled machine; no new model
acquisition or public weight redistribution is involved. Transfer only an
allowlisted source/evidence bundle and the factual SEC snapshot under its
recorded source-use decision. No contact configuration or `.env` is copied.
Use invented headlines only for CUDA extraction, resolution, sentiment and direct
prediction equivalence; do not access news labels or frozen sentiment records.
No training, calibration fitting or benchmark update is authorized by this check.
[GPU report](../reports/phase-04-target-sentiment/verification/company-mention-routing-gpu.md).

## Reviewed issuer aliases and fresh development sample (2026-10-21)

User-authorized Phase 4 scope: improve Dell/Boeing/Toyota name handling and review
on fresh headlines without changing earlier references/results. Verify aliases
from factual corporate identity statements, keeping links/attribution and brief
notes rather than copying web pages, logos or substantial text:

- Boeing's [official company site](https://www.boeing.com/) identifies Boeing as
  the aerospace company. The pinned SEC source lists BOEING CO, CIK 0000012927.
- Toyota's [official 2026 statement](https://global.toyota/en/album/images/43839258/)
  explicitly uses Toyota as shorthand for Toyota Motor Corporation. Its pinned
  SEC CIK is 0001094517. Other Toyota group companies must not be shortened to it.
- Dell's [2026 issuer-hosted filing](https://investors.delltechnologies.com/sec-filings/all-sec-filings/content/0001193125-26-271266/d159213d424b2.htm)
  defines Dell as Dell Inc., an indirect wholly owned subsidiary of Dell
  Technologies. CIK 0001571996 is the listed parent, not proof that Dell Inc. and
  Dell Technologies are the same issuer. Require explicit parent roll-up for
  that mapping; company-type evidence can still be retained without a CIK link.

These are curated factual identity relationships, not a license to redistribute
corporate content or proof of historical corporate continuity. Archive only
configuration/evidence URLs and paraphrased decisions. Existing SEC association
rights/local storage decisions apply; no new SEC request or contact transfer.

Reuse the retained, previously permitted Wikinews category responses and make
up to two bounded category API requests (Dell/Toyota) under the existing text
license/attribution decision. Freeze a deterministic disjoint sample before NER,
excluding all prior page IDs and normalized headline texts. Category choice is
explicitly targeted to known linking gaps and controls, so this is development
coverage evidence, not an unbiased or independent benchmark. Source dates are
not established; current listings cannot validate historical membership. New
assistant reference drafts require fresh user review; earlier bulk approval does
not extend to them. Keep raw text/predictions private and frozen sentiment data
untouched. No new model training, weight acquisition or Phase 5 implementation.

Fresh alias-review packet selected from the allowed category snapshots:
`ad6bb8ddb3553db5412f480a27f0839a3d8254a3df684274a1dbd0e4fd04794b`.
Twenty headlines have zero prior ID/normalized-title overlap, without asserting
event/pretraining independence. New assistant drafts remain pending; no old bulk
approval was reused. Original current-identity references/results are preserved,
including their historical Dell interpretation; the new no-parent-roll-up policy
is explicit and does not rewrite them. See the
[comparison report](../reports/phase-04-target-sentiment/evaluation/issuer-alias-review.md).

## Fresh alias-reference bulk acceptance (2026-10-22)

The user approved the fresh 20-headline packet in response to the explicit
review request. Record bulk acceptance of assistant-authored labels (including
eight flagged and two empty rows), not independent or individual annotation.
Keep original draft bytes; update only review metadata/uncertainty flags in a
new reviewed file. Company spans, types, CIK values, link statuses and source
notes remain unchanged. The no-implicit-parent-roll-up policy is preserved; Dell
remains a company without an assigned parent CIK. Score only the fixed primary
three resolver arms against that policy. Parent roll-up remains a coverage
diagnostic with no accuracy score under mismatched reference identities.
Actual CPU NER is rerun with unchanged weights and outputs, without sentiment
training/inference, source acquisition or frozen sentiment access. See the
[reviewed results](../reports/phase-04-target-sentiment/evaluation/issuer-alias-review-reviewed.md).

## Reviewed-alias GPU release verification (2026-10-22)

The user explicitly authorized installing the reviewed aliases on the GPU PC and
verifying real CUDA inference. Reuse the existing reviewed NER/RoBERTa weights and
SEC snapshot with no acquisition, training or public weight redistribution.
Transfer only allowlisted source/configuration/public deployment evidence and an
invented-input smoke script; exclude all news/reference records, `.env`, contact
configuration and credentials. Verify alias/SEC/model/source hashes. Exercise
Toyota/Boeing automatic linking under explicit company context, Dell company-only
sentiment under the default policy, and explicit Dell parent roll-up. Check direct
prediction equivalence and uncertain/context-missing load skipping. Preserve the
previous source release and launcher bytes; switch launchers only after passing.
No source terms, frozen benchmarks, reference decisions or Phase 5 scope change.
