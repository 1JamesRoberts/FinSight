# Dataset, model, and API review checklist

PhraseBank, SEntFiN, and an original-format FiQA participant copy have been acquired for reviewed local noncommercial
research; all source files remain in ignored storage. Local TF-IDF models are
stored only in ignored directories; no pretrained weights or provider integration
are present. See [dataset-loading.md](dataset-loading.md) for checksums,
loading instructions, and measured integrity results.
Phase 1's public-source candidate comparison is recorded in
[dataset-comparison.md](dataset-comparison.md), reviewed on 2026-10-10.
An independent-evaluation shortlist is recorded in
[evaluation-dataset-shortlist.md](evaluation-dataset-shortlist.md).
SEntFiN's scoped local-research assessment and immutable revision are recorded in
[sentfin-asset-review.md](sentfin-asset-review.md) and
[asset-revisions.json](../configs/asset-revisions.json).
The subsequent [evaluation contract](evaluation-contract.md) records the curated
SEntFiN freeze for local offline research; membership and gold digests are pinned
in `configs/benchmark-freeze.json`. Known accessible-corpus overlap is excluded;
full checkpoint pretraining exposure remains unknown.
Completed cards summarize the selected artifacts: [PhraseBank](dataset-cards/phrasebank.md),
[SEntFiN](dataset-cards/sentfin.md), and [FiQA copy](dataset-cards/fiqa.md).
Development-only figures and completion evidence are in
[Phase 1 findings](phase1-findings.md).
**Other asset approvals and public-text rights remain pending.** This checklist
is not a legal conclusion or authorization to acquire or redistribute an asset.

VADER package/lexicon review (2026-10-11): standalone `vaderSentiment==3.3.2`
is approved for local inference/research under its MIT grant, with attribution
and notices retained. See [VADER asset decision and baseline](vader.md). Packaged
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
decision changes. See [comparison and input-flow guide](phrasebank-class-weights.md).

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
[FiQA asset review](fiqa-asset-review.md). Acquisition of that copy does not
resolve the gated mirror’s broader MIT claim.

Bounded TF-IDF tuning review (2026-10-11): the C/class-weight grid uses the
same reviewed scikit-learn dependency, pinned development inputs, and local
noncommercial training/storage scope as the earlier LR/SVM fits. No assets or
rights decision changes; no corpus pooling or final-test scoring. Candidate
models and predictions remain ignored. See [tuning guide](tfidf-tuning.md).

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
unchanged. See [training guide](finbert-finetuning.md).

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
documented in `docs/company-targets.md`; this is not a licensed security master.
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
`docs/company-coverage.md`. Source IDs/targets/labels and the manual identity review
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
