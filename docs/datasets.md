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
