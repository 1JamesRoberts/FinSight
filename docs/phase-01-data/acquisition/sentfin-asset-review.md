# SEntFiN release review and FinBERT provenance

Reviewed: 2026-10-10. Reviewer: coding agent; evidence-based project scope decision.
Scope: SEntFiN terms/revision and candidate FinBERT checkpoint provenance.
This record describes the metadata-only review. The subsequent
[loading task](dataset-loading.md) acquired and validated the reviewed CSVs;
model weights remain unacquired. At the original review, no dataset files or
model weights were acquired. No training, inference, duplicate
analysis, or test-set freezing took place. Read-only metadata access was sufficient
for this bounded task. See [asset-revisions.json](../../../configs/asset-revisions.json).

## Immutable release identities

| Asset | Verified revision | Verification source |
| --- | --- | --- |
| SEntFiN 1.0 release | `eba43610fe3cb57a3e5773cd6c037da1f8992ad5` | GitHub commit API for `pyRis/SEntFiN` |
| ProsusAI/finbert model repository | `4556d13015211d73dccd3fdd39d39232506f3e43` | Hugging Face model metadata API |
| Separate FinBERT source code | `44995e0c5870c4ab37a189d756550654ae87cdf0` | GitHub commit API for `ProsusAI/finBERT` |

The SEntFiN [pinned release tree](https://github.com/pyRis/SEntFiN/tree/eba43610fe3cb57a3e5773cd6c037da1f8992ad5)
contains `SEntFiN.csv` (1,118,162 bytes) and `entity_list_comprehensive.csv`
(90,424 bytes). The manifest records their Git blob identities from the tree API.
Those identify repository file contents; they are not SHA-256 download checksums.
CSV SHA-256 values remain null until acquisition. SHA-256 hashes of reviewed
README/license/configuration metadata are recorded separately.

Acquisition in a later task must use these revisions, verify sizes and Git blob
identities, calculate SHA-256 over actual bytes, and store artifacts under ignored
local paths. Never replace pinned identifiers silently when a branch changes.
The manifest is documentary metadata; no acquisition pipeline is implemented.

## SEntFiN usage assessment

The [pinned repository license](https://github.com/pyRis/SEntFiN/blob/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/LICENSE)
is MIT, credited to Ankur Sinha, Satishwar Kedas, Rishu Kumar, and Pekka Malo (2021).
It permits use, modification, and distribution subject to retaining its notice,
and disclaims warranties. Its wording refers to software and documentation.
The [pinned README](https://github.com/pyRis/SEntFiN/blob/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/README.md)
identifies the repository as the dataset/code release and provides no separate
CSV restriction or rights explanation.

**Project decision:** local noncommercial research on the released CSVs is
supported by the repository-wide MIT grant. This is an interpretation of the
release context, not an explicit publisher permission statement. Record attribution
and retain the exact license alongside locally acquired files. This supports a
later loading/validation task; it does not certify all rights in underlying news.

| Intended use | Assessment |
| --- | --- |
| Local acquisition, validation, and offline evaluation | Supported under the repository-license interpretation |
| Local training on a separately designated development partition | Supported under that interpretation; never train on reserved evaluation data |
| Public aggregate metrics and original research documentation | Planned portfolio output; avoid embedding source headlines |
| Committing CSVs, sample headline text, or exposing headlines in a public demo | Not cleared by this review; publisher/source rights need separate verification |
| Publishing trained weights or a commercial service | Separate asset/output review required; no automatic conclusion about weight rights |

The [paper, section 3.1](https://arxiv.org/html/2305.12257) attributes the annotated
headlines to Economic Times coverage from 2011-2015. That establishes provenance,
not the publisher's grant to downstream users. The article's own license should
not be substituted for the CSV license. No provider account or API terms are
involved in this repository-metadata task; do not use it to authorize scraping.
No retention deadline is specified in the reviewed release documents.

## FinBERT: pinning is distinct from permission

The [pinned model card](https://huggingface.co/ProsusAI/finbert/blob/4556d13015211d73dccd3fdd39d39232506f3e43/README.md)
confirms financial-domain adaptation and PhraseBank sentiment fine-tuning.
It has no explicit license declaration. Model metadata and the file listing also
do not supply a weight-license file. The
[pinned code license](https://github.com/ProsusAI/finBERT/blob/44995e0c5870c4ab37a189d756550654ae87cdf0/LICENSE)
is Apache-2.0; it is evidence for source-code terms, not proof that the separate
hosted model weights have the same grant. Weight acquisition remains pending
an explicit applicable license or clarification of its scope. No author was contacted.

The [pinned model configuration](https://huggingface.co/ProsusAI/finbert/blob/4556d13015211d73dccd3fdd39d39232506f3e43/config.json)
uses 0=positive, 1=negative, 2=neutral. Keep this mapping separate from dataset
integer encodings; matching integer values need not represent matching classes.
Reading this metadata does not implement a model.

## Preliminary overlap assessment

- **PhraseBank supervised exposure:** confirmed by the model card; a new random
  partition cannot establish that its examples were unseen by this checkpoint.
- **FiQA exposure:** the [original FinBERT paper, section 4.2.3](https://arxiv.org/html/1908.10063)
  reports regression cross-validation. This is model-family experiment history;
  it does not identify the published classification checkpoint's exact samples.
- **SEntFiN inclusion:** not established in the reviewed training descriptions.
  Its release history alone cannot prove independence, because the news is older.
- **Raw-text pretraining exposure:** the original paper describes TRC2 financial
  domain adaptation. No complete sample manifest for the pinned checkpoint was
  found in the reviewed sources, so raw-text and syndication overlap remain unknown.

No sample-level overlap analysis was performed: the actual corpora are not local.
SEntFiN remains a candidate external benchmark, not a certified independent test.
Document unavailable pretraining corpora as a limitation even after cross-corpus
checks against accessible supervised datasets pass.

## Next bounded work and roadmap status

SEntFiN release review and revision pinning are complete for this scope.
The broader Phase 1 terms/revisions checkbox remains open: PhraseBank's exact
artifact review and FiQA's conflicting terms are still unresolved. FinBERT weight
terms remain unresolved separately. The independent-test checkbox also stays open.

Next, prepare the PhraseBank baseline asset record and pin its release, then load
and validate the permitted dataset artifacts in a separate task. An overlap audit
requires both actual SEntFiN and baseline corpora. Before any model comparison,
reserve headline groups, audit duplicates, record company coverage, and freeze a
versioned evaluation manifest. Do not inspect final-test labels to choose models.
