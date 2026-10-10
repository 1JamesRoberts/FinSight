# FiQA Task 1 asset review

Reviewed 2026-10-10 for local noncommercial FinSight research. This review selects
an accessible **original-format challenge participant copy**, not the gated
TheFinAI benchmark mirror. No model training or public text redistribution is
performed by this task.

## Terms and release selection

The [original organizer page](https://sites.google.com/view/fiqa/home) says:
“The training data is available only for non-commercial use.” Its test-data
section imposes the same noncommercial restriction. Local acquisition,
validation, and subsequent noncommercial research experiments are supported
under those original terms. Commercial use and public redistribution are not
cleared. Retain provenance and cite the challenge paper when reporting results:
[Maia et al., WWW 2018](https://doi.org/10.1145/3184558.3192301).

The organizer's linked Google Drive train/test files returned HTTP 404 during
review. The current
[TheFinAI/en-fiqa-absa mirror](https://huggingface.co/datasets/TheFinAI/en-fiqa-absa)
metadata advertises MIT, is automatically gated, and asks for name, affiliation,
country, intended use, and agreement to cite FinBen. Metadata revision:
`2484314781c82018dc97057c3d9165b390815d30`. No form was submitted, personal details
sent, or gated data downloaded. That mirror's MIT declaration does not override
the original noncommercial restriction; its broader scope remains unresolved.

The selected [participant repository](https://github.com/dayanfcosta/fiqa-2018-task1)
identifies itself as a FiQA 2018 Task 1 submission and links its published paper.
It contains the original nested JSON format used in the organizer's example.
Pin: `acd0f4cfafa6421774bc8f009e8fa3e0ec441357`. File sizes, SHA-256, and Git blob
identities are recorded in `configs/asset-revisions.json`; the participant README
is retained beside the data. The repository has no separate dataset license:
our decision relies on the organizer's original noncommercial-use statement,
not an inferred software license. This is a public participant-hosted copy with
plausible provenance, **not byte-verified against the unavailable organizer
files**. Immutable revision and checksums make this selected copy reproducible.

The [FLANG authors' release](https://github.com/SALT-NLP/FLANG) was also inspected.
Its FiQA sentiment code expects original JSON files, but its linked
`SALT-NLP/FLUE-FiQA` dataset contains question-answering retrieval data. That is
Task 2 and was not acquired for sentiment analysis.

## Acquired artifacts and measured contents

| File | Text records | Target annotations | Scored annotations |
| --- | ---: | ---: | ---: |
| task1_headline_ABSA_train.json | 438 | 503 | 503 |
| task1_post_ABSA_train.json | 675 | 675 | 675 |
| task1_headline_ABSA_test.json | 93 | 93 | 0 |
| task1_post_ABSA_test.json | 99 | 99 | 0 |

There are **1,113 training texts and 1,178 scored annotations**. This is not the
TheFinAI mirror's 822/117/234 benchmark split (1,173 rows). Do not claim benchmark
equivalence or silently reshape this copy into those splits. The participant
copy's original test inputs have targets and snippets but no gold scores/aspects;
they cannot provide supervised accuracy or regression metrics. A later labeled
evaluation release would require a separate review.

One normalized text occurs in both original train and test inputs. No normalized
exact text matches were found with PhraseBank or SEntFiN. Exact matching uses
NFKC, casefold, and whitespace collapse; it does not establish independence from
FinBERT training/pretraining. The
[FinBERT paper](https://arxiv.org/html/1908.10063) includes FiQA regression
experiments, without identifying all samples used for the published classifier.
FiQA is not accepted as the independent frozen final test.

## Preservation and limitations

Each text remains a group of target annotations. Preserve opaque source IDs,
source filename, original train/test designation, headline/post type, revision,
text, targets, raw score strings, and raw aspect/snippet strings. Scores remain
finite floats in [-1, 1]; no three-class thresholds are introduced. Original test
scores/aspects remain `None`, rather than zero/neutral or pseudo-labels.

Four training-headline snippet values are the bare token `nan` within string
lists. A restricted AST parser maps only those values to `None` and retains the
raw strings. It never evaluates expressions or substitutes the literal text
"nan". Annotation grouping is retained; no rows are dropped and no new partition
is assigned. Split and final-test decisions remain future Phase 1 work.

Neither this review nor the source schema establishes comprehensive publisher
rights, full-article validity, publication timestamps, or coverage of the five US
portfolio companies. Keep raw files and research receipts ignored by Git.
