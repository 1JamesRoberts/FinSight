# Phase 1 final report — Dataset curation and EDA

**As of:** 2026-10-22. **Status:** complete within the documented local research
scope. Phase 1 establishes curated data and evaluation boundaries; later test
consumption is described separately so its original completion record stays intact.

[All final reports](../README.md) · [Phase evidence index](README.md)

## Scope and status

Select financial sentiment assets, verify access/use decisions, inspect labels
and annotations, control duplication, prepare reproducible splits, freeze an
evaluation benchmark and produce development-only dataset cards/EDA. This phase
contains no learned sentiment or stock-prediction result.

## Data and method

Use Financial PhraseBank's ≥50% agreement collection as the master sentence view,
native SEntFiN target/headline annotations for company-targeted research, and a
reviewed FiQA copy as a continuous-score research asset. PhraseBank's four
agreement subsets overlap: “all agree” means unanimous agreement, not all rows.
Do not concatenate them as independent corpora or convert FiQA signs into gold
three-class labels without a separate protocol.

Exact normalized text and conservative three-word-shingle Jaccard near matches
form connected groups before splitting. Equivalent annotations retain lineage;
conflicting groups are quarantined. Other near duplicates stay in the same
partition. The fixed seed/group-hash policy approximates 80/10/10 partitions;
development-only groups use renormalized training/validation weights. This is a
dedicated deterministic implementation, not sklearn `train_test_split` or
stratification. Similar label proportions were observed, not optimized by seed
search. See the [split policy](../../docs/phase-01-data/benchmark/split-policy.md).

## Results

| Corpus | Development texts | Annotation unit | Use |
|---|---:|---|---|
| PhraseBank master view | 4,303 | Three-class sentence sentiment | Separate traditional/base-model research; known FinBERT supervision overlap |
| SEntFiN | 9,651 | Company/native-target sentiment per headline | Single-target and target-aware development cohorts |
| FiQA copy | 1,009 | Continuous target scores | Research asset; not pooled into three-class training |

**Attributes:** Corpus identifies a separate source view. Development texts counts
retained train-plus-validation records, not target annotations or reserved test
records. Annotation unit explains what is labeled; a SEntFiN/FiQA text can have
multiple targets. Use states the agreed role and does not establish licensing
for public source-text redistribution. Text counts sum to 14,963; target-label
counts do not share that unit and must not be pooled across corpora.

SEntFiN development contains 12,957 target annotations, including 2,571
multi-target headlines; 1,116 of those have mixed labels. PhraseBank development
is 59.4% neutral, motivating macro F1 and per-class reporting. FiQA scores remain
continuous and 48 flagged annotations retain score research eligibility with
snippet supervision disabled; keeping a row does not validate its label.
[Development findings](../../docs/phase-01-data/eda/phase1-findings.md).

A frozen SEntFiN benchmark was selected with known accessible-corpus overlap
checks and source/lineage hashes. PhraseBank remains unsuitable as independent
FinBERT evidence. The subsequent [distribution supplement](../split-label-distribution.md)
reports the actual sentence-model cohorts, including reserved aggregates, without
resplitting or selecting weights from test labels.

## Decisions

Keep corpora separate, preserve native target sets and continuous FiQA scores,
quarantine unresolved annotation conflicts, and keep raw bytes unchanged.
Use content-addressed source/split/freeze artifacts and train/validation-only EDA.
Dataset cards document source terms, label meaning, overlap and population limits.

The original SEntFiN freeze is still preserved. Its single-target cohort was later
consumed by Phase 3 and the corrected multi-target evaluation view by Phase 4;
“frozen” no longer means those evaluated cohorts are untouched. Never rerun
selection or tune against their labels. PhraseBank reserved candidates are a
separate view, not the independent final FinBERT test.

## Limitations and remaining work

Lexical duplicate checks cannot prove absence of semantic duplication or unknown
model pretraining exposure. SEntFiN is not representative US-company gold data.
FiQA copy provenance, suspect annotations and classification thresholds remain
limitations; no classifier training on FiQA is claimed. Source/model permissions
are local-use decisions, not blanket redistribution or deployment rights.
Phase 6 will need independently reviewed timestamps and prediction cutoffs.

## Evidence and reproduction

- [Original findings/completion record](../../docs/phase-01-data/eda/phase1-findings.md), [EDA report](../phase1-eda.md), [aggregate JSON](../eda-summary.json), [figures](../figures/phase1).
- Dataset cards: [PhraseBank](../../docs/phase-01-data/dataset-cards/phrasebank.md), [SEntFiN](../../docs/phase-01-data/dataset-cards/sentfin.md), [FiQA](../../docs/phase-01-data/dataset-cards/fiqa.md).
- [Duplicate/annotation review](../annotation-review.md), [split preparation](../split-preparation.md), [benchmark report](../frozen-benchmark.md), [freeze receipt](../../configs/benchmark-freeze.json).
- [Acquisition/loading guide](../../docs/phase-01-data/acquisition/dataset-loading.md), [source-use ledger](../../docs/datasets.md), [method index](../../docs/phase-01-data/README.md).

Reproduction requires the reviewed pinned source files or permitted acquisition;
raw data is not committed. Follow the existing guide to verify lineage and
reproduce descriptive aggregates. This report reads existing public results only;
it does not reacquire data, reconstruct a different holdout or open reserved texts.
