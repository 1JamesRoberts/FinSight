# Phase 2 final report — Traditional sentiment baselines

**As of:** 2026-10-22. **Status:** complete within the original validation-only
research scope. Final-test baseline results belong to Phase 3, not this table.

[All final reports](../README.md) · [Phase evidence index](README.md)

## Scope and status

Establish majority/VADER controls, TF-IDF Logistic Regression and Linear SVM,
class-weight comparisons, a bounded validation search, saved pipelines and
qualitative error analysis. The phase establishes useful sentence baselines,
not company-specific sentiment or out-of-time stock prediction.

## Data and method

Train separately on 3,785 PhraseBank sentences and 6,299 single-target SEntFiN
headlines; validate on 518 and 781 texts respectively. Do not combine the corpora
or collapse multi-target labels. Vocabulary and IDF are fitted only on training
text; transforms preserve order and never fit on validation/test data.

TF-IDF lowercases and internally tokenizes word tokens of at least two
characters, using word unigrams/bigrams, `min_df=2`, at most 50,000 features,
sublinear TF, smoothed IDF and L2 normalization. No manual pre-tokenization,
stemming or stop-word removal is required. Logistic Regression learns a
multinomial linear softmax classifier; Linear SVM uses one-versus-rest squared
hinge margins. Search C ∈ {0.1, 1, 10} and unweighted/balanced class weights on
validation, with seed 42 and fixed other settings. Balanced weights derive from
training labels. VADER uses fixed compound thresholds ±0.05 and no learned corpus
classifier. Its lexical proportions and SVM margins are not calibrated confidence.

## Results

| Method | PhraseBank validation macro F1 (%) | SEntFiN single-target validation macro F1 (%) |
|---|---:|---:|
| Training-majority control | 24.29 | 17.69 |
| VADER | 48.95 | 47.87 |
| Initial unweighted Logistic Regression | 66.96 | 81.65 |
| Validation-selected Logistic Regression | 74.46 | 81.81 |
| Initial unweighted Linear SVM | 73.44 | 80.72 |
| Validation-selected Linear SVM | 74.44 | 81.60 |

**Attributes:** Method separates fixed controls, initial C=1/unweighted models and
validation-selected models. Each column uses its own corpus's held development
validation texts, not a shared or final test. Macro F1 equally averages negative,
neutral and positive class F1, expressed as a percentage. Selected rows reuse
validation for model selection and are optimistic development estimates. Rows
share examples within a column; corpus columns are different populations and
must not be compared as proof of cross-corpus generalization.

Selected LR uses C=10/balanced for PhraseBank and C=1/balanced for SEntFiN.
Selected SVM uses C=1/balanced and C=0.1/balanced respectively. LR's lead over SVM
is small and does not establish statistical superiority. Class weighting matters
more in the neutral-heavy PhraseBank cohort. Error review highlights financial
vocabulary, implicit events, mixed cues and context/label ambiguity without
repairing gold labels or tuning after review.

Both sparse models use the same fitted feature width per corpus: 12,351 on
PhraseBank and 10,475 on SEntFiN. Three coefficient rows plus three intercepts
store 37,056 and 31,428 classifier parameters; IDF adds one value per feature.
Selected serialized pipelines are approximately 768 KiB and 642 KiB respectively.
These are artifact sizes, not RAM or degrees of freedom. Full architecture,
class metrics and recorded timing are in the [original comparison](../experiments/phase2-comparison.md).

## Decisions

Retain both sparse baselines and fixed controls, with validation-selected LR as
the Phase 3 reference. Persist full TF-IDF-plus-classifier pipelines so inference
uses exactly the fitted vocabulary/IDF. Do not pool training sets, use test labels
for class weights or present observed validation leaders as proven winners.

## Limitations and remaining work

Validation is reused across a small search and error analysis. Phase 2 does not
estimate calibration/uncertainty or guarantee five-company US-market coverage.
Corpus rights and model/source redistribution limits remain explicit. At this
phase's completion the final benchmark was unscored; subsequent Phase 3 test
consumption does not retroactively turn these validation scores into test scores.

## Evidence and reproduction

- [Original consolidated report](../experiments/phase2-comparison.md) and [aggregate JSON](../experiments/phase2-comparison.json).
- [Tuning report](../experiments/tfidf-tuning.md), [qualitative errors](../experiments/phase2-error-analysis.md), [model/experiment guides](../../docs/phase-02-baselines/README.md).
- [VADER](../../docs/phase-02-baselines/models/vader.md), [TF-IDF LR](../../docs/phase-02-baselines/models/tfidf-logreg.md), [TF-IDF SVM](../../docs/phase-02-baselines/models/tfidf-svm.md), [class weighting](../../docs/phase-02-baselines/models/phrasebank-class-weights.md).

Use the saved, verified pipelines and the existing inference guides. Detailed
records retain source hashes, fitted sizes, configuration and run IDs. This
summary performs no refitting, prediction or new benchmark scoring.
