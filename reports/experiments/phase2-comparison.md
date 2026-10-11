# FinSight Phase 2 — Traditional sentiment baselines

Train-only TF-IDF classifiers outperform VADER and training-majority controls on both development cohorts. A bounded C/class-weight search selects balanced Logistic Regression for both corpora by validation macro F1. Its lead over selected SVM is small, especially on PhraseBank; no statistical superiority claim is supported. This phase establishes reproducible local baselines, not final-test performance.

## Experimental contract

| Cohort | Training texts | Validation texts | Training policy |
| --- | ---: | ---: | --- |
| PhraseBank ≥50% agreement master view | 3,785 | 518 | Separate corpus model |
| SEntFiN single-target headlines | 6,299 | 781 | Separate corpus model |

The verified Phase 1 development loader excludes reserved/frozen groups. Known exact and conservative near-duplicate boundaries are checked by the existing split policy. No FiQA classification, multi-target headline collapse, pooled training, or final-test evaluation is performed. Nested PhraseBank agreement subsets are not separate training corpora.

TF-IDF fits vocabulary and IDF on each corpus’s training texts only. Classifier fitting uses matching training labels; balanced weights derive from training counts. Validation uses the persisted fitted pipeline. Word unigram/bigram features use lowercase tokenization (word tokens of at least two characters), min_df=2, max_features=50,000, sublinear TF, smoothed IDF and L2 normalization, with no stopword removal or stemming. Unknown validation words do not expand vocabulary.

The grid contains C=0.1, 1, 10 and null/balanced class weighting: six configurations per classifier per corpus, 24 fitted pipelines. All other settings stay fixed. Seed=42; LR max_iter=1,000, SVM max_iter=10,000. Selection maximizes validation macro F1; exact ties retain declared LR-before-SVM, then grid order. No CV, calibration, threshold search, or post-review tuning occurs.

Accuracy, macro F1 and per-class metrics retain all three labels (negative/neutral/positive); undefined precision, recall and F1 are zero. Macro F1 gives each class equal weight. Training-majority predicts the most frequent training label; VADER uses fixed inclusive compound thresholds −0.05/+0.05. These controls are scored on the same validation membership.

## Model comparison

Initial means C=1, unweighted. Selected means the best grid configuration within that classifier and cohort. VADER and majority are fixed controls. Larger scores across corpora do not establish that one dataset generalizes better: text styles, class balance, sample sizes, and annotation schemes differ.

| Cohort | Model/stage | C | Weight | Accuracy | Macro F1 | Mean ms/text | Pipeline KiB |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| phrasebank | training_majority | — | — | 0.5734 | 0.2429 | — | — |
| phrasebank | vader | — | — | 0.5560 | 0.4895 | 0.032 | — |
| phrasebank | logistic_regression / initial | 1.0 | None | 0.7490 | 0.6696 | 0.204 | 768.13 |
| phrasebank | logistic_regression / selected | 10.0 | balanced | 0.7780 | 0.7446 | 0.199 | 768.14 |
| phrasebank | linear_svm / initial | 1.0 | None | 0.7799 | 0.7344 | 0.164 | 767.99 |
| phrasebank | linear_svm / selected | 1.0 | balanced | 0.7819 | 0.7444 | 0.165 | 768.00 |
| sentfin_single_target | training_majority | — | — | 0.3611 | 0.1769 | — | — |
| sentfin_single_target | vader | — | — | 0.4738 | 0.4787 | 0.021 | — |
| sentfin_single_target | logistic_regression / initial | 1.0 | None | 0.8182 | 0.8165 | 0.188 | 641.67 |
| sentfin_single_target | logistic_regression / selected | 1.0 | balanced | 0.8195 | 0.8181 | 0.187 | 641.68 |
| sentfin_single_target | linear_svm / initial | 1.0 | None | 0.8092 | 0.8072 | 0.157 | 641.53 |
| sentfin_single_target | linear_svm / selected | 0.1 | balanced | 0.8169 | 0.8160 | 0.162 | 641.54 |

PhraseBank selected LR improves macro F1 from 0.6696 to 0.7446 versus its initial unweighted configuration, mainly after accounting for imbalance. It is only about 0.0002 above selected SVM (0.7444), while SVM has slightly higher accuracy. SEntFiN selected LR improves from 0.8165 to 0.8181; selected SVM reaches 0.8160. These are small development-selection gaps, not demonstrated statistically reliable improvements.

Latency uses serial batch-size-one inference after an invented warm-up on one local machine; training time is reported separately. This task copies recorded timing rather than rerunning benchmarks. Timing variability prevents strong speed claims. Majority timing and VADER artifact size were not profiled.

## Selected models: class metrics and confusion matrices

### phrasebank: logistic_regression

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.7500 | 0.6522 | 0.6977 |
| neutral | 297 | 0.8038 | 0.8552 | 0.8287 |
| positive | 152 | 0.7324 | 0.6842 | 0.7075 |

Confusion matrix: gold rows, predicted columns; negative/neutral/positive.

```text
[45, 17, 7]
[12, 254, 31]
[3, 45, 104]
```

Original run ID: `3d4657830b1cfe58b5f86a167b4a9216a565ac971ca1f95e13ba67041a6071cf`.

### phrasebank: linear_svm

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.7963 | 0.6232 | 0.6992 |
| neutral | 297 | 0.7964 | 0.8822 | 0.8371 |
| positive | 152 | 0.7407 | 0.6579 | 0.6969 |

Confusion matrix: gold rows, predicted columns; negative/neutral/positive.

```text
[43, 18, 8]
[8, 262, 27]
[3, 49, 100]
```

Original run ID: `0cef6943df2baa13a8234b568a7493c01b38b84abb1108df4e86de60b8f13a60`.

### sentfin_single_target: logistic_regression

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8263 | 0.7895 | 0.8075 |
| neutral | 252 | 0.7698 | 0.8095 | 0.7892 |
| positive | 282 | 0.8607 | 0.8546 | 0.8577 |

Confusion matrix: gold rows, predicted columns; negative/neutral/positive.

```text
[195, 34, 18]
[27, 204, 21]
[14, 27, 241]
```

Original run ID: `c5375a28e1498111dda90119f8a8cfb522a9169d46ceec5b4a8ec85683abd10e`.

### sentfin_single_target: linear_svm

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8428 | 0.7814 | 0.8109 |
| neutral | 252 | 0.7610 | 0.8214 | 0.7901 |
| positive | 282 | 0.8500 | 0.8440 | 0.8470 |

Confusion matrix: gold rows, predicted columns; negative/neutral/positive.

```text
[193, 36, 18]
[21, 207, 24]
[15, 29, 238]
```

Original run ID: `7740e038338174bee36238dd7612cc13ce784a6ba61b6374a086b3c60df00a2c`.

## Architecture and size

| Model | Architecture | Output interpretation |
| --- | --- | --- |
| VADER | Handwritten rules over a sentiment lexicon; no fitted corpus classifier | Compound polarity and lexical proportions; not calibrated class probabilities |
| Logistic Regression | Sparse TF-IDF → multinomial L2 logistic classifier, lbfgs | Softmax class probabilities; calibration has not been evaluated |
| Linear SVM | Sparse TF-IDF → LinearSVC, L2 squared hinge, one-versus-rest, dual=True | Signed decision scores; not probabilities |

| Cohort | TF-IDF features | Classifier coefficient shape | Intercepts | Stored classifier parameters | Separate IDF values |
| --- | ---: | --- | ---: | ---: | ---: |
| PhraseBank | 12,351 | 3 × 12,351 | 3 | 37,056 | 12,351 |
| SEntFiN single-target | 10,475 | 3 × 10,475 | 3 | 31,428 | 10,475 |

Feature dimensions and stored parameter counts are the same for LR and SVM within a corpus because the fitted feature settings and training texts match. Coefficients and IDF arrays are float64. Classifier-plus-IDF numeric arrays occupy 395,256 bytes for PhraseBank and 335,224 bytes for SEntFiN. These exclude vocabulary and Python object overhead. Pipeline file sizes include vocabulary, metadata, and serialization; they are not runtime RAM measurements or counts of mathematical degrees of freedom.

## Qualitative findings

The [validation-error review](phase2-error-analysis.md) inspects 58 stratified LR errors: 28 PhraseBank and 30 SEntFiN, with SVM and VADER comparisons. Primary sample categories include financial context (14), neutral boundary (14), mixed cues (11), event interpretation (9), numeric comparison (7), and negation scope (3). These are sample counts, not prevalence estimates.

LR and SVM share 106 PhraseBank and 136 SEntFiN errors across the full validation cohorts. They disagree on only 16 and 15 labels respectively. This motivates investigating contextual/target representations later, rather than assuming another linear classifier resolves these difficulties. It does not establish a transformer improvement in advance.

Twenty reviewed cases have possible label ambiguity or missing context. This is an exploratory Codex judgment, not confirmed mislabeling or independent human adjudication. Every original label and example remains in the evaluation. No labels were repaired, examples dropped, or hyperparameters changed after this review.

## Artifacts, inference and reproducibility

Typed baseline inference is available through `VaderSentiment`, `TfidfSentiment` and `TfidfSVM`, each with `predict(text)` and `predict_many(texts)`. Trainable models separate `fit` from inference and support checked local save/load; inference never fits preprocessing. Their score semantics deliberately differ. No web API or common probability calibration layer is introduced.

Existing commands and all 24-candidate details are in the [tuning guide](../../docs/tfidf-tuning.md) and [tuning report](tfidf-tuning.md). The [comparison JSON](phase2-comparison.json) includes initial/selected configs, complete metrics, architectures, measured sizes/times, model hashes and original run IDs, and pins the tuning/review input digests. Original run logs additionally pin split, training membership, dependency versions/lockfile, code revision and implementation hashes. Dirty-tree implementation hashes identify code used before a task commit.

Fitted models, per-record predictions, vocabulary, review text and individual notes remain in ignored local storage. Checksum-based loading verifies integrity; pickle files must still come from a trusted local source. Original baseline and tuning reports, models and predictions are preserved. The review command can reproduce the packet and aggregate review when local annotations are present; reviewer judgments are not regenerated automatically.

```bash
uv sync --locked
uv run --locked python -m finsight.sentiment.error_analysis
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

## Limitations and phase status

Phase 2 is complete within the documented local, validation-only scope: VADER, LR and SVM; train-only preprocessing; bounded tuning; metrics/latency; persisted configurations/models; qualitative error review; tested inference and the consolidated comparison. The grid is modest, selected validation scores are optimistic after repeated inspection, and uncertainty/calibration analyses are not implemented. No final-test performance is claimed.

PhraseBank agreement labels, SEntFiN entity/headline annotations, lower-quality fragments, and limited target/company coverage constrain interpretation. These are sentence baselines, not target-aware sentiment. Neither cohort supports the five-company US-market deployment claim. Dataset/model-weight redistribution remains unresolved under the documented local research permissions; only aggregates are committed.

The final SEntFiN benchmark remains frozen and unscored. ProsusAI FinBERT’s PhraseBank supervision overlap must still be disclosed; unknown pretraining exposure prevents certifying SEntFiN as unseen. Phase 3 will require its own model-terms review and explicit implementation request before acquisition/inference. No transformer, aggregation, or stock prediction functionality is added here. Hosted GitHub Actions remains unverified.

Input SHA-256 identities:

- Tuning report: `329457f5dc8b951883897dfcf0b0a3ca88357093730d86315c9c5183cefdc5e1`.
- Error review: `88b3980a09c74a4e807bbef4281d564bc009a1dd5ad725d58001776a82d6873f`.
- Freeze receipt: `ab0c9f0581612ee52f3c03a5bbbc03e7ebc621fdf76bb00d7d85bc2973b593f7`.
