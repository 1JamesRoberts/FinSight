# TF-IDF + Logistic Regression validation

Separate train-only models; PhraseBank and single-target SEntFiN

One fixed configuration; no tuning or final-test scoring.
Separate models fitted on each cohort's training texts/labels only.
Vocabulary and IDF use training only. VADER scores the same validation rows.

| Cohort | Train | Validation | Model | Accuracy | Macro F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| phrasebank | 3785 | 518 | logistic_regression | 0.7490 | 0.6696 |
| phrasebank | 3785 | 518 | vader | 0.5560 | 0.4895 |
| phrasebank | 3785 | 518 | training_majority | 0.5734 | 0.2429 |
| sentfin_single_target | 6299 | 781 | logistic_regression | 0.8182 | 0.8165 |
| sentfin_single_target | 6299 | 781 | vader | 0.4738 | 0.4787 |
| sentfin_single_target | 6299 | 781 | training_majority | 0.3611 | 0.1769 |

## phrasebank

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.9375 | 0.4348 | 0.5941 |
| neutral | 297 | 0.7200 | 0.9697 | 0.8264 |
| positive | 152 | 0.8140 | 0.4605 | 0.5882 |

Confusion matrix: rows=gold, columns=predicted.
Class order: negative/neutral/positive.

```text
[30, 30, 9]
[2, 288, 7]
[0, 82, 70]
```

Vocabulary: 12351 features. Training: 0.193 seconds. Mean inference: 0.213 ms/text (VADER 0.034).
Local model ID: `f054699006ee7d057660f753ec05ac885857d8f23888ca9241074c656c7d1524`.
Artifact SHA-256: `915ee85fc53dd054d2686b1fda530d58b067c156f9165672210a1fea76aee4cd`.

## sentfin_single_target

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8552 | 0.7652 | 0.8077 |
| neutral | 252 | 0.7727 | 0.8095 | 0.7907 |
| positive | 282 | 0.8311 | 0.8723 | 0.8512 |

Confusion matrix: rows=gold, columns=predicted.
Class order: negative/neutral/positive.

```text
[189, 35, 23]
[21, 204, 27]
[11, 25, 246]
```

Vocabulary: 10475 features. Training: 0.201 seconds. Mean inference: 0.272 ms/text (VADER 0.030).
Local model ID: `a362604910e91d138cbc592a5fe7a35f87167e8c812c83abe705e78d8cad8b9a`.
Artifact SHA-256: `d8330bcac15918283552ea6d306e238d6836a91cb2ced202431971146d8e3e0c`.

Run ID: `eff5d714ba9f99753db260e8a1c656fe277c34810956fe1fd6e3b7a5decf8566`.
Parent split: `a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898`.
Code revision: `414bf5ef5b50802a3d0753beebdadb53910795e0`; dirty: `True`.
Implementation digests identify code used before its commit.
Settings, all estimator parameters, dependency versions, hashes, and metrics
are in the accompanying aggregate JSON. Models and per-record outputs stay
in ignored storage. Vocabulary, headlines, and record IDs stay local.
These are development results, with separate corpus-specific training; they
do not measure cross-corpus transfer, target-aware sentiment, calibration,
or frozen-test generalization. Latency is descriptive on one local machine.
See `docs/tfidf-logreg.md` for usage, preprocessing, and limitations.
