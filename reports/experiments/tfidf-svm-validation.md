# TF-IDF + Linear SVM validation

Training-only fits; validation-only scores for phrasebank, sentfin_single_target

One fixed configuration; no tuning or final-test scoring.
Class weighting: `None`.
Separate models fitted on each cohort's training texts/labels only.
Vocabulary and IDF use training only. VADER scores the same validation rows.

| Cohort | Train | Validation | Model | Accuracy | Macro F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| phrasebank | 3785 | 518 | linear_svm | 0.7799 | 0.7344 |
| phrasebank | 3785 | 518 | vader | 0.5560 | 0.4895 |
| phrasebank | 3785 | 518 | training_majority | 0.5734 | 0.2429 |
| sentfin_single_target | 6299 | 781 | linear_svm | 0.8092 | 0.8072 |
| sentfin_single_target | 6299 | 781 | vader | 0.4738 | 0.4787 |
| sentfin_single_target | 6299 | 781 | training_majority | 0.3611 | 0.1769 |

## phrasebank

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.8478 | 0.5652 | 0.6783 |
| neutral | 297 | 0.7820 | 0.9057 | 0.8393 |
| positive | 152 | 0.7500 | 0.6316 | 0.6857 |

Confusion matrix: rows=gold, columns=predicted.
Class order: negative/neutral/positive.

```text
[39, 21, 9]
[5, 269, 23]
[2, 54, 96]
```

Vocabulary: 12351 features. Training: 0.114 seconds. Mean inference: 0.168 ms/text (VADER 0.034).
Local model ID: `09332f51cab46ab785a08b3bbea3711f95f47b6a852b54a79b7c4951c5ec35b0`.
Artifact SHA-256: `311792c250408f0197b6d9206a5391e34a949f0d426554cf925bd2e4e510466a`.

## sentfin_single_target

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8363 | 0.7652 | 0.7992 |
| neutral | 252 | 0.7725 | 0.7817 | 0.7771 |
| positive | 282 | 0.8200 | 0.8723 | 0.8454 |

Confusion matrix: rows=gold, columns=predicted.
Class order: negative/neutral/positive.

```text
[189, 32, 26]
[27, 197, 28]
[10, 26, 246]
```

Vocabulary: 10475 features. Training: 0.098 seconds. Mean inference: 0.157 ms/text (VADER 0.022).
Local model ID: `3ff048f7884a9f3b856f4fe65384092a85b7d2afa34b9d0165faec557c76953b`.
Artifact SHA-256: `65cb6920b65654f38a8f9f7c31b6c735387689a0da00a1b67a3f533091bcf479`.

Run ID: `65fc0ef55bb4ab4074c6e923fa8435a44ec9428fdede1153aa9759de35a91323`.
Parent split: `a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898`.
Code revision: `dc8c839cfdcb69f0de62896b98b3e812929e1eab`; dirty: `True`.
Implementation digests identify code used before its commit.
Settings, all estimator parameters, dependency versions, hashes, and metrics
are in the accompanying aggregate JSON. Models and per-record outputs stay
in ignored storage. Vocabulary, headlines, and record IDs stay local.
These are development results, with separate corpus-specific training; they
do not measure cross-corpus transfer, target-aware sentiment, calibration,
or frozen-test generalization. Latency is descriptive on one local machine.
See `docs/tfidf-svm.md` for usage, preprocessing, and limitations.
