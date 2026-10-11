# TF-IDF + Linear SVM validation

Training-only fits; validation-only scores for phrasebank, sentfin_single_target

One fixed configuration; no tuning or final-test scoring.
Class weighting: `balanced`.
Separate models fitted on each cohort's training texts/labels only.
Vocabulary and IDF use training only. VADER scores the same validation rows.

| Cohort | Train | Validation | Model | Accuracy | Macro F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| phrasebank | 3785 | 518 | linear_svm | 0.7819 | 0.7444 |
| phrasebank | 3785 | 518 | vader | 0.5560 | 0.4895 |
| phrasebank | 3785 | 518 | training_majority | 0.5734 | 0.2429 |
| sentfin_single_target | 6299 | 781 | linear_svm | 0.8092 | 0.8075 |
| sentfin_single_target | 6299 | 781 | vader | 0.4738 | 0.4787 |
| sentfin_single_target | 6299 | 781 | training_majority | 0.3611 | 0.1769 |

## phrasebank

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.7963 | 0.6232 | 0.6992 |
| neutral | 297 | 0.7964 | 0.8822 | 0.8371 |
| positive | 152 | 0.7407 | 0.6579 | 0.6969 |

Confusion matrix: rows=gold, columns=predicted.
Class order: negative/neutral/positive.

```text
[43, 18, 8]
[8, 262, 27]
[3, 49, 100]
```

Vocabulary: 12351 features. Training: 0.103 seconds. Mean inference: 0.166 ms/text (VADER 0.033).
Local model ID: `bf45ecd5b58d105064118eeb0098b231c47064e328b3bc4506e9db83f6ccab7a`.
Artifact SHA-256: `6827726895649911705593bc4008582f06ca8e93178011d252712fcfc7462894`.

### Model architecture and size

Pipeline: **TF-IDF word unigrams/bigrams -> LinearSVC**; multiclass strategy: **one-versus-rest**.

| Property | Value |
| --- | ---: |
| Input TF-IDF features | 12,351 |
| Output classes | 3 |
| Coefficient matrix | [3, 12351] |
| Intercepts | 3 |
| Stored classifier parameters | 37,056 |
| Fitted IDF values (separate preprocessing state) | 12,351 |
| Coefficient + intercept array bytes | 296,448 |
| IDF array bytes | 98,808 |
| Serialized pipeline bytes | 786,432 |
| Serialized pipeline KiB (1 KiB = 1,024 bytes) | 768.00 |

Classifier parameters count stored coefficients plus intercepts; IDF values are separate fitted preprocessing state. Numeric bytes cover these arrays only. Artifact bytes cover the entire serialized pipeline, including vocabulary/metadata/serialization overhead; neither is runtime RAM.

## sentfin_single_target

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8240 | 0.7773 | 0.8000 |
| neutral | 252 | 0.7756 | 0.7817 | 0.7787 |
| positive | 282 | 0.8265 | 0.8617 | 0.8438 |

Confusion matrix: rows=gold, columns=predicted.
Class order: negative/neutral/positive.

```text
[192, 31, 24]
[28, 197, 27]
[13, 26, 243]
```

Vocabulary: 10475 features. Training: 0.090 seconds. Mean inference: 0.154 ms/text (VADER 0.021).
Local model ID: `726fb3a650f2768d55d509d43d12548ef61383e221fbc783697e0408dd6e134b`.
Artifact SHA-256: `85d993df153e6f621f011df0805e3bcfd9fe1faf89dc958574f6744be26b5410`.

### Model architecture and size

Pipeline: **TF-IDF word unigrams/bigrams -> LinearSVC**; multiclass strategy: **one-versus-rest**.

| Property | Value |
| --- | ---: |
| Input TF-IDF features | 10,475 |
| Output classes | 3 |
| Coefficient matrix | [3, 10475] |
| Intercepts | 3 |
| Stored classifier parameters | 31,428 |
| Fitted IDF values (separate preprocessing state) | 10,475 |
| Coefficient + intercept array bytes | 251,424 |
| IDF array bytes | 83,800 |
| Serialized pipeline bytes | 656,940 |
| Serialized pipeline KiB (1 KiB = 1,024 bytes) | 641.54 |

Classifier parameters count stored coefficients plus intercepts; IDF values are separate fitted preprocessing state. Numeric bytes cover these arrays only. Artifact bytes cover the entire serialized pipeline, including vocabulary/metadata/serialization overhead; neither is runtime RAM.

Run ID: `3fca6dfa1eb732f965ecf74c1e7f67bf8151e53b9846e702e96d28548b3de502`.
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

Model profiles were added later from the original checksummed fitted
artifacts, without retraining or evaluation. Run ID/code/metrics above
refer to the original experiment; its ignored run.json remains unchanged.
Profile extraction code revision: `e976685c3615859e0f92e89f40d56b5485cbaa6b`; dirty: `True`.
Profile/report implementation hashes are logged separately in JSON.
