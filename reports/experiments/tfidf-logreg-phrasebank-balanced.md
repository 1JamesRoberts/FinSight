# TF-IDF + Logistic Regression validation

Training-only fits; validation-only scores for phrasebank

One fixed configuration; no tuning or final-test scoring.
Class weighting: `balanced`.
Separate models fitted on each cohort's training texts/labels only.
Vocabulary and IDF use training only. VADER scores the same validation rows.

| Cohort | Train | Validation | Model | Accuracy | Macro F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| phrasebank | 3785 | 518 | logistic_regression | 0.7645 | 0.7378 |
| phrasebank | 3785 | 518 | vader | 0.5560 | 0.4895 |
| phrasebank | 3785 | 518 | training_majority | 0.5734 | 0.2429 |

## phrasebank

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.7313 | 0.7101 | 0.7206 |
| neutral | 297 | 0.8000 | 0.8350 | 0.8171 |
| positive | 152 | 0.7021 | 0.6513 | 0.6758 |

Confusion matrix: rows=gold, columns=predicted.
Class order: negative/neutral/positive.

```text
[49, 13, 7]
[14, 248, 35]
[4, 49, 99]
```

Vocabulary: 12351 features. Training: 0.146 seconds. Mean inference: 0.203 ms/text (VADER 0.032).
Local model ID: `30cdc57c1f17294bd85b1888e1e73a9bf8c7b0c0064c5e9f934c62d9282d7895`.
Artifact SHA-256: `6b24bd6a8bb3f682b50c77c4c0c9b3f93568f87d9f4f3df08f69eb43c1b449a8`.

Run ID: `f9a82cf26bb9361c13f6f02e7c7b169b57c5fef9e29f0dd18ca91c345a87c5f1`.
Parent split: `a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898`.
Code revision: `cbf89346aeb9ba52b5ac7190e59dcc97b88ecc6c`; dirty: `True`.
Implementation digests identify code used before its commit.
Settings, all estimator parameters, dependency versions, hashes, and metrics
are in the accompanying aggregate JSON. Models and per-record outputs stay
in ignored storage. Vocabulary, headlines, and record IDs stay local.
These are development results, with separate corpus-specific training; they
do not measure cross-corpus transfer, target-aware sentiment, calibration,
or frozen-test generalization. Latency is descriptive on one local machine.
See `docs/tfidf-logreg.md` for usage, preprocessing, and limitations.
