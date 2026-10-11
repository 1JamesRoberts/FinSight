# TF-IDF + Logistic Regression validation

Training-only fits; validation-only scores for phrasebank

One fixed configuration; no tuning or final-test scoring.
Class weighting: `None`.
Separate models fitted on each cohort's training texts/labels only.
Vocabulary and IDF use training only. VADER scores the same validation rows.

| Cohort | Train | Validation | Model | Accuracy | Macro F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| phrasebank | 3785 | 518 | logistic_regression | 0.7490 | 0.6696 |
| phrasebank | 3785 | 518 | vader | 0.5560 | 0.4895 |
| phrasebank | 3785 | 518 | training_majority | 0.5734 | 0.2429 |

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

Vocabulary: 12351 features. Training: 0.179 seconds. Mean inference: 0.211 ms/text (VADER 0.034).
Local model ID: `882e7dd7c8dd8be4b8f42636c542b02b5f7aab9103dbb3284aecc3b85184969a`.
Artifact SHA-256: `2ba90cf1134570a8b8ebacd43c16779b09a7636fab6d7c63659252d0168eb18f`.

Run ID: `548f163366d5c4936d2592ffcc92fc28c239aaf0442b5ada057517ff86af9254`.
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
