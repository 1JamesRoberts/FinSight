# PhraseBank class-weight comparison

Phase 2 bounded experiment, 2026-10-11. Compare two predeclared settings on the
same 3,785 training sentences and 518 validation sentences. Change only classifier
class weighting; TF-IDF, C, seed, and partitions are identical. Models and detailed
outputs remain ignored. No final-test scoring, SEntFiN training, new data, or new
dependencies are involved. Dataset/model terms remain those reviewed in
[datasets.md](datasets.md).

## Run it

From the repository root, with Phase 1 local artifacts already prepared:

```sh
uv sync --locked
uv run --locked python -m finsight.sentiment validate-tfidf-logreg \
  --cohort phrasebank \
  --config configs/tfidf-logreg.json \
  --report-path reports/experiments/tfidf-logreg-phrasebank-unweighted.md

uv run --locked python -m finsight.sentiment validate-tfidf-logreg \
  --cohort phrasebank \
  --config configs/tfidf-logreg-balanced.json \
  --report-path reports/experiments/tfidf-logreg-phrasebank-balanced.md
```

Each command trains a fresh model, persists/reloads it, and scores validation.
Separate aggregate Markdown/JSON reports preserve the original two-corpus report.
Each run evaluates a fixed preset; comparing the two runs is a bounded validation
experiment, not an automatic search. Broader Phase 2 tuning remains incomplete.
The unweighted preset remains the default.

## How inputs reach the model

`src/finsight/sentiment/__main__.py` uses `argparse` to read command-line arguments.
There is no interactive `input()` prompt.

| Input | Purpose |
| --- | --- |
| `validate-tfidf-logreg` | Select the training/validation runner |
| `--config` | JSON model settings, including `class_weight` |
| `--cohort phrasebank` | Select PhraseBank development rows |
| `--data-dir` | Local storage root; defaults to `FINSIGHT_DATA_DIR`, otherwise `data` |
| `--freeze-receipt` | Pin the split and reserved benchmark; defaults to `configs/benchmark-freeze.json` |
| `--report-path` | Destination for aggregate Markdown and companion JSON |

The config contains settings, not training text. The runner calls
`load_development_records(data_dir, receipt_path)` to verify artifact digests and
duplicate-group boundaries, then return train/validation rows only. The receipt
identifies prepared records under
`data/processed/splits/<parent-split-sha256>/prepared-records.jsonl`.
Holdout bytes are checked for integrity; no holdout text or gold enters fitting,
prediction, or metrics. No new random split is made.

The cohort filter selects PhraseBank; the partition field separates training
and validation. `sentence_and_label()` extracts each native sentence and label.
The runner builds matching lists in the same order:

```python
training_examples = [sentence_and_label(row) for row in training]
train_texts = [text for text, _ in training_examples]
train_labels = [label for _, label in training_examples]

model = TfidfSentiment(config)
model.fit(train_texts, train_labels)
```

`train_texts[i]` corresponds to `train_labels[i]`. Invented examples might be
`['Profits increased.', 'Losses widened.', 'The meeting is scheduled.']` and
`['positive', 'negative', 'neutral']`. The real run uses all 3,785 training
sentences with preserved labels, not these invented examples.

`TfidfSentiment` passes `class_weight=config.class_weight` to `LogisticRegression`.
Its `fit()` calls `pipeline.fit(train_texts, train_labels)`: TF-IDF learns vocabulary
and IDF from those texts, and the classifier learns coefficients from their labels.
Validation calls `model.predict(text)` with sentence text only. Gold labels are
used afterward by `classification_metrics()`; they never enter fitting or weights.

## What balanced means

The balanced preset sets `"class_weight": "balanced"`. Omitting this setting,
as in the original preset, defaults to `None`; JSON `null` is also accepted.
Unsupported values fail validation. Scikit-learn computes weights from training
labels: `training_samples / (number_of_classes * training_class_count)`.
See the [official parameter definition](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html).
Weights scale examples' contributions to the training loss, not their TF-IDF
vectors. Rows are neither duplicated nor dropped.

| Class | Training count | Unweighted | Balanced weight |
| --- | ---: | ---: | ---: |
| Negative | 468 | 1.0000 | 2.6959 |
| Neutral | 2,257 | 1.0000 | 0.5590 |
| Positive | 1,060 | 1.0000 | 1.1903 |

Reports log effective weights, actual classifier parameters, model hashes,
code/split/config identities, and training membership digest. The saved balanced
model retains its configuration for trusted local loading.

## Results

| Validation metric | Unweighted | Balanced |
| --- | ---: | ---: |
| Accuracy | 0.7490 | 0.7645 |
| Macro F1 | 0.6696 | 0.7378 |
| Negative precision / recall | 0.9375 / 0.4348 | 0.7313 / 0.7101 |
| Neutral precision / recall | 0.7200 / 0.9697 | 0.8000 / 0.8350 |
| Positive precision / recall | 0.8140 / 0.4605 | 0.7021 / 0.6513 |

Balanced weighting improves macro F1 by 0.0683 and accuracy by 0.0154 on this
validation split. It recovers more negative/positive examples at lower precision;
neutral recall declines. It is stronger under the primary validation macro-F1
criterion, but calibration, uncertainty, and frozen-test generalization remain
unmeasured. No automatic default replacement is made.

Confusion matrices and provenance:
[unweighted](../reports/experiments/tfidf-logreg-phrasebank-unweighted.md) and
[balanced](../reports/experiments/tfidf-logreg-phrasebank-balanced.md).
