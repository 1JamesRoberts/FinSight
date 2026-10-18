# TF-IDF + Linear SVM

Phase 2 bounded experiment, 2026-10-11: four separate models, one per dataset
and class-weight setting. Training and validation use the unchanged Phase 1
partitions. No pooling, FiQA classification, multi-target sentence gold, final-test
scoring, or automatic parameter search is introduced.

## How SVM works

TF-IDF represents a sentence as a sparse vector of word/phrase weights. A linear
SVM learns a boundary separating classes in that feature space, favoring a wide
margin while allowing violations. For a binary problem, its score is `w · x + b`.
The sign chooses a side of the boundary; the coefficients describe how features
move that score. See the [SVM classification guide](https://scikit-learn.org/1.7/modules/svm.html#classification).

Our implementation uses squared hinge loss: an example with label `y` in {-1, +1}
has margin penalty `max(0, 1 - y * score)^2`. Correct classification with a small
margin can still incur a penalty. L2 regularization restrains coefficient size;
C controls the tradeoff between regularization and margin violations. Higher C
puts more emphasis on fitting training labels. This experiment fixes C=1; it
does not establish an optimal value.

There are three classes, so `LinearSVC` learns three one-versus-rest boundaries:
negative versus the others, neutral versus the others, and positive versus the
others. Prediction evaluates all three scores and returns the highest-scoring
class. They are signed decision scores, not class probabilities; they need not
sum to one or lie in [0, 1]. No probability calibration is performed.
See [LinearSVC](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.LinearSVC.html).

Logistic Regression instead fits a probabilistic classification objective.
Sharing TF-IDF features makes the classifier comparison useful, but their losses,
multiclass strategies, and intercept treatment differ. Equal numerical C does
not establish equally optimized regularization for the two model families.

## Configuration and inputs

`configs/tfidf-svm.json` is unweighted; `configs/tfidf-svm-balanced.json` changes
only `class_weight` to `"balanced"`. Both fix C=1, min_df=2, max_features=50,000,
max_iter=10,000, and seed=42. Shared `TfidfConfig` validates both model families.
The shared vectorizer preserves exactly the LR feature policy: lowercase word
unigrams/bigrams, no stopword removal, sublinear TF, smoothed IDF, and L2 norm.
Vocabulary/IDF are fitted only on training texts, including with weighted models.

Code fixes L2 penalty, squared hinge loss, one-versus-rest multiclass handling,
dual=True, tolerance 1e-4, and an intercept with scaling 1. The fixed seed controls
the dual optimizer. Convergence warnings fail runs. Exact classifier/vectorizer
parameters, dependencies, iteration count, training label counts, and resolved
weights are logged. This uses the existing reviewed scikit-learn 1.7.2 dependency;
no new package or pretrained asset is acquired. Rights decisions are in
[datasets.md](../../datasets.md).

Balanced weights are derived from each cohort's training counts using
`n_training / (3 * n_class)` and passed as class-specific penalties to LinearSVC.
They do not alter TF-IDF vectors, validation labels, or row counts. Training-only
weights are approximately 2.6959/0.5590/1.1903 for PhraseBank and
1.1039/0.9802/0.9311 for single-target SEntFiN, in negative/neutral/positive order.
SEntFiN's training class distribution is much closer to balanced.

The CLI reads flags with `argparse`. `--config` supplies settings; `--data-dir`
defaults to `FINSIGHT_DATA_DIR` or `data`; `--freeze-receipt` pins existing local
artifacts. The verified loader returns training/validation records only.
`--cohort all` (default) trains two models separately; `phrasebank` or
`sentfin_single_target` selects one. Texts and labels are extracted in matching
order, then passed to `model.fit(train_texts, train_labels)`. For validation,
`model.predict(text)` receives text only; gold labels enter metrics afterward.
The [earlier input-flow guide](phrasebank-class-weights.md) describes the shared
record/config boundaries in detail.

## Run and inspect

From the repository root after Phase 1 local artifacts have been prepared:

```sh
uv sync --locked
uv run --locked python -m finsight.sentiment validate-tfidf-svm
uv run --locked python -m finsight.sentiment validate-tfidf-svm \
  --config configs/tfidf-svm-balanced.json \
  --report-path reports/experiments/tfidf-svm-balanced-validation.md
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

The first command trains the two unweighted models; the second trains the two
balanced models. These use 3,785 training / 518 validation PhraseBank sentences
and 6,299 training / 781 validation single-target SEntFiN headlines. All
multi-target headlines, FiQA, test candidates, and frozen groups are excluded.

Reports contain aggregate accuracy, fixed-three-class macro F1, per-class
precision/recall/F1, confusion matrices, training/inference timing, and controls.
VADER and training-majority controls use the same rows and training labels.
Each report's JSON includes model/config/code/lock/split/freeze hashes, parameters,
weights, environment, and local artifact paths. Execution before committing is
identified by its base revision, dirty state, and exact implementation digests.

Fitted pipelines live under ignored
`data/interim/tfidf-svm-models/<artifact-sha256>/model.pkl`. The runner reloads the
saved model before scoring. Per-record gold/labels/scores remain in ignored
`data/interim/tfidf-svm-validation/<run-id>/`; no source text is exported.
Only trusted locally generated pickle files should be loaded; checksums detect
changed bytes, and loading requires matching scientific-library versions.
Fitted vocabularies/coefficients are not cleared for public redistribution.

Example using the logged artifact:

```python
import json
from pathlib import Path
from finsight.sentiment.svm import TfidfSVM

run = json.loads(Path("reports/experiments/tfidf-svm-validation.json").read_text())
entry = run["metrics"]["phrasebank"]
model = TfidfSVM.load(
    Path("data") / entry["artifact_relative_path"],
    expected_sha256=entry["artifact_sha256"],
)
result = model.predict("The outlook is good.")
print(result.label, result.decision_scores)
```

## Validation findings

| Cohort | Class weighting | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| PhraseBank | None | 0.7799 | 0.7344 |
| PhraseBank | Balanced | 0.7819 | 0.7444 |
| Single-target SEntFiN | None | 0.8092 | 0.8072 |
| Single-target SEntFiN | Balanced | 0.8092 | 0.8075 |

PhraseBank weighting improves macro F1 by 0.0099, with negative recall increasing
from 0.5652 to 0.6232 and positive recall from 0.6316 to 0.6579. Negative/positive
precision falls. SEntFiN changes only slightly: macro F1 rises by 0.0003 with
unchanged accuracy; this is not evidence of a meaningful advantage.

Both settings outperform VADER and majority controls on these validation cohorts.
The earlier LR macro F1 was 0.6696 unweighted / 0.7378 balanced on PhraseBank
and 0.8165 unweighted on single-target SEntFiN. Balanced SVM is slightly above
balanced LR on PhraseBank; SEntFiN SVM is below that LR starting configuration.
These are descriptive development comparisons without uncertainty estimates,
broader tuning, calibrated probabilities, cross-corpus transfer, or final-test
generalization claims. Separate corpus-specific models are not target-aware.

Detailed [unweighted report](../../../reports/experiments/tfidf-svm-validation.md) and
[balanced report](../../../reports/experiments/tfidf-svm-balanced-validation.md) include
confusion matrices and class-level metrics. Latency is a single serial local pass
after an invented warm-up; it includes transformation/classification but excludes
loading/serialization. It is not a deployment benchmark.

Synthetic tests cover train-only features and LR parity, margin/class mapping,
balanced coefficients, seeded repeatability, local artifact round trips, wrong
model/checksum rejection, invalid inputs, convergence handling, both weighted
cohorts, holdout rejection, and output privacy. Remaining Phase 2 work is bounded
development-only tuning and qualitative error analysis; no future phase starts.

## Architecture and artifact size in reports

Reports now include the actual fitted feature dimension, three output classes,
coefficient/intercept shapes, stored classifier parameter count, fitted IDF count,
numeric array bytes, and exact serialized pipeline bytes/KiB. PhraseBank uses
12,351 features and a 3 x 12,351 coefficient matrix plus three intercepts
(37,056 stored classifier parameters). SEntFiN uses 10,475 features and
31,428 stored classifier parameters. Weighting changes fitted coefficients and
class penalties, not feature or parameter counts.

Classifier parameter counts exclude IDF values and vocabulary metadata. Numeric
array storage excludes Python object overhead; serialized pipeline size includes
vocabulary and serialization overhead. Neither measures runtime RAM. The profile
does not publish vocabulary strings or coefficients.

New validation runs include profiles automatically. Existing reports were enriched
from their original checksummed local artifacts, without retraining or prediction.
Original run IDs, code provenance, metrics, artifact hashes, and ignored run.json
files stay intact; aggregate JSON separately records profile extraction provenance.
To refresh an existing report from its local fitted artifacts:

```sh
uv run --locked python -m finsight.sentiment.model_profile \
  reports/experiments/tfidf-svm-validation.md \
  reports/experiments/tfidf-svm-balanced-validation.md
```
