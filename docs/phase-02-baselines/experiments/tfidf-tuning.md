# Bounded TF-IDF tuning

This Phase 2 search compares Logistic Regression and Linear SVM on separate
PhraseBank and single-target SEntFiN development cohorts. It changes only `C`
and class weighting. `configs/tfidf-tuning.json` declares the ordered grid:
`C = 0.1, 1, 10`, each with `class_weight = null, balanced`.
That gives six candidates per classifier per corpus: 24 fitted pipelines.

## Run it

From the repository root, with the already prepared, checksummed local datasets:

```bash
uv sync --locked
uv run --locked python -m finsight.sentiment.tuning
```

To use another local data location or a separate output report:

```bash
uv run --locked python -m finsight.sentiment.tuning \
  --data-dir /absolute/path/to/data \
  --plan configs/tfidf-tuning.json \
  --freeze-receipt configs/benchmark-freeze.json \
  --report-path reports/experiments/my-tuning.md
```

The command reads the plan and the existing `configs/tfidf-logreg.json` and
`configs/tfidf-svm.json` base settings. Each candidate copies its base, replacing
only `c` and `class_weight`. Seed remains 42; LR uses 1,000 iterations and SVM
10,000. TF-IDF keeps word unigrams/bigrams, `min_df=2`, at most 50,000 features,
sublinear TF, smoothed IDF, L2 normalization, and the existing tokenization.
Invalid grids fail before training. The grid must contain the C=1 unweighted
baseline; duplicate candidates are rejected.

The verified development loader returns only training/validation rows. The
existing experiment runner selects the corpus, extracts a list of training texts
and a matching list of labels, and fits the complete TF-IDF/classifier pipeline.
Balanced weights derive from that corpus's training counts. Validation texts
are transformed with the fitted vocabulary and IDF. Each saved model is reloaded
before scoring; validation labels never enter `fit`.

## Selection and outputs

For each corpus, select the largest validation macro F1 separately for LR and
SVM, then across both classifiers. Exact ties retain declared order: LR before
SVM, then the plan's C/weight order. Accuracy and latency are reported but do not
break ties. VADER and the training-majority baseline score the same validation
rows. Membership, split, lockfile, and control metrics must agree across candidates.

The public [comparison report](../../../reports/experiments/tfidf-tuning.md) and JSON
include all candidates, selected original run IDs, class metrics, confusion
matrices, architecture/size, descriptive timing, and complete provenance.
`hyperparameters_tuned=true` describes the search; individual candidate logs
retain `false` because each runner invocation fits one fixed configuration.

Candidate configs/reports are under ignored `data/interim/tfidf-tuning/`.
Original run logs and predictions remain under the existing ignored
`tfidf-logreg-validation/` or `tfidf-svm-validation/` directories. Fitted models
remain in their content-addressed `tfidf-logreg-models/` or `tfidf-svm-models/`
directories. The summary's candidate metrics provide artifact paths/checksums.
Original public baseline reports are preserved. No weights or source text are
committed, and no default inference configuration is silently replaced.

## Why keep training sets separate?

PhraseBank contains financial sentences with its own agreement-based annotation
scheme; SEntFiN contains company-targeted headlines. Restricting SEntFiN to one
target enables sentence classification, but does not make the annotation schemes
or text distributions identical. Separate fits let us attribute changes to C and
weights while keeping earlier comparisons meaningful. Pooling is an optional
future experiment requiring an explicit hypothesis, cross-corpus leakage checks,
and a sampling/weighting policy. It is not needed to tune these baselines.

## Interpretation limits

Larger C weakens regularization; smaller C strengthens it. Balanced weighting
changes the penalty for class errors and may trade precision for recall.
Neither necessarily improves performance. The selected score has benefited from
repeated validation inspection; it is a development result, not an unbiased
estimate of final generalization. This bounded search uses the existing validation
partition, not cross-validation, and reports no statistical significance claims.
The immutable final test stays withheld. Local timing varies across runs.
The subsequent [qualitative validation-error review](../../../reports/experiments/phase2-error-analysis.md)
and [consolidated Phase 2 comparison](../../../reports/experiments/phase2-comparison.md)
complete those separate tasks without changing model selection. No FinBERT or
prediction work is started here.
