# Sentiment code guide

The sentiment package is organized by responsibility. Start with an experiment
when reading a complete workflow, then follow its calls into models, training,
evaluation, and artifacts. The refactor does not change the experimental recipes.

```text
src/finsight/sentiment/
├── models/
│   ├── finbert.py                 # Pinned FinBERT loading, prediction, label mapping
│   ├── roberta.py                 # RoBERTa classification-head loading rules
│   ├── deberta.py                 # DeBERTa head rules and custom financial adapter
│   ├── transformer.py             # Shared checked loading and sentence prediction
│   ├── target.py                  # Fresh backbones, target-pair collation, prediction
│   ├── vader.py                   # Lexicon baseline
│   ├── tfidf.py                   # TF-IDF + Logistic Regression
│   ├── svm.py                     # TF-IDF + Linear SVM
│   ├── target_tfidf.py            # Sparse target-aware baseline
│   └── saved_inference.py         # Prediction from locked local artifacts
├── training/
│   ├── config.py                 # FineTuneConfig and EarlyStopping
│   ├── sentence.py               # Shared sentence-classifier optimization
│   └── target.py                 # Shared headline-only / target-pair optimization
├── experiments/
│   ├── sentence_comparison.py    # Sentence transformer comparison
│   ├── target_comparison.py      # Matched RoBERTa / FinBERT target comparison
│   ├── target_deberta.py         # Original matched DeBERTa target study
│   ├── target_error_analysis.py  # Verified saved predictions and review packets
│   ├── final_evaluation.py       # Receipt-bound final evaluation runner
│   └── ...                       # Baseline, tuning, calibration and voting studies
├── evaluation/
│   ├── metrics.py                # Accuracy, macro F1 and confusion matrices
│   ├── calibration.py            # NLL, Brier, ECE and temperature primitives
│   ├── cohorts.py                # Development sentence selection and labels
│   ├── voting.py                 # Soft / hard voting rules
│   ├── final_metrics.py          # Locked-cohort metrics and paired bootstrap
│   ├── model_profile.py          # Linear-model architecture and artifact size
│   └── protocol.py               # Evaluation-plan and artifact verification
├── artifacts.py                 # Hashes, manifests, immutable saves, source audits
├── companies.py                 # Explicit company identity and alias registry
├── __main__.py                  # Existing baseline command interface
└── _compat/                     # Small aliases for historical flat module names
```

## Follow one training run

For the matched target study, read these functions in order:

1. `experiments/target_comparison.py:run_trial` verifies the fixed plan, prepared
   development examples, model asset and seed. It refuses to reuse a completed
   run whose configuration or implementation differs.
2. `models/target.py:load_backbone` loads the original local checkpoint. RoBERTa
   receives a new sentiment head; FinBERT retains its trained head with semantic
   output rows reordered. `load_fresh` is also used by the DeBERTa study.
3. `training/target.py:fit_target` validates partition boundaries, batches training
   examples, computes cross-entropy gradients and updates weights with AdamW.
   Both input arms use this same loop and early-stopping rule.
4. `models/target.py:collate_examples` calls the shared tokenizer helper. Pair mode
   supplies **target as sequence one and headline as sequence two**. The tokenizer
   adds its own separators; no literal `[SEP]` string is manually inserted. Only
   the headline may be truncated. Headline-only mode supplies one sequence.
5. `models/target.py:score_examples` disables dropout and gradients. Labels remain
   evaluation metadata and are never passed as inference inputs.
6. `evaluation/metrics.py:classification_metrics` measures validation performance.
   A strict macro-F1 improvement saves a checkpoint through `artifacts.save_model`;
   ties retain the earlier epoch. The experiment reloads the selected checkpoint
   and verifies that predictions reproduce the selected validation metrics.

Sentence studies follow the same separation, using
`experiments/sentence_comparison.py`, `training/sentence.py:fit_checkpoint`, and
`models/transformer.py`. Backbone loading and artifact saving are existing callable
arguments to the shared loop. RoBERTa and DeBERTa reuse checked Hugging Face loading
and scoring rather than duplicating those implementations in each model file.

TF-IDF receives raw training strings and fits its vocabulary, IDF values and
classifier together in a scikit-learn pipeline. Prediction uses the fitted
pipeline without refitting preprocessing. Its implementation remains in
`models/tfidf.py` and `models/svm.py`.

## Imports and commands

New code should import from the defining subpackage:

```python
from finsight.sentiment.training.config import FineTuneConfig
from finsight.sentiment.training.sentence import fit_checkpoint
from finsight.sentiment.training.target import fit_target
from finsight.sentiment.evaluation.metrics import classification_metrics
from finsight.sentiment.evaluation.calibration import calibration_metrics
```

Existing notebooks, trusted local pickle class references and flat imports remain
supported. For example, `finsight.sentiment.finbert` resolves to
`finsight.sentiment.models.finbert`. Compatibility aliases contain no duplicate
model or training implementations. A small legacy lookup in `training/sentence.py`
retains the historical FinBERT orchestration names; new callers import those from
`experiments/finbert_training.py`.

From the repository root, these commands show usage without training or scoring:

```bash
uv sync --locked
uv run --locked python -m finsight.sentiment --help
uv run --locked python -m finsight.sentiment.experiments.sentence_comparison --help
uv run --locked python -m finsight.sentiment.experiments.target_comparison --help
uv run --locked python -m finsight.sentiment.experiments.target_error_analysis --help
```

The old `python -m finsight.sentiment.target_backbones --help` command still works.
Actual transformer execution requires the existing optional installation:
`uv sync --locked --extra transformer`. Checkpoint, dataset and experiment
configuration paths remain relative to the repository root. Historical reruns
should use the recorded Git revision and reviewed local assets.

## Recorded experiments survive the move

Frozen plans, source data, weights, predictions and published reports are preserved.
Their old source paths and hashes describe the code at execution time. Read-only
historical target analysis verifies those bytes with
`git show <recorded-full-commit>:<original-source-path>` and separately checks the
current data/plan identity. This requires the recorded commits in local Git history;
a shallow clone may need its history fetched before auditing old runs.

New training runs record canonical repository-relative paths and SHA-256 hashes
for the complete Python source tree, including extracted helpers. Training resume
continues to reject a different implementation. The new layout must not be
presented as the code that originally trained an old checkpoint. To rerun a study
under changed code, declare a new output namespace/protocol instead of overwriting
published results. This refactor itself performs no research training, downloads,
new inference or frozen-test evaluation.

## Verification

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv build
```

Tests include synthetic tiny-transformer optimization, checkpoint reload and
validation-gradient checks when the transformer extra is installed. Additional
layout tests cover legacy import identity, old pickle class lookup, full source
manifests and historical source verification, including rejection of altered
hashes, unsafe paths and invalid revisions.
