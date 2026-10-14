# Target-level TF-IDF Logistic Regression baselines

This Phase 4 increment compares headline-only features with additive headline
and native-target features. Both arms use the same existing 11,547 SEntFiN
training annotations from 8,598 headlines and 1,410 validation annotations from
1,053 headlines. There is no PhraseBank combination, new split, five-company
filter, label change, masking, class weighting or test evaluation.

## Inputs and features

`target_control.verified_inputs` reconstructs the native train/validation view and
checks it against both locked private JSONL files. The runner selects training
examples by their existing partition, then builds aligned lists:

```python
headlines = [example.headline for example in training]
targets = [example.target for example in training]
labels = [example.label for example in training]
```

A multi-target headline appears once per annotation, with its original target
and gold label. Training rows have equal weight; more targets mean more weight
for that headline. Both headline vectorizers fit the same repeated headline list,
so their vocabulary and IDF must match. No validation text fits either vectorizer.

- Headline-only: word unigram/bigram TF-IDF, then Logistic Regression.
- Headline + target: the identical headline TF-IDF block plus a separately fitted
  word unigram/bigram TF-IDF block for native entity names, concatenated sparsely.

Each block is L2 normalized independently, with scale 1 and no normalization
after concatenation. No `[SEP]` or manually inserted target prefix is needed.
The existing vectorizer policy lowercases text, uses sublinear TF/smoothed IDF,
keeps stop words, and tokenizes words of at least two characters. There is no
stemming, lemmatization, alias normalization or manually added vocabulary.
Headline min_df=2/max_features=50,000; target min_df=1/max_features=10,000.
These settings are fixed in `configs/target-tfidf.json` before training.

Both classifiers use multinomial L2 Logistic Regression, C=1, lbfgs, tolerance
1e-4, maximum 1,000 iterations, seed 42, no class weights. A convergence warning
fails the run. One CPU thread and one deterministic fit per arm replace repeated
seed runs. No hyperparameter search, model selection or probability calibration.

## Meaning and limits

For each class, the target model adds headline evidence, target evidence and an
intercept. There are no target/headline interaction terms. It can learn entity
sentiment priors and flip labels between targets, but this does not establish
which sentiment expression refers to which entity. Target-pair transformers are
a later comparison. Adding the block also changes the feature/regularization
geometry, even though C and the headline features are shared.

Unseen words are ignored. Some unseen native target strings still have known
word/ngram features; others become all-zero target blocks. A zero block uses the
headline coefficients of the target-trained model, not the separate headline-only
classifier. Zero-feature counts are logged and no examples are dropped. Native
string novelty does not prove unseen-company identity. These financial entities
are not all verified companies, and five-company coverage remains sparse.

## Run and use saved models

With the existing reviewed private dataset and no published outputs:

```sh
uv run --locked python -m finsight.sentiment.target_tfidf_experiment \
  --data-dir data --config configs/target-tfidf.json
```

`--data-dir` selects the existing private data root. `--config` selects the fixed
experiment settings; it does not accept training text on the command line. The
runner loads the prepared view, fits using training rows only, saves both models,
reloads them and checks exact validation-probability equality before reporting.
It refuses existing public reports; inspect saved results for routine use.

Public aggregate reports: `reports/experiments/target-tfidf-logreg.json` and `.md`.
Private fitted models: `data/interim/target-tfidf-models/<artifact-sha>/model.pkl`.
Private predictions/run: `data/interim/target-tfidf-logreg/<report-sha>/`.
Fitted vocabularies and row-level results remain private alongside the data.

To use an existing model locally:

```python
import json
from pathlib import Path
from finsight.sentiment.target_tfidf import TargetTfidf

run = json.loads(Path("reports/experiments/target-tfidf-logreg.json").read_bytes())
arm = run["models"]["headline_target"]
model = TargetTfidf.load(
    Path("data") / arm["artifact_relative_path"],
    expected_sha256=arm["artifact_sha256"],
)
predictions = model.predict_many(
    ["Apple gains while Microsoft falls."] * 2,
    ["Apple", "Microsoft"],
)
```

The two lists must be nonempty and aligned. Predictions include named class
probabilities; this example does not promise correct entity-specific predictions.
Loading is for trusted local model artifacts with matching dependency versions.

Reports include target-weighted accuracy, three-class macro F1, class metrics,
confusion matrices, raw-probability NLL/Brier/ten-bin ECE and all six established
validation cohorts. Model size includes vocabulary/feature counts, coefficient
shape, classifier parameter count, IDF count, coefficient bytes and total artifact
bytes. Timings are a single local CPU pass, not a hardware benchmark. The saved
DeBERTa column provides context: it used a different, single-target training view.
All results remain exploratory development validation; reserved tests stay frozen.

## Verified results (2026-10-14)

The [comparison report](../reports/experiments/target-tfidf-logreg.md) records the
clean execution at `98e21d2fd2df29508ab62c6369592442f57791da`. Public JSON SHA-256:
`05a7e2ce603ae002499a6b9c784c6be3cac61cf8e9513cd02f1ee8269e26a962`.
Both models were trained on the Mac CPU; the fits took approximately 0.33/0.56
seconds and converged in 44/72 iterations. Timing is illustrative, not a benchmark.

| Validation cohort | Headline-only macro F1 | Headline + target macro F1 |
| --- | ---: | ---: |
| All targets | 75.20% | 74.94% |
| Single-target | 81.15% | 77.82% |
| Multi-target | 67.37% | 70.67% |
| Mixed-label headlines | 48.13% | 61.84% |

The target block raises mixed-label F1 by 13.72 percentage points and multi-target
F1 by 3.30 points, while reducing single-target F1 by 3.32 points. Overall F1 falls
by 0.26 points. Raw-probability NLL/Brier/ECE improve from 0.6468/0.3645/0.0863
to 0.5905/0.3310/0.0723. These are descriptive validation differences without
uncertainty estimates, not evidence of a globally superior target model.

The target model produces different labels among a headline's targets in 96 of
272 multi-target headlines, including 61 of 122 mixed-label headlines. Target
word features may capture useful label tendencies, but the architecture still
cannot bind arbitrary sentiment expressions to their respective entities.
There are 56 all-zero validation target blocks; those examples are retained.

The headline model has 24,985 features, 74,958 classifier parameters and a
1,595,706-byte artifact. The target model has 30,261 features, 90,786 classifier
parameters and a 1,932,304-byte artifact. Both models are saved privately.
Every validation probability is identical after trusted reload. A separate audit
reconstructs prediction bytes/hashes and checks all six cohorts against independent
scikit-learn F1/accuracy/confusion/NLL and NumPy Brier calculations.
All 472 tests and Ruff lint/format gates pass. The original final aggregate,
source freeze and immutable transformer input/comparison plan remain unchanged.

The next bounded transformer task is matched headline-only DeBERTa training,
followed by the target-pair arm under the already-fixed comparison plan.

The code/reports and both byte-identical private fitted models are also synced
to the existing Windows checkout. Windows reconstructs the stored target rows
from its own locked native view and exported IDs/probabilities; all prediction
hashes and cohort classification metrics match. Loaded-model predicted labels
are identical across machines, with maximum probability differences below
3.34e-16. All 472 tests and required Ruff checks also pass there. No Windows
retraining was needed, and temporary transfer helpers were removed afterward.
