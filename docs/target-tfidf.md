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
