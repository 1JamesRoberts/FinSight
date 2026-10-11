# TF-IDF + Logistic Regression baseline

Reviewed and implemented 2026-10-11. Phase 2 bounded task: train a first fixed
sentence classifier and evaluate development validation only. No tuning, SVM,
transformer, target-aware model, or final-test evaluation is included.

## How it works

[TF-IDF](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
turns each text into a sparse vector of word and two-word phrase weights. Repeated
terms increase their weight with logarithmic scaling; terms common across training
documents receive lower inverse-document-frequency weight. Smoothed IDF is
`log((1 + training_documents) / (1 + documents_with_term)) + 1`.
Each vector is normalized to unit L2 length.

[Logistic Regression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)
learns one weighted linear score per sentiment class, then converts the scores
into three class probabilities with softmax. Training adjusts weights to fit
the gold labels while L2 regularization limits large coefficients. The class
with the largest probability is returned. These are model probabilities, but
their calibration has not been evaluated: 0.8 does not establish 80% reliability.

Unlike VADER's hand-written sentiment rules, this model learns financial word
associations from labeled examples. It can learn phrases but has limited context
and cannot assign different sentiment to two companies in one headline.

## Initial choices and boundaries

Use scikit-learn 1.7.2, locked by `uv.lock`, under its
[BSD-3-Clause grant](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING).
The installed dependency notices are retained; no library sources are vendored.
The reviewed dataset terms support local noncommercial development training and
storage. Fitted vocabularies and coefficients remain ignored; publishing them
requires a separate rights review. See [asset decisions](datasets.md).

`configs/tfidf-logreg.json` fixes C=1, min_df=2 documents, max_features=50,000,
max_iter=1,000, and seed=42. Code fixes lowercase word unigrams/bigrams, no
stopword removal, sublinear term frequency, smoothed IDF, and L2 normalization.
The default tokenizer keeps tokens with at least two word characters, strips
punctuation, and preserves words such as “not.” There is no stemming or lemmatizing.
This is a deliberate starting point, not a validation-selected optimum.

The classifier uses multinomial logistic regression with L2 regularization,
lbfgs, tolerance 1e-4, intercept enabled, and no class reweighting. Convergence
warnings fail the run rather than silently reporting an unfinished fit. The seed
is recorded even though lbfgs does not use it. All estimator parameters and exact
scikit-learn/NumPy/SciPy versions are logged.

A scikit-learn pipeline binds the vectorizer and classifier, following its
[training-only preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).
`fit` learns vocabulary, document frequencies, IDF, and classifier coefficients
from training only. `predict` transforms with frozen features. Unknown words
are ignored; an entirely unknown text uses the learned intercepts. Empty text
fails. Training requires all three known labels; probabilities are always keyed
in negative/neutral/positive order rather than relying on integer class IDs.

The verified development loader checks the Phase 1 freeze and parent artifact
digests before returning training/validation rows. The sentence-cohort boundary
also rejects any holdout partition. Train separate models on 3,785 PhraseBank
sentences and 6,299 single-target SEntFiN headlines; evaluate their respective
518 and 781 validation texts. Do not pool corpora or assign a sentence gold label
to mixed-target headlines. FiQA has continuous scores and is excluded.
All frozen-test and other test-candidate groups remain withheld.

## Reproduce and use

From the repository root, after reconstructing the Phase 1 local data artifacts:

```sh
uv sync --locked
uv run --locked python -m finsight.sentiment validate-tfidf-logreg
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Options: `--data-dir`, `--freeze-receipt`, `--config`, `--report-path`, `--cohort`, and
`--vader-config` (defaults to the unchanged VADER baseline configuration).
The report at `reports/experiments/tfidf-logreg-validation.md` compares against
VADER rerun on exactly the same validation rows and a training-majority control.
Its companion JSON contains aggregate metrics, configurations, estimator settings,
training membership digest, model digests, parent/freeze identity, lockfile digest,
code revision, implementation digests, environment, and timing.

Models are stored under ignored `data/interim/tfidf-logreg-models/<artifact-sha256>/model.pkl`.
The aggregate JSON gives each model's relative path and SHA-256. The command
reloads each saved model before validation, so reported predictions use the
persisted pipeline. Per-record gold/predictions/probabilities and complete run
metadata stay in ignored `data/interim/tfidf-logreg-validation/<run-id>/`.
No source text is exported. Run IDs include timing, so repeated runs differ;
predictions and classification metrics should match with the same environment.
Serialized bytes can differ across fits; content-addressed storage keeps each
run's exact artifact without overwriting earlier models.
Execution before a commit records the base revision, dirty state, and exact
implementation hashes rather than claiming that the base revision contains it.

Load a **trusted locally generated** artifact using its logged digest:

```python
import json
from pathlib import Path
from finsight.sentiment.tfidf import TfidfSentiment

run = json.loads(Path("reports/experiments/tfidf-logreg-validation.json").read_text())
entry = run["metrics"]["phrasebank"]
model = TfidfSentiment.load(
    Path("data") / entry["artifact_relative_path"],
    expected_sha256=entry["artifact_sha256"],
)
prediction = model.predict("The outlook is very good!")
print(prediction.label, prediction.probabilities)
```

Pickle can execute code; load only your own trusted files. Digest verification
detects changed bytes but does not establish trust. Loading requires matching
scientific-library versions. Keep the matching source code and environment;
cross-version model portability is not promised. No new serialization dependency
or experiment framework is needed for this bounded baseline.

## Findings and limitations

Validation accuracy / macro F1: PhraseBank **0.7490 / 0.6696**, single-target
SEntFiN **0.8182 / 0.8165**. Corresponding VADER macro F1 is 0.4895 / 0.4787,
and training-majority macro F1 is 0.2429 / 0.1769. These results improve on both
controls on these cohorts. PhraseBank's neutral recall is 0.9697, but negative
and positive recall is only 0.4348 / 0.4605; aggregate accuracy hides that weakness.
The [full report](../reports/experiments/tfidf-logreg-validation.md) has class metrics
and confusion matrices. Timing is serial batch size one after an invented warm-up;
it includes feature transformation and classification but excludes model loading
and serialization. This is one local pass, not a deployment benchmark.

Separate corpus-specific training prevents interpreting these scores as transfer
performance. We have not evaluated target-aware behavior, probability calibration,
final-test generalization, or statistical uncertainty. Corpus coverage and
annotation limitations from the dataset cards still apply. Linear SVM, bounded
development-only hyperparameter comparison, and qualitative error analysis are
remaining Phase 2 tasks. Synthetic tests check train-only vocabulary/IDF,
negation/bigrams, probabilities/labels, repeatability, artifact round trips,
configuration validation, holdout rejection, CLI behavior, and output privacy.

## Subsequent class-weight experiment

`class_weight` now accepts `null` (default) or `"balanced"`. The original preset
remains unweighted. `configs/tfidf-logreg-balanced.json` changes only that setting.
`--cohort phrasebank` runs just PhraseBank; the default `all` still fits the two
corpus models separately. See [class-weight comparison and input flow](phrasebank-class-weights.md)
for the reproducible commands, training-only weights, and results. Balanced
PhraseBank validation macro F1 is 0.7378, versus 0.6696 unweighted; this bounded
comparison does not complete broader hyperparameter tuning or final evaluation.
