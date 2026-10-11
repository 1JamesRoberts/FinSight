# VADER baseline

Reviewed 2026-10-11. Phase 2, first bounded task: rule-based inference and a
validation-only sanity check. No training or threshold search is performed.

## Dependency and asset decision

Use standalone `vaderSentiment==3.3.2` from
[PyPI](https://pypi.org/project/vaderSentiment/3.3.2/), with locked distribution
hashes in `uv.lock`. The upstream [README](https://github.com/cjhutto/vaderSentiment)
explicitly covers its sentiment lexicon and code as open-source resources, and
the [MIT notice](https://github.com/cjhutto/vaderSentiment/blob/master/LICENSE.txt)
permits use/modification subject to retaining the notice. Project decision:
local inference/research with the packaged code, sentiment lexicon, and emoji
mapping is supported. Retain the installed distribution's license; cite
Hutto and Gilbert (2014), *VADER: A Parsimonious Rule-based Model for Sentiment
Analysis of Social Media Text*. No lexicon or upstream evaluation datasets are
copied into the repository. This grant does not change our news-data rights.

The standalone wheel bundles inference resources; NLTK and a network lexicon
download are unnecessary. Its declared `requests` dependency is retained by the
package resolver; our inference does not use translation services or the demo.
The experiment records package version and exact code/lexicon/emoji digests.

## How it works

VADER starts with human-rated lexical sentiment strengths. It adjusts nearby
sentiment using negation, intensifiers/dampeners, mixed-case capital emphasis,
punctuation, and contrast around “but.” It aggregates adjusted valence and
normalizes it with `s / sqrt(s*s + 15)` into a compound score in [-1, 1].
The [implementation](https://github.com/cjhutto/vaderSentiment/blob/master/vaderSentiment/vaderSentiment.py)
contains these rules; FinSight uses the packaged implementation unchanged.

The returned positive/neutral/negative proportions describe lexical sentiment
composition, not calibrated probabilities that a gold class is correct. Use
the compound score to classify according to the
[published scoring convention](https://github.com/cjhutto/vaderSentiment#about-the-scoring):

| Compound score | FinSight class |
| --- | --- |
| <= -0.05 | negative |
| > -0.05 and < 0.05 | neutral |
| >= 0.05 | positive |

Both endpoints are inclusive for the polar classes. `configs/vader.json` makes
the decision rule explicit. These defaults are a first baseline, not established
financial-domain optima. Future changes require recorded development-only tuning.
Input text is preserved: do not lowercase, remove punctuation, remove “not,”
or strip stopwords before VADER. Those cues affect its rules. Empty inputs fail.

## Limits

This is sentence-wide sentiment, not target-aware investor-impact sentiment.
General-language valence can differ from financial meaning: lower costs may
benefit investors even if individual words sound negative. Sarcasm, financial
jargon, and relationships between companies can fail. There is no fitted
vocabulary or training step, and no claim that rules understand company targets.

The development check uses only PhraseBank validation and single-target SEntFiN
validation. FiQA's continuous scores and multi-target SEntFiN are outside this
three-class sentence control. All frozen/candidate groups remain withheld.

## Use and reproduce

```sh
uv sync --locked
uv run --locked python -m finsight.sentiment validate-vader
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Run from the repository root after reconstructing the Phase 1 local artifacts.
The command accepts `--data-dir`, `--freeze-receipt`, `--config`, and `--report-path`.
It writes `reports/experiments/vader-validation.md` and aggregate JSON. Source
IDs/gold/scores remain in ignored `data/interim/vader-validation/<run-id>/` with
the full local run configuration. Latency varies, so repeated runs have different
run IDs while deterministic predictions/metrics should match. Resource hashes,
parent split/freeze identity, code revision, working-tree state, implementation
file hashes, environment, class order, and the threshold rule make runs traceable.
A dirty implementation run is identified by its file hashes, not just its base
Git revision. One invented warm-up and serial batch-size-one scoring are timed;
loading and serialization are excluded. This is not a deployment latency study.

Inference is a plain typed Python interface:

```python
from finsight.sentiment.vader import VaderSentiment

prediction = VaderSentiment().predict("The outlook is very good!")
print(prediction.label, prediction.compound)
```

Construct the analyzer once and reuse it. No learned text preprocessing is
introduced. Configurations reject unsupported fields and invalid thresholds;
scores reject nonfinite/out-of-range values. Synthetic tests cover boundary
labels, negation/emphasis, repeated prediction, metrics, cohort exclusion,
training-majority selection, and output privacy without dataset downloads.

The first validation results are descriptive: VADER macro F1 is 0.4895 on
518 PhraseBank sentences and 0.4787 on 781 single-target SEntFiN headlines.
Corresponding training-majority macro F1 values are 0.2429 and 0.1769. VADER's
PhraseBank accuracy (0.5560) is below the neutral-majority control (0.5734).
It improves macro F1 over these controls but does not establish strong performance
or independent final-test generalization. Thresholds remain unchanged. Broader
model comparisons and qualitative validation error analysis are later tasks.
