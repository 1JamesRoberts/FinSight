# Matched TF-IDF + Logistic Regression target baselines

Both arms fit the same 11,547 target annotations from 8,598 training headlines.
One fixed CPU fit per arm: C=1.0, L2 multinomial logistic regression, lbfgs, seed 42, no class weights or search. Validation only.

| Cohort | Targets | Headline LR F1 % | Headline + target LR F1 % | Saved DeBERTa F1 % |
| --- | ---: | ---: | ---: | ---: |
| all_targets | 1410 | 75.20 | 74.94 | 80.58 |
| single_target | 781 | 81.15 | 77.82 | 88.10 |
| multi_target | 629 | 67.37 | 70.67 | 71.22 |
| mixed_label_headlines | 283 | 48.13 | 61.84 | 50.15 |
| seen_native_target | 1147 | 75.51 | 75.15 | 80.30 |
| unseen_native_target | 263 | 73.84 | 74.05 | 81.74 |

The saved DeBERTa column is a three-seed mean, previously trained on 6,299
single-target headlines. Its data/update budget differs; this is context,
not a controlled model-architecture comparison. LR arms share all training rows.

| Arm | Features (headline + target) | Coefficients shape | Learned parameters | Artifact bytes |
| --- | ---: | --- | ---: | ---: |
| headline_only | 24985 + 0 | [3, 24985] | 74958 | 1595706 |
| headline_target | 24985 + 5276 | [3, 30261] | 90786 | 1932304 |

Learned parameter counts include classifier coefficients and intercepts; fitted
IDF values and vocabulary are stored separately in the saved artifact.
Headline vocabulary/IDF hashes match across arms. Each TF-IDF block uses
word unigrams/bigrams, lowercase, sublinear TF, smoothed IDF and L2 norm.
Headline min_df=2, max_features=50,000; target min_df=1, max_features=10,000.
Both are fitted only on training annotations, including repeated headlines.
The blocks are concatenated at scale 1 with no joint renormalization.

| Arm | All-target accuracy % | NLL | Brier | ECE | Zero target rows (validation) |
| --- | ---: | ---: | ---: | ---: | ---: |
| headline_only | 75.25 | 0.6468 | 0.3645 | 0.0863 | N/A |
| headline_target | 75.46 | 0.5905 | 0.3310 | 0.0723 | 56 |

## Interpretation and limitations

The target block adds target-specific class offsets, not target/headline
interactions. It can learn entity sentiment tendencies; it cannot reliably
bind individual sentiment expressions to their entity. Changes in predictions
do not demonstrate contextual aspect understanding. Native target strings
are not verified company identities, and unseen strings may share word features.
Unseen tokens are ignored; a fully zero target block leaves headline evidence
plus the intercept from the target-trained model, not the headline-only model.
Default word tokenization drops punctuation and single-character tokens.

headline_only: different predicted labels across targets in 0 validation headlines, including 0 mixed-label headlines.
headline_target: different predicted labels across targets in 96 validation headlines, including 61 mixed-label headlines.

Metrics weight target annotations equally. Cohorts overlap. Probabilities are raw;
ECE uses ten equal-width top-confidence bins. No seed-variation estimate is
appropriate for one deterministic fit. No confidence interval or superiority
claim is made from this reused validation set. Reserved tests remain untouched.
Full per-class metrics, confusion matrices, calibration bins and timings are
in the companion JSON. Both private models reproduce all probabilities after reload.
Native inputs, fitted vocabularies/weights and row predictions stay ignored/private.

Reproduce from a clean checkout with the existing private data and absent outputs:

```sh
uv run --locked python -m finsight.sentiment.target_tfidf_experiment
```

The runner refuses existing published reports. Inspect saved results for routine use.
Data manifest: `be33e779e278b2c786c48ee3205cbad4f9e6bee98839d4c3b6509009a51a856c`.
Configuration SHA-256: `1fdb30558779a9cb0ea79aa481c742f772f30a0874b0bd3deef49476a3af791a`.
Execution revision: `98e21d2fd2df29508ab62c6369592442f57791da`.
