# Experiment configuration

Reserved for versioned, nonsecret experiment settings in later roadmap phases.
Sentiment presets are documented below; no prediction presets are defined. Runtime
settings use environment variables documented in `.env.example`.

`asset-revisions.json` records reviewed, pinned dataset/model identities and usage
scope decisions. It contains metadata only; null artifact checksums mean those
files have not been downloaded. The `local_dataset_assets` and `acquisition_files` fields now configure the explicit
dataset acquisition command. Weight metadata is never part of that allowlist.

`split-policy.json` defines the Phase 1 reversible filtering rules, fixed seed,
near-duplicate screen, and provisional partition weights. Supported changes are
validated explicitly. See `docs/split-policy.md`; this is not a final-test freeze
or a model experiment configuration.

`benchmark-policy.json` declares holdout curation, cohorts, label order, and
descriptive coverage-name checks. `benchmark-freeze.json` is the authoritative
receipt pinning the final local membership/labels and its parent split by digest.
Neither contains source headlines. Preserve the receipt; do not alter the seed,
membership, or labels after inspecting results. See `docs/evaluation-contract.md`.

`vader.json` declares the Phase 2 baseline's inclusive compound-score boundaries
(-0.05 negative, +0.05 positive). It does not tune them or alter the frozen split.
See `docs/vader.md` for inference and the validation-only check.

`tfidf-logreg.json` records the first fixed Logistic Regression configuration,
including regularization, document-frequency floor, feature cap, iteration limit,
and seed. Preprocessing defaults and all estimator parameters are also logged
in each run. It performs no parameter search. See `docs/tfidf-logreg.md`.

`tfidf-logreg-balanced.json` changes only `class_weight` to `"balanced"`.
The original preset omits it and remains unweighted (`null` is also accepted).
Use `--cohort phrasebank` with the balanced preset for the bounded PhraseBank
comparison; without a cohort selector, the settings apply to both separate
models. See `docs/phrasebank-class-weights.md` for inputs and results.

`tfidf-svm.json` / `tfidf-svm-balanced.json` use the shared TF-IDF configuration
schema with unweighted / balanced LinearSVC. Both use C=1 and max_iter=10,000.
The two presets otherwise match. `validate-tfidf-svm` runs both separate sentence
cohorts by default; `--cohort` can select one. See `docs/tfidf-svm.md`.

`companies.json` is the Phase 4 versioned registry for explicit company targets.
It records the five project identities, observed share tickers, curated alias
rules and source URLs. It is separate from the frozen Phase 1 coverage-name
configuration. Bare ambiguous names and Google parent grouping require explicit
caller policies. No product aliases or free-text NER rules are implied. See
`docs/company-targets.md` for the typed lookup API and limitations.

`target-sentiment-plan.json` fixes native target/text input policy, existing
train/validation lineage, source tokenizer/model, saved headline controls,
matched training arms, seeds and metrics. `target-sentiment-lock.json` binds it
to verified private example files and tokenizer-preparation results. The lock
means **prepared, not trained**; it authorizes no reserved-test evaluation.
See `docs/target-sentiment-protocol.md`.

`target-tfidf.json` fixes a separate Phase 4 traditional baseline experiment on
the same native annotation view. It pins existing data/control identities, the
headline-only/additive-target arms, C=1/unweighted lbfgs, feature caps and one CPU
thread. It does not modify the transformer comparison plan or preparation lock.
No search or reserved-test evaluation is performed. See `docs/target-tfidf.md`.
