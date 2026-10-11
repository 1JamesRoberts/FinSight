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
