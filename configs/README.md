# Experiment configuration

Reserved for versioned, nonsecret experiment settings in later roadmap phases.
No sentiment or prediction experiment presets are defined. Runtime
settings use environment variables documented in `.env.example`.

`asset-revisions.json` records reviewed, pinned dataset/model identities and usage
scope decisions. It contains metadata only; null artifact checksums mean those
files have not been downloaded. The `local_dataset_assets` and `acquisition_files` fields now configure the explicit
dataset acquisition command. Weight metadata is never part of that allowlist.

`split-policy.json` defines the Phase 1 reversible filtering rules, fixed seed,
near-duplicate screen, and provisional partition weights. Supported changes are
validated explicitly. See `docs/split-policy.md`; this is not a final-test freeze
or a model experiment configuration.
