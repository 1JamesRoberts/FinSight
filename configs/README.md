# Experiment configuration

Reserved for versioned, nonsecret experiment settings in later roadmap phases.
No dataset, sentiment, or prediction presets are defined in Phase 0. Runtime
settings use environment variables documented in `.env.example`.

`asset-revisions.json` records reviewed, pinned dataset/model identities and usage
scope decisions. It contains metadata only; null artifact checksums mean those
files have not been downloaded. It is not an executable download configuration.
