# Experiment configuration

Reserved for versioned, nonsecret experiment settings in later roadmap phases.
No sentiment or prediction experiment presets are defined. Runtime
settings use environment variables documented in `.env.example`.

`asset-revisions.json` records reviewed, pinned dataset/model identities and usage
scope decisions. It contains metadata only; null artifact checksums mean those
files have not been downloaded. The `local_dataset_assets` and `acquisition_files` fields now configure the explicit
dataset acquisition command. Weight metadata is never part of that allowlist.
