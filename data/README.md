# Local research artifacts

Only this README and placeholders belong in Git; actual files remain ignored.
Reviewed PhraseBank and SEntFiN source artifacts live in `raw/`, with a pinned
acquisition receipt. `interim/dataset-integrity.json` holds local validation
counts. See `docs/dataset-loading.md` for reproducible acquisition and loading,
usage scope, and limitations. Provisional splits and lineage live under
`processed/splits/`; the curated final SEntFiN holdout lives under
`processed/benchmarks/`. Both use content-addressed directories. The committed
`configs/benchmark-freeze.json` receipt pins the official release; use the verified
development loader described in `docs/evaluation-contract.md` for training and
validation. Do not load frozen or other test-candidate records for development.
