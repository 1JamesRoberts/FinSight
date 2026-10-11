# Local model artifacts

Model weights and serialized artifacts are ignored by Git. Review model terms
before acquisition and redistribution; retain version and experiment metadata.

Phase 2 TF-IDF fitted pipelines currently live under ignored
`data/interim/tfidf-logreg-models/<artifact-sha256>/model.pkl`, tied to validation run
metadata. Vocabulary/coefficients have not been cleared for public redistribution.
See `docs/tfidf-logreg.md` for trusted local loading and reproduction.
