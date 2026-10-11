# Local model artifacts

Model weights and serialized artifacts are ignored by Git. Review model terms
before acquisition and redistribution; retain version and experiment metadata.

Phase 2 TF-IDF fitted pipelines currently live under ignored
`data/interim/tfidf-logreg-models/<artifact-sha256>/model.pkl`, tied to validation run
metadata. Vocabulary/coefficients have not been cleared for public redistribution.
See `docs/tfidf-logreg.md` for trusted local loading and reproduction.

Linear SVM pipelines use ignored
`data/interim/tfidf-svm-models/<artifact-sha256>/model.pkl`, with the same trusted
local/version/checksum boundaries. They return signed decision scores, not class
probabilities. See `docs/tfidf-svm.md` for the four validation experiments.

Phase 3 pretrained FinBERT files are cached on the Windows GPU computer under
ignored `data/interim/finbert-checkpoint-cache/`. The exact upstream checkpoint
is reused unchanged; no new fitted transformer is saved. SHA-256, architecture,
parameter/file sizes, and validation evidence are in
`reports/experiments/finbert-validation.json`; per-run acquisition/prediction
records remain ignored under `data/interim/finbert-validation/<run-sha256>/`.
See `docs/finbert.md` for use scope and reproduction.

Adapted FinBERT model/tokenizer artifacts live on Windows under ignored
`data/interim/finbert-finetuned-models/<manifest-sha256>/`. Every improved epoch
is stored separately; the training report selects the best manifest. Safetensors,
configuration and tokenizer hashes are verified before local inference loading.
No optimizer state is saved. See `docs/finbert-finetuning.md` for reproduction
and loading; derivative weights retain the documented private research scope.
