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

Reviewed financial RoBERTa/DeBERTa and base checkpoint sources are cached on
Windows under ignored `data/interim/transformer-checkpoint-cache/`. New base-model
sentiment artifacts use `data/interim/transformer-finetuned-models/<manifest-sha256>/`.
Every improved epoch is retained separately and the selected artifact appears
in its per-model report. Source and saved-file checksums are verified before
loading. Weights stay on Windows; Mac retains manifests and private predictions.
See `docs/transformer-comparison.md` for model identity, training scope and loading.

Adapted financial checkpoints and separate PhraseBank-only base models use the
same ignored transformer artifact directory. Their selected manifest hashes,
training corpus and loading instructions are in `docs/transformer-followup.md`.
PhraseBank runs start from original base sources; no SEntFiN-trained project
artifact is reused. Saved models contain tokenizer/configuration/weights without
optimizer state. All improved epochs remain on Windows.
