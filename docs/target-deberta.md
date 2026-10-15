# Matched DeBERTa training for native targets

Phase 4 trains two fresh arms under the unchanged
[target-input protocol](target-sentiment-protocol.md): headline-only versus
native target first / headline second. Each arm uses exactly 11,547 training
annotations and 1,410 validation annotations from the existing prepared view.
There is no PhraseBank combination or five-company filtering.

The source is the cached original `microsoft/deberta-v3-base` checkpoint at
revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. Every source file is
checksum-verified before loading, with local-only loading and remote code disabled.
A new three-class pooler/head uses negative/neutral/positive index order. Previous
fine-tuned models are controls, never initialization sources.

## Fixed comparison

- Both input formats run seeds 42, 43 and 44, independently from the original Base.
- Same complete initial state and starting Torch CPU/CUDA RNG state per seed,
  verified by fingerprints before target-pair training starts.
- Same example order per seed and epoch, checked with actual training-row hashes.
  A separate seeded shuffle generator prevents dropout from affecting row order.
- All parameters train; equal weight per annotation. Headline-only inputs repeat
  for each original target annotation, including conflicting gold labels.
- LR 1e-5, AdamW betas 0.9/0.999, eps 1e-8, weight decay 0.01 with bias/LayerNorm
  exclusions, 10% linear warm-up/decay and gradient clipping at 1.
- FP32/eager/deterministic execution, TF32 disabled; train/evaluation batches 16/32.
- Maximum five epochs, patience two. Strictly higher all-target validation macro
  F1 improves a checkpoint; exact ties retain the earlier epoch. No best seed.
- Maximum length 128. The tokenizer adds special tokens; full targets are
  preserved and only second-sequence headlines may be truncated in the pair arm.
  Tokenization eligibility was verified before training; native names remain visible.

Each complete epoch has 722 optimizer updates; the maximum is 3,610 updates,
with 361 warm-up updates. Actual stopping and selected epochs can differ because
they use the same selection rule on different learned models. Dropout draws can
also diverge as input sequence shapes differ; initial weights/RNG and row order
are matched, rather than an assertion of identical stochastic trajectories.

## Run on the existing Windows GPU checkout

From a clean checkout with the reviewed private data and cached source weights:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.target_deberta_experiment
```

`--data-dir PATH` selects a different existing reviewed data root. The default
runs all six trials and writes the aggregate after all matching checks pass.
A single declared trial can be run with both `--arm matched_headline_only`
(or `--arm target_pair`) and `--seed 42` (or 43/44). The all-trial command checks
matched initialization for each seed before starting the second arm.

If interrupted between completed trials, rerun the all-trial command. Only fully
verified completed trial reports are reused: context/source/configuration,
training membership, private prediction hashes/native validation rows, cohort
metrics and full checkpoint file hashes must agree. Incomplete epoch progress
is diagnostic and does not resume optimizer state; that trial starts fresh.
Published aggregate reports are never overwritten. Routine review uses saved
reports instead of rerunning the experiment.

Training lives in `target_training.fit_target`; `score_examples` is separate
inference code and receives no label tensor. Validation uses inference mode and
never updates gradients. Gold labels remain separately attached to metrics.
Checkpoint saves bind input mode, dataset/plan/lock, source, label order and
implementation identities. The optional context extension preserves the old
artifact format when omitted, so previous model artifacts remain unchanged.

## Outputs and interpretation

- Public trial JSON: `reports/experiments/target-deberta-runs/<arm>/seed-42.json`
  (and 43/44); aggregate JSON/Markdown: `reports/experiments/target-deberta.*`.
- Private safe checkpoints: `data/interim/transformer-finetuned-models/<manifest-sha>/`.
  Each selected artifact is identified in its trial report; intermediate improved
  checkpoints also remain private.
- Private row predictions/run copies: `data/interim/target-deberta/<trial-sha>/`.
- Private epoch diagnostics: `data/interim/target-deberta-progress/<run-id>/history.json`.

All six established validation cohorts include accuracy, three-class macro F1,
class PRF/support, confusion matrices, raw NLL/Brier and ten-bin ECE. Reports log
architecture, parameter/file sizes, runtime, selected epochs and optimizer history.
Means/sample standard deviations describe three seeds; they are not confidence
intervals. There is no calibration fitting or temperature application.

The primary comparison is matched target-pair versus matched headline-only.
The saved Phase 3 control used 6,299 single-target headlines, so its comparison
also changes training examples/update budgets. Native-string seen/unseen cohorts
are not verified company novelty. Five-company evaluation remains too sparse.
Validation also selects epochs, so results are exploratory, not independent test
evidence. Reserved tests, original final results and immutable preparation lock
remain unchanged. This task does not create a target prediction API, entity
recognizer, new labeled test collection or any later roadmap phase.
