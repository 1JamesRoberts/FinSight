# Phase 4 target evaluation runner

The runner implements the [locked contract](target-final-evaluation.md) without
changing model selection, checkpoint identities or reserved membership. Its
entry point is `finsight.sentiment.experiments.target_final_evaluation`.
Default execution verifies metadata; final inference requires an explicit receipt.

## Code flow

- `evaluation/target_data.py` reads the checksummed native development views.
  Only the explicit final-run path decodes frozen records. It rejects group and
  normalized-text overlap with development or the consumed single-target test
  cohort, validates native labels, and checks published cohort counts.
- `models/target_inference.py` receives immutable `TargetText` objects containing
  headline identity, headline and target. Gold labels never reach a model. It
  loads local artifacts without fitting/downloading, reuses the checked pair
  tokenizer, and broadcasts one headline-only prediction to that headline's targets.
- `evaluation/target_metrics.py` reports every seed, mean metrics and sample SD,
  all five declared cohorts, and seven predeclared paired differences. Bootstrap
  draws sample eligible duplicate groups within each cohort; identical draws
  apply to every variant/seed in that cohort. All cohort annotations belonging
  to each sampled group retain its multiplicity. Empty cohorts are unavailable.
  Fixed class order is retained for single-class samples. Group draws and sparse
  per-native-target metrics stay private.
- `experiments/target_final_evaluation.py` binds the plan, code, environment and
  development-view identities to a content-addressed execution receipt. It caches
  each of the 21 variants with receipt/spec/membership hashes and logs attempts.
  Resume validates completed checkpoints instead of repeating inference. Changed
  inputs, missing/corrupt completed predictions, unexpected fields, invalid
  probabilities or incomplete outputs fail. A session lock prevents concurrent
  final runs. A stale lock or unchecked orphan checkpoint requires manual review;
  the runner never automatically clears one or silently rescores it.

Public aggregates contain class/cohort metrics, timing, architecture/size metadata,
seed variation and paired intervals. Private results contain annotation predictions,
bootstrap group draws and native-target breakdowns. The original Phase 3 results
and the Phase 4 contract/lock remain unchanged.

## Safe local verification

From the repository root:

```bash
uv run --locked python -m finsight.sentiment.experiments.target_final_evaluation
uv run --locked python -m finsight.sentiment.experiments.target_final_evaluation --development-smoke
```

The first command verifies metadata without opening frozen records or models.
The smoke command uses **629 development multi-target annotations from 272
headlines**. It replays all 18 hash-verified saved transformer validation outputs
and freshly predicts both saved TF-IDF baselines plus the fixed training prior.
Transformer replay verifies evaluation/persistence integration; it is not new GPU
inference. No frozen record file is opened. It logs actual source fingerprints,
Git revision/dirty state, plan identity and private content-addressed inputs.
New results live in ignored `data/interim/target-evaluation-smoke/<UTC-run>/`.

The [verification summary](../reports/experiments/target-runner-verification.json)
records exact classification matches and probability-metric matches within 1e-12
against all 18 transformer and both original sparse validation receipts. Synthetic
tests independently recalculate whole-group paired intervals, exercise each tiny
BERT/RoBERTa/DeBERTa model without labels, gradients or weight updates, and check
receipt binding, interrupted resume, postprocessing retries, corruption, group
leakage and aggregate-only export. The full local suite has 574 passing tests.

## Prepare and execute later on the GPU machine

These commands document the explicit final path; no reserved scoring was performed
while implementing/verifying the runner. First make the intended source changes
reviewable and commit them, install the locked transformer environment, and ensure
all private weights, baseline pickles and development/frozen artifacts are present:

```powershell
uv sync --locked --extra transformer
uv run --locked python -m finsight.sentiment.experiments.target_evaluation_plan --full-artifacts
$receipt = uv run --locked python -m finsight.sentiment.experiments.target_final_evaluation --prepare
```

Preparation hashes every artifact and the frozen files as opaque bytes, verifies
native development views, requires a clean committed checkout and available CUDA,
and writes a private receipt. It does not parse test records or predict. Run only
when ready for the one locked reserved evaluation:

```powershell
uv run --locked python -m finsight.sentiment.experiments.target_final_evaluation --run $receipt
```

An interrupted execution can be resumed explicitly:

```powershell
uv run --locked python -m finsight.sentiment.experiments.target_final_evaluation --run $receipt --resume
```

After full completion, resume verifies receipt/result hashes and returns without
another test decode or inference. Source/environment/plan checks still run.
Export verifies the receipt and completed public/private result digests and refuses
existing destination files:

```powershell
$resultDirectory = Split-Path $receipt
uv run --locked python -m finsight.sentiment.experiments.target_final_evaluation --export $resultDirectory
```

The default destination is `reports/experiments/phase4-target-final.json` and `.md`.
Use `--report-path PATH_WITHOUT_EXTENSION` for a different report location. JSON
ignore exceptions remain deliberately narrow; review aggregate-only content before
adding a future final JSON to Git. Export never publishes row predictions or native
names. Confidence intervals describe test-sample uncertainty conditional on the
trained runs; seed SD is separate. Results cannot change the locked candidate.

## Current verification limits

Full Windows artifact hashing, CUDA inference and a clean execution receipt remain
pending: this Mac checkout has checkpoint manifests and development predictions,
while the complete weights remain on Windows. Existing uncommitted project and
notebook edits were preserved. The runner deliberately rejects final receipt
creation from a dirty checkout and has no CPU fallback for the locked final run.
Synthetic CPU inference and development verification pass; this does not establish
that all GPU-machine execution prerequisites have passed. No dataset acquisition,
model training, new transformer inference or reserved-test scoring occurred.
