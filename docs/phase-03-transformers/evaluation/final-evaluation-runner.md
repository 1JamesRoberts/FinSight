# Final evaluation runner

The runner implements the locked v1 primary comparison plus v2 supplementary
soft/hard voting. It adds no training, model acquisition, weight search or
calibration fit. This implementation task leaves the frozen test unscored.

**Final execution completed 2026-10-13:** see the
[locked final results](../../../reports/experiments/phase3-final.md). The commands below
document the completed workflow. Preserve its original receipt/revision and use
saved predictions for metric reproduction; do not prepare a new run to tune or
rescore the consumed test. The unscored statements describe implementation-stage
verification, before the subsequent authorized final execution.
See [the protocol](final-evaluation.md) and [ensemble extension](../ensembles/ensemble-voting.md).

## Commands and scope

Use the Windows GPU repository with all reviewed private assets and a clean,
committed checkout. Install the locked transformer extra:

```powershell
python -m uv sync --locked --extra transformer
```

Validate the entire inference/metric path on the existing validation cohort:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.final_evaluation --development-smoke
```

This scores the five fixed baselines and all nine saved selected transformer
checkpoints on the 781 validation records, reuses their outputs for calibrated
variants and soft/hard voting, and exercises the same grouped bootstrap code.
It requires exact classification/metric parity with the nine historical raw
transformer validation runs. Results stay private under
`data/interim/final-evaluation-development/<source-identity-sha>/`. They use the
full-validation temperatures, so their probability results are **in-sample
calibration diagnostics**, distinct from the earlier out-of-fold experiment.
They are not final-test scores. Repeating the smoke check reuses saved outputs.

Prepare an immutable execution receipt **without test decoding or inference**:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.final_evaluation --prepare
```

The printed receipt path is under
`data/interim/final-evaluation/<receipt-sha>/receipt.json`. It records the clean
Git revision, hashes of every package source, both protocol identities, package
versions, Python/platform, GPU and CUDA runtime. Preparation fully verifies all
65 model/tokenizer/pipeline files and the exact saved temperatures. Frozen files
are checked as opaque bytes only. No fallback to CPU or asset download is allowed.

Verified 2026-10-13: Mac and Windows pass all 384 tests and Ruff lint/format.
The real Windows GPU smoke check passes all 14 model/control runs; the nine
saved transformer labels, class metrics and probabilities reproduce history
exactly, and all five baseline classifications match prior reports. A repeat
reuses all 14 completed outputs. Independent direct group-resampling agrees
with ten bootstrap intervals across two three-seed families. Receipt preparation
was verified without scoring; the frozen file hashes remain unchanged. After
the verification-notes commit, prepare a fresh receipt for any future final run.

When the final evaluation is explicitly requested, pass that actual receipt path:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.final_evaluation --run --receipt <printed-receipt-path>
```

Only `--run` decodes the frozen JSONL, selects the 766 single-target records and
checks the original class support and scope. All preflight and execution-receipt
checks happen first. The receipt must match the current clean code revision and
environment; prepare a fresh receipt after any implementation/commit change
before the first inference. This task's preparation checks are not instructions
to execute the test now. No automatic run follows `--prepare` or smoke checking.

## Predictions, metrics and uncertainty

There are **14 model/control passes**, **29 reported prediction variants**, and
**13 aggregate families**. The 14 passes comprise five baselines plus three models
at three seeds. Calibrated and ensemble variants reuse those raw predictions.
No model ever receives gold labels: adapters receive texts and locked specs only.
Rows are joined to gold afterward for validation and metrics.

Use local-only, exact-loading transformer/tokenizer artifacts, FP32 CUDA, eager
attention, deterministic seeds, TF32 disabled, batch 32 and length 128. Verify
trusted local TF-IDF pickle bytes and dependency versions before loading; never
unpickle arbitrary files. VADER resources and thresholds must match the lock.
All warm-ups use an invented headline, and GPU inference timers synchronize CUDA.
Record loading separately, inference latency, architecture, artifact size, token
truncation and GPU peaks. Record temperature adjustment and voting timing apart
from inference; sample SD across seeds is not a latency confidence interval.

Report per-variant classification metrics and probability diagnostics, then
mean/sample SD across the three model seeds. Single fixed baselines have null
sample SD. Hard voting, SVM and VADER have no class-probability metrics. Paired
group bootstrap uses 1,000 shared seed-42 draws, retaining complete duplicate
groups and repeated draws. Recompute each seed's metric per draw and average
metrics, not probabilities. Fixed class order retains absent-class F1 as zero.
Percentile 95% intervals use linear quantiles. The 11 predeclared paired
differences include primary model/LR comparisons, raw/calibrated NLL and the
supplementary voting comparisons. No test-based winner/seed selection is made.

## Retry, resume and export

Each model/control's complete validated outputs and timing/profile metadata are
atomically checkpointed. Progress records their hashes and every attempt. An
invalid/missing prediction aborts; records are never dropped. Failure logs and
all source IDs, gold, per-record scores/votes, bootstrap draws and group IDs stay
in ignored private storage. The model pipeline has no fit/retraining path.

Resume an interrupted run with the same receipt and clean code/environment:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.final_evaluation --run --receipt <printed-receipt-path> --resume
```

Completed model outputs are reused and reverified, never silently rescored.
Missing or corrupted completed outputs fail. A fully completed run returns its
verified aggregate without decoding test records or recomputing predictions.
Exclusive lock files prevent simultaneous writers. A force-killed process can
leave `active.lock`/`evaluation.lock`: confirm that process is gone before manually
removing its lock. Do not change model choices or metrics to repair an execution.

After a successful final run, publish only its aggregate JSON and Markdown:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.final_evaluation --export --receipt <printed-receipt-path> --destination reports/experiments/phase3-final
```

Export checks the completed aggregate/receipt/private hashes and aggregate schema.
It never copies per-record predictions or bootstrap membership. Existing
different reports cannot be overwritten. Export is separate from scoring and
can precede the results commit; execution receipt checks for future resumes
remain bound to the original revision. Keep original release/protocol receipts
and record any execution failures. Final-test-driven changes need new untouched
data for subsequent confirmatory claims. The completed final scoring is linked
above; multi-target/PhraseBank reserved tests remain outside that execution.

## Architecture and tests

`saved_inference.py` contains local artifact adapters; `final_metrics.py` contains
fixed-label metrics and vectorized grouped-bootstrap calculations;
`final_evaluation.py` coordinates receipts, state, checkpoints, derived outputs and
aggregate export. Existing pinned inference/training/analysis sources and both
protocol locks remain unchanged. No dependencies are added.

Synthetic tests cover membership/probabilities, seed aggregation, group draws
against brute-force calculations, paired differences, incomplete/failing models,
resume corruption, exclusive writers, receipt/source guards, preparation without
test access, 14-pass/29-variant orchestration and aggregate-only export. The real
GPU smoke check is development-only. Run the required quality gates:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```
