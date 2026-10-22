# Detached RoBERTa target-format training

Launched 2026-10-20 (Bangkok time). This executes the
[locked nine-trial protocol](../evaluation/target-tokenizer-audit.md), not a new tuning search.
Company detection is deferred. Original labels and train/validation membership
stay unchanged; no frozen test is scored and no production model is replaced.

## Job and source

- Windows task: `FinSight-target-formats-20261019T182830Z`.
- Private job directory, relative to the primary PC project:
  `data/interim/target-format-jobs/target-formats-20261019T182830Z/`.
- Execution snapshot: `source/` inside that directory, revision
  `c9b449257ef1e9df885f14ffcacc2da20ad3e395`.
- Prepared manifest:
  `9eab447b2c6c123a5ce9fd5775a3c76a5878bc963404ce6afedce346d22a9c18`.
- Runs: original pair, target markers and target replacement × seeds 42/43/44.
  Every run starts fresh from the pinned original RoBERTa Base checkpoint.

The scheduler owns the process independently of the Mac/SSH session. A temporary
sleep-prevention request lasts only while the wrapper runs. The PC must remain
on and logged in. See the [operational guide](../../../docs/phase-04-target-sentiment/training/target-format-runner.md)
for status commands, logs and deliberate resume behavior.

## Verification and initial preflight correction

The Mac passes all 687 tests, including transformed-input train/save/reload/score
checks on tiny backbones, original native metadata/cohort preservation, matched
starts and row-order checks, aggregation and failure-status tests. Full Ruff
commands were run; only the existing scratch notebook has nine lint/format issues.
Checks excluding `notebooks/experiment.ipynb` pass; the notebook is not transferred.

The first Windows preflight passed Ruff, then stopped before training with one
test failure (686 passed). An older deployment-selection test checked for `data`
in any absolute ancestor, so it mistook this isolated source checkout for a private
dataset read. Its guard now checks project-relative paths and rejects reads outside
the checkout. The corrected snapshot has a new commit; the failed preflight logs
and original source manifest are preserved in `preflight-1/`, with `retry.json`
recording the change. Training results are not mixed across those revisions.

## Status and outputs

All nine trials completed successfully. The corrected job started at
2026-10-19 18:32:29 UTC and finished at 19:31:17 UTC (about 58 minutes 47 seconds;
Bangkok completion time 02:31 on October 20). Task Scheduler now shows Ready,
and wrapper/trial states both show complete. All 687 Windows tests and full Ruff
checks passed before training. Earlier fresh-session checks verified genuine GPU
optimizer progress after SSH disconnection; that evidence remains in the private
detachment receipt. Transfer temporary files were removed.

The completed outputs have now been retrieved and independently verified: all nine
selected checkpoint manifests/files and private prediction caches match their
hashes; native membership, gold, cohort metrics, aggregates and paired differences
recompute exactly. Frozen-record paths were forbidden during verification. The
exact source is preserved in a private Git bundle on both machines.
See the [reviewed comparison](target-format-comparison.md) and
[public verification receipt](../../experiments/target-format-verification.json).
The original input formulation remains recommended; the deployed checkpoint stays
unchanged. The job and GPU are idle.

The detached task ran full Windows checks and preparation verification before
starting any optimizer updates. Live wrapper state is `state.json`; trial/seed and
completed count are in `trials.json`; epoch/batch progress is in
`training.stdout.log`. A running task is not a completed experiment.

Each selected checkpoint and row-prediction cache stays in private primary data
storage. Completed trials are rechecked before reuse; interrupted trials start
fresh. The combined JSON/Markdown reports and per-trial aggregate JSON files are
exported to `exports/` only after all nine trials pass matching and provenance
checks. Reports include architecture/size, timing, per-seed metrics and cohort
comparisons, with definitions beneath their tables. All nine runs and their
exports are now reviewed; no new independent evaluation claim is made.
