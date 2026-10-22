# Running the locked target-format study on the GPU PC

The nine-trial execution is complete and verified. See the
[reviewed results](../../../reports/phase-04-target-sentiment/experiments/target-format-comparison.md).
The commands below describe the preserved execution and deliberate resume behavior;
no training is currently running.

The runner is `finsight.sentiment.experiments.target_format_training`. It executes
the nine trials in `configs/target-format-plan.json` using the prepared-data lock.
The three arms use the same original native examples for membership, labels and
cohorts, with separate prepared text views passed only to the tokenizer. The shared
training loop and selected-checkpoint reload/scoring are reused.

## Detached execution

The requested execution uses a Windows Task Scheduler task under `HOPE\kritt`,
with an isolated Git source snapshot inside the primary project's ignored
`data/interim/target-format-jobs/<job-id>/source/` directory. The main checkout,
previous experiments and saved production predictor stay unchanged.

`scripts/run_target_formats.ps1` runs Ruff, pytest and lock verification before
training. It launches the Python runner using the primary `.venv`, while
`PYTHONPATH` points to the isolated source. Hub/network acquisition is disabled.
The task holds a temporary sleep-prevention request for its lifetime, then releases
it. It does not change persistent Windows power settings.

The scheduler owns this process; the Mac/SSH session does not. Closing the Mac
will not stop it. The GPU PC must remain powered on and logged in. A shutdown,
restart or logout can interrupt an interactive scheduled task; it is not a service.

All nine trials run sequentially on the same GPU. Best checkpoints and native
validation prediction caches remain in ignored primary data storage. The source
snapshot receives aggregate trial reports and the combined JSON/Markdown report.
On success, public aggregates are copied into the job's `exports/` directory.

## Check progress on the PC

In PowerShell, substitute the job ID from the launch receipt:

```powershell
$job = 'C:\Hope\Project\financial-news-intelligence\data\interim\target-format-jobs\<job-id>'
Get-Content "$job\state.json"
Get-Content "$job\trials.json"
Get-Content "$job\training.stdout.log" -Tail 15
```

- `state.json`: wrapper stage (`ruff-check`, `pytest`, `preparation-check`,
  `training`, `complete` or `failed`), timestamps, process/source identity and errors.
- `trials.json`: Python runner status, current variant/seed and completed-trial
  count out of nine. It appears after the training stage starts.
- `training.stdout.log`: batch/epoch progress, validation macro F1 and selected
  epoch. The shared loop prints its generic `target_pair` mode; the preceding
  `Starting <variant> seed <seed>` line identifies the experimental format.
- `training.stderr.log`: library diagnostics and exception tracebacks if needed.
- `exports/`: final aggregate report and per-trial aggregate JSON files; only
  created after all nine trials and matching checks succeed.

## Resume behavior

Rerunning the same snapshot command verifies and reuses fully completed trials:
configuration, native membership/gold, prediction bytes, artifact files, selected
metrics, implementation hashes and input-variant context must match. A changed
implementation fails reuse. An interrupted trial starts fresh from the original
base checkpoint; mid-epoch optimizer resumption is not implemented.

Do not start a second process against the same output directory while the task
is running. The scheduled task uses `IgnoreNew` to reject concurrent starts of
that task. The preparation lock is immutable; its historical `training_started`
field is not a live status indicator.

## Direct command for the same execution snapshot

```powershell
$python = 'C:\Hope\Project\financial-news-intelligence\.venv\Scripts\python.exe'
$data = 'C:\Hope\Project\financial-news-intelligence\data'
$cache = "$data\interim\transformer-checkpoint-cache\models--FacebookAI--roberta-base\snapshots\e2da8e2f811d1448a5b465c236feacd80ffbac7b"
Set-Location "$job\source"
$env:PYTHONPATH = "$job\source\src"
& $python -u -m finsight.sentiment.experiments.target_format_training --data-dir $data --tokenizer-dir $cache --state "$job\trials.json"
```

This direct foreground command is for deliberate resume/debugging. Task Scheduler
provides the detached launch, preflight checks, sleep prevention and export step.
Validation is reused development data, not an independent benchmark. No company
detection, test scoring, calibration fitting or production-model replacement is
part of this execution.
