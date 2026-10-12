# Bounded transformer learning-rate and seed study

This Phase 3 task compares RoBERTa Base and FinBERT on the unchanged 6,299/781
single-target SEntFiN training/validation split. Each run starts from the original
pinned upstream checkpoint: a new sentiment head for RoBERTa Base, the existing
sentiment head for FinBERT. All parameters are trained; corpora are never pooled.

The predeclared plan is `configs/transformer-tuning/plan.json`. It pins the baseline
configuration, asset manifest and freeze receipt. Compare learning rates 1e-5,
2e-5 and 3e-5 at seed 42. Maximize validation macro F1 separately for each model;
exact ties retain the first declared rate (ascending). Within each run, strict
macro F1 improvement selects epochs, with the earliest epoch retained on a tie.

Persist both selected rates before running seeds 43 and 44. The selected seed-42
grid run is reused, giving three seeds per selected rate and ten runs total.
Report all seeds and mean/sample standard deviation; never select the best seed.
Seed 42 also selected the rate, so the three-seed mean includes a selection-biased
component. All scores remain development diagnostics on the repeatedly inspected
validation set; three seeds do not establish statistical superiority.

All other settings match `configs/transformer-training.json`: FP32, batch 16,
evaluation batch 32, length 128, weight decay 0.01, warmup ratio 0.1, gradient
clip 1, maximum five epochs, patience two, deterministic computation and TF32 off.
Nine rate/seed configuration files are committed; only the ten planned model/run
combinations execute. The shared existing training engine is reused without a
new framework or dependency. MLflow and calibration are separate open tasks.

## Run on Windows

From `C:\Hope\Project\financial-news-intelligence`:

```powershell
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_tuning
```

The command reads the plan and configuration files, loads verified development
records and reviewed checkpoints, executes the grid then selected-rate repeats,
and reloads each selected epoch before scoring. The frozen test is never passed
to training or inference. No learning-rate choice uses extra-seed results.

Reports are separate from historical results under
`reports/experiments/transformer-tuning-runs/<model>/lr-<rate>-seed-<seed>/`.
Each directory contains the usual aggregate JSON/Markdown. `selection.json`
records seed-42 winners before repeats; `study.json` contains progress and final
metrics/statistics with input/report/artifact identities. Weights and per-record
predictions use existing ignored artifact/run storage on Windows.

For an interrupted study, add `--resume`. Existing runs must match the expected
configuration, current training implementation, split lineage, selected artifact
file checksums and private run/prediction hashes. The runner refuses existing
reports without this flag. Completed runs are reused; no source report is
silently overwritten. Partial failed runs without reports restart from source.

An original single-configuration report remains accurate for its individual run;
the study summary describes selection across those runs. The general roadmap
item for learning rate/length/batch comparisons remains open because this task
varies learning rate only. Source supervision/exposure and licensing limitations
remain in [asset review](datasets.md).
