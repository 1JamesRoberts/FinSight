# DeBERTa-v3 Base controlled extension

Use the existing reviewed Microsoft DeBERTa-v3 Base checkpoint with a new
three-class sentiment head. Train all parameters on the unchanged 6,299/781
single-target SEntFiN train/validation split. No PhraseBank pooling or reserved
test scoring. `configs/transformer-tuning/deberta-plan.json` pins the protocol.

Compare 1e-5/2e-5/3e-5 at seed 42, maximizing selected-epoch validation macro F1
(ascending rate and earliest epoch resolve exact ties). Persist the rate before
seeds 43/44; reuse its seed-42 run. Five fresh runs, no best-seed selection.
Other settings remain FP32, batches 16/32, length 128, maximum five epochs,
patience two, AdamW weight decay .01, warmup .1, gradient clip 1 and fixed seeds.

On Windows in the GPU repository:

```powershell
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_tuning --deberta-base
```

An interrupted study can use `--resume`; existing configuration, model files,
private predictions and split lineage must verify. Results stay under
`reports/experiments/deberta-tuning-runs/`, separate from original experiments.
Every improved epoch model and selected/reloaded predictions remain in ignored
storage. No new dependency, checkpoint pin or architecture change.

For saved-prediction calibration and a private qualitative review packet:

```sh
uv run --locked python -m finsight.sentiment.transformer_analysis --deberta-base
```

This uses the same 10-bin top-label ECE (5/15 sensitivity), unnormalized
multiclass Brier, clipped log loss, training-prior control, seed persistence and
hash-ordered error sample as the earlier two-model analysis. It creates the
private packet without overwriting a finished public review. After annotating
that packet, supply `--annotations <private-annotations.json>` to write
`reports/experiments/deberta-analysis.json`. Plot from its aggregate JSON:

```sh
uv run --locked --group eda python scripts/plot_transformer_analysis.py \
  --report reports/experiments/deberta-analysis.json \
  --output reports/figures/phase3/deberta-calibration.png
```

Treat all scores as validation diagnostics: the cohort has already selected
rates/epochs and been reviewed. Three seeds are not independent datasets or an
ensemble. Compare against existing RoBERTa/FinBERT without rerunning their
experiments or silently replacing their historical reports. Model weights,
headlines, IDs, individual predictions and review notes remain private.
