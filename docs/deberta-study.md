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

## Seed-42 rate selection

| Learning rate | Selected epoch | Validation macro F1 |
|---|---:|---:|
| 1e-5 | 2 | 0.882560 |
| 2e-5 | 5 | 0.873757 |
| 3e-5 | 4 | 0.882375 |

The declared rule selected 1e-5. Its 0.000185 advantage over 3e-5 is a narrow
validation difference, not evidence of a generally optimal rate. The selection
receipt was captured before completed repeat results and has SHA-256
`39095f2a04622bac71f92f5add9a9c919b2669b75d608be47f0f973f07a6e838`.
The fresh 2e-5/seed-42 control exactly reproduces historical class metrics,
confusion matrix, selected artifact identity and prediction-byte hash.

## Completed results and private review

All five runs complete. At selected LR 1e-5, macro F1 is 0.8810 ± 0.0044 and
accuracy 0.8818 ± 0.0041 across seeds 42/43/44 (sample SD). Mean confidence is
0.9642, ECE 0.0858, Brier 0.2082 and log loss 0.4995. These observed means compare
favorably with the earlier candidates but do not establish unseen superiority.

See [study results](../reports/experiments/deberta-tuning.md) and
[calibration/error report](../reports/experiments/deberta-analysis.md), including
model architecture/size, per-seed/per-class scores and reliability plots.
The private review packet SHA is
`4ccde94183b871494ebeb2c53863e4372afc3c04302bca248052c9d4372f3c9a`.
It contains 26 sampled headlines; annotations flag 18 cases as possibly ambiguous,
without changing gold. To reproduce the reviewed aggregate JSON:

```sh
uv run --locked python -m finsight.sentiment.transformer_analysis --deberta-base \
  --annotations data/interim/phase3-error-review/4ccde94183b871494ebeb2c53863e4372afc3c04302bca248052c9d4372f3c9a/annotations.json
uv run --locked --group eda python scripts/plot_deberta_study.py
```

Private inputs are required; the public checkout alone has no dataset, weight,
prediction or review-note redistribution clearance. The first command verifies
saved predictions and writes the reviewed public JSON. The second produces
`reports/figures/phase3/deberta-learning-rate-seeds.png`. Use the calibration
plot command above for the other figure. Neither command retrains a model.
