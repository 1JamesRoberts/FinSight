# Financial checkpoint adaptation and PhraseBank base training

Four separate experiments extend the first transformer comparison. Original
reports and source pins remain unchanged. No model starts from a previous
project-trained artifact.

| Key | Original checkpoint | Training corpus | Validation selects epoch |
| --- | --- | --- | --- |
| financial_roberta_adapted | Financial RoBERTa Large | SEntFiN | SEntFiN |
| financial_deberta_adapted | Financial DeBERTa-v3 Small | SEntFiN | SEntFiN |
| roberta_phrasebank | FacebookAI/RoBERTa Base | PhraseBank | PhraseBank |
| deberta_phrasebank | Microsoft/DeBERTa-v3 Base | PhraseBank | PhraseBank |

The financial models retain their existing negative/neutral/positive heads.
Base models initialize new positive/negative/neutral heads with seed 42.
The training loop explicitly maps gold labels and logits to the checkpoint's
index order, preserving head weights. All parameters receive training gradients.
Training data are never pooled. Each invocation restarts from its pinned source.

PhraseBank-only uses 3,785 training phrases and 518 validation phrases. SEntFiN
uses 6,299/781 single-target headlines. PhraseBank validation selects the base
models' epochs, then their saved best checkpoints score the 781 SEntFiN validation
headlines to measure transfer. Those transfer scores do not select epochs, change
settings, or update parameters. They are development diagnostics, not final-test
results. The frozen SEntFiN test and PhraseBank test candidates remain withheld.
FinBERT's PhraseBank scores have known supervision exposure and are diagnostic.

## Run on Windows

From `C:\Hope\Project\financial-news-intelligence`, run one experiment at a time:

```powershell
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison financial_roberta_adapted
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison financial_deberta_adapted
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison roberta_phrasebank
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison deberta_phrasebank
```

Settings remain `configs/transformer-training.json`: learning rate 2e-5, batch
16, evaluation batch 32, length 128, maximum five epochs, patience two, seed 42,
FP32 AdamW, warm-up/decay and clipping. Financial RoBERTa Large needs more memory;
record any necessary adjustment rather than silently change the experiment.

Per-model reports describe the actual training strategy/cohort, architecture,
size, epoch history, class metrics, GPU usage, lineage, model and code hashes.
PhraseBank reports also log transfer metrics/membership, with separate ignored
`transfer-predictions.jsonl` files. Saved artifacts and `load_saved` follow the
[same guide](transformer-comparison.md). Source checkpoint licenses and data
limitations remain in [asset review](../../datasets.md).

A shared cohort selector rejects reserved records, unsupported corpora and
cross-partition groups. The existing FinBERT default remains SEntFiN-only.
Small-model tests exercise both classifier index orders, both architectures,
PhraseBank training and saved-artifact reloading. No new training framework or
dependency is added.

## Verified results (2026-10-11)

| Run | Selection validation macro F1 | SEntFiN transfer macro F1 | Selected epoch |
| --- | ---: | ---: | ---: |
| Financial RoBERTa Large, SEntFiN adaptation | 0.8696 | — | 5 |
| Financial DeBERTa-v3 Small, SEntFiN adaptation | 0.8741 | — | 5 |
| RoBERTa Base, PhraseBank only | 0.8868 | 0.7845 | 5 |
| DeBERTa-v3 Base, PhraseBank only | 0.8857 | 0.7816 | 4 |

All four ran five epochs with the original fixed settings. Financial checkpoints
improve over their unchanged SEntFiN scores. PhraseBank-only bases transfer less
well than the earlier SEntFiN-trained bases; sample counts and epoch-selection
corpora differ, so this does not isolate a causal corpus effect. No final test
was scored. See the [consolidated report](../../../reports/experiments/transformer-followup.md)
for accuracy, architecture, model size, timing, memory and comparison details.

Selected manifests under the ignored Windows artifact directory:

| Run | Manifest SHA-256 |
| --- | --- |
| financial_roberta_adapted | `77e886132b9ea71dd4412f4a44b40246796447b4b5c28c2820cc262498ec686c` |
| financial_deberta_adapted | `b3e29be91c8d166b715ca31d8f2a1ca440beacea2cbc597b607f2825c2bc9ceb` |
| roberta_phrasebank | `9f151e0c4a70285d2f75038297bc9baf4b3181846546d97d8798f03060a8af96` |
| deberta_phrasebank | `976d80967708729fa1649282a643133b8aa18f71f178e6d114c718859881ab72` |

Source/saved weight hashes, ordered validation IDs/gold labels, recomputed metrics,
code/configuration lineage and frozen benchmark hashes pass verification. Saved
models reproduce selected epoch metrics. Source and saved tokenizer input IDs
and masks match on all selection/transfer validation texts. Transformers 4.57.6
emits a Mistral regex warning for saved Financial DeBERTa because its local-config
heuristic includes this non-Mistral model/version; no regex patch is applied.

Mac and Windows Ruff checks and all 280 tests pass. Tests cover both label orders,
both cohorts, reserved-record rejection and actual tiny-model training/reloading.
This completes these four runs; hyperparameter comparisons, MLflow, calibration,
error review and final evaluation remain open.
