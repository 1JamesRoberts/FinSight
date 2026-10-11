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
limitations remain in [asset review](datasets.md).

A shared cohort selector rejects reserved records, unsupported corpora and
cross-partition groups. The existing FinBERT default remains SEntFiN-only.
Small-model tests exercise both classifier index orders, both architectures,
PhraseBank training and saved-artifact reloading. No new training framework or
dependency is added.
