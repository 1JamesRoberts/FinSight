# FinBERT fine-tuning on SEntFiN

This first experiment adapts the existing `ProsusAI/finbert` sentiment checkpoint,
pinned to revision `4556d13015211d73dccd3fdd39d39232506f3e43`. Its classifier
already learned from Financial PhraseBank. We continue updating its encoder and
sentiment head, rather than initialize a new head on a base language model.

Only the 6,299 single-target SEntFiN training headlines update parameters.
The 781 validation headlines select the epoch. PhraseBank, FiQA, multi-target
headlines and the frozen final test are excluded. Existing duplicate-group
splits and labels are retained. No training sets are pooled.

## Run on the Windows GPU computer

From `C:\Hope\Project\financial-news-intelligence` in PowerShell:

```powershell
python -m uv sync --locked --extra transformer --group eda
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment fine-tune-finbert
```

The default `--config` is `configs/finbert-finetune.json`. You may supply a
separate JSON configuration with the same fields. `--data-dir` locates the
reviewed local data; `--freeze-receipt` identifies its pinned split/freeze.
`--baseline-report` and `--pretrained-report` supply prior comparison results.
`--report-path` chooses the aggregate Markdown and JSON output location.
The default report is `reports/experiments/finbert-finetuning.md`.

Each invocation starts from the pinned original model; it does not resume a
previous run. The optional transformer dependencies are already locked.

## What the code does

`run_finetuning` verifies the development inputs and the saved baselines' split
lineage and ordered membership. `sentfin_partitions` selects single-target
SEntFiN records and rejects reserved records and overlapping groups.
`sentence_and_label` extracts each headline and its gold sentiment. Labels map
to the checkpoint's indices: positive=0, negative=1, neutral=2.

`fit_checkpoint` tokenizes each batch with the existing BERT WordPiece tokenizer,
dynamically pads it and truncates at 128 tokens. Training batches are shuffled
with seed 42. All 109,484,547 parameters receive gradients from cross-entropy
loss. AdamW uses learning rate 2e-5, weight decay 0.01 (excluding bias and
LayerNorm weights), a 10% linear warm-up, linear decay and gradient clipping at
norm 1. Training batch size is 16; validation batch size is 32. Computation uses
FP32, deterministic algorithms and disabled TF32.

After each epoch, validation runs in evaluation/inference mode with gradients
disabled. Strictly higher validation macro F1 saves a new artifact; exact ties
retain the earlier epoch. Stop after two epochs without improvement or five
epochs total. This is one fixed configuration and seed, not a hyperparameter grid.
The saved best checkpoint is reloaded, then its validation metrics must match
those recorded at selection. Model size, timings, truncation, class metrics,
source hashes, code revision and lineage are included in the report.

## Saved artifacts and inference

Each improved checkpoint is stored on the GPU computer under ignored
`data/interim/finbert-finetuned-models/<manifest-sha256>/`. It contains
`model.safetensors`, model/tokenizer configuration, vocabulary, and a manifest
with hashes of every file. The report identifies the selected artifact.
Private predictions/run logs remain in
`data/interim/finbert-finetuning/<run-sha256>/`; live epoch history is recorded
under `data/interim/finbert-training-progress/`. No optimizer state is saved;
these are inference artifacts, not resumable training checkpoints.

To load the selected model in Python, use the path and manifest hash from the
run report:

```python
from pathlib import Path
from finsight.sentiment.finbert import FinBertConfig, FinBertSentiment

model = FinBertSentiment.from_artifact(
    FinBertConfig(device="cuda"),
    Path("data/interim/finbert-finetuned-models/<manifest-sha256>"),
    "<manifest-sha256>",
)
predictions = model.predict_batch(["The company reported higher earnings."])
print(predictions[0].label.value, predictions[0].probabilities)
```

The loader verifies hashes before loading local files and freezes parameters.
Probabilities remain uncalibrated. Validation selected the model and is not an
unbiased final performance estimate. Full upstream raw-news exposure is unknown.
The user's documented local research decision covers this adaptation; upstream
weight-license scope remains unresolved. Weights and derivative artifacts remain
private and ignored. The source checkpoint and frozen test are preserved.

The [experiment report](../../../reports/experiments/finbert-finetuning.md) records the
actual result. This bounded task does not complete Phase 3's grid, MLflow,
calibration/error analysis, or final evaluation.

## First verified result

Five epochs took 196.55 seconds on the RTX 4070 SUPER. Epoch 3 was selected:
validation accuracy 0.8771, macro F1 0.8762, compared with pretrained FinBERT
0.7478/0.7492 and selected Logistic Regression 0.8195/0.8181. Neutral F1 improved
from 0.6922 to 0.8378. Later epochs reduce training loss but worsen validation
macro F1. Patience triggers at epoch 5, which also reaches the configured cap;
this run does not save epochs through early stopping.

Validation loss is lowest at epoch 2 and then rises despite higher classification
F1 at epoch 3. Selection follows the predefined macro F1 rule. Calibration and
confidence errors deserve separate review before any final evaluation.

The selected Windows directory is:

```text
C:\Hope\Project\financial-news-intelligence\data\interim\finbert-finetuned-models\276b014dec08139a12ec8ea85f5ba94d6f271148c62b49f26d37ea8e472936a7
```

Its manifest SHA-256 is the directory name. Mac storage has the manifest and
private prediction/run records for verification; weights remain on Windows.
Ruff passes on both machines. Mac has 269 passing tests and one skipped optional
backend test; Windows passes all 270, including actual tiny-BERT training,
validation gradient isolation and saved-checkpoint reload. Frozen-test bytes
and the original checkpoint hashes are unchanged.
