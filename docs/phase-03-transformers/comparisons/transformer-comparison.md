# RoBERTa and DeBERTa comparison

Four bounded Phase 3 experiments use the same frozen development protocol:

| Experiment key | Checkpoint | Project training |
| --- | --- | --- |
| financial_roberta | soleimanian/financial-roberta-large-sentiment | None |
| financial_deberta | mrm8488/deberta-v3-ft-financial-news-sentiment-analysis | None |
| roberta_base | FacebookAI/roberta-base | New sentiment head + encoder |
| deberta_base | microsoft/deberta-v3-base | New sentiment head + encoder |

The financial DeBERTa checkpoint uses **v3 Small**, while the newly trained model
uses **v3 Base**. Financial RoBERTa uses **Large**, while we train **Base**. These
are checkpoint comparisons with different sizes and training histories, not a
controlled experiment isolating architecture. FinBERT already had PhraseBank
sentiment supervision before our SEntFiN adaptation. The base models have no
trained sentiment head; we initialize one with fixed seed 42.

## Reproduce on Windows

In `C:\Hope\Project\financial-news-intelligence`, run one task at a time:

```powershell
python -m uv sync --locked --extra transformer --group eda
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison financial_roberta
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison financial_deberta
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison roberta_base
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison deberta_base
```

`--data-dir` defaults to `data`. `--config` defaults to
`configs/transformer-training.json`; it shares the fixed first FinBERT adaptation
settings. The immutable asset pins live in `configs/transformer-assets.json`.
Each invocation starts from its upstream checkpoint; it does not resume training.
Only base-model keys train. Financial checkpoints run unchanged inference.

Only 6,299 single-target SEntFiN training records receive gradients; the same 781
validation headlines are scored. No PhraseBank/FiQA pooling, multi-target
training, or frozen-test scoring. The shared verified loader checks lineage and
duplicate groups before any model acquisition. Baseline comparisons verify
ordered validation membership and the prior FinBERT training membership.

## Implementation and settings

`transformer_comparison.py` loads only reviewed, checksum-pinned checkpoint files
with local-only model loading, disabled remote code and restricted weight loading.
Unchanged financial classifiers must load with no missing or unexpected keys.
Base training explicitly permits only the expected new classifier/pooler and
unused upstream language-model head. Missing encoder weights cause failure.
Financial checkpoint indices are negative/neutral/positive; outputs are explicitly
reordered into the common probability mapping. Labels are never inferred from
scores. Base training uses positive/negative/neutral, matching the shared loop.

`finbert_training.fit_checkpoint` is reused through small loader/saver hooks.
No experiment framework or duplicated training loop is introduced. RoBERTa uses
byte-level BPE tokenization; base DeBERTa uses its SentencePiece tokenizer.
Inputs are original headlines with each checkpoint's tokenizer and normalization;
no TF-IDF preprocessing is applied. Padding is dynamic and max length is 128.
All encoder and head parameters are trained with unweighted cross-entropy.

Use FP32, eager attention, deterministic algorithms, disabled TF32, seed 42,
AdamW learning rate 2e-5, weight decay 0.01 (excluding biases and LayerNorm),
10% linear warm-up followed by linear decay, clipping at norm 1, training batch
16 and evaluation batch 32. Select maximum validation macro F1, retaining the
earliest exact tie; patience 2, maximum 5 epochs. This compares one fixed
configuration per base model, not each model's best achievable performance.

After training, reload the saved best artifact and require its validation metrics
to match its epoch. No optimizer state is saved. Each improved epoch is retained
under ignored `data/interim/transformer-finetuned-models/<manifest-sha256>/`.
Use `load_saved(directory, manifest_sha256, asset, config)` and
`score_model(torch, tokenizer, model, texts, config)` for local inference.
`load_saved` checks source ancestry, configuration and every saved file's hash.
Downloaded sources stay under `data/interim/transformer-checkpoint-cache/`.
Private record-level predictions and run logs use
`data/interim/transformer-comparison/<run-sha256>/`.

Public per-model reports under `reports/experiments/<experiment-key>.md`/JSON
contain metrics, architecture/parameter/storage size, epoch history, loss,
truncation, GPU timing/memory, configs, hashes, membership and environment.
Inference timings include tokenization, length audit and output transfer with
synthetic warm-up and CUDA synchronization. Historical CPU baseline timings
cannot establish GPU speed superiority. Weights and source text remain ignored.

## Interpretation

The [asset review](../../datasets.md) records licenses and upstream training limitations.
Financial DeBERTa's data description matches PhraseBank. Financial RoBERTa lacks
complete labeled-data provenance; absence of reported SEntFiN exposure is not
proof of independence. Raw-news pretraining exposure is unknown for all models.
Validation is used to select trained epochs and is not an unbiased test estimate.
Probabilities remain uncalibrated. Preserve the frozen test until final model
selection. MLflow, calibration, grid search and final evaluation remain separate
roadmap tasks.

## Verified first results

| Model | Validation accuracy | Macro F1 | Selected epoch |
| --- | ---: | ---: | ---: |
| Financial RoBERTa Large, unchanged | 0.7439 | 0.7333 | None |
| Financial DeBERTa-v3 Small, unchanged | 0.7388 | 0.7414 | None |
| RoBERTa Base, SEntFiN-trained | 0.8860 | 0.8849 | 4 |
| DeBERTa-v3 Base, SEntFiN-trained | 0.8745 | 0.8738 | 5 |

Both bases ran five epochs. RoBERTa training/validation/saving took 199.55s;
DeBERTa took 393.30s. Peak training allocated memory was 2.40/3.53 GiB,
respectively, fitting the RTX 4070 SUPER in FP32. Selected artifacts on Windows:

| Model | Manifest SHA-256 / directory name |
| --- | --- |
| RoBERTa Base | `9a3e11b759fc223eb467e7393b73f002b01095a4eb261a5aee64288cf4d1ad59` |
| DeBERTa-v3 Base | `a90155447ce3f0910bf0b9c8bf341a2c932c97f6e01e1250466ed1a351174cf3` |

Use the directory name under `data/interim/transformer-finetuned-models/` and
pass the same hash to `load_saved`. Choose the corresponding asset entry from
`configs/transformer-assets.json` and load `FineTuneConfig` from the training
configuration. Both saved models reproduce selected-epoch validation metrics.
Weights remain on Windows; Mac has the manifests and private prediction/run logs.

The [consolidated report](../../../reports/experiments/transformer-comparison.md)
includes FinBERT and Phase 2 controls, model architectures, sizes and timing.
All 275 tests and Ruff checks passed on both machines. Tiny-model tests cover
actual updates and artifact reload for RoBERTa and DeBERTa; inference rejects
untrained heads, label mappings are explicit, and reserved records are rejected.
A narrow test filter accommodates the pinned upstream DeBERTa import's deprecated
TorchScript API; it does not alter model execution or suppress runtime errors.

All four models' predictions, metrics, membership and source/code hashes were
independently verified. Source checkpoint and frozen-test hashes are unchanged.
Later experiments truthfully record a dirty working tree from earlier untracked
report outputs; recorded implementation/configuration/lockfile hashes match the
Git commit. This task completes four experiments, not the full Phase 3 roadmap.

## Unchanged financial checkpoints on PhraseBank

Reuse the original financial sentiment checkpoints, without project training,
on the 518 PhraseBank validation phrases:

```powershell
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison financial_roberta_phrasebank
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment.transformer_comparison financial_deberta_phrasebank
```

These entries perform one inference evaluation, with zero training samples, no
checkpoint selection, no derived weights and no additional SEntFiN scoring.
Financial DeBERTa's apparent upstream PhraseBank supervision makes its score
diagnostic; Financial RoBERTa has incomplete supervised provenance. No reserved
PhraseBank or frozen SEntFiN test is scored. Original source pins stay unchanged.

Verified on 2026-10-12: unchanged Financial RoBERTa accuracy/macro F1
**0.7973/0.8142**; unchanged Financial DeBERTa **0.9826/0.9851** on the same
518 PhraseBank validation phrases. The latter is an apparent upstream-supervision
diagnostic, not independent generalization. Exact training/validation overlap
cannot be established from the source description. Both zero-training runs pass
ordered prediction/gold/metric, source checksum, code-lineage and frozen benchmark
checks on Mac/Windows. Ruff and all 282 tests pass on both.
Results are included in [all-model cohort report](../../../reports/experiments/all-models-by-cohort.md).
