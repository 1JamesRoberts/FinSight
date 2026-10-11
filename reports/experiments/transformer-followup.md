# Transformer follow-up: financial adaptation and PhraseBank training

Four independent runs reuse pinned upstream checkpoints and fixed settings. No pooling
or sequential PhraseBank-to-SEntFiN training. The frozen test remains unscored.

## Financial checkpoints: SEntFiN validation before and after adaptation

| Checkpoint | Unchanged accuracy | Unchanged macro F1 | Adapted accuracy | Adapted macro F1 | Selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| [financial_roberta](financial_roberta_adapted.md) | 0.7439 | 0.7333 | 0.8707 | 0.8696 | 5 |
| [financial_deberta](financial_deberta_adapted.md) | 0.7388 | 0.7414 | 0.8745 | 0.8741 | 5 |

These adaptations use 6,299 training / 781 validation single-target SEntFiN headlines.
Existing sentiment heads and all encoder parameters are updated. Compare each source
checkpoint before versus after adaptation; model size and architecture remain fixed.

## Base models trained on PhraseBank only

| Base checkpoint | PhraseBank validation accuracy | PhraseBank macro F1 | SEntFiN transfer accuracy | Transfer macro F1 | Selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| [roberta_phrasebank](roberta_phrasebank.md) | 0.8938 | 0.8868 | 0.7836 | 0.7845 | 5 |
| [deberta_phrasebank](deberta_phrasebank.md) | 0.8900 | 0.8857 | 0.7798 | 0.7816 | 4 |

Start from the original base checkpoints with new sentiment heads; train on 3,785
PhraseBank phrases and select epochs using only the 518 PhraseBank validation phrases.
SEntFiN transfer scores use the 781 development validation headlines afterward. They
do not select epochs or update parameters. No reserved PhraseBank or SEntFiN test scored.

## Same architecture, different training corpus

| Base checkpoint | SEntFiN-trained macro F1 on SEntFiN validation | PhraseBank-trained transfer macro F1 on SEntFiN validation |
| --- | ---: | ---: |
| roberta_base | 0.8849 | 0.7845 |
| deberta_base | 0.8738 | 0.7816 |

These figures have different selection procedures: SEntFiN-trained models selected
epochs on SEntFiN validation, whereas PhraseBank models selected on PhraseBank. This
is a transfer diagnostic, not an unbiased comparison on a final test. Training-set sizes
also differ; corpus effects cannot be separated from sample count.

## Training, architecture, and storage

All runs use seed 42, FP32, AdamW learning rate 2e-5, batch 16, eval batch 32, max length
128, weight decay 0.01 excluding bias/LayerNorm, 10% linear warm-up/decay and norm-1
clipping. Maximum five epochs; patience two; strict validation macro F1 improvement.
GPU: RTX 4070 SUPER, CUDA 12.8. No configuration was silently adjusted.

| Experiment | Layers | Hidden | Parameters | Epochs | Train/validation/save seconds | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| financial_roberta_adapted | 24 | 1024 | 355,362,819 | 5 | 446.28 | 6.71 |
| financial_deberta_adapted | 6 | 768 | 141,897,219 | 5 | 220.90 | 2.72 |
| roberta_phrasebank | 12 | 768 | 124,647,939 | 5 | 119.90 | 3.02 |
| deberta_phrasebank | 12 | 768 | 184,424,451 | 5 | 241.86 | 5.57 |

Every improved artifact is retained separately in ignored Windows storage. Reloading
the selected artifacts reproduced their epoch metrics. Mac retains private prediction
records and manifests, without model weights. No optimizer state is saved. Per-model
reports include architecture/storage, epoch history, class metrics, confusion matrices,
truncation, runtime, source and code hashes, split membership and environment.

## Interpretation and verification

Financial DeBERTa has apparent upstream PhraseBank supervision; Financial RoBERTa
lacks complete supervised sample provenance. Raw-text pretraining exposure is unknown.
FinBERT on PhraseBank is diagnostic because it used that corpus upstream. Probabilities
remain uncalibrated; one configuration and seed do not establish statistical superiority.

Original source checkpoint and frozen benchmark hashes are unchanged. Validation and
transfer predictions were checked against exact ordered IDs/gold labels and metrics
recomputed. Recorded code/config/lockfile hashes match their Git revision. Later runs
record generated reports as working-tree changes. Ruff and all 280 tests pass on Mac
and Windows, including both label orders and both training cohorts. Source and saved
tokenizer IDs/masks match for every selection and transfer validation text; see the
guide for the pinned library’s Financial DeBERTa reload warning.

See [run/code guide](../../docs/transformer-followup.md), [prior comparison](transformer-comparison.md)
and [asset review](../../docs/datasets.md). Weights and source text remain private.
MLflow, hyperparameter comparisons, calibration/error analysis and final evaluation
remain open. Phase 3 is in progress.
