# Phase 3: first transformer comparison

All models score the same **781 single-target SEntFiN validation headlines**. Only the
project-trained transformers update parameters on the same **6,299 training headlines**.
The frozen final test remains unscored. No corpus pooling or multi-target training.

| Model | Additional SEntFiN training | Validation accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| linear_svm | Training split (Phase 2) | 0.8169 | 0.8160 |
| logistic_regression | Training split (Phase 2) | 0.8195 | 0.8181 |
| training_majority | Training majority only | 0.3611 | 0.1769 |
| vader | None | 0.4738 | 0.4787 |
| finbert-finetuning | Yes; existing sentiment checkpoint | 0.8771 | 0.8762 |
| finbert-validation | None | 0.7478 | 0.7492 |
| [financial_roberta](financial_roberta.md) | None | 0.7439 | 0.7333 |
| [financial_deberta](financial_deberta.md) | None | 0.7388 | 0.7414 |
| [roberta_base](roberta_base.md) | Yes; new sentiment head | 0.8860 | 0.8849 |
| [deberta_base](deberta_base.md) | Yes; new sentiment head | 0.8745 | 0.8738 |

## Model size, architecture, and execution

| Checkpoint | Layers | Hidden | Heads | Parameters | FP32 parameter MiB | Inference seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| financial_roberta | 24 | 1024 | 16 | 355,362,819 | 1355.60 | 1.051 |
| financial_deberta | 6 | 768 | 12 | 141,897,219 | 541.29 | 1.346 |
| roberta_base | 12 | 768 | 12 | 124,647,939 | 475.49 | 0.533 |
| deberta_base | 12 | 768 | 12 | 184,424,451 | 703.52 | 1.381 |

GPU: RTX 4070 SUPER / CUDA 12.8. FP32, eager attention, deterministic algorithms,
TF32 disabled. Inference batch 32; synthetic warm-up and synchronization. Timing
includes tokenizer, full length audit and output transfer. Per-model reports retain
vocabulary/feed-forward size, file sizes, class metrics, confusion matrices, truncation
and GPU memory. Historical CPU baseline timings are not a direct speed comparison.

## Base-model training

Both bases use learning rate 2e-5, batch 16, length 128, seed 42, AdamW, weight decay
0.01 excluding bias/LayerNorm, 10% linear warm-up then decay, norm-1 gradient clipping,
unweighted cross-entropy, maximum five epochs and patience two. Select validation
macro F1 by strict improvement; ties keep the earliest epoch. Train all parameters.

| Model | Epochs run | Selected epoch | Training/validation/saving seconds | Peak training allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| roberta_base | 5 | 4 | 199.55 | 2.40 |
| deberta_base | 5 | 5 | 393.30 | 3.53 |

The saved best checkpoints were reloaded and reproduced selected-epoch validation
metrics. Every improved checkpoint is retained privately on Windows; optimizer state
is not saved. Mac holds manifests and private prediction/run logs for verification.

## Interpretation and limits

Compare unchanged financial checkpoints with unchanged FinBERT to assess available
classifiers. Compare SEntFiN-trained models to assess this fixed local training setup.
Financial RoBERTa is Large, financial DeBERTa is Small, and our newly trained models
are Base. FinBERT also starts with prior PhraseBank sentiment supervision. Differences
therefore combine architecture, size, pretraining and prior task supervision.

Financial DeBERTa's training description matches PhraseBank; precise splits are unclear.
Financial RoBERTa does not disclose complete labeled-sample provenance. SEntFiN
exposure cannot be ruled out from silence. Full raw-news pretraining exposure remains
unknown. Our fine-tuned results use validation for epoch selection; one seed and fixed
settings do not establish statistical superiority. Probabilities are uncalibrated.

Original checkpoint and frozen-test digests are unchanged. Private saved predictions
were checked against ordered validation IDs and gold labels; metrics were recomputed.
Later runs log a dirty working tree because earlier generated reports were untracked;
implementation/configuration/lockfile hashes match their recorded Git revision.

## Provenance and remaining work

Each per-model JSON is a content-addressed run record with exact model revision/file
checksums, source-card license, configuration, split membership, code and environment.
This aggregate JSON pins those report hashes. See [guide](../../docs/transformer-comparison.md)
and [asset review](../../docs/datasets.md). No weights or source text are published.

MLflow, hyperparameter comparisons, multiple seeds, calibration/error analysis and
final frozen-test evaluation remain open. Phase 3 is not complete.
