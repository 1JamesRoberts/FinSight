# roberta_base

sentfin_single_target selects trained epochs; development validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `FacebookAI/roberta-base` at `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
Mode: new_sentiment_head; train 6299, validation 781.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.8195 | 0.8181 |
| vader | 0.4738 | 0.4787 |
| training_majority | 0.3611 | 0.1769 |
| linear_svm | 0.8169 | 0.8160 |
| finbert-validation | 0.7478 | 0.7492 |
| finbert-finetuning | 0.8771 | 0.8762 |
| roberta_base | 0.8707 | 0.8686 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 43, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "RobertaForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 498591756, "parameters": 124647939, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 503,431,895.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 0.556s; 0.712 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 29.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 5; patience triggered False; total steps 1970.
Training/validation/saving: 202.96s; peak allocated/reserved bytes 2579305472/2902458368.
Training truncation: `{"training": {"maximum_tokens": 39, "truncated_texts": 0}, "validation": {"maximum_tokens": 29, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/fca701557f897870f9ede87be99df97b2b480ee5a8ada8c1eee7ac5e4a540455`; manifest `fca701557f897870f9ede87be99df97b2b480ee5a8ada8c1eee7ac5e4a540455`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6056 | 0.4767 | 0.8361 | 0.8338 |
| 2 | 0.3063 | 0.4259 | 0.8643 | 0.8626 |
| 3 | 0.2110 | 0.5759 | 0.8617 | 0.8589 |
| 4 | 0.1441 | 0.6517 | 0.8681 | 0.8664 |
| 5 | 0.1033 | 0.6945 | 0.8707 | 0.8686 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9013 | 0.8502 | 0.8750 |
| neutral | 252 | 0.8465 | 0.8095 | 0.8276 |
| positive | 282 | 0.8664 | 0.9433 | 0.9032 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[210, 26, 11]
[18, 204, 30]
[5, 11, 266]
```

Run `3c2fe01e9f9f283983921313aaf3c61b08633985a2f07e4edb46be79d82034be`; code `c62c1132c071d37ff386563ca350f8b354bb3d53`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
