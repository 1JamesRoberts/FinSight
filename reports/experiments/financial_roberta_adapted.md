# financial_roberta_adapted

sentfin_single_target selects trained epochs; development validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `soleimanian/financial-roberta-large-sentiment` at `f8804d31111d7c3569e88abaad6969918e858fbd`.
Mode: adapt_existing_sentiment_head; train 6299, validation 781.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.8195 | 0.8181 |
| vader | 0.4738 | 0.4787 |
| training_majority | 0.3611 | 0.1769 |
| linear_svm | 0.8169 | 0.8160 |
| finbert-validation | 0.7478 | 0.7492 |
| finbert-finetuning | 0.8771 | 0.8762 |
| financial_roberta_adapted | 0.8707 | 0.8696 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 16, "class": "RobertaForSequenceClassification", "hidden_size": 1024, "intermediate_size": 4096, "layers": 24, "parameter_bytes": 1421451276, "parameters": 355362819, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 1,426,316,466.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.255s; 1.607 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 29.

## Training

Strategy: adapt_existing_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 5; patience triggered False; total steps 1970.
Training/validation/saving: 446.28s; peak allocated/reserved bytes 7204306944/7719616512.
Training truncation: `{"training": {"maximum_tokens": 39, "truncated_texts": 0}, "validation": {"maximum_tokens": 29, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/77e886132b9ea71dd4412f4a44b40246796447b4b5c28c2820cc262498ec686c`; manifest `77e886132b9ea71dd4412f4a44b40246796447b4b5c28c2820cc262498ec686c`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.5154 | 0.4724 | 0.8323 | 0.8278 |
| 2 | 0.2881 | 0.5161 | 0.8630 | 0.8622 |
| 3 | 0.2024 | 0.6475 | 0.8643 | 0.8629 |
| 4 | 0.1248 | 0.7780 | 0.8694 | 0.8684 |
| 5 | 0.0829 | 0.7819 | 0.8707 | 0.8696 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8958 | 0.8704 | 0.8830 |
| neutral | 252 | 0.8449 | 0.8214 | 0.8330 |
| positive | 282 | 0.8716 | 0.9149 | 0.8927 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[215, 21, 11]
[18, 207, 27]
[7, 17, 258]
```

Run `5739c37f8d6b07fec1127ae0cf12cf47a077df4efde7fcab1bb1ad952e2ce607`; code `fb810900b14e58f9593e43f0b382f88a03edd193`; dirty `False`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
