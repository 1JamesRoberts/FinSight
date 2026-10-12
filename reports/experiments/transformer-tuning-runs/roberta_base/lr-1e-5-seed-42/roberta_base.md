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
| roberta_base | 0.8822 | 0.8810 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 1e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "RobertaForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 498591756, "parameters": 124647939, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 503,431,895.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 0.555s; 0.711 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 29.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 2; patience triggered True; total steps 1576.
Training/validation/saving: 159.94s; peak allocated/reserved bytes 2578125824/2900361216.
Training truncation: `{"training": {"maximum_tokens": 39, "truncated_texts": 0}, "validation": {"maximum_tokens": 29, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/9800a1b42f80297c00519a68977e63e7ee386bb3e1e2b8e79bb792b8f43ff71a`; manifest `9800a1b42f80297c00519a68977e63e7ee386bb3e1e2b8e79bb792b8f43ff71a`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6631 | 0.4783 | 0.8297 | 0.8237 |
| 2 | 0.3045 | 0.4023 | 0.8822 | 0.8810 |
| 3 | 0.2245 | 0.4740 | 0.8784 | 0.8769 |
| 4 | 0.1739 | 0.5498 | 0.8796 | 0.8785 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8840 | 0.8947 | 0.8893 |
| neutral | 252 | 0.8607 | 0.8333 | 0.8468 |
| positive | 282 | 0.8990 | 0.9149 | 0.9069 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[221, 18, 8]
[21, 210, 21]
[8, 16, 258]
```

Run `ecfc57859fff3641c4ed21092a3f19145effed67da070b917f4c2a1555174e4f`; code `c62c1132c071d37ff386563ca350f8b354bb3d53`; dirty `False`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
