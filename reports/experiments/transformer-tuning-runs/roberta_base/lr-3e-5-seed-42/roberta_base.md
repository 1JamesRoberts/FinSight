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
| roberta_base | 0.8758 | 0.8749 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 3e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "RobertaForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 498591756, "parameters": 124647939, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 503,431,895.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 0.564s; 0.722 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 29.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 2; patience triggered True; total steps 1576.
Training/validation/saving: 159.98s; peak allocated/reserved bytes 2578125824/2900361216.
Training truncation: `{"training": {"maximum_tokens": 39, "truncated_texts": 0}, "validation": {"maximum_tokens": 29, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/4c4aea2402c73e752a542aa1d4d44132919ec380aadd2137c49d006fae530120`; manifest `4c4aea2402c73e752a542aa1d4d44132919ec380aadd2137c49d006fae530120`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.5969 | 0.4994 | 0.8246 | 0.8212 |
| 2 | 0.3110 | 0.4818 | 0.8758 | 0.8749 |
| 3 | 0.2090 | 0.6409 | 0.8553 | 0.8533 |
| 4 | 0.1280 | 0.6548 | 0.8707 | 0.8695 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8621 | 0.9109 | 0.8858 |
| neutral | 252 | 0.8601 | 0.8294 | 0.8444 |
| positive | 282 | 0.9025 | 0.8865 | 0.8945 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[225, 16, 6]
[22, 209, 21]
[14, 18, 250]
```

Run `ad79bfc0973048c71804c70a2caebc0117fc2640f4b2f1c918b0796c4398df28`; code `c62c1132c071d37ff386563ca350f8b354bb3d53`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
