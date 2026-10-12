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
| roberta_base | 0.8758 | 0.8746 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 44, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "RobertaForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 498591756, "parameters": 124647939, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 503,431,895.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 0.588s; 0.752 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 29.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 3; patience triggered True; total steps 1970.
Training/validation/saving: 201.76s; peak allocated/reserved bytes 2580942848/2898264064.
Training truncation: `{"training": {"maximum_tokens": 39, "truncated_texts": 0}, "validation": {"maximum_tokens": 29, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/fb86dab0bf886e387a959c5c3a56d660cb3695e5dddcd5480e7fec8122528715`; manifest `fb86dab0bf886e387a959c5c3a56d660cb3695e5dddcd5480e7fec8122528715`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6010 | 0.4901 | 0.8553 | 0.8528 |
| 2 | 0.2875 | 0.4691 | 0.8707 | 0.8703 |
| 3 | 0.2263 | 0.5292 | 0.8758 | 0.8746 |
| 4 | 0.1439 | 0.6703 | 0.8668 | 0.8648 |
| 5 | 0.1041 | 0.6689 | 0.8681 | 0.8663 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9038 | 0.8745 | 0.8889 |
| neutral | 252 | 0.8723 | 0.8135 | 0.8419 |
| positive | 282 | 0.8567 | 0.9326 | 0.8930 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[216, 17, 14]
[17, 205, 30]
[6, 13, 263]
```

Run `49e8904513df4a0d8fa977589d31de03b3bffb16c80e0a39e3902ee9d958dbee`; code `c62c1132c071d37ff386563ca350f8b354bb3d53`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
