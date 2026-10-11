# financial_deberta_adapted

sentfin_single_target selects trained epochs; development validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis` at `9e10915c245a80a89b18d1ac51350e093c7bb35a`.
Mode: adapt_existing_sentiment_head; train 6299, validation 781.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.8195 | 0.8181 |
| vader | 0.4738 | 0.4787 |
| training_majority | 0.3611 | 0.1769 |
| linear_svm | 0.8169 | 0.8160 |
| finbert-validation | 0.7478 | 0.7492 |
| finbert-finetuning | 0.8771 | 0.8762 |
| financial_deberta_adapted | 0.8745 | 0.8741 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 6, "parameter_bytes": 567588876, "parameters": 141897219, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 578,718,827.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 0.607s; 0.777 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 27.

## Training

Strategy: adapt_existing_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 5; patience triggered False; total steps 1970.
Training/validation/saving: 220.90s; peak allocated/reserved bytes 2924172288/3462397952.
Training truncation: `{"training": {"maximum_tokens": 36, "truncated_texts": 0}, "validation": {"maximum_tokens": 27, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/b3e29be91c8d166b715ca31d8f2a1ca440beacea2cbc597b607f2825c2bc9ceb`; manifest `b3e29be91c8d166b715ca31d8f2a1ca440beacea2cbc597b607f2825c2bc9ceb`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.5504 | 0.4387 | 0.8451 | 0.8410 |
| 2 | 0.2978 | 0.4677 | 0.8694 | 0.8682 |
| 3 | 0.2064 | 0.5475 | 0.8707 | 0.8698 |
| 4 | 0.1448 | 0.6655 | 0.8745 | 0.8740 |
| 5 | 0.1015 | 0.6731 | 0.8745 | 0.8741 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9149 | 0.8704 | 0.8921 |
| neutral | 252 | 0.8379 | 0.8413 | 0.8396 |
| positive | 282 | 0.8737 | 0.9078 | 0.8904 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[215, 22, 10]
[13, 212, 27]
[7, 19, 256]
```

Run `85d7e5e0020da9529d20e65df1e8dde37c3475d42f68b98ec2354d75ece85df9`; code `fb810900b14e58f9593e43f0b382f88a03edd193`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
