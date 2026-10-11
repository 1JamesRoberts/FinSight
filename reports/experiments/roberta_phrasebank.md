# roberta_phrasebank

phrasebank selects trained epochs; development validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `FacebookAI/roberta-base` at `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
Mode: new_sentiment_head; train 3785, validation 518.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.7780 | 0.7446 |
| vader | 0.5560 | 0.4895 |
| training_majority | 0.5734 | 0.2429 |
| linear_svm | 0.7819 | 0.7444 |
| finbert-validation | 0.8822 | 0.8823 |
| roberta_phrasebank | 0.8938 | 0.8868 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "RobertaForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 498591756, "parameters": 124647939, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 503,431,895.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 0.681s; 1.315 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 86.

## Training

Strategy: new_sentiment_head; all parameters trained on phrasebank only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 5; patience triggered False; total steps 1185.
Training/validation/saving: 119.90s; peak allocated/reserved bytes 3238147072/3860856832.
Training truncation: `{"training": {"maximum_tokens": 133, "truncated_texts": 1}, "validation": {"maximum_tokens": 86, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/9f151e0c4a70285d2f75038297bc9baf4b3181846546d97d8798f03060a8af96`; manifest `9f151e0c4a70285d2f75038297bc9baf4b3181846546d97d8798f03060a8af96`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6561 | 0.4006 | 0.8475 | 0.8491 |
| 2 | 0.3079 | 0.3680 | 0.8822 | 0.8700 |
| 3 | 0.1794 | 0.4353 | 0.8764 | 0.8739 |
| 4 | 0.1261 | 0.5381 | 0.8803 | 0.8730 |
| 5 | 0.0647 | 0.5636 | 0.8938 | 0.8868 |

## SEntFiN transfer validation

Not used for checkpoint selection; no SEntFiN training.
Accuracy 0.7836; macro F1 0.7845 on 781 texts.
PhraseBank FinBERT comparison is diagnostic: known upstream supervision.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.8732 | 0.8986 | 0.8857 |
| neutral | 297 | 0.9271 | 0.8990 | 0.9128 |
| positive | 152 | 0.8428 | 0.8816 | 0.8617 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[62, 4, 3]
[8, 267, 22]
[1, 17, 134]
```

Run `e71a9a56f66fbf81d0c5f57d848472b9f2aaea475a256b8b98bf89e622d67763`; code `fb810900b14e58f9593e43f0b382f88a03edd193`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
