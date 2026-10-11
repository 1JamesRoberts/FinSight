# deberta_phrasebank

phrasebank selects trained epochs; development validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `microsoft/deberta-v3-base` at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
Mode: new_sentiment_head; train 3785, validation 518.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.7780 | 0.7446 |
| vader | 0.5560 | 0.4895 |
| training_majority | 0.5734 | 0.2429 |
| linear_svm | 0.7819 | 0.7444 |
| finbert-validation | 0.8822 | 0.8823 |
| deberta_phrasebank | 0.8900 | 0.8857 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 737697804, "parameters": 184424451, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 740,189,720.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.075s; 2.074 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 81.

## Training

Strategy: new_sentiment_head; all parameters trained on phrasebank only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 4; patience triggered False; total steps 1185.
Training/validation/saving: 241.86s; peak allocated/reserved bytes 5985523200/6652166144.
Training truncation: `{"training": {"maximum_tokens": 128, "truncated_texts": 0}, "validation": {"maximum_tokens": 81, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/976d80967708729fa1649282a643133b8aa18f71f178e6d114c718859881ab72`; manifest `976d80967708729fa1649282a643133b8aa18f71f178e6d114c718859881ab72`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.7236 | 0.4959 | 0.8147 | 0.8208 |
| 2 | 0.3006 | 0.4211 | 0.8668 | 0.8685 |
| 3 | 0.1801 | 0.4489 | 0.8822 | 0.8780 |
| 4 | 0.1118 | 0.5049 | 0.8900 | 0.8857 |
| 5 | 0.0553 | 0.5716 | 0.8822 | 0.8801 |

## SEntFiN transfer validation

Not used for checkpoint selection; no SEntFiN training.
Accuracy 0.7798; macro F1 0.7816 on 781 texts.
PhraseBank FinBERT comparison is diagnostic: known upstream supervision.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.9508 | 0.8406 | 0.8923 |
| neutral | 297 | 0.9233 | 0.8923 | 0.9075 |
| positive | 152 | 0.8118 | 0.9079 | 0.8571 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[58, 8, 3]
[3, 265, 29]
[0, 14, 138]
```

Run `15bfcd783eda29d2e9cd4934ae2fadbda7225218ae0d65ef010b64b09f34a564`; code `fb810900b14e58f9593e43f0b382f88a03edd193`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
