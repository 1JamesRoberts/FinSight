# deberta_base

sentfin_single_target selects trained epochs; development validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `microsoft/deberta-v3-base` at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
Mode: new_sentiment_head; train 6299, validation 781.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.8195 | 0.8181 |
| vader | 0.4738 | 0.4787 |
| training_majority | 0.3611 | 0.1769 |
| linear_svm | 0.8169 | 0.8160 |
| finbert-validation | 0.7478 | 0.7492 |
| finbert-finetuning | 0.8771 | 0.8762 |
| deberta_base | 0.8848 | 0.8844 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 1e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 44, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 737697804, "parameters": 184424451, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 740,189,720.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.337s; 1.712 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 27.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 5; patience triggered False; total steps 1970.
Training/validation/saving: 391.15s; peak allocated/reserved bytes 3774094336/4462739456.
Training truncation: `{"training": {"maximum_tokens": 36, "truncated_texts": 0}, "validation": {"maximum_tokens": 27, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/93f85da57d6f83bcaa17cbeb5b690f7eadfa2cb82a2fba82b1522d8e2a5ae209`; manifest `93f85da57d6f83bcaa17cbeb5b690f7eadfa2cb82a2fba82b1522d8e2a5ae209`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6629 | 0.3975 | 0.8553 | 0.8536 |
| 2 | 0.2865 | 0.4229 | 0.8694 | 0.8695 |
| 3 | 0.2128 | 0.4562 | 0.8809 | 0.8806 |
| 4 | 0.1618 | 0.5357 | 0.8809 | 0.8802 |
| 5 | 0.1357 | 0.5374 | 0.8848 | 0.8844 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9313 | 0.8785 | 0.9042 |
| neutral | 252 | 0.8526 | 0.8492 | 0.8509 |
| positive | 282 | 0.8754 | 0.9220 | 0.8981 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[217, 19, 11]
[12, 214, 26]
[4, 18, 260]
```

Run `c6f14e9f38d9d6272a6da9918826097854e42a698bf8681cdb2ae695b0c1d7ca`; code `d691078a57d57cd8a8b5dddc2f0da9e0e08edbfb`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
