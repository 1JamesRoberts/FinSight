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
| deberta_base | 0.8835 | 0.8826 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 1e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 737697804, "parameters": 184424451, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 740,189,720.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.195s; 1.530 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 27.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 2; patience triggered True; total steps 1576.
Training/validation/saving: 327.42s; peak allocated/reserved bytes 3786812928/4857004032.
Training truncation: `{"training": {"maximum_tokens": 36, "truncated_texts": 0}, "validation": {"maximum_tokens": 27, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/90aa5ca7fbef01f2fe2ed92a7f10a9f6f0e5b2c3c15aeff32a316b34ebc54fce`; manifest `90aa5ca7fbef01f2fe2ed92a7f10a9f6f0e5b2c3c15aeff32a316b34ebc54fce`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6876 | 0.4764 | 0.8425 | 0.8386 |
| 2 | 0.2992 | 0.4141 | 0.8835 | 0.8826 |
| 3 | 0.2139 | 0.4486 | 0.8796 | 0.8787 |
| 4 | 0.1670 | 0.5271 | 0.8822 | 0.8819 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9174 | 0.8988 | 0.9080 |
| neutral | 252 | 0.8761 | 0.8135 | 0.8436 |
| positive | 282 | 0.8623 | 0.9326 | 0.8961 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[222, 15, 10]
[15, 205, 32]
[5, 14, 263]
```

Run `d763a66242396d4e7a81fc6dc53cd738f6ce9dcac538b51d5d76a885a53abc24`; code `d691078a57d57cd8a8b5dddc2f0da9e0e08edbfb`; dirty `False`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
