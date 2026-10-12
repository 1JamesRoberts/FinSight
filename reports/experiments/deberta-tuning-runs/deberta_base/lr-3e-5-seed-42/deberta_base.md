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
| deberta_base | 0.8835 | 0.8824 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 3e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 737697804, "parameters": 184424451, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 740,189,720.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.247s; 1.597 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 27.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 4; patience triggered False; total steps 1970.
Training/validation/saving: 385.13s; peak allocated/reserved bytes 3786812928/4857004032.
Training truncation: `{"training": {"maximum_tokens": 36, "truncated_texts": 0}, "validation": {"maximum_tokens": 27, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/f8349a549720569bfc91edcb479667a91bcfeaa92a64bf5643c7f77bfb1ff5b9`; manifest `f8349a549720569bfc91edcb479667a91bcfeaa92a64bf5643c7f77bfb1ff5b9`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.5904 | 0.5319 | 0.8169 | 0.8074 |
| 2 | 0.2657 | 0.4368 | 0.8835 | 0.8823 |
| 3 | 0.1671 | 0.5469 | 0.8694 | 0.8677 |
| 4 | 0.0989 | 0.6342 | 0.8835 | 0.8824 |
| 5 | 0.0623 | 0.6556 | 0.8809 | 0.8794 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9129 | 0.8907 | 0.9016 |
| neutral | 252 | 0.8766 | 0.8175 | 0.8460 |
| positive | 282 | 0.8656 | 0.9362 | 0.8995 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[220, 17, 10]
[15, 206, 31]
[6, 12, 264]
```

Run `179b2b4abc5a4cbe269909a0c7cc9bb15e85f53e38f4817be8d46a09cd428213`; code `d691078a57d57cd8a8b5dddc2f0da9e0e08edbfb`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
