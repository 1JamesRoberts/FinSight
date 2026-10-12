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
| roberta_base | 0.8860 | 0.8849 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "RobertaForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 498591756, "parameters": 124647939, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 503,431,895.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 0.600s; 0.768 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 29.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 4; patience triggered False; total steps 1970.
Training/validation/saving: 198.20s; peak allocated/reserved bytes 2578125824/2900361216.
Training truncation: `{"training": {"maximum_tokens": 39, "truncated_texts": 0}, "validation": {"maximum_tokens": 29, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/9a3e11b759fc223eb467e7393b73f002b01095a4eb261a5aee64288cf4d1ad59`; manifest `9a3e11b759fc223eb467e7393b73f002b01095a4eb261a5aee64288cf4d1ad59`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6096 | 0.5242 | 0.8131 | 0.8043 |
| 2 | 0.2935 | 0.3985 | 0.8848 | 0.8832 |
| 3 | 0.1977 | 0.5560 | 0.8809 | 0.8791 |
| 4 | 0.1419 | 0.6350 | 0.8860 | 0.8849 |
| 5 | 0.0984 | 0.6841 | 0.8771 | 0.8754 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9012 | 0.8866 | 0.8939 |
| neutral | 252 | 0.8787 | 0.8333 | 0.8554 |
| positive | 282 | 0.8796 | 0.9326 | 0.9053 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[219, 18, 10]
[16, 210, 26]
[8, 11, 263]
```

Run `9cbe8a6f4216456883766c9830017006867f62bbb99bbf1e34c3bf5a87ef276b`; code `c62c1132c071d37ff386563ca350f8b354bb3d53`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
