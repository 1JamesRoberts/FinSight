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
| deberta_base | 0.8771 | 0.8760 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 1e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 43, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 737697804, "parameters": 184424451, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 740,189,720.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.327s; 1.699 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 27.

## Training

Strategy: new_sentiment_head; all parameters trained on sentfin_single_target only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 4; patience triggered False; total steps 1970.
Training/validation/saving: 387.32s; peak allocated/reserved bytes 3776869888/4857004032.
Training truncation: `{"training": {"maximum_tokens": 36, "truncated_texts": 0}, "validation": {"maximum_tokens": 27, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/bc029914f8cabb7a888980b3778788a1cbb70516a6155eed268c8a34a8b35dd5`; manifest `bc029914f8cabb7a888980b3778788a1cbb70516a6155eed268c8a34a8b35dd5`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.7130 | 0.4267 | 0.8617 | 0.8607 |
| 2 | 0.3096 | 0.4227 | 0.8694 | 0.8692 |
| 3 | 0.2141 | 0.4977 | 0.8668 | 0.8665 |
| 4 | 0.1620 | 0.5469 | 0.8771 | 0.8760 |
| 5 | 0.1326 | 0.5867 | 0.8707 | 0.8696 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9061 | 0.8988 | 0.9024 |
| neutral | 252 | 0.8707 | 0.8016 | 0.8347 |
| positive | 282 | 0.8586 | 0.9255 | 0.8908 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[222, 17, 8]
[15, 202, 35]
[8, 13, 261]
```

Run `19995a8ed377624f4ac06bdcf75eb45adb875c9568e4a30dbacca580d7a31dbb`; code `d691078a57d57cd8a8b5dddc2f0da9e0e08edbfb`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
