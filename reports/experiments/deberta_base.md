# deberta_base

SEntFiN validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `microsoft/deberta-v3-base` at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
Mode: fine_tune_base; train 6299, validation 781.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.8195 | 0.8181 |
| vader | 0.4738 | 0.4787 |
| training_majority | 0.3611 | 0.1769 |
| linear_svm | 0.8169 | 0.8160 |
| finbert-validation | 0.7478 | 0.7492 |
| finbert-finetuning | 0.8771 | 0.8762 |
| deberta_base | 0.8745 | 0.8738 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 12, "parameter_bytes": 737697804, "parameters": 184424451, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 740,189,720.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.381s; 1.768 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 27.

## Training

New three-class sentiment head; all encoder/head parameters trained on SEntFiN only.
AdamW (betas 0.9/0.999, eps 1e-8); bias/LayerNorm excluded from decay; linear warm-up/decay; gradient clipping.
Best epoch 5; patience triggered False; total steps 1970.
Training/validation/saving: 393.30s; peak allocated/reserved bytes 3786812928/4857004032.
Training truncation: `{"training": {"maximum_tokens": 36, "truncated_texts": 0}, "validation": {"maximum_tokens": 27, "truncated_texts": 0}}`.
Best artifact: `data/interim/transformer-finetuned-models/a90155447ce3f0910bf0b9c8bf341a2c932c97f6e01e1250466ed1a351174cf3`; manifest `a90155447ce3f0910bf0b9c8bf341a2c932c97f6e01e1250466ed1a351174cf3`.
Selected checkpoint reloaded and metrics reproduced. No optimizer state saved.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6162 | 0.4844 | 0.8425 | 0.8376 |
| 2 | 0.2694 | 0.4947 | 0.8643 | 0.8628 |
| 3 | 0.1808 | 0.5352 | 0.8707 | 0.8698 |
| 4 | 0.1144 | 0.6790 | 0.8732 | 0.8725 |
| 5 | 0.0818 | 0.6664 | 0.8745 | 0.8738 |

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9020 | 0.8947 | 0.8984 |
| neutral | 252 | 0.8512 | 0.8175 | 0.8340 |
| positive | 282 | 0.8707 | 0.9078 | 0.8889 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[221, 18, 8]
[16, 206, 30]
[8, 18, 256]
```

Run `a3b9f03d1d019e2830b0cebf02b399a2d8fa5993f2a1507540034fd8056fc66a`; code `631d21211262b6cba3fba2be53324785190ac7c5`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
