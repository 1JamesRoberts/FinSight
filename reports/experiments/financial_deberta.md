# financial_deberta

SEntFiN validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis` at `9e10915c245a80a89b18d1ac51350e093c7bb35a`.
Mode: unchanged_sentiment; train 0, validation 781.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.8195 | 0.8181 |
| vader | 0.4738 | 0.4787 |
| training_majority | 0.3611 | 0.1769 |
| linear_svm | 0.8169 | 0.8160 |
| finbert-validation | 0.7478 | 0.7492 |
| finbert-finetuning | 0.8771 | 0.8762 |
| financial_deberta | 0.7388 | 0.7414 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 6, "parameter_bytes": 567588876, "parameters": 141897219, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 578,729,878.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.346s; 1.723 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 27.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8514 | 0.7652 | 0.8060 |
| neutral | 252 | 0.6246 | 0.7063 | 0.6629 |
| positive | 282 | 0.7664 | 0.7447 | 0.7554 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[189, 45, 13]
[23, 178, 51]
[10, 62, 210]
```

Run `f2c8cac19238ca68ed55a07d26bafc70e9e635d9fbe65220c9be298c68c69c27`; code `631d21211262b6cba3fba2be53324785190ac7c5`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
