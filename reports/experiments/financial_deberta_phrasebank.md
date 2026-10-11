# financial_deberta_phrasebank

Unchanged checkpoint inference on phrasebank; no project training; development validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis` at `9e10915c245a80a89b18d1ac51350e093c7bb35a`.
Mode: unchanged_inference; train 0, validation 518.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.7780 | 0.7446 |
| vader | 0.5560 | 0.4895 |
| training_majority | 0.5734 | 0.2429 |
| linear_svm | 0.7819 | 0.7444 |
| finbert-validation | 0.8822 | 0.8823 |
| financial_deberta_phrasebank | 0.9826 | 0.9851 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 12, "class": "DebertaV2ForSequenceClassification", "hidden_size": 768, "intermediate_size": 3072, "layers": 6, "parameter_bytes": 567588876, "parameters": 141897219, "vocabulary_size": 128100}`.
Model/tokenizer/source-notice file bytes: 578,729,878.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.159s; 2.237 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 81.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 1.0000 | 1.0000 | 1.0000 |
| neutral | 297 | 0.9865 | 0.9832 | 0.9848 |
| positive | 152 | 0.9673 | 0.9737 | 0.9705 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[69, 0, 0]
[0, 292, 5]
[0, 4, 148]
```

Run `a609d031440430f665b6cc69060fbe594949ce91b3a4ecc034bd9d52697c432b`; code `acccf656bb1f314cb331a70cba02e2a3f5af1a85`; dirty `True`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
