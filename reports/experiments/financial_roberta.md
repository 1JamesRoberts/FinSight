# financial_roberta

SEntFiN validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `soleimanian/financial-roberta-large-sentiment` at `f8804d31111d7c3569e88abaad6969918e858fbd`.
Mode: unchanged_sentiment; train 0, validation 781.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.8195 | 0.8181 |
| vader | 0.4738 | 0.4787 |
| training_majority | 0.3611 | 0.1769 |
| linear_svm | 0.8169 | 0.8160 |
| finbert-validation | 0.7478 | 0.7492 |
| finbert-finetuning | 0.8771 | 0.8762 |
| financial_roberta | 0.7439 | 0.7333 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 16, "class": "RobertaForSequenceClassification", "hidden_size": 1024, "intermediate_size": 4096, "layers": 24, "parameter_bytes": 1421451276, "parameters": 355362819, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 1,424,229,411.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.051s; 1.346 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 29.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8140 | 0.8502 | 0.8317 |
| neutral | 252 | 0.7241 | 0.5000 | 0.5915 |
| positive | 282 | 0.7020 | 0.8688 | 0.7765 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[210, 27, 10]
[32, 126, 94]
[16, 21, 245]
```

Run `02bb45821689a958c7fc4ea035fc8a1dc73dcdf1b6db88fae6ed00cc9cdbfa29`; code `631d21211262b6cba3fba2be53324785190ac7c5`; dirty `False`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
