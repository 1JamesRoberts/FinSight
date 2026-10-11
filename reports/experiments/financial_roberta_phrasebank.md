# financial_roberta_phrasebank

Unchanged checkpoint inference on phrasebank; no project training; development validation only; no final test; one seed/configuration. Models differ in size and prior training. Raw-text pretraining exposure unknown. Financial RoBERTa supervised provenance incomplete. Financial DeBERTa appears PhraseBank-supervised. Probabilities uncalibrated.

Checkpoint `soleimanian/financial-roberta-large-sentiment` at `f8804d31111d7c3569e88abaad6969918e858fbd`.
Mode: unchanged_inference; train 0, validation 518.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.7780 | 0.7446 |
| vader | 0.5560 | 0.4895 |
| training_majority | 0.5734 | 0.2429 |
| linear_svm | 0.7819 | 0.7444 |
| finbert-validation | 0.8822 | 0.8823 |
| financial_roberta_phrasebank | 0.7973 | 0.8142 |

## Configuration and model

Configuration: `{"device": "cuda", "evaluation_batch_size": 32, "learning_rate": 2e-05, "max_epochs": 5, "max_grad_norm": 1.0, "max_length": 128, "patience": 2, "seed": 42, "train_batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01}`.
Architecture: `{"attention_heads": 16, "class": "RobertaForSequenceClassification", "hidden_size": 1024, "intermediate_size": 4096, "layers": 24, "parameter_bytes": 1421451276, "parameters": 355362819, "vocabulary_size": 50265}`.
Model/tokenizer/source-notice file bytes: 1,424,229,411.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8; FP32; eager attention; deterministic; TF32 disabled.
Inference: 1.532s; 2.958 ms/text at batch size 32.
Timing includes tokenization, length audit and output transfer after synthetic warm-up; synchronized.
Truncated validation texts: 0; maximum tokens: 86.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.8514 | 0.9130 | 0.8811 |
| neutral | 297 | 0.9409 | 0.6970 | 0.8008 |
| positive | 152 | 0.6384 | 0.9408 | 0.7606 |

Confusion matrix: rows gold, columns predicted; negative/neutral/positive.

```text
[63, 4, 2]
[11, 207, 79]
[0, 9, 143]
```

Run `79cbf3b2c770ab61493016807e47fd4c957c820d4b57198b3a0f6e2a342dd380`; code `acccf656bb1f314cb331a70cba02e2a3f5af1a85`; dirty `False`.
JSON pins source/model/configuration/code hashes, membership, baseline provenance and environment.
Weights and individual predictions remain ignored. Validation selects trained checkpoints; no unbiased test claim.
