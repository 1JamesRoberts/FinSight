# Pretrained FinBERT validation

Unchanged pretrained checkpoint; verified validation cohorts only

No project fine-tuning, tuning, calibration, or final-test scoring.
PhraseBank has known FinBERT supervision overlap; its scores are diagnostic.
SEntFiN is development validation; full pretraining exposure is unknown.

| Cohort | Model | Validation texts | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| phrasebank | logistic_regression | 518 | 0.7780 | 0.7446 |
| phrasebank | vader | 518 | 0.5560 | 0.4895 |
| phrasebank | training_majority | 518 | 0.5734 | 0.2429 |
| phrasebank | linear_svm | 518 | 0.7819 | 0.7444 |
| phrasebank | finbert_pretrained | 518 | 0.8822 | 0.8823 |
| sentfin_single_target | logistic_regression | 781 | 0.8195 | 0.8181 |
| sentfin_single_target | vader | 781 | 0.4738 | 0.4787 |
| sentfin_single_target | training_majority | 781 | 0.3611 | 0.1769 |
| sentfin_single_target | linear_svm | 781 | 0.8169 | 0.8160 |
| sentfin_single_target | finbert_pretrained | 781 | 0.7478 | 0.7492 |

## Model architecture and size

Checkpoint: `ProsusAI/finbert` at `4556d13015211d73dccd3fdd39d39232506f3e43`.
BertForSequenceClassification: 12 encoder layers, hidden size 768, 12 heads, feed-forward size 3072, vocabulary 30522; three sentiment outputs.
Parameters: **109,484,547**; gradient-enabled: 0.
FP32 parameter storage: 437,938,188 bytes; downloaded checkpoint files: 438,226,859 bytes.
Index mapping: 0=positive, 1=negative, 2=neutral. Softmax probabilities are uncalibrated; exact ties retain checkpoint index order.

## Inference settings and timing

Device: cuda / NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8. Batch size 32; max length 128 including special tokens; dynamic padding, right truncation, FP32, eager attention, evaluation mode, no gradients, deterministic algorithms, TF32 disabled.
one pass after synthetic warm-up; tokenization/length audit, GPU transfer/forward/softmax and CPU output; CUDA synchronized; excludes loading, metrics, hashing and serialization.
Phase 2 CPU timings used batch size one on another machine; no direct speed comparison.

| Cohort | Seconds | Amortized ms/text | Texts/sec | Truncated | Max tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| phrasebank | 0.577 | 1.113 | 898.2 | 0 | 93 |
| sentfin_single_target | 0.562 | 0.719 | 1390.2 | 0 | 30 |

Peak GPU allocated bytes: 601229824; reserved bytes: 861929472. Includes model residency and inference after the warm-up reset.
Loading time (including acquisition/cache lookup): 38.091s.

## Class results

### phrasebank

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.8375 | 0.9710 | 0.8993 |
| neutral | 297 | 0.9508 | 0.8451 | 0.8948 |
| positive | 152 | 0.7989 | 0.9145 | 0.8528 |

Confusion matrix (rows=gold, columns=predicted; negative/neutral/positive):

```text
[67, 1, 1]
[12, 251, 34]
[1, 12, 139]
```

### sentfin_single_target

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.7924 | 0.7571 | 0.7743 |
| neutral | 252 | 0.6397 | 0.7540 | 0.6922 |
| positive | 282 | 0.8347 | 0.7340 | 0.7811 |

Confusion matrix (rows=gold, columns=predicted; negative/neutral/positive):

```text
[187, 49, 11]
[32, 190, 30]
[17, 58, 207]
```

## Reproducibility and use scope

Run: `ee63070bdb16c74bef562cddd8a1f30c92ba757489d8203477c6fb3d45b43e2c`; code revision: `a83794d0323e3380e763500b75ea2c1850c0d280`; working tree dirty: `False`.
Parent split: `a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898`.
Frozen benchmark: `b7cc249017ddd87f0b2c52a3abf3ff5b96f6920832b411f18eac547fec0b0f1c` remains unscored.
JSON records configuration, environment, source/checkpoint hashes, validation membership digests, and saved baseline run IDs. Local run, acquisition receipt, and predictions are under ignored data/interim/finbert-validation/.
Weights are cached locally; weight-license scope remains unresolved. Use follows the user's local noncommercial research decision; public weight/text redistribution is outside scope.
Historical Phase 1 source acquisition records remain unchanged.
