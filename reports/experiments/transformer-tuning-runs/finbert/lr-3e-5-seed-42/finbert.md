# FinBERT adaptation on SEntFiN

SEntFiN development only; single configuration/seed; full adaptation of a PhraseBank-supervised checkpoint; weight license scope unresolved; local research decision

Continues training all parameters of the existing financial sentiment checkpoint. This is adaptation, not training a sentiment head on a base language model.

Train: 6299; validation: 781; PhraseBank, FiQA, multi-target headlines, and frozen test groups excluded from this run.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.8195 | 0.8181 |
| vader | 0.4738 | 0.4787 |
| training_majority | 0.3611 | 0.1769 |
| linear_svm | 0.8169 | 0.8160 |
| finbert_pretrained | 0.7478 | 0.7492 |
| finbert_adapted | 0.8771 | 0.8766 |

## Fixed configuration and checkpoint selection

Learning rate 3e-05; train batch 16; evaluation batch 32; max length 128; seed 42; up to 5 epochs; patience 2.
AdamW, betas=(0.9,0.999), eps=1e-8; bias/LayerNorm excluded from decay.
linear warm-up then linear decay over maximum configured steps.
FP32; deterministic algorithms; TF32 disabled; no AMP.
Weight decay 0.01; warm-up 197 steps; gradient clip norm 1.0. Training order shuffled with fixed generator; no validation gradients or class weighting.
maximum validation macro F1; strict improvement; earliest exact tie.
Selected epoch **2**; early stopped: True. Optimizer steps 1576/1970.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 | Improved | Seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 0.4864 | 0.4080 | 0.8489 | 0.8448 | True | 38.63 |
| 2 | 0.2552 | 0.4294 | 0.8771 | 0.8766 | True | 38.77 |
| 3 | 0.1366 | 0.5896 | 0.8694 | 0.8685 | False | 38.29 |
| 4 | 0.0680 | 0.6972 | 0.8732 | 0.8719 | False | 39.23 |

## Model architecture, storage, and timing

BertForSequenceClassification: 12 layers, hidden 768, 12 heads, feed-forward 3072, vocabulary 30522.
Parameters trained: 109,484,547; FP32 parameter bytes 437,938,188; saved model/tokenizer files 438,907,218 bytes.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8. Training wall time (epochs/validation/saves): 154.93s.
Peak training allocated/reserved memory: 2275302400 / 2598371328 bytes.
Reloaded inference: 0.550s, 0.704 amortized ms/text at evaluation batch size; synchronized, synthetic warm-up, includes tokenization/length audit and output transfer.
Historical baseline timings use different hardware/batches; no direct speed claim.
Truncation audit: `{"training": {"maximum_tokens": 38, "truncated_texts": 0}, "validation": {"maximum_tokens": 30, "truncated_texts": 0}}`.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8954 | 0.8664 | 0.8807 |
| neutral | 252 | 0.8178 | 0.8730 | 0.8445 |
| positive | 282 | 0.9194 | 0.8901 | 0.9045 |

Confusion matrix (rows=gold, columns=predicted; negative/neutral/positive):

```text
[214, 26, 7]
[17, 220, 15]
[8, 23, 251]
```

## Reproducibility and limitations

Run `3171cfcdcf7992a7fda939aeadf84b06d91a4a6a7ea06d9ee0d54af142a8fd86`; code revision `c62c1132c071d37ff386563ca350f8b354bb3d53`; working tree dirty `True`.
Best artifact: ignored data/interim/finbert-finetuned-models/428d36bf1eb516f708b6420f8bf75591f29d989911571046e56b49fc80edc7b5; manifest SHA-256 `428d36bf1eb516f708b6420f8bf75591f29d989911571046e56b49fc80edc7b5`.
The best saved artifact was reloaded and its validation metrics matched the selected epoch. JSON logs source/model/artifact/configuration hashes, frozen lineage, ordered membership, per-epoch metrics, baselines, timing and environment. Predictions and weights stay ignored.
Validation chooses the epoch and therefore is not an unbiased final performance estimate. One configuration and seed; no hyperparameter grid or statistical superiority claim. FinBERT previously used PhraseBank supervision; complete raw-news pretraining exposure is unknown. Probabilities remain uncalibrated. The frozen test remains unscored.
Local research and derivative storage follow the documented user decision; hosted weight license scope remains unresolved and public redistribution is outside scope.
