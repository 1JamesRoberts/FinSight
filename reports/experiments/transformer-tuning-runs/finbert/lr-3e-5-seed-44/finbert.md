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
| finbert_adapted | 0.8732 | 0.8722 |

## Fixed configuration and checkpoint selection

Learning rate 3e-05; train batch 16; evaluation batch 32; max length 128; seed 44; up to 5 epochs; patience 2.
AdamW, betas=(0.9,0.999), eps=1e-8; bias/LayerNorm excluded from decay.
linear warm-up then linear decay over maximum configured steps.
FP32; deterministic algorithms; TF32 disabled; no AMP.
Weight decay 0.01; warm-up 197 steps; gradient clip norm 1.0. Training order shuffled with fixed generator; no validation gradients or class weighting.
maximum validation macro F1; strict improvement; earliest exact tie.
Selected epoch **3**; early stopped: True. Optimizer steps 1970/1970.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 | Improved | Seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 0.4744 | 0.4409 | 0.8553 | 0.8538 | True | 39.31 |
| 2 | 0.2565 | 0.5157 | 0.8694 | 0.8690 | True | 39.73 |
| 3 | 0.1441 | 0.5679 | 0.8732 | 0.8722 | True | 39.62 |
| 4 | 0.0778 | 0.6952 | 0.8720 | 0.8706 | False | 39.35 |
| 5 | 0.0318 | 0.7358 | 0.8694 | 0.8680 | False | 38.62 |

## Model architecture, storage, and timing

BertForSequenceClassification: 12 layers, hidden 768, 12 heads, feed-forward 3072, vocabulary 30522.
Parameters trained: 109,484,547; FP32 parameter bytes 437,938,188; saved model/tokenizer files 438,907,218 bytes.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8. Training wall time (epochs/validation/saves): 196.62s.
Peak training allocated/reserved memory: 2274249216 / 2594177024 bytes.
Reloaded inference: 0.567s, 0.726 amortized ms/text at evaluation batch size; synchronized, synthetic warm-up, includes tokenization/length audit and output transfer.
Historical baseline timings use different hardware/batches; no direct speed claim.
Truncation audit: `{"training": {"maximum_tokens": 38, "truncated_texts": 0}, "validation": {"maximum_tokens": 30, "truncated_texts": 0}}`.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.8848 | 0.8704 | 0.8776 |
| neutral | 252 | 0.8314 | 0.8413 | 0.8363 |
| positive | 282 | 0.9011 | 0.9043 | 0.9027 |

Confusion matrix (rows=gold, columns=predicted; negative/neutral/positive):

```text
[215, 25, 7]
[19, 212, 21]
[9, 18, 255]
```

## Reproducibility and limitations

Run `7c834d6588517132cf30b05f76852f8ffd2155d5a32d17f1b54efbecc31ecee8`; code revision `c62c1132c071d37ff386563ca350f8b354bb3d53`; working tree dirty `True`.
Best artifact: ignored data/interim/finbert-finetuned-models/23f0ac0774c2f64233010fe21c3adcd5bbd7bbdde2f5d119318a31d918c581e6; manifest SHA-256 `23f0ac0774c2f64233010fe21c3adcd5bbd7bbdde2f5d119318a31d918c581e6`.
The best saved artifact was reloaded and its validation metrics matched the selected epoch. JSON logs source/model/artifact/configuration hashes, frozen lineage, ordered membership, per-epoch metrics, baselines, timing and environment. Predictions and weights stay ignored.
Validation chooses the epoch and therefore is not an unbiased final performance estimate. One configuration and seed; no hyperparameter grid or statistical superiority claim. FinBERT previously used PhraseBank supervision; complete raw-news pretraining exposure is unknown. Probabilities remain uncalibrated. The frozen test remains unscored.
Local research and derivative storage follow the documented user decision; hosted weight license scope remains unresolved and public redistribution is outside scope.
