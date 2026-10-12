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
| finbert_adapted | 0.8771 | 0.8756 |

## Fixed configuration and checkpoint selection

Learning rate 3e-05; train batch 16; evaluation batch 32; max length 128; seed 43; up to 5 epochs; patience 2.
AdamW, betas=(0.9,0.999), eps=1e-8; bias/LayerNorm excluded from decay.
linear warm-up then linear decay over maximum configured steps.
FP32; deterministic algorithms; TF32 disabled; no AMP.
Weight decay 0.01; warm-up 197 steps; gradient clip norm 1.0. Training order shuffled with fixed generator; no validation gradients or class weighting.
maximum validation macro F1; strict improvement; earliest exact tie.
Selected epoch **5**; early stopped: False. Optimizer steps 1970/1970.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 | Improved | Seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 0.4712 | 0.4497 | 0.8617 | 0.8604 | True | 39.55 |
| 2 | 0.2514 | 0.4897 | 0.8707 | 0.8706 | True | 39.17 |
| 3 | 0.1355 | 0.5905 | 0.8668 | 0.8657 | False | 37.99 |
| 4 | 0.0625 | 0.7102 | 0.8745 | 0.8728 | True | 38.72 |
| 5 | 0.0318 | 0.7178 | 0.8771 | 0.8756 | True | 38.85 |

## Model architecture, storage, and timing

BertForSequenceClassification: 12 layers, hidden 768, 12 heads, feed-forward 3072, vocabulary 30522.
Parameters trained: 109,484,547; FP32 parameter bytes 437,938,188; saved model/tokenizer files 438,907,218 bytes.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8. Training wall time (epochs/validation/saves): 194.28s.
Peak training allocated/reserved memory: 2275759616 / 2596274176 bytes.
Reloaded inference: 0.695s, 0.890 amortized ms/text at evaluation batch size; synchronized, synthetic warm-up, includes tokenization/length audit and output transfer.
Historical baseline timings use different hardware/batches; no direct speed claim.
Truncation audit: `{"training": {"maximum_tokens": 38, "truncated_texts": 0}, "validation": {"maximum_tokens": 30, "truncated_texts": 0}}`.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9021 | 0.8583 | 0.8797 |
| neutral | 252 | 0.8406 | 0.8373 | 0.8390 |
| positive | 282 | 0.8881 | 0.9291 | 0.9081 |

Confusion matrix (rows=gold, columns=predicted; negative/neutral/positive):

```text
[212, 25, 10]
[18, 211, 23]
[5, 15, 262]
```

## Reproducibility and limitations

Run `8f9835fb19da34458b5f1da25f2f8731be92c5210fe30a9483b060f4ac31ac37`; code revision `c62c1132c071d37ff386563ca350f8b354bb3d53`; working tree dirty `True`.
Best artifact: ignored data/interim/finbert-finetuned-models/f294fdb6c30b641e960dca9a15775a870ada89f5ca027c0f5f95c14cd560a057; manifest SHA-256 `f294fdb6c30b641e960dca9a15775a870ada89f5ca027c0f5f95c14cd560a057`.
The best saved artifact was reloaded and its validation metrics matched the selected epoch. JSON logs source/model/artifact/configuration hashes, frozen lineage, ordered membership, per-epoch metrics, baselines, timing and environment. Predictions and weights stay ignored.
Validation chooses the epoch and therefore is not an unbiased final performance estimate. One configuration and seed; no hyperparameter grid or statistical superiority claim. FinBERT previously used PhraseBank supervision; complete raw-news pretraining exposure is unknown. Probabilities remain uncalibrated. The frozen test remains unscored.
Local research and derivative storage follow the documented user decision; hosted weight license scope remains unresolved and public redistribution is outside scope.
