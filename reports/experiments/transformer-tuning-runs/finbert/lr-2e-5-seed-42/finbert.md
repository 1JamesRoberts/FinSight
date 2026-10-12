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
| finbert_adapted | 0.8771 | 0.8762 |

## Fixed configuration and checkpoint selection

Learning rate 2e-05; train batch 16; evaluation batch 32; max length 128; seed 42; up to 5 epochs; patience 2.
AdamW, betas=(0.9,0.999), eps=1e-8; bias/LayerNorm excluded from decay.
linear warm-up then linear decay over maximum configured steps.
FP32; deterministic algorithms; TF32 disabled; no AMP.
Weight decay 0.01; warm-up 197 steps; gradient clip norm 1.0. Training order shuffled with fixed generator; no validation gradients or class weighting.
maximum validation macro F1; strict improvement; earliest exact tie.
Selected epoch **3**; early stopped: True. Optimizer steps 1970/1970.

| Epoch | Train loss | Validation loss | Accuracy | Macro F1 | Improved | Seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 0.4971 | 0.4310 | 0.8399 | 0.8354 | True | 39.64 |
| 2 | 0.2615 | 0.4236 | 0.8707 | 0.8698 | True | 39.54 |
| 3 | 0.1474 | 0.5642 | 0.8771 | 0.8762 | True | 38.85 |
| 4 | 0.0785 | 0.6671 | 0.8707 | 0.8693 | False | 37.78 |
| 5 | 0.0452 | 0.6799 | 0.8707 | 0.8691 | False | 38.06 |

## Model architecture, storage, and timing

BertForSequenceClassification: 12 layers, hidden 768, 12 heads, feed-forward 3072, vocabulary 30522.
Parameters trained: 109,484,547; FP32 parameter bytes 437,938,188; saved model/tokenizer files 438,907,218 bytes.
GPU: NVIDIA GeForce RTX 4070 SUPER; CUDA 12.8. Training wall time (epochs/validation/saves): 193.88s.
Peak training allocated/reserved memory: 2275302400 / 2598371328 bytes.
Reloaded inference: 0.615s, 0.788 amortized ms/text at evaluation batch size; synchronized, synthetic warm-up, includes tokenization/length audit and output transfer.
Historical baseline timings use different hardware/batches; no direct speed claim.
Truncation audit: `{"training": {"maximum_tokens": 38, "truncated_texts": 0}, "validation": {"maximum_tokens": 30, "truncated_texts": 0}}`.

## Class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.9060 | 0.8583 | 0.8815 |
| neutral | 252 | 0.8158 | 0.8611 | 0.8378 |
| positive | 282 | 0.9110 | 0.9078 | 0.9094 |

Confusion matrix (rows=gold, columns=predicted; negative/neutral/positive):

```text
[212, 28, 7]
[17, 217, 18]
[5, 21, 256]
```

## Reproducibility and limitations

Run `f286df58936cf1e0aba03da79803deb4e56c208b1316daab2d8829a76b9c4fdc`; code revision `c62c1132c071d37ff386563ca350f8b354bb3d53`; working tree dirty `True`.
Best artifact: ignored data/interim/finbert-finetuned-models/276b014dec08139a12ec8ea85f5ba94d6f271148c62b49f26d37ea8e472936a7; manifest SHA-256 `276b014dec08139a12ec8ea85f5ba94d6f271148c62b49f26d37ea8e472936a7`.
The best saved artifact was reloaded and its validation metrics matched the selected epoch. JSON logs source/model/artifact/configuration hashes, frozen lineage, ordered membership, per-epoch metrics, baselines, timing and environment. Predictions and weights stay ignored.
Validation chooses the epoch and therefore is not an unbiased final performance estimate. One configuration and seed; no hyperparameter grid or statistical superiority claim. FinBERT previously used PhraseBank supervision; complete raw-news pretraining exposure is unknown. Probabilities remain uncalibrated. The frozen test remains unscored.
Local research and derivative storage follow the documented user decision; hosted weight license scope remains unresolved and public redistribution is outside scope.
