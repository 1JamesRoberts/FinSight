# Phase 3 — Sentence-level transformers reports

**Start here:** [Final report](final-report.md). This is the canonical phase summary; the task evidence below retains its original scope.

Financial/base transformer comparisons, seed studies, calibration, ensembles and locked evaluation.

[All reports](../README.md) · [Methods and reproduction guides](../../docs/phase-03-transformers/README.md)

## Start here: consolidated comparisons

| Readable report | Machine-readable artifact |
| --- | --- |
| [FinSight — All-model comparison by cohort](../experiments/all-models-by-cohort.md) | — |
| [Locked Phase 3 final sentiment evaluation](../experiments/phase3-final.md) | [JSON](../experiments/phase3-final.json) |

## Model and dataset reports

| Readable report | Machine-readable artifact |
| --- | --- |
| [deberta_base](../experiments/deberta_base.md) | [JSON](../experiments/deberta_base.json) |
| [deberta_phrasebank](../experiments/deberta_phrasebank.md) | [JSON](../experiments/deberta_phrasebank.json) |
| [financial_deberta](../experiments/financial_deberta.md) | [JSON](../experiments/financial_deberta.json) |
| [financial_deberta_adapted](../experiments/financial_deberta_adapted.md) | [JSON](../experiments/financial_deberta_adapted.json) |
| [financial_deberta_phrasebank](../experiments/financial_deberta_phrasebank.md) | [JSON](../experiments/financial_deberta_phrasebank.json) |
| [financial_roberta](../experiments/financial_roberta.md) | [JSON](../experiments/financial_roberta.json) |
| [financial_roberta_adapted](../experiments/financial_roberta_adapted.md) | [JSON](../experiments/financial_roberta_adapted.json) |
| [financial_roberta_phrasebank](../experiments/financial_roberta_phrasebank.md) | [JSON](../experiments/financial_roberta_phrasebank.json) |
| [Pretrained FinBERT validation](../experiments/finbert-validation.md) | [JSON](../experiments/finbert-validation.json) |
| [Financial DeBERTa Base: SEntFiN adaptation](../experiments/proprak-deberta-adaptation.md) | [JSON](../experiments/proprak-deberta-adaptation.json) |
| [Exploratory ProPrak01 financial DeBERTa comparison](../experiments/proprak-deberta-validation.md) | [JSON](../experiments/proprak-deberta-validation.json) |
| [roberta_base](../experiments/roberta_base.md) | [JSON](../experiments/roberta_base.json) |
| [roberta_phrasebank](../experiments/roberta_phrasebank.md) | [JSON](../experiments/roberta_phrasebank.json) |
| [Phase 3: first transformer comparison](../experiments/transformer-comparison.md) | [JSON](../experiments/transformer-comparison.json) |
| [Transformer follow-up: financial adaptation and PhraseBank training](../experiments/transformer-followup.md) | [JSON](../experiments/transformer-followup.json) |

## Tuning and seed studies

| Readable report | Machine-readable artifact |
| --- | --- |
| [DeBERTa-v3 Base — learning-rate and three-seed study](../experiments/deberta-tuning.md) | [JSON](../experiments/deberta-tuning.json) |
| [FinBERT adaptation on SEntFiN](../experiments/finbert-finetuning.md) | [JSON](../experiments/finbert-finetuning.json) |
| [FinSight — SEntFiN learning-rate and seed comparison](../experiments/transformer-tuning.md) | [JSON](../experiments/transformer-tuning.json) |

## Diagnostics and error review

| Readable report | Machine-readable artifact |
| --- | --- |
| [DeBERTa-v3 Base — calibration and qualitative errors](../experiments/deberta-analysis.md) | [JSON](../experiments/deberta-analysis.json) |
| [Phase 3 — Transformer calibration and qualitative errors](../experiments/transformer-analysis.md) | [JSON](../experiments/transformer-analysis.json) |

## Probability calibration

| Readable report | Machine-readable artifact |
| --- | --- |
| [Three-model temperature-scaling experiment](../experiments/temperature-scaling.md) | [JSON](../experiments/temperature-scaling.json) |

## Ensembles

| Readable report | Machine-readable artifact |
| --- | --- |
| [Soft and democratic voting on SEntFiN validation](../experiments/ensemble-voting.md) | [JSON](../experiments/ensemble-voting.json) |

## Trial records

Detailed per-rate/per-seed records remain in their original run archives:

- [transformer-tuning-runs](../experiments/transformer-tuning-runs)
- [deberta-tuning-runs](../experiments/deberta-tuning-runs)

- [Exploratory financial DeBERTa adaptation: per-seed JSON](../experiments/proprak-deberta-adapted)

## Figures

- [Transformer/calibration/ensemble figures](../figures/phase3)
