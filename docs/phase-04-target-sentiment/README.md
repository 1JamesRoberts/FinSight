# Phase 4 — Company-targeted sentiment

Native target conditioning, matched baselines, error analysis and reserved evaluation.

[All documentation](../README.md) · [Report navigation](../../reports/README.md)

**Execution update:** v1 stopped at its overlap audit before inference. The authorized v2 correction excludes that reviewed group; all 21 variants completed and their exported results verified. See the [current status](../../reports/phase-04-target-sentiment/README.md#evaluation-execution-status).

## Data

- [Phase 4 development coverage: method and findings](data/company-coverage.md)
- [Phase 4: company target registry](data/company-targets.md)
- [Phase 4 target input and baseline protocol](data/target-sentiment-protocol.md)

## Baselines

- [Target-level TF-IDF Logistic Regression baselines](baselines/target-tfidf.md)

## Training

- [Matched target conditioning: RoBERTa Base and FinBERT](training/target-backbones.md)
- [Matched DeBERTa training for native targets](training/target-deberta.md)

[Results and original artifacts](../../reports/phase-04-target-sentiment/README.md)
## Analysis

- [Phase 4 target-error review](analysis/target-error-analysis.md)

## Evaluation

- [Phase 4 target evaluation runner](evaluation/target-evaluation-runner.md)
- [Current v2 locked evaluation](evaluation/target-final-evaluation-v2.md)
- [Historical v1 evaluation contract](evaluation/target-final-evaluation.md)


## Prediction

- [Company mention detection and automatic sentiment](company-mentions.md)

- [Reusable RoBERTa target prediction: Python and CLI](prediction.md)
