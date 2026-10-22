# Phase 4 — Company-targeted sentiment

**Phase outcome:** [Final report](../../reports/phase-04-target-sentiment/final-report.md) · [All final reports](../../reports/README.md).

Native target conditioning, matched baselines, error analysis and reserved evaluation.

[All documentation](../README.md) · [Report navigation](../../reports/README.md)

**Execution update:** v1 stopped at its overlap audit before inference. The authorized v2 correction excludes that reviewed group; all 21 variants completed and their exported results verified. See the [current status](../../reports/phase-04-target-sentiment/README.md#evaluation-execution-status).

## Data

- [Completed reference review and word-aware grouping decision](../../reports/phase-04-target-sentiment/evaluation/organization-benchmark-reviewed.md)

- [Organization mention benchmark: sample, annotation and human review](data/organization-benchmark.md)
- [Company/organization span annotation policy](data/organization-annotation-policy.md)

- [Phase 4 development coverage: method and findings](data/company-coverage.md)
- [Phase 4: company target registry](data/company-targets.md)
- [Phase 4 target input and baseline protocol](data/target-sentiment-protocol.md)

## Baselines

- [Target-level TF-IDF Logistic Regression baselines](baselines/target-tfidf.md)

## Training

- [Completed target-format comparison and recommendation](../../reports/phase-04-target-sentiment/experiments/target-format-comparison.md)
- [Detached target-format runner and progress commands](training/target-format-runner.md)
- [Verified target-format tokenizer audit and nine-run lock](../../reports/phase-04-target-sentiment/evaluation/target-tokenizer-audit.md)
- [Target markers and replacement: span audit and experiment preparation](training/target-format-study.md)
- [Matched target conditioning: RoBERTa Base and FinBERT](training/target-backbones.md)
- [Matched DeBERTa training for native targets](training/target-deberta.md)

[Results and original artifacts](../../reports/phase-04-target-sentiment/README.md)
## Analysis

- [Synthetic company-name sensitivity: fixed method and reproduction](analysis/target-name-stress.md)
- [Reviewed name-sensitivity results](../../reports/phase-04-target-sentiment/experiments/target-name-stress.md)

- [Phase 4 target-error review](analysis/target-error-analysis.md)

## Evaluation

- [Phase 4 target evaluation runner](evaluation/target-evaluation-runner.md)
- [Current v2 locked evaluation](evaluation/target-final-evaluation-v2.md)
- [Historical v1 evaluation contract](evaluation/target-final-evaluation.md)


## Prediction

- [Conservative organization resolution and verified-only sentiment routing](company-resolution.md)

- [Pretrained organization candidates and experimental sentiment bridge](organization-detection.md)
- [Registry versus NER validation comparison and review](../../reports/phase-04-target-sentiment/experiments/organization-detection.md)

- [Company mention detection and automatic sentiment](company-mentions.md)

- [Reusable RoBERTa target prediction: Python and CLI](prediction.md)

- [US-listed SEC issuer directory: setup and lookup](us-listed-directory.md)

- [US-company news mention/linking review packet and scoring](data/us-company-review.md)

- [Run the verified GPU release: PowerShell launchers](prediction-gpu-release.md)

- [Reviewed issuer aliases, parent policy and fresh reference review](issuer-aliases.md)

- [Reviewed-alias GPU launchers, examples and rollback](issuer-alias-gpu.md)
