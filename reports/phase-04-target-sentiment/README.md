# Phase 4 — Company-targeted sentiment reports

Native target conditioning, matched baselines, error analysis and reserved evaluation.

[All reports](../README.md) · [Methods and reproduction guides](../../docs/phase-04-target-sentiment/README.md)

## Evaluation execution status

The detached v1 evaluation on 2026-10-18 passed 574 Windows tests and full artifact verification, then stopped at the frozen membership audit before inference. One duplicate group is shared with the consumed Phase 3 single-target cohort (one reserved headline, two target annotations); there is no development group or exact-text overlap. V1 correctly forbids silent exclusion.

The authorized v2 protocol excludes that reviewed group from the evaluation view,
leaving 262 headlines / 592 annotations. All 21 fixed variants completed on the RTX 4070 SUPER in a detached Windows
job lasting 201.5 seconds, including verification. Mac and Windows each passed
583 tests and Ruff. All prediction caches, receipt/membership/private-result hashes
and recomputed primary-cohort metrics verify. The original freeze, v1 contract
and prior results remain unchanged. This v2 test cohort is now consumed and must
not be used for further tuning or model selection.

- [Development-only runner verification JSON](../experiments/target-runner-verification.json)
- [Locked evaluation contract](../../docs/phase-04-target-sentiment/evaluation/target-final-evaluation.md)
- [Overlap review and proposed cohort correction](overlap-review.md)
- [Authorized v2 evaluation protocol](../../docs/phase-04-target-sentiment/evaluation/target-final-evaluation-v2.md)
## Final test results

[Full evaluation report](../experiments/phase4-target-final.md) ·
[Metrics, architecture, size, timing and provenance JSON](../experiments/phase4-target-final.json)

| Backbone | Headline-only macro F1 (%) | Target-aware macro F1 (%) | Conditioning gain (pp) |
| --- | ---: | ---: | ---: |
| DeBERTa Base | 74.53 | 88.75 | +14.23 |
| RoBERTa Base | 72.82 | 88.34 | +15.52 |
| FinBERT | 73.19 | 86.84 | +13.65 |
| TF-IDF Logistic Regression | 68.47 | 76.15 | +7.68 |

Transformer values average the three fixed seeds; gains use unrounded metrics.
The primary RoBERTa conditioning contrast is +15.52 pp, paired 95% interval
[12.16, 19.10]. On mixed-sentiment headlines, RoBERTa improves from 49.45% to
85.96% macro F1 (+36.51 pp). On same-label headlines its gain is only +0.16 pp,
with an interval spanning zero. Conditioning is most useful when targets differ
in sentiment; it does not establish an improvement for every subgroup.

RoBERTa minus DeBERTa is -0.41 pp, interval [-1.62, 0.87]; RoBERTa minus FinBERT
is +1.50 pp, interval [-0.23, 3.29]. These exploratory contrasts do not establish
a clear backbone winner. The validation-selected RoBERTa candidate remains fixed.
Native entities are not all verified companies; this is not a five-ticker benchmark.

Execution receipt: `a712c2c921b6a5966f6d9986a2dadb493e768da6af2b9b2301f2ba76236da88b`.
Aggregate SHA-256: `cded5333aa6dc0716fbcb8ac250b3cd4a8c974c4549d1ecaaa6142a8d2f9c9cf`.
The clean execution source commit is `ce1207b75e450695def45db3853fc7afe16331a0`;
private caches and logs remain in ignored storage on both research machines.

## Start here: consolidated comparisons

| Readable report | Machine-readable artifact |
| --- | --- |
| [Target conditioning across three backbones](../experiments/target-backbones.md) | [JSON](../experiments/target-backbones.json) |

## Target inputs and matched baselines

| Readable report | Machine-readable artifact |
| --- | --- |
| [Five-company SEntFiN development coverage](../company-coverage.md) | [JSON](../company-coverage.json) |
| [Matched DeBERTa target-conditioning comparison](../experiments/target-deberta.md) | [JSON](../experiments/target-deberta.json) |
| [Saved DeBERTa headline control: target-level validation](../experiments/target-saved-control.md) | [JSON](../experiments/target-saved-control.json) |
| [Matched TF-IDF + Logistic Regression target baselines](../experiments/target-tfidf-logreg.md) | [JSON](../experiments/target-tfidf-logreg.json) |
| [Target-aware input preparation](../target-input-preparation.md) | [JSON](../target-input-preparation.json) |


## Diagnostics and error review

| Readable report | Machine-readable artifact |
| --- | --- |
| [Phase 4 target-sentiment error analysis](../experiments/target-error-analysis.md) | [JSON](../experiments/target-error-analysis.json) |


## Prediction interface

- [RoBERTa deployment selection and verification](target-prediction.md)
- [Python/CLI usage](../../docs/phase-04-target-sentiment/prediction.md)
