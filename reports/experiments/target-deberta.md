# Matched DeBERTa target-conditioning comparison

Each arm starts fresh from the pinned original Microsoft DeBERTa-v3 Base.
Both use the same 11,547 native target training annotations and 1,410 validation annotations.
Seeds 42/43/44; LR 1e-5; FP32; batch 16/32; max five epochs; patience two.
Checkpoint selection uses all-target validation macro F1; earliest exact tie.

| Cohort | Headline-only F1 % | Target-pair F1 % | Difference (pp) |
| --- | ---: | ---: | ---: |
| all_targets | 81.19 ± 0.36 | 88.96 ± 0.42 | +7.77 |
| single_target | 87.47 ± 0.20 | 88.11 ± 0.41 | +0.64 |
| multi_target | 73.15 ± 0.79 | 89.81 ± 0.42 | +16.65 |
| mixed_label_headlines | 51.65 ± 0.59 | 91.91 ± 0.12 | +40.26 |
| seen_native_target | 81.10 ± 0.46 | 89.70 ± 0.32 | +8.60 |
| unseen_native_target | 81.54 ± 1.53 | 85.70 ± 1.06 | +4.16 |

Values are mean ± sample standard deviation across three seeds, not confidence intervals.
Metrics weight target annotations equally; cohorts overlap. Seen/unseen means exact native target string, not verified company identity.

| Arm | Seed | Selected epoch | All-target F1 % | Multi-target F1 % | Mixed-label F1 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched_headline_only | 42 | 4 | 81.60 | 74.05 | 51.65 |
| matched_headline_only | 43 | 3 | 81.00 | 72.57 | 52.25 |
| matched_headline_only | 44 | 4 | 80.97 | 72.84 | 51.06 |
| target_pair | 42 | 4 | 88.87 | 89.77 | 91.83 |
| target_pair | 43 | 3 | 89.41 | 90.24 | 91.86 |
| target_pair | 44 | 5 | 88.59 | 89.41 | 92.05 |

| Arm | Accuracy % | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| matched_headline_only | 81.09 | 0.5146 | 0.2828 | 0.0804 |
| target_pair | 88.94 | 0.5059 | 0.1973 | 0.0882 |

Architecture: DebertaV2ForSequenceClassification, 12 layers, hidden size 768, 12 attention heads, 184,424,451 parameters.
Six selected safe checkpoints, input context manifests and private row predictions are retained locally.
Full per-trial class metrics/confusion matrices, calibration bins, model bytes, timings, optimizer history and provenance are in the companion JSON.

## Interpretation boundaries

Full initial-state/RNG fingerprints and common-epoch training-order hashes match between arms for every seed.
Both train on the same examples with equal annotation weights; the input representation is the intended conditioning difference.
The saved Phase 3 control used only 6,299 single-target training headlines, so comparisons against it also change training data/update budgets.
TF-IDF baselines use the same native annotation view but a different architecture and training procedure; their report remains separate.
No masking, alias normalization, new splits, new LR search, calibration fitting or reserved-test inference occurred.
These are reused validation/epoch-selection results. They do not establish independent test generalization or reliable five-company performance.
No seed was selected as a winner. The original final evaluation and preparation lock remain unchanged.

Plan SHA-256: `d8c87e545ec2546cace4a41f3f5840b1ad077d04cdcefec81447973524185a81`.
