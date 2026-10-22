# RoBERTa target-format development comparison

Three seeds per format; identical native train/validation annotations. No frozen test scoring.

| Format | Cohort | Macro F1 % ± pp | Difference vs pair (pp) |
| --- | --- | ---: | ---: |
| target_pair | all_targets | 89.70 ± 0.73 | +0.00 |
| target_pair | single_target | 88.75 ± 0.89 | +0.00 |
| target_pair | multi_target | 90.51 ± 0.62 | +0.00 |
| target_pair | mixed_label_headlines | 92.33 ± 1.23 | +0.00 |
| target_pair | seen_native_target | 89.99 ± 0.63 | +0.00 |
| target_pair | unseen_native_target | 88.40 ± 1.48 | +0.00 |
| target_markers | all_targets | 89.48 ± 0.46 | -0.22 |
| target_markers | single_target | 88.45 ± 0.31 | -0.30 |
| target_markers | multi_target | 90.56 ± 1.30 | +0.05 |
| target_markers | mixed_label_headlines | 91.63 ± 1.03 | -0.69 |
| target_markers | seen_native_target | 89.88 ± 0.53 | -0.11 |
| target_markers | unseen_native_target | 87.69 ± 1.38 | -0.71 |
| target_replacement | all_targets | 88.83 ± 0.68 | -0.86 |
| target_replacement | single_target | 87.60 ± 0.71 | -1.15 |
| target_replacement | multi_target | 90.08 ± 0.79 | -0.43 |
| target_replacement | mixed_label_headlines | 90.53 ± 1.93 | -1.80 |
| target_replacement | seen_native_target | 89.19 ± 0.81 | -0.80 |
| target_replacement | unseen_native_target | 87.22 ± 0.39 | -1.18 |

Format is the original pair, name-preserving markers, or replacement in both sequences. Cohort denotes all targets, single-/multi-target headlines, mixed-label headlines, or exact original targets seen/unseen in retained training annotations; cohorts overlap. Macro F1 equally averages the three class F1 scores; values show mean percent ± sample standard deviation in percentage points across seeds, not confidence intervals. Difference is mean F1 minus the fresh pair baseline in percentage points (pp).

| Format | Seed | Selected epoch | Parameters | Layers / hidden / heads | Saved bytes | Training seconds |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| target_pair | 42 | 3 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 381.6 |
| target_markers | 42 | 3 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 383.9 |
| target_replacement | 42 | 3 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 379.7 |
| target_pair | 43 | 5 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 381.3 |
| target_markers | 43 | 5 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 382.5 |
| target_replacement | 43 | 5 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 381.9 |
| target_pair | 44 | 5 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 382.3 |
| target_markers | 44 | 3 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 383.0 |
| target_replacement | 44 | 3 | 124,647,939 | 12 / 768 / 12 | 500,175,018 | 380.2 |

Format uses the definitions above. Seed fixes initialization and shuffling. Selected epoch maximizes validation macro F1, keeping the earliest tie. Parameters counts all model parameters. Layers/hidden/heads are encoder depth, hidden-vector width and attention heads per layer. Saved bytes includes selected model/tokenizer files. Training seconds includes optimization, epoch validation and checkpoint saving, excluding initial loading and final reload scoring.

The JSON includes all histories, class metrics/confusion matrices, raw NLL/Brier/10-bin ECE, same-seed F1 differences, architecture, timing and artifact/prediction provenance. Identical starts and common-epoch row orders are verified. Full predictions and weights stay private. This is reused development validation, not independent evaluation; masking does not remove all identity clues. No production model is replaced.
