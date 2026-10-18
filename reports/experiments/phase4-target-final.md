# Phase 4 reserved multi-target evaluation

Protocol: `phase4-sentfin-multi-target-v2`.

One previously evaluated near-duplicate group is excluded: 262 multi-target headlines / 592 annotations remain. The original frozen benchmark and Phase 3 results are unchanged.

The validation-selected candidate remains RoBERTa Base; test results do not change selection.
Raw probabilities are uncalibrated. Cross-backbone and subgroup comparisons are secondary.

## mixed_label_headlines

Annotations: 253; headlines: 110; duplicate groups: 110.

| Model / input | Macro F1 mean (%) | Seed SD (pp) | Accuracy (%) |
| --- | ---: | ---: | ---: |
| deberta_base.matched_headline_only | 50.90 | 0.25 | 50.99 |
| deberta_base.target_pair | 86.83 | 1.47 | 86.82 |
| finbert.matched_headline_only | 49.56 | 0.66 | 50.20 |
| finbert.target_pair | 85.61 | 1.06 | 85.51 |
| roberta_base.matched_headline_only | 49.45 | 1.07 | 50.07 |
| roberta_base.target_pair | 85.96 | 1.05 | 85.77 |
| tfidf.headline_only | 47.41 | n/a | 47.04 |
| tfidf.headline_target | 68.68 | n/a | 69.57 |
| training_prior | 20.58 | n/a | 44.66 |

| Paired contrast | Difference (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: |
| deberta_conditioning | 35.93 | [31.98, 40.14] |
| finbert_conditioning | 36.05 | [32.19, 40.07] |
| roberta_conditioning | 36.51 | [32.33, 40.83] |
| roberta_vs_deberta | -0.87 | [-2.61, 0.78] |
| roberta_vs_finbert | 0.34 | [-1.63, 2.45] |
| roberta_vs_sparse | 17.28 | [11.97, 22.48] |
| sparse_conditioning | 21.27 | [16.70, 26.39] |

## multi_target

Annotations: 592; headlines: 262; duplicate groups: 262.

| Model / input | Macro F1 mean (%) | Seed SD (pp) | Accuracy (%) |
| --- | ---: | ---: | ---: |
| deberta_base.matched_headline_only | 74.53 | 0.24 | 74.27 |
| deberta_base.target_pair | 88.75 | 0.55 | 88.63 |
| finbert.matched_headline_only | 73.19 | 0.20 | 73.03 |
| finbert.target_pair | 86.84 | 1.16 | 86.82 |
| roberta_base.matched_headline_only | 72.82 | 0.25 | 72.97 |
| roberta_base.target_pair | 88.34 | 0.63 | 88.12 |
| tfidf.headline_only | 68.47 | n/a | 68.58 |
| tfidf.headline_target | 76.15 | n/a | 76.69 |
| training_prior | 21.77 | n/a | 48.48 |

| Paired contrast | Difference (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: |
| deberta_conditioning | 14.23 | [11.23, 17.29] |
| finbert_conditioning | 13.65 | [10.30, 17.19] |
| roberta_conditioning | 15.52 | [12.16, 19.10] |
| roberta_vs_deberta | -0.41 | [-1.62, 0.87] |
| roberta_vs_finbert | 1.50 | [-0.23, 3.29] |
| roberta_vs_sparse | 12.19 | [8.75, 15.88] |
| sparse_conditioning | 7.68 | [4.13, 11.15] |

## same_label_headlines

Annotations: 339; headlines: 152; duplicate groups: 152.

| Model / input | Macro F1 mean (%) | Seed SD (pp) | Accuracy (%) |
| --- | ---: | ---: | ---: |
| deberta_base.matched_headline_only | 91.49 | 0.63 | 91.64 |
| deberta_base.target_pair | 90.07 | 0.98 | 89.97 |
| finbert.matched_headline_only | 89.90 | 0.67 | 90.07 |
| finbert.target_pair | 87.63 | 1.35 | 87.81 |
| roberta_base.matched_headline_only | 89.86 | 0.49 | 90.07 |
| roberta_base.target_pair | 90.02 | 0.68 | 89.87 |
| tfidf.headline_only | 84.05 | n/a | 84.66 |
| tfidf.headline_target | 81.65 | n/a | 82.01 |
| training_prior | 22.61 | n/a | 51.33 |

| Paired contrast | Difference (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: |
| deberta_conditioning | -1.42 | [-3.52, 0.61] |
| finbert_conditioning | -2.26 | [-5.58, 0.65] |
| roberta_conditioning | 0.16 | [-2.89, 3.28] |
| roberta_vs_deberta | -0.05 | [-1.79, 1.88] |
| roberta_vs_finbert | 2.39 | [-0.23, 5.48] |
| roberta_vs_sparse | 8.37 | [3.98, 13.24] |
| sparse_conditioning | -2.41 | [-6.88, 1.72] |

## seen_native_target

Annotations: 491; headlines: 251; duplicate groups: 251.

| Model / input | Macro F1 mean (%) | Seed SD (pp) | Accuracy (%) |
| --- | ---: | ---: | ---: |
| deberta_base.matched_headline_only | 74.47 | 0.23 | 74.20 |
| deberta_base.target_pair | 89.87 | 0.70 | 89.88 |
| finbert.matched_headline_only | 73.51 | 0.35 | 73.25 |
| finbert.target_pair | 87.62 | 1.33 | 87.64 |
| roberta_base.matched_headline_only | 73.06 | 0.30 | 73.12 |
| roberta_base.target_pair | 89.43 | 0.73 | 89.41 |
| tfidf.headline_only | 69.29 | n/a | 69.45 |
| tfidf.headline_target | 77.29 | n/a | 78.00 |
| training_prior | 21.83 | n/a | 48.68 |

| Paired contrast | Difference (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: |
| deberta_conditioning | 15.40 | [12.35, 18.94] |
| finbert_conditioning | 14.11 | [10.53, 17.91] |
| roberta_conditioning | 16.37 | [12.86, 19.97] |
| roberta_vs_deberta | -0.44 | [-1.65, 0.82] |
| roberta_vs_finbert | 1.80 | [-0.10, 3.81] |
| roberta_vs_sparse | 12.14 | [8.51, 16.16] |
| sparse_conditioning | 8.00 | [3.87, 12.09] |

## unseen_native_target

Annotations: 101; headlines: 85; duplicate groups: 85.

| Model / input | Macro F1 mean (%) | Seed SD (pp) | Accuracy (%) |
| --- | ---: | ---: | ---: |
| deberta_base.matched_headline_only | 74.82 | 1.06 | 74.59 |
| deberta_base.target_pair | 83.60 | 0.86 | 82.51 |
| finbert.matched_headline_only | 70.89 | 0.41 | 71.95 |
| finbert.target_pair | 82.88 | 0.38 | 82.84 |
| roberta_base.matched_headline_only | 71.01 | 1.14 | 72.28 |
| roberta_base.target_pair | 83.91 | 0.47 | 81.85 |
| tfidf.headline_only | 64.07 | n/a | 64.36 |
| tfidf.headline_target | 70.73 | n/a | 70.30 |
| training_prior | 21.48 | n/a | 47.52 |

| Paired contrast | Difference (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: |
| deberta_conditioning | 8.77 | [1.81, 17.23] |
| finbert_conditioning | 11.99 | [3.36, 21.17] |
| roberta_conditioning | 12.89 | [5.17, 22.08] |
| roberta_vs_deberta | 0.31 | [-3.09, 3.92] |
| roberta_vs_finbert | 1.02 | [-4.62, 6.25] |
| roberta_vs_sparse | 13.18 | [5.50, 21.69] |
| sparse_conditioning | 6.66 | [-1.67, 14.73] |

Intervals use 1,000 paired duplicate-group bootstrap draws and average seed metrics within each draw.
Seed SD describes variation across trained seeds; it is not a confidence interval.
Native entities are not verified companies; sparse per-target results remain private.
Checkpoint pretraining exposure is unknown. These results do not establish five-company generalization.
