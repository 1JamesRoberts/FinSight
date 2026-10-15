# Target conditioning across three backbones

Development validation only: 11,547 training target annotations; 1,410 validation annotations from 1,053 headlines. Seeds 42/43/44.

| Backbone | Cohort | Headline F1 % | Target-pair F1 % | Difference pp |
| --- | --- | ---: | ---: | ---: |
| deberta_base | all_targets | 81.19 ± 0.36 | 88.96 ± 0.42 | +7.77 |
| deberta_base | mixed_label_headlines | 51.65 ± 0.59 | 91.91 ± 0.12 | +40.26 |
| deberta_base | multi_target | 73.15 ± 0.79 | 89.81 ± 0.42 | +16.65 |
| deberta_base | seen_native_target | 81.10 ± 0.46 | 89.70 ± 0.32 | +8.60 |
| deberta_base | single_target | 87.47 ± 0.20 | 88.11 ± 0.41 | +0.64 |
| deberta_base | unseen_native_target | 81.54 ± 1.53 | 85.70 ± 1.06 | +4.16 |
| finbert | all_targets | 81.29 ± 0.24 | 88.65 ± 0.74 | +7.36 |
| finbert | single_target | 87.76 ± 0.22 | 87.82 ± 0.74 | +0.05 |
| finbert | multi_target | 73.02 ± 0.46 | 89.44 ± 0.79 | +16.42 |
| finbert | mixed_label_headlines | 51.73 ± 0.17 | 89.80 ± 0.44 | +38.08 |
| finbert | seen_native_target | 80.88 ± 0.28 | 88.86 ± 0.89 | +7.98 |
| finbert | unseen_native_target | 83.09 ± 0.98 | 87.76 ± 0.28 | +4.66 |
| roberta_base | all_targets | 80.95 ± 0.35 | 89.07 ± 0.79 | +8.11 |
| roberta_base | single_target | 87.33 ± 0.17 | 87.76 ± 1.12 | +0.43 |
| roberta_base | multi_target | 72.57 ± 0.52 | 90.43 ± 0.51 | +17.85 |
| roberta_base | mixed_label_headlines | 51.35 ± 0.86 | 92.60 ± 0.42 | +41.25 |
| roberta_base | seen_native_target | 80.82 ± 0.63 | 89.43 ± 0.59 | +8.61 |
| roberta_base | unseen_native_target | 81.53 ± 1.04 | 87.46 ± 1.85 | +5.93 |

Mean ± sample standard deviation across three seeds; not confidence intervals.
Cohorts overlap; all metrics weight target annotations equally. Seen/unseen means exact native target string, not verified company identity.

| Backbone | Arm | Accuracy % | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| deberta_base | matched_headline_only | 81.09 | 0.5146 | 0.2828 | 0.0804 |
| deberta_base | target_pair | 88.94 | 0.5059 | 0.1973 | 0.0882 |
| finbert | matched_headline_only | 81.25 | 0.6268 | 0.2924 | 0.0912 |
| finbert | target_pair | 88.70 | 0.5756 | 0.2070 | 0.0921 |
| roberta_base | matched_headline_only | 80.80 | 0.5083 | 0.2769 | 0.0680 |
| roberta_base | target_pair | 89.08 | 0.4979 | 0.1932 | 0.0827 |

| Backbone | Parameters | Layers | Hidden size | Heads | Selected checkpoint bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| deberta_base | 184,424,451 | 12 | 768 | 12 | 740,189,720 |
| finbert | 109,484,547 | 12 | 768 | 12 | 438,226,091 |
| roberta_base | 124,647,939 | 12 | 768 | 12 | 500,175,018 |

| Backbone | Arm | Seed | Selected epoch | All-target F1 % |
| --- | --- | ---: | ---: | ---: |
| deberta_base | matched_headline_only | 42 | 4 | 81.60 |
| deberta_base | target_pair | 42 | 4 | 88.87 |
| deberta_base | matched_headline_only | 43 | 3 | 81.00 |
| deberta_base | target_pair | 43 | 3 | 89.41 |
| deberta_base | matched_headline_only | 44 | 4 | 80.97 |
| deberta_base | target_pair | 44 | 5 | 88.59 |
| finbert | matched_headline_only | 42 | 5 | 81.03 |
| finbert | target_pair | 42 | 4 | 88.40 |
| finbert | matched_headline_only | 43 | 5 | 81.51 |
| finbert | target_pair | 43 | 2 | 88.08 |
| finbert | matched_headline_only | 44 | 5 | 81.34 |
| finbert | target_pair | 44 | 5 | 89.49 |
| roberta_base | matched_headline_only | 42 | 4 | 81.10 |
| roberta_base | target_pair | 42 | 5 | 88.22 |
| roberta_base | matched_headline_only | 43 | 2 | 80.55 |
| roberta_base | target_pair | 43 | 2 | 89.77 |
| roberta_base | matched_headline_only | 44 | 3 | 81.21 |
| roberta_base | target_pair | 44 | 5 | 89.22 |

All parameters are updated. FP32, deterministic eager attention, batch 16/32, maximum five epochs, patience two, max length 128; strict all-target validation macro F1 selects the earliest tied epoch.
Previously selected learning rates: DeBERTa 1e-5, RoBERTa 2e-5, FinBERT 3e-5. No new rate search. FinBERT retains its original supervised financial head; the base backbones receive seeded new heads. Different tokenizers, rates and pretraining limit causal claims across backbones.
Within each backbone, initial state/RNG and common epoch row orders match between arms. Original source checkpoints are used, with no sequential adaptation.
Raw probabilities only; no calibration fitting, masking, alias normalization, five-company filtering, new split or reserved-test scoring.
Reused validation guides epoch selection, so these results do not establish independent target-test generalization. FinBERT has known PhraseBank supervision; pretraining exposure to SEntFiN remains unknown.
Weights and row predictions remain private. Full per-seed histories, class metrics, confusion matrices, calibration bins, timings, tokenization audits and artifact checksums are in the companion JSON and per-trial reports.
