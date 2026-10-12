# Three-model temperature-scaling experiment

SEntFiN single-target validation: 781 records. Shared five-fold grouped cross-fitting, seeds 42/43/44; no new model inference or training.

Each assessed row uses a temperature fitted on the other four folds. Pool out-of-fold probabilities per seed, then report mean/sample SD. Earlier epoch/rate selection used these records: these remain exploratory development results.

| Model | ECE before → after | Brier before → after | NLL before → after | Confidence before → after |
|---|---:|---:|---:|---:|
| roberta_base | 0.1061 ± 0.0077 → 0.0419 ± 0.0173 | 0.2253 ± 0.0107 → 0.1989 ± 0.0068 | 0.6196 ± 0.0837 → 0.3871 ± 0.0134 | 0.9807 ± 0.0070 → 0.8893 ± 0.0025 |
| finbert | 0.0950 ± 0.0194 → 0.0247 ± 0.0075 | 0.2179 ± 0.0129 → 0.1972 ± 0.0063 | 0.5717 ± 0.1442 → 0.3825 ± 0.0200 | 0.9708 ± 0.0190 → 0.8874 ± 0.0013 |
| deberta_base | 0.0858 ± 0.0208 → 0.0186 ± 0.0063 | 0.2082 ± 0.0133 → 0.1907 ± 0.0072 | 0.4995 ± 0.0741 → 0.3679 ± 0.0130 | 0.9642 ± 0.0149 → 0.8894 ± 0.0039 |

Accuracy, macro F1, per-class metrics and confusion matrices remain unchanged. Temperature scaling adjusts confidence, not the chosen class. Lower ECE/Brier/NLL is better; ECE alone does not measure discrimination.

| Model | Seed | Full-validation T (future use) | Fold T range | NLL before → OOF after |
|---|---:|---:|---:|---:|
| roberta_base | 42 | 2.3314 | 2.2516–2.3893 | 0.6350 → 0.3871 |
| roberta_base | 43 | 2.4835 | 2.3799–2.5396 | 0.6945 → 0.4005 |
| roberta_base | 44 | 2.0035 | 1.9357–2.0457 | 0.5292 → 0.3736 |
| finbert | 42 | 1.6194 | 1.5791–1.6644 | 0.4294 → 0.3606 |
| finbert | 43 | 2.5466 | 2.4652–2.6302 | 0.7178 → 0.4000 |
| finbert | 44 | 2.0781 | 2.0358–2.1168 | 0.5679 → 0.3869 |
| deberta_base | 42 | 1.5467 | 1.5229–1.5660 | 0.4141 → 0.3581 |
| deberta_base | 43 | 2.0278 | 2.0027–2.0721 | 0.5469 → 0.3826 |
| deberta_base | 44 | 2.0928 | 2.0758–2.1314 | 0.5374 → 0.3629 |

Full-validation temperatures are separately fitted saved candidates for a future locked evaluation. Their in-sample improvement is not reported as evidence. No best seed or ensemble is selected.

Saved probabilities are converted to log probabilities with floor 1e-15; this is equivalent to logit temperature scaling up to an additive constant for positive, unfloored probabilities. FP32 rounding/flooring limits recovery of the original logits. All floor counts, optimizer boundaries, per-seed reliability bins, 5/15-bin sensitivity and unchanged classification metrics are in JSON.

Bounds T=[0.05,20], objective NLL, fixed 80-iteration inverse-temperature bisection. Method: [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html). The class mapping remains negative/neutral/positive; no labels or parent partitions change.

Final tests remain frozen and unscored. Source data, per-record adjusted outputs, fold IDs and calibrator artifacts stay ignored; public files contain aggregates and hashes only.

Protocol SHA-256: `d9c24f9a05e5264dec5ec5da8770b44db7ac145c88792406a35c1df3f8898357`. Private packet SHA-256: `37d66d5c2c03b58c24bc44dc92914b4ca961abe0613451c3513f22f9fd1c88aa`.

See [reproduction and limitations](../../docs/temperature-scaling.md).
