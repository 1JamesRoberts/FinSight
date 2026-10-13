# Soft and democratic voting on SEntFiN validation

Three matching-seed ensembles (42/43/44), each combining calibrated FinBERT, RoBERTa Base and DeBERTa Base with equal weights. 781 validation records; cross-fitted member temperatures, no new training or calibration. Results remain exploratory because this validation already selected rates/epochs.

| Model/method | Macro F1 ± SD | Accuracy ± SD | NLL ± SD | Brier ± SD | ECE ± SD |
|---|---:|---:|---:|---:|---:|
| finbert | 0.8748 ± 0.0023 | 0.8758 ± 0.0022 | 0.3825 ± 0.0200 | 0.1972 ± 0.0063 | 0.0247 ± 0.0075 |
| roberta_base | 0.8760 ± 0.0082 | 0.8775 ± 0.0078 | 0.3871 ± 0.0134 | 0.1989 ± 0.0068 | 0.0419 ± 0.0173 |
| deberta_base | 0.8810 ± 0.0044 | 0.8818 ± 0.0041 | 0.3679 ± 0.0130 | 0.1907 ± 0.0072 | 0.0186 ± 0.0063 |
| soft | 0.8846 ± 0.0010 | 0.8856 ± 0.0007 | 0.3415 ± 0.0119 | 0.1752 ± 0.0056 | 0.0354 ± 0.0104 |
| hard | 0.8859 ± 0.0042 | 0.8869 ± 0.0039 | N/A | N/A | N/A |

Soft voting averages calibrated probabilities then takes argmax; exact ties use negative/neutral/positive order. Democratic/hard voting takes the majority label; three-way ties resolve to neutral. Vote counts are not calibrated confidence, so hard voting has no probability metrics.

| Method | Seed | Macro F1 | Corrected DeBERTa errors | Lost DeBERTa correct cases | Hard three-way ties |
|---|---:|---:|---:|---:|---:|
| soft | 42 | 0.8854 | 15 | 13 | — |
| hard | 42 | 0.8905 | 14 | 8 | 3 |
| soft | 43 | 0.8835 | 18 | 12 | — |
| hard | 43 | 0.8821 | 17 | 12 | 2 |
| soft | 44 | 0.8850 | 15 | 14 | — |
| hard | 44 | 0.8851 | 12 | 11 | 0 |

No weight search, stacking, best-seed selection or fitted ensemble calibration. The soft average itself is not guaranteed calibrated. Per-seed class metrics, confusion matrices, reliability bins, confidence, ECE sensitivity and member agreement are in JSON; individual votes/probabilities remain private.

Both voting methods were declared as supplementary final comparisons before this analysis. The additive v2 protocol retains the original primary individual-model comparison. Final test predictions must use the locked full-validation member temperatures, not these assessment-fold temperatures. Frozen tests remain unscored.

Architecture: three existing classifiers, 418,556,937 total member parameters, zero learned voting parameters and no new weight files. Serving requires three member inferences per seed, rather than one; measured validation reuse here does not establish serving latency or throughput.

See [reproduction and interpretation](../../docs/ensemble-voting.md).
