# TF-IDF validation tuning

Separate training-only corpus fits; repeated validation selection, no CV or pooled training.

Selection: validation macro F1. Exact ties retain declared order: LR before SVM, then grid order.
TF-IDF settings remain fixed; only C and class weighting vary.
These validation scores informed selection and are not unbiased final performance estimates.
The frozen test was not scored. Existing baseline reports are preserved.

## phrasebank

| Model | C | Weight | Accuracy | Macro F1 | ms/text | KiB | Selected within model |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.1 | None | 0.6274 | 0.3848 | 0.202 | 768.13 |  |
| logistic_regression | 0.1 | balanced | 0.7220 | 0.6720 | 0.198 | 768.14 |  |
| logistic_regression | 1.0 | None | 0.7490 | 0.6696 | 0.204 | 768.13 |  |
| logistic_regression | 1.0 | balanced | 0.7645 | 0.7378 | 0.199 | 768.14 |  |
| logistic_regression | 10.0 | None | 0.7722 | 0.7183 | 0.198 | 768.13 |  |
| logistic_regression | 10.0 | balanced | 0.7780 | 0.7446 | 0.199 | 768.14 | yes |
| linear_svm | 0.1 | None | 0.7297 | 0.6285 | 0.162 | 767.99 |  |
| linear_svm | 0.1 | balanced | 0.7683 | 0.7205 | 0.168 | 768.00 |  |
| linear_svm | 1.0 | None | 0.7799 | 0.7344 | 0.164 | 767.99 |  |
| linear_svm | 1.0 | balanced | 0.7819 | 0.7444 | 0.165 | 768.00 | yes |
| linear_svm | 10.0 | None | 0.7606 | 0.7176 | 0.165 | 767.99 |  |
| linear_svm | 10.0 | balanced | 0.7568 | 0.7146 | 0.232 | 768.00 |  |

Overall selected: **logistic_regression**, `{'c': 10.0, 'min_df': 2, 'max_features': 50000, 'max_iter': 1000, 'seed': 42, 'class_weight': 'balanced'}`.
Macro F1 change versus its C=1 unweighted baseline: +0.0751.
Controls: VADER 0.4895; training majority 0.2429 macro F1.

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| negative | 0.7500 | 0.6522 | 0.6977 |
| neutral | 0.8038 | 0.8552 | 0.8287 |
| positive | 0.7324 | 0.6842 | 0.7075 |

Selected confusion matrix (gold rows/predicted columns; negative/neutral/positive):

```text
[45, 17, 7]
[12, 254, 31]
[3, 45, 104]
```

Architecture: TF-IDF word unigrams/bigrams -> LogisticRegression; 12,351 features, 37,056 stored classifier parameters.
Selected artifact SHA-256: `09cf4459530a6b9a1acf188e609a687a84b324a5ca5b1e0b838c4dc9e0295bfa`.
Selected original run ID: `3d4657830b1cfe58b5f86a167b4a9216a565ac971ca1f95e13ba67041a6071cf`.

## sentfin_single_target

| Model | C | Weight | Accuracy | Macro F1 | ms/text | KiB | Selected within model |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.1 | None | 0.7439 | 0.7384 | 0.185 | 641.67 |  |
| logistic_regression | 0.1 | balanced | 0.7657 | 0.7646 | 0.187 | 641.68 |  |
| logistic_regression | 1.0 | None | 0.8182 | 0.8165 | 0.188 | 641.67 |  |
| logistic_regression | 1.0 | balanced | 0.8195 | 0.8181 | 0.187 | 641.68 | yes |
| logistic_regression | 10.0 | None | 0.8156 | 0.8141 | 0.189 | 641.67 |  |
| logistic_regression | 10.0 | balanced | 0.8169 | 0.8154 | 0.190 | 641.68 |  |
| linear_svm | 0.1 | None | 0.8156 | 0.8145 | 0.161 | 641.53 |  |
| linear_svm | 0.1 | balanced | 0.8169 | 0.8160 | 0.162 | 641.54 | yes |
| linear_svm | 1.0 | None | 0.8092 | 0.8072 | 0.157 | 641.53 |  |
| linear_svm | 1.0 | balanced | 0.8092 | 0.8075 | 0.155 | 641.54 |  |
| linear_svm | 10.0 | None | 0.7913 | 0.7902 | 0.154 | 641.53 |  |
| linear_svm | 10.0 | balanced | 0.7913 | 0.7902 | 0.157 | 641.54 |  |

Overall selected: **logistic_regression**, `{'c': 1.0, 'min_df': 2, 'max_features': 50000, 'max_iter': 1000, 'seed': 42, 'class_weight': 'balanced'}`.
Macro F1 change versus its C=1 unweighted baseline: +0.0016.
Controls: VADER 0.4787; training majority 0.1769 macro F1.

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| negative | 0.8263 | 0.7895 | 0.8075 |
| neutral | 0.7698 | 0.8095 | 0.7892 |
| positive | 0.8607 | 0.8546 | 0.8577 |

Selected confusion matrix (gold rows/predicted columns; negative/neutral/positive):

```text
[195, 34, 18]
[27, 204, 21]
[14, 27, 241]
```

Architecture: TF-IDF word unigrams/bigrams -> LogisticRegression; 10,475 features, 31,428 stored classifier parameters.
Selected artifact SHA-256: `3613ef35c9cee527698e8d6429fc6565e88deabf313ee64736308b661adc8ea4`.
Selected original run ID: `c5375a28e1498111dda90119f8a8cfb522a9169d46ceec5b4a8ec85683abd10e`.

Search ID: `329457f5dc8b951883897dfcf0b0a3ca88357093730d86315c9c5183cefdc5e1`.
Complete candidate configs, per-class metrics, confusion matrices, model sizes, timings,
split identities, dependency versions, code provenance, and artifact hashes are in JSON.
Models, vocabulary, record IDs, and predictions remain in ignored local storage.
Corpus styles/annotation schemes differ; scores do not measure cross-corpus transfer.
See `docs/tfidf-tuning.md` for setup, commands, selection rules, and limitations.
