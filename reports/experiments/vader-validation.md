# VADER validation sanity check

validation only; PhraseBank and single-target SEntFiN

Thresholds: negative <= -0.05, positive >= 0.05, neutral between.
No threshold tuning and no final-test evaluation. Majority control uses
training labels from the corresponding sentence cohort only.

| Cohort | Validation texts | VADER accuracy | VADER macro F1 | Majority accuracy | Majority macro F1 | Mean inference ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phrasebank | 518 | 0.5560 | 0.4895 | 0.5734 | 0.2429 | 0.0264 |
| sentfin_single_target | 781 | 0.4738 | 0.4787 | 0.3611 | 0.1769 | 0.0151 |

### phrasebank: per-class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 69 | 0.4571 | 0.2319 | 0.3077 |
| neutral | 297 | 0.7371 | 0.5286 | 0.6157 |
| positive | 152 | 0.4259 | 0.7566 | 0.5450 |

Confusion matrix: rows=gold, columns=predicted; order negative, neutral, positive.

```text
[16, 23, 30]
[15, 157, 125]
[4, 33, 115]
```


### sentfin_single_target: per-class metrics

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| negative | 247 | 0.6471 | 0.4453 | 0.5276 |
| neutral | 252 | 0.3858 | 0.4960 | 0.4340 |
| positive | 282 | 0.4704 | 0.4787 | 0.4745 |

Confusion matrix: rows=gold, columns=predicted; order negative, neutral, positive.

```text
[110, 77, 60]
[35, 125, 92]
[25, 122, 135]
```

Run ID: `955195fc41bdd3369ac76947b8453d1a4619704248115dfd024027a7767d5007`.
Parent split: `a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898`.
Code revision at execution: `94c8d06446908b5291b803f39c65a685d2b07af1`; dirty: `True`.
Implementation file digests distinguish uncommitted implementation changes.
Aggregate settings, resources, environment, and timing accompany it as JSON.
The complete local run is also preserved in ignored `run.json`.
Local prediction IDs/gold/scores remain ignored; no source text is exported.
Latency is a single-pass local measurement, not a deployment benchmark.
See `docs/vader.md` for scoring, reproduction, and financial-domain limits.
