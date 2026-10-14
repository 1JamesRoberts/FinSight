# Five-company SEntFiN development coverage

Provided annotations only; no test analysis, label changes or training.
Literal candidates are diagnostic name matches, not verified entity labels.

## Complete development cohort

| Split | Headlines | Target annotations | Single-target | Multi-target | Mixed-label |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 8598 | 11547 | 6299 | 2299 | 994 |
| validation | 1053 | 1410 | 781 | 272 | 122 |

## strict

Registry matches treat the annotated targets as financial entities.
The separate parent-rollup view does not rewrite source targets.

| Split | Ticker | Labeled headlines | Targets | Negative | Neutral | Positive | Single-target | Multi-target | Mixed-label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | AAPL | 21 | 21 | 11 | 6 | 4 | 13 | 8 | 2 |
| train | MSFT | 4 | 4 | 1 | 2 | 1 | 1 | 3 | 0 |
| train | NVDA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | AMZN | 2 | 2 | 1 | 1 | 0 | 1 | 1 | 1 |
| train | GOOGL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | AAPL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | MSFT | 2 | 2 | 1 | 0 | 1 | 1 | 1 | 1 |
| validation | NVDA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | AMZN | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| validation | GOOGL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

| Split | Ticker | Literal target | Alias only | No registered alias | Candidate headlines | Unannotated candidates | Conflicts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | AAPL | 21 | 0 | 0 | 22 | 1 | 0 |
| train | MSFT | 4 | 0 | 0 | 5 | 1 | 0 |
| train | NVDA | 0 | 0 | 0 | 0 | 0 | 0 |
| train | AMZN | 2 | 0 | 0 | 2 | 0 | 0 |
| train | GOOGL | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | AAPL | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | MSFT | 2 | 0 | 0 | 2 | 0 | 0 |
| validation | NVDA | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | AMZN | 1 | 0 | 0 | 1 | 0 | 0 |
| validation | GOOGL | 0 | 0 | 0 | 0 | 0 | 0 |

## with_parent_rollup

Registry matches treat the annotated targets as financial entities.
The separate parent-rollup view does not rewrite source targets.

| Split | Ticker | Labeled headlines | Targets | Negative | Neutral | Positive | Single-target | Multi-target | Mixed-label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | AAPL | 21 | 21 | 11 | 6 | 4 | 13 | 8 | 2 |
| train | MSFT | 4 | 4 | 1 | 2 | 1 | 1 | 3 | 0 |
| train | NVDA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | AMZN | 2 | 2 | 1 | 1 | 0 | 1 | 1 | 1 |
| train | GOOGL | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| validation | AAPL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | MSFT | 2 | 2 | 1 | 0 | 1 | 1 | 1 | 1 |
| validation | NVDA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | AMZN | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| validation | GOOGL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

| Split | Ticker | Literal target | Alias only | No registered alias | Candidate headlines | Unannotated candidates | Conflicts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | AAPL | 21 | 0 | 0 | 22 | 1 | 0 |
| train | MSFT | 4 | 0 | 0 | 5 | 1 | 0 |
| train | NVDA | 0 | 0 | 0 | 0 | 0 | 0 |
| train | AMZN | 2 | 0 | 0 | 2 | 0 | 0 |
| train | GOOGL | 1 | 0 | 0 | 1 | 0 | 0 |
| validation | AAPL | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | MSFT | 2 | 0 | 0 | 2 | 0 | 0 |
| validation | NVDA | 0 | 0 | 0 | 0 | 0 | 0 |
| validation | AMZN | 1 | 0 | 0 | 1 | 0 | 0 |
| validation | GOOGL | 0 | 0 | 0 | 0 | 0 | 0 |

## Resolution and annotation diagnostics

| Split | Resolved annotations | Parent grouping pending | Outside registry | Unknown targets with name cue | Nonliteral targets (whole corpus) |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 27 | 1 | 11519 | 2 | 87 |
| validation | 3 | 0 | 1407 | 0 | 8 |

Count headlines once per company, and annotations as provided.
Company rows overlap; their sum is not a unique headline count.
Mixed-label means the original headline has different target sentiments.
Nonliteral targets can be abbreviations or implicit references, not errors.
Outside-registry targets generally belong to other entities, not errors.
Literal candidates do not create sentiment labels. No rows are dropped.

Registry `companies-v1`.
Registry SHA-256 `3414bd464620472b37f23944f0c6d8011f77fc8460491c4001010d2aa820d285`.
Plan SHA-256 `1b6ca24e45f351c8db3d0a356706ce788a7accab7f5c3162e53eb24bc1ce2752`; code `e45324fdd53113da1a1ab4053122458011963be2`.
Private review flags SHA-256 `eec051d0af5f6afb43c83b7843747b041b5208fc67326e33db8d888d268f0076`.
See [method and interpretation](../docs/company-coverage.md). Full provenance is in the sibling JSON.
