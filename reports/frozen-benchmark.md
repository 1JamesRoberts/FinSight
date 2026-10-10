# Frozen SEntFiN evaluation benchmark

Release: `sentfin-independent-v1`. Final holdout frozen before models.

Headlines: 1029; target annotations: 1360; duplicate groups: 1028.
Curation excluded 2 candidate records with malformed target annotations; their groups stay withheld.

| Cohort | Headlines | Targets | Negative | Neutral | Positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 1029 | 1360 | 333 | 548 | 479 |
| single_target | 766 | 766 | 215 | 259 | 292 |
| multi_target | 263 | 594 | 118 | 289 | 187 |
| mixed_target_sentiment | 110 | 253 | 51 | 113 | 89 |

Unique normalized targets: 833.
Nonliteral target names: 0 annotations (retained; canonical names need not appear literally in headlines).

MVP exact target-name matches (descriptive; no entity resolution):
- AAPL: 2
- MSFT: 0
- NVDA: 0
- AMZN: 0
- GOOGL: 0

This is held out from FinSight development. Known accessible-corpus
exact/near overlap is excluded. Full checkpoint training exposure is unknown.
Financial entities are not all verified companies; this is not a five-company
US-market benchmark. See `docs/evaluation-contract.md` for the protocol.

Benchmark manifest SHA-256: `b7cc249017ddd87f0b2c52a3abf3ff5b96f6920832b411f18eac547fec0b0f1c`.
Records SHA-256: `144db27f984376867a04dae3724ad69ab4de9e6dbca8fdd43f41d1bca558f3ab`.

Raw and frozen records remain ignored local research data. The checked-in
receipt pins membership and labels through content digests, without text.
No model training, inference, or evaluation metrics have been implemented.
