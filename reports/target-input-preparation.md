# Target-aware input preparation

Status: **prepared, not trained**. Native target–headline pairs and matched
headline-only rows share source IDs, partitions, groups and labels.

| Split | Headlines | Examples | Negative | Neutral | Positive | Multi-target examples | Mixed-label examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 8598 | 11547 | 3078 | 4395 | 4074 | 5248 | 2276 |
| validation | 1053 | 1410 | 385 | 523 | 502 | 629 | 283 |

| Split | Input | Max tokens | Truncated headlines | Targets with UNK | Identical inputs with conflicting labels |
| --- | --- | ---: | ---: | ---: | ---: |
| train | matched_headline_only | 36 | 0 | 0 | 995 |
| train | target_pair | 40 | 0 | 0 | 0 |
| validation | matched_headline_only | 28 | 0 | 0 | 122 |
| validation | target_pair | 34 | 0 | 0 | 0 |

The full first-sequence target is verified in every encoded pair.
Only the headline may be right-truncated; target overflow fails preparation.
Headline-only label conflicts are expected for mixed-sentiment headlines.
The label-conflict count is distinct encoded input patterns, not row count.
UNK and nonliteral targets remain in the view; no labels are changed.

Plan SHA-256: `d8c87e545ec2546cace4a41f3f5840b1ad077d04cdcefec81447973524185a81`.
Private data manifest: `be33e779e278b2c786c48ee3205cbad4f9e6bee98839d4c3b6509009a51a856c`.
Execution revision: `867857f380ca89b3a0292b6d7718aab4ff2cc1f0`.
Native text/labels remain private under `data/interim/target-pairs/`.
No training, inference, calibration fitting or test analysis occurred.
See [protocol and use](../docs/target-sentiment-protocol.md).
