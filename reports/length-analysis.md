# FinSight text-length analysis

Phase 1 descriptive EDA. Character counts measure Unicode code points in
original text, including whitespace; word counts use whitespace splitting.
Words are not FinBERT subword tokens. No cleaning, filtering, tokenization,
FiQA class thresholds, or new partitions are applied. Duplicate source rows
remain counted. FiQA uses original training texts only.

P95 is the inclusive interpolated 95th percentile; fractional counts arise
from interpolation. Median/P95/max are shown here; JSON also includes mean,
minimum, P90/P99, a word histogram, and >100-word counts. Flags <=3, >50,
and >100 words are fixed descriptive markers, not outlier rejection rules.

## PhraseBank agreement subsets

| Group | Texts | Words median | Words P95 | Words max | Chars median | Chars P95 | Chars max | <=3 words | >50 words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 4,846 | 21.0 | 42.0 | 81.0 | 119.0 | 233.8 | 315.0 | 4 | 35 |
| 66 | 4,217 | 21.0 | 42.0 | 81.0 | 119.0 | 233.0 | 315.0 | 4 | 34 |
| 75 | 3,453 | 21.0 | 42.4 | 81.0 | 116.0 | 231.0 | 315.0 | 4 | 28 |
| all | 2,264 | 21.0 | 42.0 | 81.0 | 111.0 | 227.0 | 315.0 | 3 | 22 |

Subsets overlap. `all` means unanimous annotator agreement.

## PhraseBank by source sentiment label

| Group | Texts | Words median | Words P95 | Words max | Chars median | Chars P95 | Chars max | <=3 words | >50 words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50/negative | 604 | 22.0 | 44.0 | 56.0 | 118.0 | 226.0 | 296.0 | 0 | 4 |
| 50/neutral | 2,879 | 21.0 | 41.0 | 81.0 | 116.0 | 232.1 | 315.0 | 4 | 21 |
| 50/positive | 1,363 | 23.0 | 43.0 | 57.0 | 128.0 | 238.0 | 298.0 | 0 | 10 |
| 66/negative | 514 | 22.0 | 44.0 | 56.0 | 116.5 | 231.0 | 296.0 | 0 | 4 |
| 66/neutral | 2,535 | 21.0 | 41.0 | 81.0 | 115.0 | 232.0 | 315.0 | 4 | 21 |
| 66/positive | 1,168 | 23.0 | 43.0 | 57.0 | 128.0 | 234.0 | 298.0 | 0 | 9 |
| 75/negative | 420 | 22.0 | 44.0 | 56.0 | 116.5 | 231.1 | 296.0 | 0 | 3 |
| 75/neutral | 2,146 | 20.0 | 40.0 | 81.0 | 111.0 | 227.0 | 315.0 | 4 | 16 |
| 75/positive | 887 | 23.0 | 44.7 | 57.0 | 126.0 | 234.0 | 298.0 | 0 | 9 |
| all/negative | 303 | 22.0 | 44.0 | 56.0 | 114.0 | 232.8 | 296.0 | 0 | 3 |
| all/neutral | 1,391 | 19.0 | 39.0 | 81.0 | 107.0 | 223.0 | 315.0 | 3 | 14 |
| all/positive | 570 | 23.0 | 45.0 | 57.0 | 122.0 | 231.0 | 272.0 | 0 | 5 |

## SEntFiN headline lengths

Overall counts each headline record once. Label groups count a headline
once if any target has that label; multiple targets with the same label
do not multiply its weight. Mixed-label headlines appear in several
label groups, so group counts do not sum to the overall count.

| Group | Texts | Words median | Words P95 | Words max | Chars median | Chars P95 | Chars max | <=3 words | >50 words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 10,753 | 10.0 | 14.0 | 23.0 | 57.0 | 87.0 | 135.0 | 1 | 0 |
| negative | 3,294 | 10.0 | 14.0 | 23.0 | 56.0 | 84.0 | 120.0 | 1 | 0 |
| neutral | 4,535 | 10.0 | 14.0 | 23.0 | 58.0 | 90.0 | 135.0 | 0 | 0 |
| positive | 4,169 | 10.0 | 15.0 | 21.0 | 57.0 | 89.0 | 129.0 | 0 | 0 |

The supplied `Words` field differs from whitespace counting for 88 headlines.
This report consistently uses its stated counting rule. Differences may
reflect source counting conventions; neither source field nor text is changed.

## FiQA training text lengths by source type

Each original source text record contributes once, regardless of its
annotation count. Continuous scores are not converted to classes.

| Group | Texts | Words median | Words P95 | Words max | Chars median | Chars P95 | Chars max | <=3 words | >50 words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| headline | 438 | 9.0 | 14.0 | 18.0 | 60.5 | 80.0 | 116.0 | 0 | 0 |
| post | 675 | 13.0 | 24.0 | 30.0 | 83.0 | 138.3 | 142.0 | 22 | 0 |
| combined | 1,113 | 10.0 | 23.0 | 30.0 | 67.0 | 135.0 | 142.0 | 22 | 0 |

## Reproduction and limitations

Run `uv run --locked python -m finsight.data analyze-lengths`.
JSON output: `data/interim/length-analysis.json` (ignored).
All manifest artifact bytes are verified before reading source records.
This report contains only aggregates; no source text is committed.

Manifest SHA-256: `e937655fe70b8db4576b9e1652a6d574e4c172dcf4a80f2b759fd17540df4793`.

- financial_phrasebank: `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`
- sentfin_1_0: `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`
- fiqa_task1_original: `acd0f4cfafa6421774bc8f009e8fa3e0ec441357`

These distributions describe the selected copies, not all financial news.
They do not establish a transformer context limit or measure truncation.
Model-specific token length analysis belongs to later model work. No
conclusions about sentiment quality or causal effects follow from length
differences alone. Duplicate/ambiguity review and split design remain open.
