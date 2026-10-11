# Phase 1 supplement: label distributions across splits

Reviewed 2026-10-11, before Phase 3. This compares the two sentence-classification cohorts used in Phase 2: PhraseBank ≥50% agreement and single-target SEntFiN. Counts and proportions are descriptive; they do not measure model performance. No split, label, model configuration, or class-weight choice changes.

## Method and provenance

Train/validation counts are recomputed through the verified development loader and cross-checked against the saved tuning report. PhraseBank reserved counts come from the existing split-preparation aggregate; frozen SEntFiN counts come from the committed freeze receipt. No reserved text is inspected or model scored. Raw counts are the source of truth; displayed percentages are rounded to two decimals and may not sum to exactly 100%.

The original assignment hashes seed 42 and each stable duplicate-group ID into approximately 80/10/10 train/validation/test-candidate buckets. It does not inspect labels or enforce class balance. Development-only flagged groups use renormalized train/validation weights; duplicate groups stay together. Close proportions were observed after assignment, not obtained by searching seeds or adjusting groups.

Each row below counts sentences/headlines with one label, not all SEntFiN target annotations. FiQA is outside this comparison because its scores are continuous. Each delta is partition percentage minus training percentage, in percentage points (pp), not a relative percentage change.

## phrasebank

Test role: 527 reserved parent-split candidates; not an independent FinBERT test or a separately curated final benchmark.

| Partition | Texts | Negative n (%) | Neutral n (%) | Positive n (%) | Majority/minority ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,785 | 468 (12.36%) | 2,257 (59.63%) | 1,060 (28.01%) | 4.82 |
| validation | 518 | 69 (13.32%) | 297 (57.34%) | 152 (29.34%) | 4.30 |
| test_candidate | 527 | 67 (12.71%) | 314 (59.58%) | 146 (27.70%) | 4.69 |

| Compared with training | Negative Δ pp | Neutral Δ pp | Positive Δ pp | Largest absolute gap pp |
| --- | ---: | ---: | ---: | ---: |
| validation | +0.96 | -2.29 | +1.34 | 2.29 |
| test_candidate | +0.35 | -0.05 | -0.30 | 0.35 |

## sentfin_single_target

Test role: 766 single-target headlines from the frozen sentfin-independent-v1 benchmark.

| Partition | Texts | Negative n (%) | Neutral n (%) | Positive n (%) | Majority/minority ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 6,299 | 1,902 (30.20%) | 2,142 (34.01%) | 2,255 (35.80%) | 1.19 |
| validation | 781 | 247 (31.63%) | 252 (32.27%) | 282 (36.11%) | 1.14 |
| frozen_test | 766 | 215 (28.07%) | 259 (33.81%) | 292 (38.12%) | 1.36 |

| Compared with training | Negative Δ pp | Neutral Δ pp | Positive Δ pp | Largest absolute gap pp |
| --- | ---: | ---: | ---: | ---: |
| validation | +1.43 | -1.74 | +0.31 | 1.74 |
| frozen_test | -2.13 | -0.19 | +2.32 | 2.32 |

## Interpretation

PhraseBank is neutral-dominated in every partition: roughly 57–60% neutral, 12–13% negative, and 28–29% positive. Validation has 2.29 pp less neutral than training; reserved candidates differ from training by at most 0.35 pp. The majority/minority ratios are 4.82 in training, 4.30 in validation, and 4.69 in the reserved candidates. Close split proportions do not remove class imbalance; macro F1 and per-class precision/recall remain useful.

Single-target SEntFiN is more evenly distributed. Validation differs from training by at most 1.74 pp; the frozen single-target test differs by at most 2.32 pp. The frozen test has somewhat more positive and fewer negative labels than training, without a large descriptive class shift. No formal equivalence test or guarantee of similar text/target distributions follows from these proportions.

These modest observed gaps provide no compelling label-distribution reason to replace the current splits. They do not establish that group hashing is preferable to stratification. Group-aware stratification would explicitly aim to preserve class proportions while respecting duplicate groups; our hash assignment makes similar proportions an expected but unguaranteed outcome. Stratification preserves the existing class mixture rather than making classes equally common.

## Test-role distinctions and boundaries

PhraseBank’s 527 rows are reserved test candidates from the parent split. They were not promoted to the independent frozen benchmark, remain unused in training/tuning, and cannot provide independent evidence for the PhraseBank-trained ProsusAI FinBERT checkpoint. Their distribution is shown to describe the original partition, not authorize a new evaluation.

The SEntFiN test row is the **766-headline frozen single-target cohort**, not the parent split’s 1,031 all-headline candidates and not all 1,029 frozen headlines. Training/validation are likewise restricted to single-target headlines: 6,299 and 781, rather than 8,598 and 1,053 all-headline rows. The 263 multi-target frozen headlines and their multiple annotations belong to separate evaluation cohorts and are not pooled into this table.

Test aggregates were already recorded at freeze/preparation and are published here solely as a user-requested descriptive audit. Do not use these proportions to choose class weights, tune models, resample groups, or change the frozen benchmark. Class weights must continue to use training counts only. There is no new test-text review, model prediction, score, or Phase 3 implementation.

## Exact values and sources

The [aggregate JSON](split-label-distribution.json) stores exact counts, unrounded percentages, deltas, ratios, cohort definitions, and input identities. See the [split-preparation report](split-preparation.md), [frozen-benchmark report](frozen-benchmark.md), [split policy](../docs/split-policy.md), and [evaluation contract](../docs/evaluation-contract.md). The original Phase 1 notebook and figures remain development-only.

Parent split manifest SHA-256: `a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898`.
Frozen benchmark manifest SHA-256: `b7cc249017ddd87f0b2c52a3abf3ff5b96f6920832b411f18eac547fec0b0f1c`.
Freeze receipt SHA-256: `ab0c9f0581612ee52f3c03a5bbbc03e7ebc621fdf76bb00d7d85bc2973b593f7`.
