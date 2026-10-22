# Synthetic company-name sensitivity: reviewed results

Reviewed 2026-10-20. [Fixed protocol](../../../docs/phase-04-target-sentiment/analysis/target-name-stress.md) · [Plan](../../../configs/target-name-stress-plan.json) · [Aggregate results](../../experiments/target-name-stress.json) · [Verification receipt](../../experiments/target-name-stress-verification.json).

Retain the original target–headline pair formulation. Markers show lower name sensitivity on this small diagnostic, but this was not a model-selection experiment. The deployed checkpoint remains unchanged. Replacement produces identical masked inputs by construction, so its stability does not demonstrate learned generalization.

## Scope and method

Reuse all nine selected RoBERTa checkpoints: three input formats, each trained with seeds 42, 43 and 44. No training, calibration fitting, company detection or frozen-test access was performed. All models have 124,647,939 parameters, 12 layers, hidden size 768, 12 attention heads and a three-class classification head; FP32 parameter storage is 498,591,756 bytes. Architecture and inference timing are recorded per checkpoint in the JSON.

A fixed allowlist selects 67 native company-target annotations from 64 validation headlines, covering twelve companies. The assistant reviewed the contexts before inference without changing membership. For each annotation, replace every audited literal occurrence of the selected company with `Cedar Group`, then separately with `Northstar Holdings`; change the requested target to match. All other text remains unchanged. Apply each checkpoint's original input transformation after renaming. This gives 134 paired comparisons and 201 scored inputs per checkpoint (67 originals plus 134 renamed versions). All inputs fit the 128-token budget without truncation; maximum encoded lengths are 33/41/35 for pair/markers/replacement.

No synthetic gold labels are assigned. We measure changes from the model's original prediction, not counterfactual accuracy or macro F1. Changing a real company's identity can change factual meaning; a flip is not automatically an error. Literal replacement also leaves aliases, affiliations and product cues untouched.

## Main results

| Input format | Label flips (%) | Mean probability mass moved (pp) | Largest class-probability change (pp) |
| --- | ---: | ---: | ---: |
| Original pair | 4.73 ± 0.43 | 5.15 ± 0.54 | 99.70 |
| Target markers | 2.74 ± 1.55 | 2.80 ± 0.73 | 94.45 |
| Target replacement | 0.00 ± 0.00 | ≈0 | 0.000113 |

**Table attributes:** Input format is the representation each checkpoint was trained on: target–headline pair, literal target markers, or selected-target masking. Label flips is the percentage of 134 substitutions that change the highest-probability class, averaged over three seeds. The ± value is sample standard deviation across those seeds, expressed in percentage points (pp), not a confidence interval. Mean probability mass moved is total variation: half the sum of absolute changes in the three probabilities, multiplied by 100 and averaged over substitutions and seeds; its ± is also seed sample standard deviation in pp. For example, moving 0.10 probability from positive to neutral gives 10 pp total variation. Largest class-probability change is the maximum absolute change for any one class across all substitutions and seeds; it is not an average. Probabilities are raw, uncalibrated model outputs.

Pair and markers sometimes make very large probability changes despite relatively low average flip rates. Replacement's residual probability change is floating-point noise below the predeclared absolute tolerance of 1e-5 (0.001 pp); every replacement label is unchanged. Both renamed and original masked inputs were scored separately rather than reusing predictions.

## Per-checkpoint results

| Input format | Seed | Flips / comparisons | Label flips (%) | Mean probability mass moved (pp) |
| --- | ---: | ---: | ---: | ---: |
| Original pair | 42 | 6 / 134 | 4.48 | 4.81 |
| Target markers | 42 | 3 / 134 | 2.24 | 2.34 |
| Target replacement | 42 | 0 / 134 | 0.00 | ≈0 |
| Original pair | 43 | 7 / 134 | 5.22 | 5.78 |
| Target markers | 43 | 6 / 134 | 4.48 | 3.64 |
| Target replacement | 43 | 0 / 134 | 0.00 | ≈0 |
| Original pair | 44 | 6 / 134 | 4.48 | 4.86 |
| Target markers | 44 | 2 / 134 | 1.49 | 2.42 |
| Target replacement | 44 | 0 / 134 | 0.00 | ≈0 |

**Table attributes:** Seed identifies the fixed training run whose selected checkpoint was reloaded; no model was retrained for this probe. Flips counts changed predicted labels out of 134 original/substitution pairs. Label flips is that fraction as a percentage. Mean probability mass moved uses the total-variation definition above, in pp; ≈0 denotes only numerical differences. The same 67 annotations are reused for both substitute names, every seed and every format. These rows therefore do not represent independent samples or independent test sets.

There are 19/402 flips for pair, 11/402 for markers, and 0/402 for replacement when pooling seeds within each format. Across the nine checkpoints there are 1,206 paired comparisons, all derived from the same 67 annotations. These counts should not be used as an independent-sample confidence calculation.

## Company support

| Native target | Validation annotations |
| --- | ---: |
| BPCL | 6 |
| HDFC Bank | 3 |
| HUL | 6 |
| ICICI Bank | 4 |
| Infosys | 11 |
| ONGC | 7 |
| SBI | 6 |
| Sun Pharma | 6 |
| TCS | 6 |
| Tata Steel | 4 |
| Tech Mahindra | 4 |
| Wipro | 4 |

**Table attributes:** Native target is the original annotation's exact target string, before renaming. Validation annotations counts selected target–headline pairs, not distinct headlines or substitutions. Counts sum to 67; some annotations share a headline, so the distinct-headline total is 64. The allowlist is manually scoped and does not establish exhaustive company detection or coverage of the five US tickers.

## Interpretation and decision

Markers are less sensitive than pair to these two invented names, but the sample is small and manually scoped, and reuses development validation. It does not establish superior performance on unseen companies or real news. In the [primary development comparison](target-format-comparison.md), pair macro F1 was 89.70% versus markers 89.48% and replacement 88.83%. This diagnostic supplies additional descriptive evidence; it does not overturn the original recommendation or select another checkpoint. No counterfactual accuracy or F1 is reported because the renamed headlines have no validated gold labels.

Company detection remains a separate deferred task. This check neither modifies the registry nor resolves aliases/coreference. No Phase 5 functionality is included.

## Verification and provenance

All nine original-input replays matched their saved native validation classes. Maximum probability replay difference was 4.35114e-6, within the fixed 1e-5 absolute tolerance. The runner verified checkpoint files and saved native cache hashes on the GPU PC. After transfer, all 1,206 private paired predictions were independently checked locally for membership, repeated-original consistency, and all per-checkpoint and aggregate metrics; all nine artifact identities match the public training reports. Plan, case, runner and dependency hashes also match the frozen preparation.

The immutable preparation packet is `634edaf74cb58f9ef096ef7b710a179daaffd86db846698fded345965746a090`. Aggregate result SHA-256 is `9eabd893fb32e5d0e9e53ac218b638e14a9afeed5d5b0597710d31f412e11d29`. The self-hashed diagnostic runner uses the preserved training source at commit `c9b449257ef1e9df885f14ffcacc2da20ad3e395`; that snapshot was not modified. Headlines, transformed text and paired row predictions stay in ignored local storage on both machines; only aggregates and provenance are public.

Thirteen new behavior tests cover substitution, collisions, alignment, probability validation and sensitivity calculations. All 700 pytest tests pass. Full required Ruff commands encounter existing lint/format problems only in the unrelated `notebooks/experiment.ipynb`; excluding that unchanged scratch notebook passes. No new dependencies were introduced.
