# RoBERTa target-format comparison: reviewed results

Reviewed 2026-10-20. [Full metrics and histories](../../experiments/target-format.json) · [Execution-generated report](../../experiments/target-format.md) · [Verification receipt](../../experiments/target-format-verification.json) · [Locked protocol](../../../configs/target-format-plan.json).

**Keep the original target–headline pair formulation.** On this fixed development comparison, markers do not improve mean overall or unseen-target macro F1; replacement performs worse overall in all three seeds. This supports retaining the simpler existing input format. It does not establish statistical superiority or independent generalization. The deployed checkpoint is unchanged.

All nine trials completed on the RTX 4070 SUPER. Each starts from the same pinned original RoBERTa Base snapshot, with matched initialization and row orders within seeds 42/43/44. Training uses 11,395 native annotations from 8,497 headlines; validation uses 1,396 annotations from 1,044 headlines. Both transformed arms and the freshly trained original-pair baseline use identical examples. No frozen test, company detector, calibration fitting or new hyperparameter search is involved.

## Validation cohorts

| Cohort | Target annotations | Original pair F1 (%) | Markers F1 (%) | Replacement F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| All targets | 1,396 | 89.70 ± 0.73 | 89.48 ± 0.46 | 88.83 ± 0.68 |
| Single-target headlines | 774 | 88.75 ± 0.89 | 88.45 ± 0.31 | 87.60 ± 0.71 |
| Multi-target headlines | 622 | 90.51 ± 0.62 | 90.56 ± 1.30 | 90.08 ± 0.79 |
| Mixed-label headlines | 283 | 92.33 ± 1.23 | 91.63 ± 1.03 | 90.53 ± 1.93 |
| Seen native target | 1,134 | 89.99 ± 0.63 | 89.88 ± 0.53 | 89.19 ± 0.81 |
| Unseen native target | 262 | 88.40 ± 1.48 | 87.69 ± 1.38 | 87.22 ± 0.39 |

What each attribute means:

- **Cohort:** the overlapping validation subsets defined below; these rows must not be added together.
- **Target annotations:** native target–label pairs, not unique headlines. A headline with three labeled targets contributes three annotations.
- **Original pair F1 (%):** target name as the first tokenizer sequence, unchanged full headline as the second. This is the new subset-matched baseline.
- **Markers F1 (%):** original target name first, with all audited literal target occurrences surrounded by `[TARGET]` / `[/TARGET]` in the headline.
- **Replacement F1 (%):** `[TARGET]` first, with all audited literal target occurrences replaced by `[TARGET]` in the headline. The selected name is hidden from both sequences.
- **F1 and ±:** macro F1 equally averages the negative, neutral and positive class F1 scores. Each cell shows the mean percentage across three seeds ± sample standard deviation in percentage points. This is seed variability, not a confidence interval.
- **All targets:** every retained validation annotation.
- **Single-target headlines:** annotations from headlines with one native target.
- **Multi-target headlines:** annotations from headlines with multiple native targets, whether their labels agree or differ.
- **Mixed-label headlines:** annotations from multi-target headlines whose native target labels differ; a subset of the multi-target cohort.
- **Seen native target:** the exact original target string appears as an annotated target in retained training. Appearing somewhere in a training headline is insufficient.
- **Unseen native target:** the exact original target string does not appear as an annotated target in retained training. This is not an entity-disjoint company benchmark: aliases can refer to the same entity, and native targets also include sectors, indices and commodities.

Compared with the original pair, markers change overall macro F1 by **−0.22 percentage points (pp)** and replacement by **−0.86 pp**. On unseen native targets, changes are **−0.71 pp** and **−1.18 pp** respectively. Markers have only a +0.05 pp mean difference on the multi-target cohort, alongside −0.69 pp on mixed-label headlines. These overlapping cohorts and three-seed summaries do not establish a reliable benefit for markers.

## Individual runs and matched differences

| Format | Seed | Selected epoch | Macro F1 (%) | Difference vs same-seed pair (pp) | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original pair | 42 | 3 | 90.10 | +0.00 | 381.6 |
| Markers | 42 | 3 | 89.90 | -0.20 | 383.9 |
| Replacement | 42 | 3 | 89.57 | -0.53 | 379.7 |
| Original pair | 43 | 5 | 88.86 | +0.00 | 381.3 |
| Markers | 43 | 5 | 89.55 | +0.70 | 382.5 |
| Replacement | 43 | 5 | 88.70 | -0.16 | 381.9 |
| Original pair | 44 | 5 | 90.14 | +0.00 | 382.3 |
| Markers | 44 | 3 | 88.99 | -1.15 | 383.0 |
| Replacement | 44 | 3 | 88.23 | -1.90 | 380.2 |

What each attribute means:

- **Format:** the same three input representations defined above; they share the backbone, labels and data.
- **Seed:** fixes fresh head initialization and training shuffle order. Initialization state/RNG and common-epoch annotation order hashes match across formats within each seed.
- **Selected epoch:** the epoch with the highest all-target validation macro F1 for that run, keeping the earlier epoch on an exact tie. Selection uses unrounded values.
- **Macro F1 (%):** all-target validation score from the saved and reloaded selected checkpoint for this individual run, not a three-seed average.
- **Difference vs same-seed pair (pp):** this run’s macro F1 minus its original-pair baseline with the same seed, multiplied by 100. Negative values favor the baseline; baseline rows are zero by construction.
- **Fit seconds:** wall time for optimization, epoch validation and checkpoint saving. Initial loading and final selected-checkpoint reload/scoring are excluded.

Replacement loses overall macro F1 in all three seeds. Markers win overall only for seed 43 and lose on unseen targets in every seed. That consistency is useful descriptive evidence, but three seeds and reused validation do not support a formal significance claim. Every run reached epoch 5; four of nine selected epoch 5. The five-epoch budget limits conclusions about other training schedules. No post-result extension or tuning was performed.

## Raw probability diagnostics

| Format | Accuracy (%) | NLL | Brier | ECE (%) |
| --- | ---: | ---: | ---: | ---: |
| Original pair | 89.71 | 0.5023 | 0.1852 | 8.46 |
| Markers | 89.45 | 0.4622 | 0.1868 | 7.97 |
| Replacement | 88.85 | 0.4792 | 0.1938 | 8.07 |

What each attribute means:

- **Format:** the same three input representations; each value is the mean across their three selected checkpoints on all 1,396 validation annotations.
- **Accuracy (%):** percentage of annotations whose highest-probability predicted class matches the native label; higher is better.
- **NLL:** negative log likelihood of the native gold label. Lower is better; it penalizes assigning very little probability to the correct label, even when accuracy is similar.
- **Brier:** mean sum of squared errors across the three class probabilities, using one-hot native labels. This unnormalized multiclass score ranges from 0 to 2; lower is better.
- **ECE (%):** top-label expected calibration error using 10 equal-width confidence bins. It measures the weighted gap between confidence and accuracy, expressed as a percentage; lower is better. It depends on binning and sample support.

Markers have lower mean NLL and ECE than the original pair, but slightly worse Brier and lower macro F1. Replacement also lowers mean NLL/ECE while worsening Brier and classification. The metrics therefore do not show a uniform probability-quality winner. These are raw, uncalibrated softmax outputs; lower validation ECE is not evidence that deployment confidence is calibrated. The locked primary comparison remains macro F1, so probability diagnostics do not justify changing the input-format recommendation.

## Model architecture, size and runtime

- All nine runs use `RobertaForSequenceClassification`: 12 encoder layers, hidden width 768 (values per token representation), intermediate width 3,072 and 12 attention heads per layer.
- Every model has **124,647,939 parameters**. FP32 parameter tensors occupy **498,591,756 bytes**; this counts parameter storage, not training activation/optimizer memory.
- The vocabulary stays at 50,265 entries. Markers use existing subwords, with no added embeddings or vocabulary resize.
- Each selected model’s weights, tokenizer and configuration files total **500,175,018 bytes** (500.18 decimal MB), excluding the artifact-manifest file. All formats have the same saved footprint.
- Recorded fit time totals **3,436.3 seconds (57 minutes 16 seconds)** across nine sequential trials. The corrected detached job took about **58 minutes 47 seconds** including preflight, loading, reload/scoring and export.

## Verification and provenance

The post-run verifier rechecks all nine checkpoint manifests and their file hashes on the PC, reads the checksummed private native prediction caches, validates original annotation IDs/text/targets/gold and recomputes every cohort’s classification/probability metrics. It also recomputes the three-seed aggregates and same-seed differences and confirms exports equal the execution reports byte-for-byte. No new training or model inference is performed. Frozen-record paths are explicitly forbidden during result verification.

The verified aggregate SHA-256 is `74beb760cfba0ede12a74c1b90451ef3e8f1f8b8db47bf7001eaa4545c481b89`. The exact execution source is revision `c9b449257ef1e9df885f14ffcacc2da20ad3e395`, preserved in a private Git bundle on both machines. The public verification receipt lists each trial report, selected artifact manifest and private prediction-cache digest. Native row predictions and weights remain private. Public files were checked to contain aggregates and provenance rather than source examples.

The preserved bundle was cloned into a temporary bare repository, and all 115
Python source fingerprints at that exact revision match every trial's recorded
implementation. The temporary verification clone was removed afterward. All 687
pytest tests pass. Required full Ruff commands were run; only the existing
untracked scratch notebook retains nine lint issues and formatting failures.
Lint/format checks excluding `notebooks/experiment.ipynb` pass; it is unchanged.

The frozen contract, previous final target evaluation and current RoBERTa deployment configuration retain their prior checksums. No existing benchmark or predictor is replaced. The newly trained subset baseline is not directly comparable to historical full-data scores because both training membership and evaluation support differ.

## Decision and limits

Retain the original target-pair formulation and the current deployed checkpoint. This experiment does not support adding marker/replacement preprocessing for better unseen-target generalization. Do not select a new best seed or deploy one of these subset-trained checkpoints from this analysis.

The observed loss from replacement does not isolate its cause: useful target semantics, changed subword inputs and the fixed optimization schedule could all matter. Literal spans are not semantic/coreference gold, contextual identity clues remain after replacement, and difficult alignment cases were filtered out. Validation was reused for development and epoch selection. Fresh independent evaluation is required for new generalization claims; the consumed test stays untouched.

A separately declared [name-substitution diagnostic](target-name-stress.md) is now complete using the saved checkpoints. It does not change this primary comparison or its recommendation. Company detection remains deferred. No further model training or Phase 5 implementation is part of this review.
