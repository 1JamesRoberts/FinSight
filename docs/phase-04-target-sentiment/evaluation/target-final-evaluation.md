# Locked company-targeted sentiment evaluation

Historical v1 protocol. For current execution, see the [authorized v2 correction](target-final-evaluation-v2.md).

Protocol: `phase4-sentfin-multi-target-v1`, declared 2026-10-17 after development
comparison and qualitative review, before reserved multi-target inference.
The authoritative [plan](../../../configs/target-evaluation-plan.json) is bound by its
[lock receipt](../../../configs/target-evaluation-lock.json). This task creates and
verifies the contract; it does not implement or execute the test runner.

## Validation-only selection

Choose the backbone with the highest **unrounded arithmetic mean of all-target
validation macro F1 across seeds 42, 43 and 44**, using each target-pair run's
already-selected checkpoint. The cohort is the same 1,410 development validation
annotations used in the comparison. Exact ties prefer DeBERTa, then RoBERTa, then
FinBERT, retaining the previous incumbent when tied. Do not choose a best seed.
This rule is declared now after validation analysis; it is not claimed to have
been registered before seeing validation results.

| Backbone | Target-pair validation macro F1, mean ± sample SD |
| --- | ---: |
| RoBERTa Base | 89.07% ± 0.79 pp |
| DeBERTa Base | 88.96% ± 0.42 pp |
| FinBERT | 88.65% ± 0.74 pp |

**RoBERTa Base is the validation-selected candidate.** Its approximately 0.11 pp
lead over DeBERTa does not establish statistical or practical superiority.
Validation was reused for epoch selection and development. The final results
will characterize this fixed choice; they must not be used to select a different
backbone, best seed, ensemble, calibrator or retraining recipe. All three seeds
remain experimental replicates, not a voting ensemble. A deployment checkpoint
or serving policy is separate future work and cannot be chosen by test scores.

## The remaining test cohort

Use the original `sentfin-independent-v1` freeze and **only its 263 multi-target
headlines, containing 594 native target annotations**. These counts were already
published during Phase 1; no frozen records were decoded to create this contract.
The 110 mixed-sentiment headlines, containing 253 annotations, form a declared
secondary subset. Also report same-label headlines and seen/unseen native-target
strings. These subsets overlap and are not independent samples.

The 766 single-target test headlines were evaluated in Phase 3 and are excluded.
PhraseBank tests remain reserved. Do not combine old single-target results with
new multi-target results to claim an untouched all-headline test. Existing
source-label curation and aggregate inspection are disclosed in the original
[evaluation contract](../../phase-01-data/benchmark/evaluation-contract.md); reserved means no multi-target
model scoring or performance-based selection has occurred.

Before future inference, verify frozen bytes and audit duplicate-group separation
from training, validation and the consumed single-target cohort. Abort on overlap;
do not silently remove or replace examples. Selection uses original native target
strings and labels, with no company filtering, alias rollup, masking, gold repairs
or model-dependent exclusions. Preserve every target of each admitted headline.
The unit is a native financial entity; entities are not all verified companies.

## Fixed comparisons and checkpoint strategy

The plan pins **18 transformer checkpoints**: three backbones × two input arms ×
three seeds. Each entry binds its source trial report, saved artifact path, complete
file inventory, manifest digest, training configuration and selected epoch.
The original run manifests retain the full model-source and training lineage.
No fresh training or train-plus-validation fitting is permitted.

The arms are the existing matched headline-only and target-pair models. Headline
predictions are broadcast to the native targets; target-pair predictions use native
target first and headline second. Tokenizers insert separators, preserve the full
target, and truncate only the headline at the fixed length of 128.

Additional baselines are both saved Phase 4 TF-IDF Logistic Regression arms
(headline-only and separate headline/target features), plus the class-frequency
prior learned from the 11,547 training annotations. The prior predicts neutral;
its probabilities are fixed training counts divided by the training total.
No vocabulary, thresholds or probabilities are fitted on test records.

The **primary paired contrast** is RoBERTa target-pair minus RoBERTa matched
headline-only macro F1 on all 594 reserved annotations. Six secondary contrasts
are declared: DeBERTa conditioning, FinBERT conditioning, RoBERTa versus DeBERTa,
RoBERTa versus FinBERT, RoBERTa versus target-feature TF-IDF, and the two TF-IDF arms.
All differences are left minus right. Report every declared comparison regardless
of direction. Secondary/subgroup findings are exploratory without multiplicity
correction and do not select a new winner.

## Metrics, uncertainty and confidence

Primary performance is pooled target-level macro F1 with fixed class order
negative/neutral/positive. Every native annotation has equal weight. Also report
accuracy, class support, per-class precision/recall/F1 and confusion matrices;
undefined class metrics are zero. Report each seed and the arithmetic mean with
sample standard deviation, rather than averaging probabilities across seeds.

Use 1,000 paired duplicate-group bootstrap replicates, seed 42, with percentile
95% intervals. Apply the same sampled group multiplicities to every method and
seed, retaining all annotations in each sampled group. Compute each seed's metric
and then average the three seed metrics inside each replicate. These intervals
measure test-sample uncertainty conditional on the trained runs; training seeds
are not themselves resampled. Report seed variation separately. If an interval
cannot be computed for an empty subgroup, report it unavailable rather than
changing the cohort. Fixed class order remains even in single-class replicates.

NLL (probability floor 1e-15), unnormalized multiclass Brier and ten-bin equal-width
ECE are secondary descriptive probability metrics. Use raw probabilities; neither
old sentence calibrators nor new temperatures are applied. Confidence is not
claimed to be calibrated. Per-native-target support and fixed-class metrics may
be retained privately, flagging support below 20 as sparse; do not drop those
examples from pooled metrics or claim reliable five-US-company performance.

Record loading and warmed prediction times separately, batch size, hardware,
lengths, peak memory, model/source identity, code revision and all outputs.
Missing artifacts, changed identities, invalid probabilities or failed predictions
abort execution. Do not silently drop failures. Text, weights and row-level results
remain private under the existing local research terms. Public results are aggregate.
Checkpoint pretraining exposure remains unknown; FinBERT's PhraseBank supervision
and the dataset's limited five-company coverage remain limitations.

## Verify without scoring

From the repository root:

```bash
uv run --locked python -m finsight.sentiment.experiments.target_evaluation_plan
```

This verifies source receipts and all 18 checkpoint manifests using metadata. It
never loads a model, unpickles an artifact, or opens the frozen record file. Full
weights remain on the Windows GPU computer; metadata verification on Mac therefore
reports `full_artifacts_verified: false`. This does not mean execution is ready.

On the machine containing every artifact, an additional audit can hash weights,
baseline pickles and frozen files as opaque bytes without parsing/scoring them:

```bash
uv run --locked python -m finsight.sentiment.experiments.target_evaluation_plan --full-artifacts
```

`--write` creates a contract once and refuses an existing plan or lock. Source
fingerprints and the dirty-working-tree flag in the creation receipt honestly
record the current uncommitted refactor; the recorded Git HEAD alone does not
represent those source bytes. Future execution requires committed clean runner
code, a separate execution receipt, full artifact checks and the group-disjointness
audit. Preserve the Phase 3 plans/results and publish Phase 4 results separately.

The next bounded task is to implement that receipt-bound target test runner and
verify it on synthetic/development examples before reserved inference. Entity
identification, independent five-company annotation, reliable per-company testing,
calibration and the prediction API remain open.

## Runner implementation (2026-10-18)

The [target evaluation runner](target-evaluation-runner.md) now implements the
receipt-bound execution, verified resume, cohort/paired-bootstrap reporting and
aggregate-only export paths. Synthetic inference/persistence tests and the native
multi-target development smoke run pass. Full GPU artifact/CUDA verification and
reserved inference remain pending; the original contract and lock are unchanged.
