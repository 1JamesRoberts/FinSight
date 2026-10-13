# Locked Phase 3 final evaluation

Protocol `phase3-sentfin-single-target-v1`, locked 2026-10-13, before final-test
inference. The [plan](../configs/final-evaluation-plan.json) pins settings,
checkpoints, calibrators, source inputs and model-file inventories. The
[lock receipt](../configs/final-evaluation-lock.json) pins the complete plan hash.
This task verifies readiness; it does not implement or execute the test scorer.

## Population and comparison

Evaluate all 766 single-target headlines from `sentfin-independent-v1`, retaining
their existing gold labels, duplicate groups and fixed class order:
negative/neutral/positive. This is the sentence-model comparison defined in the
[original benchmark contract](evaluation-contract.md). It does not replace that
contract's later target-aware evaluation across all annotations. Multi-target
headlines remain for Phase 4, and PhraseBank reserved tests stay unscored.

| Candidate | Locked choice | Probability reporting |
|---|---|---|
| Training-prior control | SEntFiN training counts 1902/2142/2255; predict positive | Training-prior probabilities |
| VADER | Existing ±0.05 compound thresholds and lexicon identities | None; proportions are not calibrated probabilities |
| TF-IDF Logistic Regression | SEntFiN-selected C=1, balanced, seed 42 | Native probabilities, no adjustment |
| TF-IDF Linear SVM | SEntFiN-selected C=0.1, balanced, seed 42 | None; margins are not probabilities |
| Unchanged FinBERT | Existing pinned ProsusAI checkpoint | Raw probabilities |
| Fine-tuned FinBERT | LR 3e-5; saved selected epochs, seeds 42/43/44 | Raw and corresponding saved temperature |
| RoBERTa Base | LR 2e-5; saved selected epochs, seeds 42/43/44 | Raw and corresponding saved temperature |
| DeBERTa-v3 Base | LR 1e-5; saved selected epochs, seeds 42/43/44 | Raw and corresponding saved temperature |

Use the already saved weights selected on development data. No retraining on
train-plus-validation, new calibration fit, best-seed selection or ensemble.
Three-seed results report each seed and mean/sample SD of metrics. The same
trained checkpoint supplies its raw and calibrated variants; inference is shared.
Use the exact published full-validation temperatures, not fold-specific values
or slightly different platform refits. Each scalar belongs to its own checkpoint.
Calibration preserves class predictions; do not present duplicate F1 results as
independent experiments or an accuracy gain.

Financial adapted/unchanged RoBERTa/DeBERTa and PhraseBank-only models remain
development diagnostics; this bounded final comparison contains the selected
three-model study, traditional controls and unchanged FinBERT. No hidden choice
of additional candidates after seeing test outcomes.

## Metrics and uncertainty

Primary metric: three-class macro F1, with undefined precision/recall/F1 set to
zero. Also report accuracy, per-class support/precision/recall/F1 and confusion
matrices (rows gold, columns predicted). Native probabilities are assessed with
NLL (natural log, gold-probability floor 1e-15), unnormalized multiclass Brier
(0–2), 10-bin top-label ECE and reliability bins, plus ECE sensitivity at 5/15 bins.
Include mean confidence and confidence minus accuracy. Keep VADER/SVM probability
metrics explicitly unavailable; do not softmax margins or reinterpret proportions.

Use paired duplicate-group bootstrap: 1,000 replicates, seed 42, percentile 95%
intervals. Resample complete groups with replacement, retaining all their records
and repeated draws. Use the same sampled groups across every model/seed/variant.
For transformers, recompute each seed's metric on each replicate and average
those three metrics; do not combine predictions or resample seeds as independent
datasets. Sample SD across seeds and bootstrap uncertainty describe different
sources of variation. Use the fixed class order even if a replicate lacks a class.

Predeclared paired macro-F1 differences: DeBERTa minus RoBERTa, DeBERTa minus
FinBERT, and each fine-tuned transformer minus selected TF-IDF LR. Predeclare
calibration differences as calibrated minus raw NLL per model (negative is an
improvement). No multiplicity-adjusted significance claim or causal explanation
of architecture superiority. Calibration comparisons use identical checkpoints,
examples and bootstrap draws. Report unfavorable or inconclusive outcomes too.

## Execution and failure policy

Transformers keep CUDA FP32, eager attention, TF32 disabled, deterministic
settings, batch 32 and token limit 128. Existing TF-IDF/VADER preprocessing and
thresholds stay fixed. Explicitly validate label mappings. Log token truncation,
hardware, environment, dependency lock, model architecture/size and model-file
hashes. Use one invented warm-up headline; synchronize CUDA around batched
inference. Separate loading, inference, calibration adjustment and metric times;
CPU baseline timings use batch 1 and are not directly comparable to GPU timings.

Implement and test the evaluation runner as a separate bounded task, then record
its exact clean code revision, implementation hashes and environment before the
first test inference. The current lock pins existing training/inference sources
and the verifier, not a future scorer that does not yet exist. New runner code
must implement this protocol; it must not change its candidate choices/settings.
Run the read-only verifier before inference and require full artifact checks.

A failure aborts result publication and is logged; never silently omit a record.
Execution repairs/retries must retain this lock, membership and model settings.
Resume from saved completed predictions rather than rescoring them; record every
attempt. No metric-driven revision, recalibration, label correction or seed choice
after inspecting results. Later test-inspired improvements are exploratory and
need new untouched data for a subsequent confirmatory evaluation.

## Verify the lock without scoring

From the repository root with reviewed private inputs:

```sh
uv run --locked python -m finsight.sentiment.evaluation_protocol
```

This checks the plan/receipt, repository input hashes, private development
run/prediction identities, saved temperatures and selected checkpoint manifests.
It streams frozen files as opaque bytes for checksum verification; it never
decodes test texts/labels, loads model weights, unpickles a pipeline or predicts.
The Mac has selected transformer manifests, but the full weights are on Windows.
On Windows with all saved artifacts present:

```powershell
python -m uv run --locked python -m finsight.sentiment.evaluation_protocol --full-artifacts
```

The full check additionally hashes all nine saved transformer/tokenizer file
sets, unchanged FinBERT files and both selected TF-IDF pipelines. It still
performs no inference. Missing or mismatched artifacts fail; do not regenerate
models to satisfy the lock. Synthetic tests exercise scope drift, seed and
calibration ancestry, hash corruption, paths, metadata-only checking and opaque
non-JSON test bytes. Public files contain aggregate metadata and hashes only;
source records, labels, individual outputs and model weights remain ignored.

Verified 2026-10-13: metadata/lineage verification passes on Mac; full verification
passes on Windows for all 65 model/tokenizer/pipeline files. Both machines pass
all 346 tests and required Ruff lint/format checks. The frozen files retain their
original hashes; no test records were decoded or scored. The two selected TF-IDF
artifacts and their development-run lineage are also available privately on
Windows. Protocol SHA-256:
`e749b1879fd3337d947184106ea06cd32b41143fcf9279f51b57b2055f8a2601`.

## Remaining scope decisions

Length/batch ablations and MLflow integration are explicitly deferred. This
comparison keeps length128, train16/eval32 and existing checksummed local JSON
experiment records. MLflow and the broader ablation tasks are not marked complete;
they are not prerequisites to this bounded locked comparison. DeBERTa Large stays
deferred. Phase 3 final evaluation remains pending until its runner and results
are verified; this protocol task alone does not close Phase 3.

Use the qualification **held-out SEntFiN evaluation; no known accessible-corpus
overlap; pretraining exposure unknown**. SEntFiN Indian-market headlines do not
establish generalization to US tickers or full articles. The previously authorized
private FinBERT research scope and unresolved hosted-weight license remain;
nothing in this protocol authorizes public weights or headline redistribution.
