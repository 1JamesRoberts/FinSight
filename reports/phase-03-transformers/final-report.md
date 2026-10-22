# Phase 3 final report — Sentence-level transformers

**As of:** 2026-10-22. **Status:** bounded sentence-model research and locked
single-target evaluation complete; MLflow integration and length/batch ablations
remain deferred in the roadmap. Later exploratory checkpoints are not part of
the original final-test comparison.

[All final reports](../README.md) · [Phase evidence index](README.md)

## Scope and status

Compare traditional baselines with unchanged/fine-tuned financial and general
transformers, select training settings on development data, study calibration
and ensembles, and execute a receipt-bound frozen single-target test evaluation.
This phase does not establish company-level sentiment or market prediction.

## Data and method

Sentence models use separate PhraseBank or single-target SEntFiN cohorts. The
principal SEntFiN training/validation sizes are 6,299/781 headlines. Initial
financial RoBERTa Large/DeBERTa-v3 Small inference and adaptation, original Base
RoBERTa/DeBERTa fine-tuning, and separate PhraseBank-only runs are documented in
[all-model development comparisons](../experiments/all-models-by-cohort.md).
Their validation tables are not the locked test table below.

The final protocol evaluates 766 SEntFiN single-target headlines/duplicate groups
(215 negative, 259 neutral, 292 positive). All candidate artifacts, calibration,
seeds 42/43/44, voting and paired bootstrap contrasts were predeclared. No best
seed or new temperature is selected on test. FinBERT uses a 12-layer, hidden-768
BERT classifier (109,484,547 parameters); RoBERTa Base uses 12 layers/hidden 768
(124,647,939); DeBERTa-v3 Base uses 12 layers/hidden 768 (184,424,451). Length is
128 and recorded GPU evaluation batch size is 32. Saved sizes and runtime
measurements remain in the [original final report](../experiments/phase3-final.md).

## Results

| Candidate | Frozen single-target macro F1 (%) | Seed SD (pp) |
|---|---:|---:|
| VADER | 51.21 | — |
| TF-IDF Logistic Regression | 81.50 | — |
| TF-IDF Linear SVM | 82.04 | — |
| Unchanged FinBERT | 71.82 | — |
| Fine-tuned FinBERT | 88.28 | 0.27 |
| RoBERTa Base | 88.68 | 0.55 |
| DeBERTa Base | 89.32 | 0.22 |
| Calibrated soft voting, supplementary | 90.28 | 0.51 |
| Democratic/hard voting, supplementary | 90.03 | 0.67 |

**Attributes:** Candidate names fixed controls, saved classifiers or the
predeclared voting supplement. Macro F1 equally averages negative/neutral/positive
F1 on the same 766 frozen examples; trained transformer/voting rows are means
across the three fixed seeds, not a selected deployment checkpoint. Seed SD is
sample standard deviation in percentage points, not a confidence interval. A
dash means a fixed result without seed SD. Rows reuse the same data and are not
independent benchmarks. Raw/calibrated individual classifiers have identical
accuracy/F1 because temperature scaling preserves each class ranking.

DeBERTa has the highest individual-model mean; soft voting has the highest
observed ensemble mean. The paired soft-minus-DeBERTa F1 difference is +0.96 pp,
with a 95% grouped-bootstrap interval approximately [-0.02, 2.01] pp. It does
not establish an ensemble advantage. Transformer-minus-LR predeclared intervals
are positive. Neutral remains the weakest class for the selected transformers.

| Model | Raw NLL | Temperature-scaled NLL |
|---|---:|---:|
| FinBERT | 0.5228 | 0.3597 |
| RoBERTa Base | 0.5160 | 0.3431 |
| DeBERTa Base | 0.3867 | 0.3098 |

**Attributes:** Model identifies the same three-seed individual classifier family.
NLL is mean negative natural log probability assigned to the correct label,
without percentage units; lower is better. Raw uses original probabilities;
temperature-scaled uses calibrators fitted on development validation before
final scoring. These paired rows reuse the same test examples. Lower NLL does
not require different hard predictions or prove every confidence bucket is
calibrated. Brier/ECE and bin sensitivity are retained in the original report.

Calibration improves held-out NLL for all three families under the declared
paired intervals. The ensemble adds three-model inference cost and does not
improve every calibration metric. Its 418,556,937 combined member parameters are
not a single larger learned model. A post-final ProPrak01 financial DeBERTa Base
adaptation reaches validation mean F1 0.8815; it is explicitly exploratory and
was not rescored on the consumed test or substituted into the locked comparison.
[Post-final adaptation](../experiments/proprak-deberta-adaptation.md).

## Decisions

Keep the validation-based DeBERTa sentence reference and retain the original
final comparison without test-inspired replacement or best-seed selection.
Treat ensembles as a measured, inconclusive cost/quality tradeoff. Persist
calibrators/artifact manifests and use aggregate-only public reports with private
predictions retained for audit. The later target-aware RoBERTa deployment in
Phase 4 solves a different task and is not a reversal of this sentence result.

## Limitations and remaining work

PhraseBank is known FinBERT supervision, so its unchanged FinBERT score is an
overlap-qualified diagnostic. SEntFiN has no known accessible-corpus overlap under
the recorded checks, but unknown pretraining exposure and Indian-market coverage
prevent certifying independence or US-stock generalization. Repeated development
selection is optimistic. Paired intervals are not multiplicity-adjusted tests.
FinBERT hosted-weight license scope remains a documented local research decision,
not a public redistribution grant. MLflow and controlled length/batch ablations
remain deferred; source/version/seed/metric tracking currently uses local JSON.
The single-target test is consumed; new confirmatory claims need untouched data.

## Evidence and reproduction

- [Locked final report](../experiments/phase3-final.md), [aggregate JSON](../experiments/phase3-final.json), [figure](../figures/phase3/final-evaluation.png).
- [All-model development comparison](../experiments/all-models-by-cohort.md), [model/seed-study evidence](README.md).
- [Calibration](../experiments/temperature-scaling.md), [voting](../experiments/ensemble-voting.md), [runner/receipt instructions](../../docs/phase-03-transformers/evaluation/final-evaluation-runner.md).
- [Source/model use decisions](../../docs/datasets.md), [phase methods](../../docs/phase-03-transformers/README.md).

Prefer checking existing aggregate/receipt hashes and recomputing metrics from
verified private caches. Do not run new inference or training just to regenerate
this narrative. Exact historical execution/source/artifact identities stay in
the original report and JSON; this summary does not alter them.
