# Matched DeBERTa training for native targets

Phase 4 trains two fresh arms under the unchanged
[target-input protocol](target-sentiment-protocol.md): headline-only versus
native target first / headline second. Each arm uses exactly 11,547 training
annotations and 1,410 validation annotations from the existing prepared view.
There is no PhraseBank combination or five-company filtering.

The source is the cached original `microsoft/deberta-v3-base` checkpoint at
revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. Every source file is
checksum-verified before loading, with local-only loading and remote code disabled.
A new three-class pooler/head uses negative/neutral/positive index order. Previous
fine-tuned models are controls, never initialization sources.

## Fixed comparison

- Both input formats run seeds 42, 43 and 44, independently from the original Base.
- Same complete initial state and starting Torch CPU/CUDA RNG state per seed,
  verified by fingerprints before target-pair training starts.
- Same example order per seed and epoch, checked with actual training-row hashes.
  A separate seeded shuffle generator prevents dropout from affecting row order.
- All parameters train; equal weight per annotation. Headline-only inputs repeat
  for each original target annotation, including conflicting gold labels.
- LR 1e-5, AdamW betas 0.9/0.999, eps 1e-8, weight decay 0.01 with bias/LayerNorm
  exclusions, 10% linear warm-up/decay and gradient clipping at 1.
- FP32/eager/deterministic execution, TF32 disabled; train/evaluation batches 16/32.
- Maximum five epochs, patience two. Strictly higher all-target validation macro
  F1 improves a checkpoint; exact ties retain the earlier epoch. No best seed.
- Maximum length 128. The tokenizer adds special tokens; full targets are
  preserved and only second-sequence headlines may be truncated in the pair arm.
  Tokenization eligibility was verified before training; native names remain visible.

Each complete epoch has 722 optimizer updates; the maximum is 3,610 updates,
with 361 warm-up updates. Actual stopping and selected epochs can differ because
they use the same selection rule on different learned models. Dropout draws can
also diverge as input sequence shapes differ; initial weights/RNG and row order
are matched, rather than an assertion of identical stochastic trajectories.

## Run on the existing Windows GPU checkout

From a clean checkout with the reviewed private data and cached source weights:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.target_deberta_experiment
```

`--data-dir PATH` selects a different existing reviewed data root. The default
runs all six trials and writes the aggregate after all matching checks pass.
A single declared trial can be run with both `--arm matched_headline_only`
(or `--arm target_pair`) and `--seed 42` (or 43/44). The all-trial command checks
matched initialization for each seed before starting the second arm.

If interrupted between completed trials, rerun the all-trial command. Only fully
verified completed trial reports are reused: context/source/configuration,
training membership, private prediction hashes/native validation rows, cohort
metrics and full checkpoint file hashes must agree. Incomplete epoch progress
is diagnostic and does not resume optimizer state; that trial starts fresh.
Published aggregate reports are never overwritten. Routine review uses saved
reports instead of rerunning the experiment.

Training lives in `target_training.fit_target`; `score_examples` is separate
inference code and receives no label tensor. Validation uses inference mode and
never updates gradients. Gold labels remain separately attached to metrics.
Checkpoint saves bind input mode, dataset/plan/lock, source, label order and
implementation identities. The optional context extension preserves the old
artifact format when omitted, so previous model artifacts remain unchanged.

## Outputs and interpretation

- Public trial JSON: `reports/experiments/target-deberta-runs/<arm>/seed-42.json`
  (and 43/44); aggregate JSON/Markdown: `reports/experiments/target-deberta.*`.
- Private safe checkpoints: `data/interim/transformer-finetuned-models/<manifest-sha>/`.
  Each selected artifact is identified in its trial report; intermediate improved
  checkpoints also remain private.
- Private row predictions/run copies: `data/interim/target-deberta/<trial-sha>/`.
- Private epoch diagnostics: `data/interim/target-deberta-progress/<run-id>/history.json`.

All six established validation cohorts include accuracy, three-class macro F1,
class PRF/support, confusion matrices, raw NLL/Brier and ten-bin ECE. Reports log
architecture, parameter/file sizes, runtime, selected epochs and optimizer history.
Means/sample standard deviations describe three seeds; they are not confidence
intervals. There is no calibration fitting or temperature application.

The primary comparison is matched target-pair versus matched headline-only.
The saved Phase 3 control used 6,299 single-target headlines, so its comparison
also changes training examples/update budgets. Native-string seen/unseen cohorts
are not verified company novelty. Five-company evaluation remains too sparse.
Validation also selects epochs, so results are exploratory, not independent test
evidence. Reserved tests, original final results and immutable preparation lock
remain unchanged. This task does not create a target prediction API, entity
recognizer, new labeled test collection or any later roadmap phase.

## Verified results (2026-10-15)

See [the aggregate report](../reports/experiments/target-deberta.md) and companion
JSON for every seed, selected checkpoint, class metric and provenance field.
All six runs executed cleanly at `88e308634e669f2e066e083a2de7b11f4340c110`.
Aggregate JSON SHA-256:
`bb564efcff73577393e8be002c1cc50315e6950db7ae6eb699f7cdbc592e629f`.

| Validation cohort | Target annotations | Headline-only macro F1 | Target-pair macro F1 |
| --- | ---: | ---: | ---: |
| All targets | 1,410 | 81.19% ± 0.36 | 88.96% ± 0.42 |
| Single-target | 781 | 87.47% ± 0.20 | 88.11% ± 0.41 |
| Multi-target | 629 | 73.15% ± 0.79 | 89.81% ± 0.42 |
| Mixed-label headlines | 283 | 51.65% ± 0.59 | 91.91% ± 0.12 |
| Seen native target string | 1,147 | 81.10% ± 0.46 | 89.70% ± 0.32 |
| Unseen native target string | 263 | 81.54% ± 1.53 | 85.70% ± 1.06 |

Values summarize three seeds, with sample SD in percentage points. Cohorts overlap;
no confidence interval or independent-test claim follows from this table.
Single-target validation is the same 781-headline cohort used in Phase 3, but
both fresh models use the broader native-target training view. It is separate
from the already-consumed 766-headline single-target final test.

The primary target-pair gain is **7.77 percentage points overall** and **40.26
points on mixed-label examples**, calculated from unrounded means. The largest
gain occurs where a headline-wide prediction cannot represent differing target
labels. For these 283 mixed-label validation annotations, even an oracle that
chooses the best single label separately for each headline is limited to 56.54%
accuracy. This gold-derived structural ceiling is an accuracy bound, not a trained
baseline or an F1 bound. No model predictions were chosen using that oracle.
Target-pair predictions can differ across targets in 135/128/131 validation
headlines for seeds 42/43/44; headline-only labels never differ within a headline.

These findings support useful target conditioning on this validation corpus;
they do not prove general contextual understanding or unseen-company identity.
The seen-string gain is larger than the unseen-string gain, so entity familiarity
and native label tendencies may still contribute. Qualitative error analysis and
a separately locked target-level evaluation remain necessary before stronger claims.

Raw probability quality is mixed: overall NLL changes from 0.5146 to 0.5059,
Brier from 0.2828 to 0.1973, and ten-bin ECE from 0.0804 to 0.0882.
Accuracy/F1 gains do not imply better calibration. No old temperature calibrator
was applied and no new one fitted. Review/calibrate probabilities under a separate
protocol before treating their magnitudes as reliable confidence estimates.

## Saved models and verification

Both input formats use exactly the same architecture: 12 layers, hidden size 768,
12 attention heads, 184,424,451 parameters, FP32. The tokenizer input changes;
there is no separate classifier head per company. Each saved model/tokenizer
file set totals 740,189,720 bytes (excluding its artifact context manifest).
All six actually trained five epochs/3,610 updates, taking approximately 71.5
minutes combined inside the training loops on the RTX 4070 SUPER. This is a single
local execution, not a hardware benchmark. Selected epochs are:

| Seed | Headline-only | Target pair |
| --- | ---: | ---: |
| 42 | 4 | 4 |
| 43 | 3 | 3 |
| 44 | 4 | 5 |

Every selected checkpoint is safely serialized and fully checksummed on Windows.
To locate one, read `artifact_relative_path` and `artifact_manifest_sha256` from
its public trial JSON under `reports/experiments/target-deberta-runs/` and prepend
the Windows data root `C:\Hope\Project\financial-news-intelligence\data`.
The input mode is also bound in its manifest `context`; retain it when building
any later inference pipeline. The generic checkpoint loader alone does not infer
the correct target/headline input formulation.

Mac holds the same public reports, verified artifact manifests and reconstructed
private target-level predictions; full trained weights remain on Windows.
The export contained IDs/probabilities, not raw text/targets/gold labels. Mac
rebuilt all native rows using its existing locked view, reproducing all six
prediction hashes. Independent scikit-learn/NumPy calculations agree on accuracy,
macro F1, confusion matrices, clipped NLL, Brier and ECE in all 36 run/cohort cells.
Initial states/RNG, every shared epoch order, source code/data context and selected
checkpoint reload NLL also verify. All 484 tests and required Ruff checks pass
on both machines. Source freeze, original final result and immutable input plan/
preparation lock are unchanged. The historical lock remains prepared_not_trained;
these new trial reports, rather than edits to that receipt, record actual training.

This training/comparison increment is complete. Next: qualitative target-level
error analysis, followed by an explicit reserved-target evaluation contract.
No reserved test was scored or used for any choice in this task.

## Earlier baseline context

| Model | All-target F1 | Single-target F1 | Multi-target F1 | Mixed-label F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF headline LR | 75.20% | 81.15% | 67.37% | 48.13% |
| TF-IDF headline + target LR | 74.94% | 77.82% | 70.67% | 61.84% |
| Saved DeBERTa headline control | 80.58% | 88.10% | 71.22% | 50.15% |
| Matched DeBERTa headline-only | 81.19% | 87.47% | 73.15% | 51.65% |
| Matched DeBERTa target-pair | 88.96% | 88.11% | 89.81% | 91.91% |

All columns use the same development validation annotations. LR is one fixed
fit; DeBERTa entries are three-seed means. The saved control trained on 6,299
single-target headlines; every new LR/DeBERTa arm uses all 11,547 training
annotations. The primary controlled comparison remains the two matched DeBERTa
arms, not a causal comparison between architectures or against the saved control.
