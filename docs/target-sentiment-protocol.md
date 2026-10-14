# Phase 4 target input and baseline protocol

Scope: native SEntFiN development inputs and a fixed comparison plan. This
increment prepares data and verifies tokenization; it does not train a model,
run model inference, or evaluate a reserved test set.

## Representation

Create one example per original target annotation. Preserve the exact native
headline, target and negative/neutral/positive label. A private example contains
its source record ID, duplicate group, annotation index, existing partition,
original text/target, label, multi-target/mixed-label flags and existing review
flags. IDs are stable hashes of source ID, annotation index and native target.

For the target-aware arm, the first tokenizer sequence is the target string and
the second is the complete headline. The tokenizer supplies its special-token
layout; do not manually concatenate separator strings. Company names remain
visible. No masking, ticker conversion, five-company restriction, parent grouping,
manual alias normalization or extra vocabulary tokens are introduced.

`finsight.data.target_pairs.encode_batch` returns padded tokenizer fields as
lists, ready for later tensor collation. It never includes gold labels in model
text. The future trainer must create the classification-label tensor separately.
The three-class order in the plan is negative, neutral, positive. Existing saved
control outputs must be reordered from their own artifact mapping; their native
logit order must not be assumed to match the new heads.

## Eligibility and grouping

- Keep every valid source annotation in the existing development view: 11,547
  training examples from 8,598 headlines, and 1,410 validation examples from
  1,053 headlines. Do not assign new splits after target expansion.
- Preserve all annotations from the same headline and all duplicate-connected
  groups within their existing partition. Reject reserved records before corpus
  selection, duplicate source IDs, cross-partition groups and normalized exact
  headline overlap. Reject blank inputs, unsupported labels and repeated identical
  native target strings within one headline.
- Keep the 95 nonliteral target annotations identified by the earlier audit.
  A target need not appear verbatim for target-plus-text modeling. This does not
  solve span alignment; masking/mention-span experiments would need separate rules.
- Retain all 11 existing development-only training headlines (17 target examples)
  and their review reasons: ten target-coverage differences and one cross-corpus
  overlap. They are allowed in training by the original split policy, not silently
  declared clean. Fail if a development-only row enters validation.
- Both new training arms use identical examples and equal weight per annotation.
  Thus multi-target headlines contribute multiple examples. No headline-level
  reweighting, majority-label replacement, class weighting or adjudication occurs.

The derived JSONL files contain actual target/headline/label values and lineage,
not just pointers. They stay in ignored local storage. Source files, parent
splits, native labels and original final-evaluation artifacts remain unchanged.

## Tokenization safeguards

Use the cached reviewed tokenizer from `microsoft/deberta-v3-base`, revision
`8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, with the existing locked dependencies.
Verify the source config, tokenizer config, SentencePiece model and README bytes.
Load with `local_files_only=True`, `trust_remote_code=False`, `use_fast=False`.
There are no model downloads, hub calls, weight loading or forward passes.

For target pairs, use `truncation='only_second'`, maximum length 128, right
truncation and right padding. Verify every encoded sequence against the complete
original target IDs plus the correctly budgeted headline prefix and tokenizer
special tokens. Reject a target if it leaves no room for at least one headline
token. Do not silently truncate the target, drop its row or increase the limit
without a reviewed protocol revision. Log headline truncation and target UNK
occurrences. UNK diagnostics do not automatically change eligibility.

The headline-only arm tokenizes only the headline, with the same 128-token cap.
Different targets from a headline receive identical baseline inputs. We audit
identical token sequences associated with different labels; those conflicts are
expected in the headline-only arm and are not an instruction to change gold.

## Three planned comparison arms

| Arm | Starting point | New training data | Model input |
| --- | --- | --- | --- |
| Saved headline control | Existing selected-rate Base artifacts, seeds 42/43/44 | None; previously trained on 6,299 single-target headlines | Headline only |
| Matched headline-only | Original Microsoft Base plus new sentiment head | All 11,547 native training target examples | Headline only, repeated for each target |
| Target pair | Same original Base and same seeded head initialization | Exactly the same 11,547 examples and labels | Target first, headline second |

The primary conditioning comparison is **target pair versus matched headline-only**.
Comparing only against the saved control would also change training data and
update counts. Saved-control predictions will be generated once per unique
validation headline and joined to all its targets; existing single-target
predictions may be reused only after source/order/probability validation.
No labels or gold targets are used to choose its headline prediction.

The two fresh training arms use LR 1e-5, seeds 42/43/44, train/evaluation batches
16/32, maximum five epochs, patience two and earliest exact ties. Both begin
independently from the original source checkpoint, without sequential adaptation
or initialization from the already SEntFiN-trained control. Use all encoder/head
parameters, FP32, deterministic settings and disabled TF32, as in the prior study.
AdamW uses betas 0.9/0.999, eps 1e-8, weight decay 0.01 (bias/LayerNorm excluded),
10% linear warm-up/decay and gradient clipping at 1. There is no new LR grid.
Select epochs using target-level validation macro F1 across all 1,410 examples.
Report all three seeds; do not select the best seed.

Report raw-probability accuracy, three-class macro F1, class metrics, confusion
matrices, NLL, Brier and ten-bin ECE. Do not apply the old single-target validation
calibrators to the expanded validation set or fit temperatures in this comparison.
Report all-target, single-target, multi-target and mixed-label headline cohorts,
plus seen/unseen native target strings relative to training. Native-string
novelty is not independently verified unseen-company identity. Any later
resampling uses headline duplicate groups, preserving related target examples.
These reused checkpoint-selection validation results remain exploratory.

## Provenance, operation and remaining work

`configs/target-sentiment-plan.json` fixes data/model/configuration/report
identities, seeds, input policies, training budget and evaluation scope. The
preparation command validates the three saved control manifest identities, but
does not load their weights or score them. Weight-file verification remains a
mandatory preflight for their later inference runner.

On the Windows project, the tokenizer is already under the reviewed checkpoint
cache. From a clean execution checkout with local data and without published
preparation outputs:

```powershell
$tokenizer = 'data/interim/transformer-checkpoint-cache/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
python -m uv run --locked --extra transformer python -m finsight.data.target_preparation --tokenizer-dir $tokenizer
```

`--data-dir PATH` can point to the reviewed data view in a separate checkout.
The runner refuses existing preparation reports/lock. Routine inspection should
use the saved outputs rather than rerun. An existing immutable private view also
requires a separate reviewed workspace for deliberate repetition.

Successful preparation writes private `train.jsonl`, `validation.jsonl`,
`manifest.json` and `preparation.json` under
`data/interim/target-pairs/<manifest-sha>/`. Public aggregate diagnostics are in
[the preparation report](../reports/target-input-preparation.md) and JSON.
`configs/target-sentiment-lock.json` binds the plan, private view manifest and
readiness report. Its status is **prepared_not_trained**; it is not a trained
model, a performance result or a final-test execution receipt.

No future reserved-test inference is authorized by preparing these files. The
already-consumed single-target final benchmark must not be reused for choosing
this model. Reserved multi-target examples stay untouched until a separate
applicable evaluation contract is fixed. The five application companies still
need additional labeled evaluation data; sparse coverage is documented separately.

Next bounded task: run the saved headline-only control on the expanded development
validation cohort and verify target-level evaluation/joining. Matched headline
training and target-pair training follow after that baseline runner is verified.

## Verified preparation results (2026-10-14)

The private view is complete: 11,547/1,410 training/validation examples, with
5,248/629 examples from multi-target headlines and 2,276/283 from mixed-label
headlines. All source labels and existing groups are preserved. Validation has
1,147 examples with native target strings seen in training and 263 with unseen
strings; this does not establish unseen-company identity.

Every pair passed the full-target token-ID check. Pair lengths are at most 40
training tokens and 34 validation tokens; headline-only maxima are 36/28. No
headline is truncated and no target contains an UNK token. Headline-only encoding
has 995/122 distinct token patterns with conflicting labels; pairs have zero.
These are input diagnostics, not model scores. The training pattern count differs
from the number of mixed-label source headlines because it groups encoded text
across records, including existing same-group coverage variants.

The three saved control artifact manifests were verified without loading weights.
No model was loaded, trained or scored. The original final aggregate remains
unchanged. All 15 new behavior cases and the full **455-test** suite pass with
required Ruff checks on Mac and Windows. Mac independently reproduced both
private JSONL hashes, summary counts, the data manifest, and the complete tokenizer
audit using checksum-verified tokenizer files; no source text or weights were
transferred for that verification.

Private view manifest SHA-256:
`be33e779e278b2c786c48ee3205cbad4f9e6bee98839d4c3b6509009a51a856c`.
Plan SHA-256:
`d8c87e545ec2546cace4a41f3f5840b1ad077d04cdcefec81447973524185a81`.
Preparation report SHA-256:
`a5f3ff3d343a77cd052f298f110722a9ca93de6bba5981e29c480605f4744042`.
Execution code revision:
`867857f380ca89b3a0292b6d7718aab4ff2cc1f0`.

This input/protocol increment is complete. It provides no target-level model
performance result and does not complete entity identification, target-aware
training, independent target evaluation, or five-company evaluation coverage.

## Saved headline-control runner

The validation-only runner verifies the preparation lock, reconstructs the source
train/validation membership, checks exact private JSONL bytes, verifies all saved
weight/tokenizer files and loads each selected artifact locally. It scores all
1,053 validation headline records for each seed and joins predictions to 1,410
annotations. All 781 single-target predicted labels and classification metrics
must reproduce the earlier report; maximum probability differences must stay
within 1e-5 to allow small FP32 differences from batch padding.

From the clean Windows GPU checkout with the existing private data/weights:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.target_control
```

Optional `--data-dir PATH` points to the existing reviewed private data root.
There are no downloads, training steps, temperature fitting or reserved-test
inference. Public results go to `reports/experiments/target-saved-control.json`
and `.md`; private target-level predictions go to ignored
`data/interim/target-saved-control/<run-sha>/seed-42.jsonl` (and seeds 43/44).
The runner refuses to overwrite published reports. Use the saved reports for
routine inspection. Mean and sample standard deviation summarize the three
seeds, with target annotations as the metric unit. Seed variation is not a
confidence interval. Native seen/unseen strings are not verified company IDs.
