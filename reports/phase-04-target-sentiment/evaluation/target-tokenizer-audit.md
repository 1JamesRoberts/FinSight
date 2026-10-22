# RoBERTa target-format tokenizer audit and training lock

Verified 2026-10-20. [Aggregate JSON](target-tokenizer-audit.json) ·
[Training plan](../../../configs/target-format-plan.json) ·
[Preparation lock](../../../configs/target-format-lock.json) ·
[Input rules and commands](../../../docs/phase-04-target-sentiment/training/target-format-study.md).

All three formats pass. **No additional examples are excluded.** This audit did
not start training; the subsequent [detached job](../experiments/target-format-training.md) is
now complete and [reviewed](../experiments/target-format-comparison.md). This completes the Phase 4 tokenizer/locking task. Company detection
remains deferred. Frozen test records are not opened, and no weights are loaded.
This audit measures input preservation, not sentiment performance.

## Finalized shared membership

| Measure | Train | Validation |
| --- | ---: | ---: |
| Character-eligible headlines entering this audit | 8,497 | 1,044 |
| Character-eligible annotations entering this audit | 11,395 | 1,396 |
| Final retained headlines | 8,497 | 1,044 |
| Final retained annotations | 11,395 | 1,396 |
| Additional excluded headlines | 0 | 0 |
| Additional excluded annotations | 0 | 0 |

What each measure means:

- **Train / Validation:** the existing partitions; no examples move between them.
- **Character-eligible headlines entering this audit:** distinct headlines where
  every native target passed the earlier literal-span checks.
- **Character-eligible annotations entering this audit:** target–label pairs from
  those headlines. A headline with three targets contributes three annotations.
- **Final retained headlines:** headlines where every annotation passes every
  input format's tokenizer checks. A failure in any arm would exclude the whole
  headline from every arm, keeping comparison membership identical.
- **Final retained annotations:** annotations belonging to final retained
  headlines. All nine training runs will use these same examples.
- **Additional excluded headlines:** character-eligible headlines newly rejected
  by tokenization, beyond the earlier span exclusions. None were rejected here.
- **Additional excluded annotations:** all annotations from those newly rejected
  headlines, including otherwise eligible sibling targets. This equals entering
  minus final annotations; it excludes the original span-audit rejections.

Labels and cohort supports remain those in the
[span audit](target-span-audit.md#retained-label-and-cohort-support), including
283 mixed-label and 262 unseen-native-target validation annotations. Membership
hashes match that audit exactly. Original labels, splits and prior results remain
unchanged.

## Maximum complete input lengths

| Input format | Maximum train tokens | Maximum validation tokens |
| --- | ---: | ---: |
| Original target pair | 50 | 39 |
| Target markers | 58 | 47 |
| Target replacement | 46 | 39 |

What each attribute means:

- **Original target pair:** native target first, unchanged headline second;
  for example, `Acme` paired with `Acme rose; Beta fell.`
- **Target markers:** retain the target name in the first sequence and surround
  every selected literal occurrence in the headline with `[TARGET]` and
  `[/TARGET]`, such as `[TARGET] Acme [/TARGET] rose; Beta fell.`
- **Target replacement:** use `[TARGET]` as the first sequence and replace every
  selected literal occurrence in the headline with `[TARGET]`. This hides that
  literal name from both sequences; other names and contextual clues remain.
- **Maximum train tokens / Maximum validation tokens:** the largest complete
  pair per format/partition, including subwords from both sequences and four
  native RoBERTa pair special tokens. These are maxima, not averages; rows can
  reach their maxima on different examples. Padding is not counted.

All 38,373 annotation/format inputs fit in full within 128 tokens. Both content
sequences decode exactly back to their inputs, and native pair construction
preserves both full token sequences. Therefore every target and marker survives,
including repeated occurrences. No truncation, empty sequence or unknown-token
failure occurred. Slow/fast content IDs, paired IDs and attention masks agree.

The tokenizer is `FacebookAI/roberta-base` at revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Six existing tokenizer/config files
were copied from the authorized PC cache to ignored local storage and verified
against original asset hashes. `RobertaTokenizer` is the training implementation;
`RobertaTokenizerFast` equivalence supports the existing saved-model loader.
Vocabulary size stays 50,265. Markers use existing subwords: `[TARGET]` takes four
tokens in the first sequence. No vocabulary entries, resized embeddings or
pretrained MLM mask tokens are introduced.

## Locked training protocol

1. **Nine fresh runs:** three formats × seeds 42/43/44, starting from the original
   pinned RoBERTa Base snapshot with a new seeded three-class head. Initial
   state/RNG and common-epoch row order must match across formats within each seed.
   The original-pair baseline is retrained on this subset; no adapted checkpoint
   or historical full-data score substitutes for it.
2. **Existing optimization settings:** learning rate 2e-5; weight decay 0.01;
   training/validation batch sizes 16/32; maximum length 128; maximum five epochs;
   patience two; warmup ratio 0.1; clipping norm 1.0. AdamW betas are 0.9/0.999,
   epsilon 1e-8; biases and LayerNorm weights have no decay. Reuse the linear
   warmup/decay scheduler. Use FP32 deterministic eager attention with TF32 off,
   no class weights, no accumulation and no new hyperparameter search.
3. **Same selection rule:** each trial selects strict highest all-retained-target
   validation macro F1, keeping the earlier epoch on an exact tie. Report all
   three seeds; do not choose a best seed or combine train/validation.
4. **Evaluation:** all-target, single-target, multi-target, mixed-label, seen-native
   and unseen-native validation cohorts. Seen/unseen uses original annotated target
   strings in retained training, never replacement placeholders. Score saved and
   reloaded weights; report classification metrics, NLL, Brier and 10-bin ECE on
   raw softmax probabilities, without calibration fitting.
5. **Reporting:** every trial, mean/sample standard deviation across seeds, and
   same-seed macro F1 differences against the new original-pair baseline. Record
   architecture, parameter count, saved bytes, timings, histories, configurations
   and artifact/prediction hashes. Name-substitution stress checks remain a
   separate development diagnostic, not a model-selection metric.

## Artifacts and verification

The plan pins the original span report, backbone asset and training config. The
lock pins that plan, this JSON audit, `uv.lock` and the private prepared manifest.
Source hashes identify the exact uncommitted preparation code alongside its parent
Git revision. `training_started: false` describes the preparation state; the lock
is not a mutable job-status file.

Private native annotations and all three input views are under
`data/interim/target-format/9eab447b2c6c123a5ce9fd5775a3c76a5878bc963404ce6afedce346d22a9c18/`.
Its `tokenizer-exclusions.jsonl` is empty. Public outputs contain aggregates,
configuration and digests only.

All 673 pytest tests pass, including 13 new tokenizer/preparation tests covering
length boundaries, target overflow, unknown tokens, decoding differences,
slow/fast disagreements, padding/masks, whole-headline exclusions, private output,
non-overwrite and corruption rejection. Required full Ruff checks were run;
only the existing untracked scratch notebook has nine lint issues and formatting
failures. Checks excluding `notebooks/experiment.ipynb` pass; it remains untouched.

A second real-data audit reproduced every prepared view byte-for-byte while
explicitly forbidding frozen-record and weight-file paths. All lock/source/file
hashes verify. The frozen public contract, previous final target results and
saved RoBERTa deployment config keep their prior checksums. The private proof is
`data/interim/target-format-verification/20261020/verification.json`.

## Remaining work

The [nine-trial execution and review](../experiments/target-format-comparison.md) are complete.
This preparation task launches no
training jobs and changes no production predictor. Reused validation is not an
independent benchmark. Literal alignment does not establish identity/coreference,
replacement leaves contextual clues, and the filtered subset omits difficult span
cases. A newly selected variant needs fresh independent evaluation before a new
generalization claim; the consumed test remains untouched.
