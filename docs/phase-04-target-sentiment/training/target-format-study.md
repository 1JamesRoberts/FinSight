# Target markers and replacement: development preparation

Character alignment and the pinned RoBERTa tokenizer audit are complete. The
nine-run training protocol is locked; its [reviewed execution](../../../reports/phase-04-target-sentiment/experiments/target-format-comparison.md)
is complete. The original pair remains the recommended input format, and the
deployed checkpoint is unchanged. Company detection is deferred. Native SEntFiN target annotations provide the requested target;
no five-company registry or NER model is used.

## Inputs to compare

For the invented headline `Acme rose while Beta fell.`, predicting Acme:

| Variant | First tokenizer sequence | Second tokenizer sequence |
| --- | --- | --- |
| Current target pair | `Acme` | `Acme rose while Beta fell.` |
| Target markers | `Acme` | `[TARGET] Acme [/TARGET] rose while Beta fell.` |
| Target replacement | `[TARGET]` | `[TARGET] rose while Beta fell.` |

What each attribute means:

- **First tokenizer sequence:** the requested native target for pairs/markers,
  or the constant placeholder for replacement.
- **Second tokenizer sequence:** the headline, with only the selected target's
  literal occurrences changed in the marker/replacement variants.
- **Current target pair:** retains the original target and unchanged headline.
- **Target markers:** retains the name and adds visible start/end markers.
- **Target replacement:** hides the selected literal name from both sequences.

Predicting Beta creates a separate example marking/replacing Beta instead. The
other target remains visible. Replacement hides the selected literal target name
from **both sequences**; keeping Acme in the first sequence would not test that
identity-removal hypothesis. Other names, aliases, products or contextual clues
can still reveal identity. This is not complete entity anonymization.

`[TARGET]` and `[/TARGET]` are ordinary text strings in this preparation, not
pretrained `[MASK]` tokens or newly added tokenizer vocabulary. Tokenization and
full visibility have been verified: the longest input is 58 tokens, so no further
exclusions are needed at length 128. The existing
predictor and training input modes are unchanged; these transformations are not
automatically applied to a saved model that was trained on original text.

## Fixed character-alignment policy

`native-literal-boundary-all-mentions-v1`:

1. Search the unchanged headline for each native target literally, with
   case-insensitive matching and Unicode word-character boundaries. Preserve
   original start/end offsets; end is exclusive. No alias mapping, fuzzy matching,
   stemming, whitespace repair or target correction is performed.
2. Retain every non-overlapping occurrence of the selected literal target. Mark
   or replace all of them, applying edits from right to left to preserve offsets.
3. Flag a target with no boundary match, a span overlapping any other annotated
   target's span, or an existing reserved marker string. Nested names are excluded
   conservatively, even if a separate occurrence might be unambiguous.
4. Retain a headline only if **every native annotation** passes. Every experimental
   arm will use identical retained target annotations. Preserve original labels,
   train/validation assignments and headline/duplicate-group membership.
5. Save source hashes, private eligible IDs, counts and dry-run transformed-text
   hashes. Full excluded/repeated examples stay private. Selection never depends
   on polarity or predictions.

This checks lexical alignment, not entity identity or correct coreference.
Repeated text can denote a different sense, an instrument, or a geographic phrase.
Unannotated aliases and entities are outside this audit. Successful alignment does
not prove every marked span is semantically the intended target.

## Run on existing development files

From the repository root:

```bash
uv run --locked python -m finsight.sentiment.experiments.target_span_audit
```

No GPU, tokenizer download or model weights are required. `--root` and `--data-dir`
can locate an existing checkout/data directory. The command reads only the locked
prepared native train/validation files through `development_inputs`.

- Public counts: `reports/phase-04-target-sentiment/evaluation/target-span-audit.json`.
- Private content-addressed output:
  `data/interim/target-span-audit/<private_manifest_sha256>/`.
- Private files: `membership.json`, `review.jsonl`, `manifest.json`.
- Text transformation: `finsight.data.target_spans.transform_input`.
- [Findings and limits](../../../reports/phase-04-target-sentiment/evaluation/target-span-audit.md).

The private manifest guards its existing bytes on rerun. The public audit is
regenerated with the current revision/source hashes. Separate qualitative review
notes are exploratory assistant judgments, not gold span annotations.

## Tokenizer audit and locked training protocol

Every complete pair must fit within 128 tokens, decode to the unchanged input,
contain no unknown/empty content sequence, and have matching slow/fast IDs and
masks. A failure excludes the entire headline in every arm. None occurred:
final membership remains 11,395 training and 1,396 validation annotations.

See the [verified audit and protocol](../../../reports/phase-04-target-sentiment/evaluation/target-tokenizer-audit.md).
`configs/target-format-plan.json` pins nine trials and original backbone/config
hashes; `configs/target-format-lock.json` pins the audit and private prepared views.
Full native examples and all three transformed inputs stay under ignored
`data/interim/target-format/<data_manifest_sha256>/`. Frozen records are not read.

Verify the existing lock using the copied local tokenizer cache:

```bash
uv run --locked python -m finsight.sentiment.experiments.target_format_preparation \
  --tokenizer-dir data/interim/target-format-tokenizer/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --check
```

To reproduce preparation in a separate checkout/data workspace, omit `--check`.
Preparation refuses to overwrite a published audit/lock. Use `--root` and
`--data-dir` for that workspace. The GPU PC's original cached snapshot at
`data/interim/transformer-checkpoint-cache/models--FacebookAI--roberta-base/snapshots/<revision>`
can also supply `--tokenizer-dir`; no hub download or weight read is needed.

The training job uses fresh RoBERTa Base variants on this same subset, including a
fresh original-pair baseline. Reuse the reviewed checkpoint, learning rate,
seeds 42/43/44, epoch-selection rule and optimizer settings. Do not substitute
the historical full-data baseline or start from its already-finetuned weights.
Use the same seeded initialization and row order for matched arms; let each arm
select its epoch using the same validation criterion.

Report all-target, mixed-label and exact-native unseen-target development results,
with unseen status computed from the **retained original training target names**,
not the constant replacement string. Add clearly labeled company-name substitution
stress checks; these are synthetic robustness checks, not independent evaluation.

Validation has already been reused for development. Masking improvement must be
measured, not assumed. The consumed test cohort stays untouched, and a newly chosen
variant needs fresh independent evaluation before a new generalization claim.
This preparation neither trains those variants nor changes production inference.

The [detached training runner](target-format-runner.md) implements execution of
this locked protocol. Its job status is separate from these immutable preparation
records; results are only complete after all nine verified trials finish.
