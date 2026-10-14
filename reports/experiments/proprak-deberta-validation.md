# Exploratory ProPrak01 financial DeBERTa comparison

Verified 2026-10-14. This experiment follows the completed locked final
evaluation and uses **development validation only**. It adds no final-test score
and does not change the final report or current model recommendation.

## Checkpoint and architecture

[ProPrak01/deberta-v3-financial-sentiment](https://huggingface.co/ProPrak01/deberta-v3-financial-sentiment)
is pinned at `f008a38828d87fbf80bea4c841333d9c493972e7`; the model card declares MIT.
The author reports joint PhraseBank/Twitter Financial News/FiQA supervision and
a continuation phase not included in the public notebook. Upstream memberships
and SEntFiN exposure are not established. PhraseBank/FiQA cannot be treated as
independent benchmarks for this checkpoint. Hosted benchmark scores use different
splits and must not be compared directly with our results.

DeBERTa-v3 Base: 12 encoder layers, hidden size 768, 12 attention heads,
intermediate size 3072, vocabulary 128,100. Masked mean pooling includes special
tokens and excludes padding; dropout 0.1 feeds a 768→3 classification head and
a separate 768→1 regression head. Unlike our standard DeBERTa classifier, it
does not use the sequence-classification context pooler.

Total parameters: **183,834,628**; FP32 parameter bytes: 735,338,512.
Published weight file: **735,425,035 bytes** (701.36 MiB).
Both heads were loaded strictly. Only the classification head was evaluated;
the regression head was not thresholded or used for stock prediction.

## Validation results

All rows use the same 781 single-target SEntFiN validation records and class
supports: negative 247, neutral 252, positive 282. The first four rows below are
one-seed runs; the final row is the existing three-seed mean. Adapted models
use our SEntFiN training split; this new checkpoint received **zero project
training**, no epoch selection and no calibration fitting.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| ProPrak01 DeBERTa-v3 Base, unchanged | 0.7900 | 0.7915 |
| Earlier financial DeBERTa-v3 Small, unchanged | 0.7388 | 0.7414 |
| Earlier financial DeBERTa-v3 Small, SEntFiN-adapted | 0.8745 | 0.8741 |
| Original DeBERTa-v3 Base, first SEntFiN training run | 0.8745 | 0.8738 |
| SEntFiN-trained DeBERTa Base, selected LR, seeds 42/43/44 mean | 0.8818 | 0.8810 |

The unchanged ProPrak01 model has higher observed F1 than the earlier unchanged
financial Small checkpoint, but lower observed F1 than SEntFiN-adapted Small and
our SEntFiN-trained Base models. This mixes upstream training histories, pooling
architectures, sizes and project adaptation; it does not isolate the effect of
financial training. No paired significance or superiority claim is made.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.8955 | 0.7976 | 0.8437 | 247 |
| neutral | 0.6823 | 0.7500 | 0.7146 | 252 |
| positive | 0.8134 | 0.8191 | 0.8163 | 282 |

Neutral has the lowest class F1. Confusion matrix (rows gold, columns predicted;
negative/neutral/positive):

```text
[197, 42, 8]
[18, 189, 45]
[5, 46, 231]
```

## Execution and compatibility

Batch 32, max length 128, seed 42, FP32 eager attention, deterministic operations,
TF32 disabled, RTX 4070 SUPER with CUDA 12.8. Inference
took 1.772s (2.269 ms/text) including tokenization,
length audit and output transfer after synthetic warm-up. Timing is descriptive;
it is not an isolated production benchmark. Peak allocated/reserved GPU bytes:
943,373,824/1,033,895,936.
No validation text was truncated; maximum observed token length was 27.

The source configuration was saved by Transformers 5.13.1. Exact state loading
succeeded under our locked 4.57.6 without changing dependencies. Its v5 tokenizer
metadata contains a list-valued extra-special-token field incompatible with the
old auto-loader. We load the full published tokenizer JSON via the fast wrapper,
assign existing token roles, verify identical vocabulary and probe token IDs,
and explicitly set inference padding/truncation. The serialized backend's saved
padding/truncation settings are disabled for the *untruncated probe comparison*.
No source file, token normalization, model weight or classification label order
was modified. Two compatibility preflights failed before validation inference;
only the final successful attempt produced scored validation outputs.

## Provenance and reproduction

Execution revision: `9e0190a33b8e3622ab9e85f31be758cf47ecc628`; clean checkout.
Aggregate/private run identity: `ca29d04131e40c4b3626815021820fe2317f79205a128a04a686f79c8233df1c`.
Validation membership: `df68a4fa8844a432385a0f23487f3e582513ca9b7e04e76c8c17b0f1584130da`.
Full file pins, runtime versions, historical-report hashes and implementation
identity are in [the aggregate JSON](proprak-deberta-validation.json).
Weights and ordered predictions remain ignored under Windows `data/interim/`.
No original final aggregate or frozen dataset bytes changed.

From the Windows project directory, run on a checkout without this output report:

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.proprak_deberta
```

The runner rejects an existing output before inference. Inspect the saved report
instead of repeating it; a deliberate rerun requires preserving/archiving that
report first. It verifies pinned source files and development memberships,
reconstructs the custom model locally, uses tensor-only strict loading and
never executes downloaded Python code. `--device cpu` is available.

The next optional experiment is SEntFiN adaptation with a protocol fixed before
training; this task does not implement it. Any confirmatory evaluation of new
models requires fresh untouched data, rather than choosing models using the
already-consumed final benchmark. The general sentiment model also does not
establish company-targeted capability; Phase 4 remains separate.

## Verification

All 388 tests and required Ruff lint/format checks pass on Mac and Windows.
Unit tests cover masked pooling, exact loading of both heads, serialized
padding/token-ID preservation and exclusion of reserved records. An independent
saved-output audit verifies all 781 ordered predictions against development
gold, probabilities, sklearn accuracy/F1/confusion matrix, code/source/lockfile
hashes and unchanged frozen bytes. The original final aggregate SHA-256 remains
`1c5a6334bc0f532075a1b3d9f6d821cb53901e6c83b14ee11dbf9156aa559dfe`.
This bounded unchanged-checkpoint validation task is complete; adaptation and
company-targeted modeling were not implemented.
