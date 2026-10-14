# Financial DeBERTa Base: SEntFiN adaptation

Post-final exploratory development experiment. No final-test inference,
calibration fitting, best-seed selection or changes to the locked final results.

## Protocol

Start each seed from ProPrak01/deberta-v3-financial-sentiment at
`f008a38828d87fbf80bea4c841333d9c493972e7`. Train the encoder and three-class
head on 6,299 single-target SEntFiN training headlines. Freeze both regression
head parameters; verify exact equality to source weights after reload.
Use 781 validation headlines (247 negative, 252 neutral, 282 positive).

Fixed LR 1e-5 inherits the prior Base validation choice; no new LR search.
Seeds 42/43/44; FP32; train/eval batch 16/32; length 128; maximum five epochs;
patience two; strict validation macro-F1 improvement and earliest exact tie.
AdamW betas 0.9/0.999, eps 1e-8, decay 0.01 (bias/LayerNorm excluded),
10% linear warm-up/decay and gradient clipping at 1. No class weights.
Budget matches the selected-rate Base runs, not their full preceding LR search.

## Results

| Seed | Selected epoch | Accuracy | Macro F1 | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 5 | 0.8835 | 0.8828 | 0.5842 | 0.2075 | 0.0972 |
| 43 | 2 | 0.8745 | 0.8747 | 0.4472 | 0.2121 | 0.0873 |
| 44 | 5 | 0.8873 | 0.8869 | 0.5525 | 0.2033 | 0.0906 |

| Metric | Three-seed mean | Sample SD |
| --- | ---: | ---: |
| accuracy | 0.8818 | 0.0066 |
| macro_f1 | 0.8815 | 0.0062 |
| nll | 0.5279 | 0.0717 |
| brier | 0.2076 | 0.0044 |
| ece | 0.0917 | 0.0050 |

## Same-cohort reference results

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Unchanged ProPrak01 financial Base (fixed checkpoint) | 0.7900 | 0.7915 |
| Adapted ProPrak01 financial Base (three-seed mean) | 0.8818 | 0.8815 |
| Original Microsoft Base, SEntFiN-trained (three-seed mean) | 0.8818 | 0.8810 |
| Selected TF-IDF Logistic Regression | 0.8195 | 0.8181 |

Means average metrics across independent training seeds; they are not a voting ensemble.
Sample SD measures variation across these seeds, not test-sampling uncertainty.
These are checkpoint-selection validation scores, not unbiased test estimates.

## Raw probability calibration

Lower is better for NLL, Brier and ECE. No temperatures were fitted here.
Base reference values below also use raw probabilities; do not compare
this model against the older calibrated Base values as an architecture test.

| Model (three-seed mean) | NLL | Brier | ECE |
| --- | ---: | ---: | ---: |
| Adapted financial Base | 0.5279 | 0.2076 | 0.0917 |
| SEntFiN-trained original Base | 0.4995 | 0.2082 | 0.0858 |

ECE uses ten equal-width top-confidence bins; multiclass Brier is the
unnormalized sum over classes (range 0–2). Lower ECE alone is not sufficient.

## Class results

| Class | Mean F1 | Sample SD |
| --- | ---: | ---: |
| negative | 0.9065 | 0.0023 |
| neutral | 0.8465 | 0.0074 |
| positive | 0.8915 | 0.0105 |

## Size, speed and saved models

12 layers, hidden 768, 12 attention heads; masked mean pooling and dropout 0.1.
Total parameters: **183,834,628**; trainable parameters: **183,833,859**; 769 regression parameters frozen.
Classification head 768→3; retained unused regression head 768→1.
The encoder changes during adaptation, so retaining regression weights
does not preserve upstream FiQA performance. These are classification-only
artifacts; continuous regression outputs have not been validated.
Strict reload uses all backbone and both-head tensors. Derived weights are safetensors.

| Seed | Training/validation/saving seconds | Inference ms/text | Artifact MiB | Manifest SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 42 | 382.19 | 1.593 | 709.25 | `f32b966f94a89aa8ed70dbcfb527ae6def2aa097b67d824ae4cd692b85738594` |
| 43 | 301.45 | 1.574 | 709.25 | `ab4d7eb6f40961df783665faa7c7d74a4a2fb459fd04e458cc36143a8237794a` |
| 44 | 381.00 | 1.653 | 709.25 | `89cdc2ce2eaa776a869f876e84f680000543daa7477e96c9b80b9aa912bdb85a` |

Artifact MiB includes model, tokenizer, configuration and manifest files;
it is not runtime RAM. Inference includes tokenization, length audit and
output transfer after warm-up at batch 32 on RTX 4070 SUPER. Timings are
descriptive and not an isolated production benchmark.

Saved models remain private on Windows under `data/interim/transformer-finetuned-models/<manifest>`.
All selected models were reloaded and reproduced selected-epoch validation metrics.

## Provenance, limitations and reproduction

Execution code: `efee37bd278d521d64d147e3e587094939357a96`; all three seeds use this revision.
Plan SHA-256: `868a31cd6456233d944f8635d3552b0d8d8552d4821a7457de1a3982e93edac0`.
Per-seed JSON records source files, memberships, artifact identities, epoch
history, trained/frozen parameter counts, predictions SHA-256 and implementation hashes.
Seed 42 starts from a clean checkout; later seeds may record only preceding
untracked seed reports. The runner rejects any other working-tree change.

Use the clean execution revision above in a separate checkout with access to
the reviewed local data directory. That revision predates the output reports.
Do not delete tracked reports in this checkout to bypass the safeguards.
Copy the reviewed data view into its ignored data directory, or add
`--data-dir PATH` to the commands below to point to the existing local view.

```powershell
python -m uv run --locked --extra transformer python -m finsight.sentiment.proprak_training --seed 42
python -m uv run --locked --extra transformer python -m finsight.sentiment.proprak_training --seed 43
python -m uv run --locked --extra transformer python -m finsight.sentiment.proprak_training --seed 44
```

Completed reports are refused before training. Inspect saved outputs rather
than repeating training. Use a fresh checkout for any deliberate rerun.
This checksummed plan also requires unchanged code during execution; only
the declared seed output files may be untracked. On a checkout containing
these committed reports, run the following to rebuild this summary only:

```powershell
python -m uv run --locked python scripts/report_proprak_adaptation.py
```

Upstream training includes PhraseBank/Twitter/FiQA; SEntFiN exposure is unknown.
The comparison changes financial supervision, pooling architecture and prior
training history together; it does not isolate financial pretraining causally.
Validation selected epochs for every trained model. We make no statistical
superiority claim and do not choose a seed or replace the final model based
on this exploratory comparison. New confirmatory claims require fresh untouched
evaluation data. No weights, text or private predictions are redistributed.

## Interpretation and verification

Adaptation raises observed macro F1 by about 9.00 percentage points over
the unchanged checkpoint. Relative to SEntFiN-trained original Base, the
mean difference is only +0.05 percentage points; mean accuracy is identical.
Seed variation is larger (0.62 versus 0.44 percentage-point sample SD).
Raw NLL/ECE are slightly worse; raw Brier is marginally lower. These results
give no compelling reason to replace the existing calibrated Base model.
They do not establish equivalence or statistical superiority. Neutral is
the weakest class on average. No extra LR search or calibration was added.

The independent saved-output audit verifies all 2,343 ordered validation
predictions, sklearn metrics, independent NumPy NLL/Brier/ECE, earliest
best-epoch selection, progress histories, source/artifact/code hashes,
real encoder/classifier changes and frozen regression tensors. Token IDs
match source for every train/validation text in all three saved tokenizers
(21,240 text encodings). A generic Mistral-regex warning from the pinned
tokenizer reload was retained in the logs; no regex fix was applied.
Token-ID checks verify faithful reproduction of the published tokenizer,
rather than independently validating its upstream design choices. Frozen data
bytes and the original final aggregate are unchanged. Audit code/results
remain private beside the run artifacts under
`data/interim/proprak-deberta-adaptation/audit.py` and `audit.json`.
The generator refuses inputs differing from the audited run hashes.
Ruff lint/format checks and all 390 tests pass on Mac and Windows.
