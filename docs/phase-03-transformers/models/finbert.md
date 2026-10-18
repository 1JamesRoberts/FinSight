# Pretrained FinBERT validation

This Phase 3 task runs the existing `ProsusAI/finbert` sentiment checkpoint with
no project fine-tuning. Financial PhraseBank and SEntFiN single-target validation
cohorts retain the exact membership used by Phase 2. Training records are never
scored or fitted, FiQA continuous labels and multi-target headlines are excluded,
and the frozen final test remains unscored.

## Run on the Windows GPU computer

In PowerShell, from `C:\Hope\Project\financial-news-intelligence`:

```powershell
python -m uv sync --locked --extra transformer --group eda
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment validate-finbert
```

The CLI defaults to `configs/finbert.json`, the existing freeze receipt, and the
saved Phase 2 tuning report. It verifies dataset and baseline lineage before
acquiring the pinned checkpoint. To select one validation cohort or another
aggregate report path:

```powershell
python -m uv run --locked --extra transformer --group eda python -m finsight.sentiment validate-finbert --cohort sentfin_single_target --report-path reports/experiments/finbert-sentfin-validation.md
```

Input flow: verified prepared development records → sentence-level cohort →
validation rows only → original texts → checkpoint tokenizer → padded tensor
batches → unchanged BERT sentiment head → softmax and explicit label mapping →
aggregate metrics. Vocabulary/tokenizer are provided by the checkpoint, not
fitted on our dataset. Source text is not exported to public reports.

## Fixed settings and outputs

`configs/finbert.json` fixes revision
`4556d13015211d73dccd3fdd39d39232506f3e43`, batch size 32, maximum sequence
length 128 including special tokens, CUDA, and seed 42. Inference uses FP32,
dynamic padding, right truncation, eager attention, evaluation mode, disabled
gradients, deterministic algorithms, and disabled TF32. CUDA failure raises an
error instead of silently switching devices. CPU is supported by explicitly
changing the device in a separate configuration; its timing is a separate run.

The checkpoint output order is **positive, negative, neutral**. The public
metric axes remain negative, neutral, positive. Softmax probabilities sum to
one but have not been calibrated; they are not measured probabilities of being
correct. Exact ties use checkpoint index order. Full token lengths and
truncation counts are recorded, including any discarded tail tokens.

The first run acquires six checkpoint files into ignored
`data/interim/finbert-checkpoint-cache/`. PyTorch weights and reviewed metadata
are checked against pinned upstream SHA-256 values before loading. The model
must load without missing/unexpected/mismatched keys. The project never creates
or trains a replacement classification head. Subsequent runs reuse the cache
and verify it again; there is no extra fitted model artifact to save.

Public aggregate results: `reports/experiments/finbert-validation.md` and JSON.
Local run, acquisition receipt, and per-record probabilities/gold labels:
`data/interim/finbert-validation/<run-sha256>/`. These remain ignored.
The JSON records checkpoint/file identities, environment, code revision and
source hashes, configuration, membership digests, metrics, architecture,
parameter count/bytes, downloaded file size, timing, and peak GPU memory.
MLflow integration and fine-tuning belong to later Phase 3 tasks.

## Comparison and interpretation

The runner checks the selected LR/SVM runs in the saved Phase 2 tuning report
against the current parent split, frozen benchmark, ordered validation IDs,
sample counts, and gold class supports. It copies their scores and fixed
VADER/majority controls; it does not retrain or rescore them. The TF-IDF models
were trained and selected using our development data; this FinBERT experiment
measures transfer of an existing externally trained checkpoint. Their different
training histories must remain visible when comparing scores.

PhraseBank is known FinBERT supervised-training data; our local train/validation
split cannot undo that exposure. Its result is a diagnostic, not an independent
generalization claim. SEntFiN validation has no known accessible-corpus overlap,
but full checkpoint pretraining exposure is unknown. Neither cohort is the final
frozen benchmark.

Timing is a single GPU pass after an invented warm-up, with CUDA synchronization.
It includes tokenization, the full-length audit, device transfer, forward pass,
softmax, and CPU results, but excludes acquisition/loading, hashing, metrics,
and serialization. Amortized milliseconds per text at batch size 32 are not
single-request latency. Historical Phase 2 CPU timings used another machine
and batch size one; they do not establish a direct speed advantage.

The user's local research decision permits proceeding with unresolved weight
license scope; see [asset review](../../datasets.md#gpu-preparation-asset-review-2026-10-11).
Weights/source text are not committed or redistributed. The immutable Phase 1
asset manifest stays unchanged; a subsequent per-run receipt records this
acquisition. No other transformer or training experiment is included.

## Verified result (2026-10-11)

The [validation report](../../../reports/experiments/finbert-validation.md) and JSON
record run `ee63070bdb16c74bef562cddd8a1f30c92ba757489d8203477c6fb3d45b43e2c`,
executed from clean code revision `a83794d0323e3380e763500b75ea2c1850c0d280`
on the RTX 4070 SUPER, Python 3.12.3, PyTorch 2.11.0+cu128, and Transformers
4.57.6. Ruff/formatting and all 257 tests passed on both Mac and Windows.

| Validation cohort | FinBERT accuracy | FinBERT macro F1 | Selected LR macro F1 |
| --- | ---: | ---: | ---: |
| PhraseBank (known supervision overlap) | 0.8822 | 0.8823 | 0.7446 |
| SEntFiN single-target | 0.7478 | 0.7492 | 0.8181 |

FinBERT beats the fixed VADER/majority controls on both cohorts. On SEntFiN,
it trails selected LR by about 0.0689 macro F1 (6.89 percentage points), so the
pretrained checkpoint does not replace the stronger development baseline.
Different training histories matter: LR learned from our SEntFiN training
partition, whereas this FinBERT run received none of it. PhraseBank performance
cannot support an independent advantage because upstream sentiment training
used that corpus.

SEntFiN's weakest FinBERT class is neutral (F1 0.6922, precision 0.6397).
58 positive and 49 negative validation texts were predicted neutral; total
neutral predictions are 297 versus 252 gold neutral labels. These aggregate
confusions suggest a useful direction for later error review and adaptation;
they do not by themselves explain the wording or annotation causes.

The model has 12 BERT encoder layers, hidden size 768, 12 attention heads,
and 109,484,547 parameters. Six downloaded files total 438,226,859 bytes
(about 438 MB); FP32 parameter storage is 437,938,188 bytes. All gradient flags
are disabled and no weights were updated. Neither validation cohort required
truncation (maximum token lengths 93 and 30). The measured GPU inference passes
took 0.577s and 0.562s, with peak allocated memory 601,229,824 bytes and reserved
memory 861,929,472 bytes. These timings are single-pass batched measurements,
not direct comparisons with historical CPU baselines.

Checkpoint files remain cached on Windows. The private run, acquisition receipt,
and predictions are retained on both machines; the Mac does not need a second
copy of the pretrained weights. No frozen test predictions were produced.
