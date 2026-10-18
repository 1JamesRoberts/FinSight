# Reusable RoBERTa target prediction

The local Python/CLI interface uses the saved target-aware RoBERTa checkpoint,
with model loading performed once for repeated single/batch requests. The
[usage guide](../../docs/phase-04-target-sentiment/prediction.md) explains inputs,
output fields, pair tokenization, device overrides and optional company checks.

## Selection and scope

The [deployment config](../../configs/target-roberta-inference.json) selects seed
43, epoch 2, by highest unrounded saved all-target **validation** macro F1:

| Seed | Validation macro F1 (%) |
| --- | ---: |
| 42 | 88.2169 |
| 43 | 89.7683 |
| 44 | 89.2187 |

Exact ties select the lowest seed. No test data/results, private predictions or
weights are read to select the deployment checkpoint. This deployment choice was
made after final evaluation using saved validation evidence only. The completed
[benchmark](../experiments/phase4-target-final.md) remains a three-seed comparison;
its averaged scores are not claimed as this single checkpoint's performance.

The original checkpoint version is its artifact manifest SHA-256:
`2f742f69854f097b5b1351e4a1de126b54c28fe2c900edb3c2e41526cd32e855`.
Responses also identify the deployment-config digest. The model has 12 transformer
layers, hidden size 768, 12 attention heads and 124,647,939 parameters. Scores use
raw, uncalibrated probabilities in negative/neutral/positive order. The target is
sequence one; the headline is sequence two. Max length 128 truncates only the
headline; a target leaving no headline-token budget is rejected.

The optional registry wrapper abstains on ambiguous/unsupported identities and
preserves the original target string for scoring. It does not perform mention
extraction or establish five-company generalization. No new training, dataset
acquisition, model download, label changes or frozen-test scoring is involved.

## Verification

Twenty behavior tests cover validation-only selection/ties and drift, local loading
and reuse, exact pair encoding and label mapping, no gradients/weight changes,
batching/order, invalid inputs, target overflow, company abstention, invalid model
probabilities, CLI inputs and explicit CPU loading of a CUDA-provenance artifact.
The corruption fixture uses a separate copy to support Windows safetensors locks.

All **603 tests passed on Mac and Windows**. Full Ruff lint/format checks passed
in the isolated implementation checkout, including Windows. The Mac working tree's
full Ruff commands encounter the unrelated empty `notebooks/experiment.ipynb`;
checks excluding only that file pass. The notebook was left untouched.

On the RTX 4070 SUPER, the real selected artifact reproduced **all 1,410 saved
validation classifications with maximum probability difference 0.0**. Loading took
6.94 seconds; validation inference took 1.85 seconds. Repeated single/batch requests
reused the same model; gradients were disabled. The actual CLI also scored two
invented targets and returned the expected artifact/config identifiers. A read
guard rejected any attempted frozen-test/result reads; none occurred. Current
source hashes match the verified GPU source at commit
`4b38fe42d2cddf6b3569d5adfe3410cac47189a2`.

Private verification and CLI outputs are in ignored
`data/interim/target-inference-smoke/20261018/` on both machines. All original
research artifacts, freezes, evaluation contracts and final scores retain their
hashes. No existing evaluation receipt or result was rewritten.
