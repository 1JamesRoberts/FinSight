# Windows GPU research setup

This preparation task installs the optional transformer runtime and transfers
existing reviewed data. It adds no inference or training pipeline. The frozen
SEntFiN final test remains unscored. The user subsequently authorized local
FinBERT research use with the missing weight-license declaration documented;
see [asset review and decision](datasets.md#gpu-preparation-asset-review-2026-10-11).

## Installation

From PowerShell in `C:\Hope\Project\financial-news-intelligence`, with Python
3.12 available:

```powershell
python -m pip install --user uv==0.11.20
python -m uv sync --locked --extra transformer --group eda
python -m uv run --locked --extra transformer --group eda ruff check .
python -m uv run --locked --extra transformer --group eda ruff format --check .
python -m uv run --locked --extra transformer --group eda pytest
```

`python -m uv` works even when the user's Scripts folder is absent from PATH.
Keep `--extra transformer` on later `uv run` commands that need the GPU runtime;
uv synchronizes the requested extras and can remove unrequested packages.
The optional `eda` group supplies plotting libraries used by the full test suite.
Dependencies install into this checkout's `.venv`, with no activation required.

The `transformer` extra pins PyTorch 2.11.0 and Transformers 4.57.6. On Windows,
uv resolves PyTorch from its explicit official CUDA 12.8 index; other platforms
use PyPI. The complete resolution is in `uv.lock`. Default installation and CI
omit this extra, keeping baseline development lightweight. This is a tested
pinned environment, not a claim that these are the newest releases. A separate
CUDA toolkit is unnecessary for the prebuilt wheel and tensor check performed
here; the NVIDIA driver must support its bundled runtime.

## Data transfer and integrity

Transfer is limited to the existing local noncommercial research scope. No
source text or weights are committed. The transferred artifacts are:

- `data/raw/acquisition-receipt.json` and the pinned source files/notices for
  PhraseBank, SEntFiN, and FiQA.
- `data/processed/splits/a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898/`:
  parent manifest and prepared records.
- `data/processed/benchmarks/b7cc249017ddd87f0b2c52a3abf3ff5b96f6920832b411f18eac547fec0b0f1c/`:
  frozen manifest and test records.

The transfer contains 16 files (21,182,818 uncompressed bytes). Source assets
were checked against `configs/asset-revisions.json` before packaging. Each
archive file was checked for its length, SHA-256, and safe relative path before
installation; differing existing files would abort the transfer. Installed
bytes were then rechecked. The local ignored receipt is
`data/interim/windows-transfer-receipt.json`.

For a later integrity check that returns only train/validation records:

```powershell
python -m uv run --locked --extra transformer --group eda python -c "from pathlib import Path; from finsight.data.benchmark import load_development_records; rows = load_development_records(Path('data'), Path('configs/benchmark-freeze.json')); print(len(rows), sorted({r['partition'] for r in rows}))"
```

Expected: `14963 ['train', 'validation']`. This verifies frozen and parent
checksums, duplicate-group partition consistency, and exclusion of reserved
benchmark groups. FiQA is preserved for later analysis; its continuous scores
are not silently converted into training classes. The two classification
training cohorts remain separate.

## GPU check

This command imports the supported BERT class and checks CUDA without loading a
checkpoint or accessing dataset text:

```powershell
python -m uv run --locked --extra transformer --group eda python -c "import torch, transformers; from transformers import BertForSequenceClassification; assert torch.cuda.is_available(); torch.manual_seed(42); a = torch.randn(256, 256); b = torch.randn(256, 256); c = a.cuda() @ b.cuda(); torch.cuda.synchronize(); assert torch.allclose(c.cpu(), a @ b, atol=0.001, rtol=0.001); print(torch.__version__, transformers.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

No `from_pretrained` call occurs. A successful check proves imports and a GPU
matrix multiplication work; it does not establish FinBERT accuracy, training
memory requirements, or inference performance.

## Repository synchronization

The Windows checkout uses the project's GitHub SSH remote and Windows OpenSSH
for Git. `core.autocrlf=false` preserves committed bytes used by freeze checks.
Git operations from the Mac require an SSH terminal (`ssh -tt`) on this machine;
without it, remote Git operations previously hung despite successful key
loading. Project updates for this task were transferred with Git bundles and
fast-forwarded only after checking that the Windows checkout was clean.
No commits were pushed to GitHub. Datasets and receipts remain ignored.

## Verification

Verified on 2026-10-11 with Python 3.12.3, uv 0.11.20, PyTorch
`2.11.0+cu128`, Transformers 4.57.6, and NVIDIA driver 610.88:

| Check | Result |
| --- | --- |
| CUDA availability | True; runtime 12.8 |
| GPU | NVIDIA GeForce RTX 4070 SUPER; 12,878,086,144 device memory bytes reported by PyTorch |
| Synthetic GPU/CPU matrix comparison | Passed |
| BERT sequence-classification class import | Passed; no checkpoint loaded |
| Reviewed transfer | 16/16 length and SHA-256 checks passed |
| Development loader | 14,963 verified train/validation records |
| Windows Ruff and formatting | Passed |
| Windows pytest | 233 passed |
| Mac Ruff, formatting, and pytest | Passed; 233 tests |
| Mac and Windows package build | Source distribution and wheel built |

The initial Windows run exposed newline translation in existing validation
artifact writers: calculated run IDs did not match bytes saved with CRLF.
VADER and shared TF-IDF LR/SVM writers now persist exact hashed UTF-8 bytes.
Regression assertions check run-directory hashes, prediction hashes, and
public/local JSON equality. The existing model-profile enrichment test also
passes on Windows. Historical experiment files were not rewritten or rescored.
CI now includes Windows/Python 3.12 alongside Linux/Python 3.11 and 3.12;
hosted execution remains unverified until the commits are pushed and CI runs.

GPU/development-loader evidence is retained locally in ignored
`data/interim/windows-gpu-smoke.json`. No pretrained weights or real-data
training/evaluation runs were performed. The user subsequently authorized
acquiring the pinned checkpoint for local research despite unresolved license
scope. The next model task is development-only batched inference with this
decision recorded in the experiment. No Phase 3 model task is marked complete.
