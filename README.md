# FinSight

FinSight is a financial news sentiment intelligence portfolio project for Data
Scientist, ML Engineer, and AI Engineer internships. Its primary research focus
is comparing traditional NLP methods, FinBERT, and company-specific sentiment.
A secondary study will test whether sentiment improves stock prediction over
market-only baselines, using time-respecting evaluation.

**Status: Phase 0 foundation only.** No datasets, sentiment models, ingestion,
stock prediction, API, or dashboard are implemented. See [ROADMAP.md](ROADMAP.md)
for the phase sequence and acceptance criteria.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
Python 3.11+ is supported; `.python-version` selects Python 3.12 for local work.
Run these commands from the repository root:

```sh
uv python install 3.12
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

`uv sync` creates `.venv`, installs FinSight in editable mode, and installs the
development dependencies from `uv.lock`. No environment activation is required.
Commit `uv.lock`; use `uv lock` deliberately when changing dependencies, then
rerun the checks. To apply formatting, run `uv run --locked ruff format .`.

## Configuration

Defaults work without a `.env` file. For local overrides:

```sh
cp .env.example .env
uv run --locked --env-file .env python -c 'from finsight.config import Settings; print(Settings.from_env())'
```

`FINSIGHT_DATA_DIR` defaults to `data` and is relative to the current working
directory unless absolute. `FINSIGHT_LOG_LEVEL` defaults to `INFO`; allowed values
are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` (case-insensitive).
`Settings.from_env()` reads the environment only when called; it does not load
`.env` automatically, create directories, or configure logging. An explicit
mapping can be supplied for tests. No API credentials are needed yet.

Keep secrets in the ignored `.env` file or deployment environment. Add only
redacted variable examples when a later phase introduces a provider.

## Structure

```text
.github/workflows/ci.yml   # Formatting, lint, tests, and package build
configs/                  # Future experiment configuration; no presets yet
src/finsight/
  config.py               # Typed environment settings
  data/                   # Future data adapters and preparation
  sentiment/              # Future sentiment research
  features/               # Future aggregation and features
  prediction/             # Future downstream prediction study
  pipelines/              # Future workflow orchestration
  api/                    # Future delivery adapters
tests/unit/              # Configuration tests
tests/integration/       # Reserved for later external integrations
data/{raw,interim,processed}/  # Ignored local research data
 models/                  # Ignored model artifacts
 notebooks/               # Future exploration
 scripts/                 # Future command entry points
 reports/{figures,experiments}/ # Research outputs
 frontend/                # Reserved for Phase 8
 infrastructure/          # Reserved for Phase 9
 docs/                    # Architecture and data-use review checklist
```

The module folders are empty boundaries, not implemented services. Details and
tradeoffs are in [docs/architecture.md](docs/architecture.md). Review the
[dataset/model/API checklist](docs/datasets.md) before acquiring any research
assets. No third-party data or model licenses have been approved by this setup.

## Verification and CI

GitHub Actions runs the same locked checks on Python 3.11 and 3.12 for pushes
and pull requests, and builds the package with `uv build`. CI uses
[setup-uv](https://github.com/astral-sh/setup-uv) to install uv and select Python.
After pushing to GitHub, confirm the workflow results before claiming CI passed.
Local checks alone cannot verify a hosted Actions run.

Work one roadmap task at a time, add tests for reusable behavior, and update
checkboxes only after verification. Keep the final test set separate from
training and tuning; record dataset provenance and model versions; compare
against baselines; prevent look-ahead bias. Negative research results are valid.
