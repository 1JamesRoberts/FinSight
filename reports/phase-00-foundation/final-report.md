# Phase 0 final report — Repository foundation

**As of:** 2026-10-22. **Status:** foundation implementation complete; historical
hosted-CI acceptance remains unverified. This report summarizes the foundation
milestone, not the dependencies/features subsequently added by other phases.

[All final reports](../README.md) · [Phase evidence index](README.md)

## Scope and status

Establish an installable, reproducible Python project with clear boundaries,
configuration, tests, linting and CI. Phase 0 itself adds no dataset processing,
sentiment model, prediction service or market experiment. All eight foundation
implementation tasks are checked in the [roadmap](../../ROADMAP.md).

## Data and method

Use one typed Python package under `src/finsight/`, built with `uv_build` and
installed using the locked `uv` environment. Ruff handles lint/format checks;
pytest verifies behavior. Configuration uses explicit environment values with
secret/data/model paths ignored by Git. GitHub Actions declares installation,
lint, format, tests and package builds; later changes add Windows coverage.

[Architecture decisions](../../docs/architecture.md) retain the original phase
record and subsequent extensions. Domain/research code remains separate from
external I/O, model training and orchestration. Add interfaces when a concrete
use case requires them; no generic dependency-injection or microservice framework.

## Results

The recorded 2026-10-10 local foundation verification passed locked installation
in the working tree and a fresh copy on Python 3.12.2, 11 unit tests, Ruff lint
and formatting, source/wheel builds and installed imports outside the repository.
Secret/data/model ignore patterns were checked. These are historical phase-level
checks, not the current total test count or proof of hosted Actions success.

The current dependency set includes later NLP work. The original architecture's
empty runtime dependency list describes the foundation snapshot, not today's
`pyproject.toml`.

## Decisions

Keep the `src` layout, one package, typed configuration and small functions with
explicit collaborators. Preserve reproducibility through `uv.lock` and run the
same quality commands locally and in CI. No extra framework was needed to create
module boundaries. Later sentiment work uses `models`, `training`, `experiments`,
`evaluation` and `artifacts` without replacing these principles.

## Limitations and remaining work

A configured workflow is not a verified hosted CI run. The original acceptance
criterion remains qualified until a GitHub Actions result is independently
checked. Current full Ruff commands also flag the existing untracked scratch
notebook; phase-specific historical green checks do not imply the current tree
is entirely lint-clean. Do not add credentials or redistribute datasets/weights
without the separate rights review.

## Evidence and reproduction

- [Foundation guide](../../docs/phase-00-foundation/README.md), [project setup](../../README.md), [architecture](../../docs/architecture.md).
- [Build/dependency configuration](../../pyproject.toml), [lockfile](../../uv.lock), [CI workflow](../../.github/workflows/ci.yml).
- [Coding/research rules](../../AGENTS.md), [terms ledger](../../docs/datasets.md), [phase acceptance record](../../ROADMAP.md).

For the current checkout use `uv sync --locked`, `uv run --locked ruff check .`,
`uv run --locked ruff format --check .` and `uv run --locked pytest`. These check
today's code, not a reconstruction of the original Phase 0 dependency snapshot.
