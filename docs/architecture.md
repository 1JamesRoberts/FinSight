# Phase 0 architecture decisions

Recorded: 2026-10-10. Scope: repository foundation only.

## One package, explicit boundaries

FinSight starts as one Python package in a `src/` layout. Tests import the installed
package rather than modifying the import path, so packaging mistakes can surface
early. Empty subpackages mirror the roadmap's data, sentiment, features,
prediction, pipelines, and API areas. They contain no business behavior yet.

As later phases introduce behavior, keep research/domain logic independent of
HTTP frameworks, database clients, and providers. Pipelines coordinate use cases;
data and API adapters handle external I/O and translate to internal types.
Inject external collaborators at those boundaries when a real use case needs
them. Add interfaces only when they clarify a concrete integration or test.
No dependency-injection framework, generic repositories, microservices, or empty
class hierarchies are needed in Phase 0.

## Minimal dependencies

Python 3.11+ supplies dataclasses, pathlib, and typed environment access.
`uv_build` builds the installable package. Runtime dependencies are empty;
Ruff and pytest form the development dependency group. `uv.lock` records exact
resolved versions; `uv sync --locked` refuses to silently change that lock.
Local development defaults to Python 3.12; CI covers 3.11 and 3.12.

Defer pandas, scikit-learn, Transformers, PyTorch, LightGBM, MLflow, FastAPI,
PostgreSQL clients, and scheduling tools until their roadmap phases require
them. Frontend and infrastructure directories reserve locations only.

## Configuration at the boundary

`Settings` is an immutable typed dataclass. `Settings.from_env()` validates
nonempty paths and known logging levels, accepts a mapping for isolated tests,
and reads process environment variables only on demand. Relative paths use the
working directory; there is no guessed repository root, automatic `.env` loading,
or import-time filesystem/logging initialization. Callers load `.env` through
uv locally. Nonsecret experiment configuration belongs in `configs/` once those
experiments exist; no speculative YAML schema or parsing dependency is added.

Secrets belong in the environment and never in committed configuration. Add
provider-specific settings only when selecting a provider in a later phase.

## Artifact and research boundaries

Code, tests, configuration examples, and reviewed Markdown reports belong in Git.
Data and model artifacts are ignored except for explanatory files/placeholders.
The dataset/model/API checklist records unresolved licensing decisions before
acquisition. Future training and inference stay separate; final test sets must
remain independent of tuning, with FinBERT/PhraseBank overlap explicitly audited.
Stock research must use information available at the stated prediction cutoff.

## Quality gates

Ruff checks style, imports, modern Python usage, and common bugs; its formatter
provides one consistent layout. pytest verifies configuration behavior, environment
isolation, immutability, validation, and absence of filesystem side effects.
GitHub Actions uses a small two-version matrix, read-only repository permissions,
locked dependencies, and a package-build check. Hosted CI success requires an
actual run after the repository is pushed.


## Phase 1 extension: reviewed source files (2026-10-10)

Acquisition, source parsing, and integrity auditing now live in `finsight.data`.
They use the standard library and immutable typed records; no new runtime
framework or dependency is introduced. The explicit acquisition command uses
pinned manifest identities, preserves raw files, and verifies downloads before
atomic persistence. Source loaders preserve original metadata and expose common
string labels without assigning splits. A separate offline audit produces
aggregate integrity counts. Tests use synthetic source formats; CI never depends
on research data, network access, or model weights.

FiQA’s original nested schema has a dedicated `data/fiqa.py` loader. It retains
continuous target scores and unscored test inputs rather than forcing all sources
into a classification record. Acquisition reuses the manifest/checksum workflow;
no new runtime dependencies or model interfaces are introduced.

## Phase 1 extension: development EDA and freeze boundaries

The protected split and benchmark receipts bind local records to reviewed source
revisions and duplicate-group assignments. Development consumers verify checksums
and load only train/validation records; frozen/candidate groups remain withheld.
The benchmark contract keeps unknown pretraining exposure explicit.

EDA separates pure aggregate counting from verified input loading and optional
plot rendering. The CLI and notebook call the same functions, so figures do not
depend on manually edited notebook state. Matplotlib and ipykernel are in the
optional `eda` group; imports are deferred until plotting is requested. No plotting
framework enters ingestion or domain records. Five aggregate PNGs and JSON are
committed under a narrow ignore exception, while raw/processed text and notebook
outputs remain uncommitted. CI uses synthetic plotting examples with no downloads.

## Phase 2 extension: VADER baseline

`sentiment/vader.py` wraps the unchanged standalone analyzer with typed predictions
and immutable threshold settings. It preserves raw text cues and exposes scores
without calling them calibrated probabilities. No generic model hierarchy or
training abstraction is introduced for this rule-based baseline.

The separate validation command loads verified development records, selects
sentence-compatible validation cohorts, and compares predictions to a majority
control learned only from the corresponding training labels. Metrics use fixed
class order and explicit zero-division rules. Configuration, scoring-resource
hashes, source/freeze digests, code provenance, and timing accompany aggregates;
source IDs/gold/predictions remain local. No threshold tuning or frozen-test
inference is performed. The pinned VADER dependency contains its own lexicon;
inference is offline and does not require NLTK or resource downloads.

## Phase 2 extension: TF-IDF + Logistic Regression

`sentiment/tfidf.py` owns the typed training/inference interface and a scikit-learn
pipeline. The vectorizer and classifier fit together on training only; inference
uses frozen vocabulary/IDF. `tfidf_validation.py` coordinates verified split
loading, separate corpus-specific fits, saved-model validation, baseline metrics,
and artifact logging. It reuses sentence cohort selection and fixed-class metrics
from the VADER check rather than duplicating evaluation logic.

One reviewed runtime dependency, scikit-learn, supplies sparse features and
logistic regression. No pandas, model-service hierarchy, MLflow, or custom
optimizer is required. JSON stores settings and aggregate run metadata; standard
pickle stores trusted local fitted pipelines with version/checksum checks.
Artifact paths use serialized-content digests to preserve older runs. Models,
vocabularies, and per-record outputs stay ignored; aggregate reports are committed.
No target-aware behavior, hyperparameter search, or final-test scoring is added.

## Phase 2 extension: Linear SVM

`sentiment/svm.py` adds a concrete fitted model with typed label/decision-score
outputs, using shared TF-IDF configuration, vectorizer construction, and input
validation. Logistic Regression's probability interface remains distinct.
The existing development runner supports an explicit classifier choice, reusing
split verification, cohort selection, metrics, timing, controls, and artifact
logging. There is no generic model registry or new service/framework dependency.
Both weighted and unweighted presets run on each corpus separately. Source text,
fitted vocabularies/coefficients, and per-record outputs stay local; aggregate
validation reports are committed. The frozen benchmark is preserved.

## Phase 4 extension: explicit company identities (2026-10-14)

`finsight.sentiment.companies` holds immutable identity/alias/resolution records
and pure explicit-target lookup. One local JSON loader validates a versioned
registry; it has no import-time I/O. Sentiment, HTTP, storage and model frameworks
are not involved. Ambiguous names and optional subsidiary grouping are explicit
policy choices; unsupported targets return structured results. Ticker observations
are retained separately from the company-level project ticker. No generic entity
service or NER dependency is introduced. See `docs/company-targets.md`.

## Phase 4 extension: native target inputs (2026-10-14)

`finsight.data.target_pairs` expands verified development records into immutable
examples with source/group/annotation lineage. It preserves native text and labels;
no company registry substitution occurs. A shared tokenizer helper checks complete
first-sequence targets and second-sequence truncation without model inference.
`finsight.data.target_preparation` explicitly coordinates pinned local tokenizer
checks, private JSONL serialization, aggregate diagnostics and preparation lock
creation. Training/inference remain separate future runners; no generic trainer
framework, new dependency or model API is added here.
