# Registry company-mention baseline

The first mention detector uses the unchanged `companies-v1` registry and no new
model or dependency. The [guide](../../docs/phase-04-target-sentiment/company-mentions.md)
describes exact matching rules, ambiguity policy, CLI and Python usage.

The detector records original spans and uses leftmost longest matching. Accepted
and ambiguous mentions stay separate. Names are case-insensitive; bare tickers
require uppercase, while cashtags and NASDAQ prefixes are case-insensitive.
Unknown/suffixed exchanges are not inferred. Known names inside longer words do
not match. Ordinary ambiguous aliases do not inherit company context from a
financial headline or a different company elsewhere in the text.

Automatic sentiment uses the first resolved mention per company as its unchanged
model target and batches the accepted companies against the complete original
headline. Repeated aliases and GOOG/GOOGL metadata remain in the output for audit;
they do not create duplicate company predictions. Unresolved-only and unknown-only
headlines produce no sentiment and do not load a model through the CLI.

## Invented checks, not a mention accuracy benchmark

| Example surface | Default behavior |
| --- | --- |
| Apple Inc. | Accepted as Apple |
| Microsoft Corporation | Accepted as Microsoft; nested Microsoft suppressed |
| Apple / Amazon / Alphabet | Flagged ambiguous |
| Google LLC | Flagged until parent rollup is explicitly allowed |
| MSFT / $msft / NASDAQ: MSFT | Accepted ticker forms; original surface retained |
| msft / NYSE: MSFT / MSFT.O | Not accepted as supported ticker forms |
| pineapple / MICROSOFTISH | No match |
| Tesla | Outside detector coverage; explicit-target prediction still available |

Thirty synthetic detector checks and six new inference/CLI cases cover spans,
longest matches, repeated names, company grouping, ambiguity/rollup, boundaries,
invalid inputs, no-model abstention and integration with tiny RoBERTa inference.
No frozen headline, annotation, final score or dataset split is used to design or
evaluate these rules. The five-company registry remains byte-identical. This
baseline does not establish real-news mention precision/recall or five-company
sentiment accuracy. General NER, coreference, expanded names and a labeled mention
benchmark remain separate work.

All **639 tests passed on Mac and Windows**. Full Ruff lint/format passed in the
isolated Windows source snapshot; local checks excluding only the existing
untracked scratch notebook also pass. Full Mac workspace checks still report that
notebook's existing lint/format issues; its contents remain untouched.

The selected saved RoBERTa checkpoint passed an invented-headline CUDA smoke:
automatic and equivalent explicit-target predictions match exactly (maximum
probability difference 0.0). Repeated names group correctly; unresolved/unknown
names are unscored, while explicit Tesla input still works. The model is reused
across calls, with evaluation mode and disabled gradients. The actual automatic
CLI also returned the expected accepted targets and artifact identifiers.

Private logs, source fingerprints and invented-output verification are retained
under ignored `data/interim/company-mentions-smoke/20261019/`. The current source
matches the verified snapshot. No original research artifact, registry entry,
frozen data, model weight or benchmark score was modified.
