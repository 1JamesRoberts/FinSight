# Company mention detection and sentiment

The `registry-mentions-v1` baseline finds literal names, aliases and tickers from
our existing five-company registry. It returns original headline spans and keeps
accepted and ambiguous candidates separate. It uses Python regular expressions;
no new model, external service, dataset or dependency is introduced.

## Workflow and rules

1. Match known forms in the original headline, preserving spelling and start/end
   character offsets (end exclusive). Names are case-insensitive. Bare tickers
   must be uppercase; cashtags and `NASDAQ:` prefixes are case-insensitive.
2. Keep the longest match at each position: `Apple Inc.` suppresses its nested
   `Apple`. Word boundaries prevent matches inside `pineapple` or `MICROSOFTISH`.
   Unsupported exchange/suffixed ticker forms are not interpreted as bare tickers.
3. Apply the existing identity policy. Bare Apple/Amazon/Alphabet stay ambiguous
   unless the caller sets `company_context=True`. Financial words or another
   company elsewhere in the headline do not automatically resolve ambiguity.
   Google-to-Alphabet grouping requires `allow_parent_rollup=True` separately.
4. For sentiment, choose the **first resolved mention per company** as the target
   and batch all accepted companies against the unchanged complete headline.
   The target retains its original surface form; it is not masked or replaced by
   a ticker. Selection never depends on model probabilities.
5. Return one prediction per accepted company, retaining all resolved mentions
   and all ambiguous candidates for review. GOOG/GOOGL mention metadata remains
   distinguishable even though sentiment is grouped at company level.

No accepted company means an empty prediction list, not neutral sentiment. The
CLI then skips loading RoBERTa entirely. A registry miss does not establish that
no company is present: Tesla is outside automatic detection, while explicit
`--target Tesla` continues to work.

The baseline does not perform general organization NER, semantic disambiguation,
pronoun resolution, fuzzy matching or product identification. It can miss unseen
spellings/Unicode-width variants and can match a known name used in another sense.
A financial headline alone is not independent confirmation of identity. Broader
NER and a separately labeled mention benchmark remain future work. Synthetic unit
checks do not establish real-news precision/recall or five-company accuracy.

## Detection only: works on the Mac without model weights

```bash
uv run --locked python -m finsight.sentiment.mentions \
  --text "Apple Inc. profits rose while Microsoft expanded near the Amazon river."
```

This returns `Apple Inc.` and `Microsoft` under `resolved`, and `Amazon` under
`ambiguous`. `--registry PATH` can locate the existing registry file; no acquisition
occurs. Both policy flags are available but should reflect explicit caller choices.

## Windows prediction with the saved GPU checkpoint

Run in PowerShell from the primary repository (the old execution checkout was
archived during cleanup):

```powershell
cd C:\Hope\Project\financial-news-intelligence
$python = '.\.venv\Scripts\python.exe'
& $python -m finsight.sentiment.mentions --text 'Apple Inc. profits rose while Microsoft reported losses.'
& $python -m finsight.sentiment.predict --text 'Apple Inc. profits rose while Microsoft reported losses.' --detect-companies --device cuda
```

`--detect-companies` replaces manual `--target`/`--company`; those modes cannot be
combined. `--company-context` and `--allow-parent-rollup` are accepted in automatic
mode as explicit policy assertions. Direct target prediction remains unchanged:

```powershell
& $python -m finsight.sentiment.predict --text 'Tesla reported lower profits.' --target Tesla --device cuda
```

Automatic prediction returns an object with `detection`, `companies` and
`target_selection`. The detection includes `resolved`, `ambiguous`, registry version
and SHA-256, and detector version. Each company result includes its identity,
representative target, resolved mentions, and the existing raw-probability sentiment
response with model/config identifiers. Raw probabilities remain uncalibrated.

## Python

```python
from pathlib import Path

from finsight.sentiment.companies import load_registry
from finsight.sentiment.mentions import detect_companies
from finsight.sentiment.models.target_predictor import TargetSentimentPredictor

registry = load_registry(Path("configs/companies.json"))
text = "Apple Inc. profits rose while Microsoft reported losses."
mentions = detect_companies(text, registry)  # No model or dataset needed.

predictor = TargetSentimentPredictor.load(device="cuda")  # Saved weights required.
result = predictor.predict_headline(text, registry)
```

Repeated calls reuse the loaded RoBERTa model. Ambiguous-only requests make no model
forward call. Detection does not change the selected checkpoint, tokenizer pair
encoding, training configuration, evaluation results or frozen datasets.

## Code

- `sentiment/mentions.py`: matching, original spans, ambiguity and detection CLI.
- `models/target_predictor.py`: grouping accepted mentions and batched sentiment.
- `sentiment/predict.py`: explicit versus automatic CLI modes and no-model empty path.
- `tests/unit/test_company_mentions.py`: invented matching/ambiguity cases.
- `tests/unit/test_target_predictor.py`: synthetic inference and CLI behavior.
