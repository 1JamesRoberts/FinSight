# Reviewed company aliases and fresh review

Resolver v4 adds an opt-in source-reviewed alias catalog and legal suffix handling
for `Co`/`Company`. It does not remove arbitrary business words such as Motor,
Technologies, Holdings or Group, nor use fuzzy matching. The original five-company
registry and archived GPU release remain unchanged.

Use the new code from this Mac's checkout with the existing local SEC/NER files:

```bash
uv run --locked python -m finsight.sentiment.organizations \
  --text "Toyota and Boeing announced new contracts while Dell expanded its product range." \
  --us-directory data/interim/us-listed-directory/d76a4e6fd5d7cc540b3ef6974ece2ca508b3fd16360e8d9eca7b94ddce9f528b \
  --company-aliases configs/us-company-aliases.json --company-context
```

BERT still has to detect each exact name. `--company-context` is an explicit caller
assertion that short names are company references, not an inference from the ORG
tag. The reviewed catalog names Toyota/Boeing as issuer aliases and Dell/Dell Inc.
as subsidiary names. No `--confirmed-company` is needed for a recognized alias
under this context policy. The catalog is pinned to the existing SEC manifest;
unknown CIKs, fund/trust-held issuers, duplicate entries, missing evidence and
wrong snapshot versions are rejected.

Dell is classified as a company, but linking it to listed Dell Technologies
requires `--allow-parent-rollup`. Without that flag it retains null issuer fields
and `aggregation_eligible: false`; the company-mention sentiment path can still
score it. The optional flag permits the documented parent mapping explicitly.
Toyota does not match Toyota Tsusho, Toyota Motor Sales or longer subsidiary
names by substring. Any alias/canonical-name collision retains all candidate
issuers and abstains from selecting one.

Add `--with-sentiment --device cuda` only where the saved sentiment checkpoint is
available. **The reviewed aliases are now installed and CUDA-verified on the GPU PC.**
Use the [updated launchers](issuer-alias-gpu.md), which supply the pinned SEC
and catalog paths. The original target sentiment model/input remains unchanged.

## Fresh reference review

**Status:** The user bulk-approved all 20 drafts. `annotations.reviewed.jsonl`
and `review-acceptance.json` beside the original draft record acceptance of the
existing span/type/CIK decisions and clearing of uncertainty flags. The original
sheet remains marked pending for provenance; use the acceptance metadata for
current status. [Reviewed metrics](../../reports/phase-04-target-sentiment/evaluation/issuer-alias-review-reviewed.md).
The review instructions below describe the original workflow.

The [20-row local review sheet](../../data/interim/issuer-alias-review/ad6bb8ddb3553db5412f480a27f0839a3d8254a3df684274a1dbd0e4fd04794b/review.md)
contains source links, manual assistant draft spans, CIK decisions, uncertainty
flags and explanations without model predictions. The sheet was written before
running NER on the sample. Previous approval of the first 20 headlines does not
approve these new references. All 20 rows require review, including the two empty
rows and eight flagged rows. The frozen sample excludes old page IDs and
normalized titles but does not claim different underlying news events or freedom
from model pretraining exposure.

The reference policy is current issuer identity without implicit parent roll-up.
Dell remains a company mention without a standalone listed-parent CIK. Review
historical brands/identities, the split Microsoft–Nortel names, manufacturer names
inside product references, and other organizations. Reply with numbered
approvals/corrections, or copy `annotations.provisional.jsonl` to
`annotations.reviewed.jsonl`, correct spans/types/identity decisions, resolve
uncertainty and record reviewer/status per row. Keep original drafts intact.
See [field definitions](data/us-company-review.md). No software can establish that
bulk approval means independent or individually verified annotation.

## Reproduction

The local retained response bytes and plan reproduce the sample:

```bash
uv run --locked python scripts/prepare_issuer_alias_review.py

uv run --locked python -m finsight.sentiment.experiments.issuer_alias_review \
  --packet data/interim/issuer-alias-review/ad6bb8ddb3553db5412f480a27f0839a3d8254a3df684274a1dbd0e4fd04794b
```

The preparation plan records five source response paths, URLs, retrieval times
and hashes. Three responses reuse the previously permitted source pool; Dell and
Toyota were newly acquired through two bounded Wikinews API requests. Those
responses, the archived baseline release and model weights are ignored assets;
a fresh clone alone cannot reproduce this run. The earlier script
`prepare_us_company_review.py acquire` implements the same bounded API acquisition
pattern, but re-downloading today's category membership may produce other bytes.
There is no automatic re-download during preparation/evaluation.

The runner verifies the archived baseline's complete source manifest, runs its
original resolver in a fresh subprocess on the same current NER spans, and records
all current source/catalog/plan/reference hashes. The separate `suffix_only` arm
isolates general legal suffix changes. A fourth arm with explicit parent roll-up
is a counts-only policy diagnostic; it is not scored against no-roll-up reference
CIKs. Add `--annotations <packet>/annotations.reviewed.jsonl` after approval to
score the primary three arms. Pending/uncertain labels are rejected before model
inference; no final accuracy is reported from unapproved assistant drafts.

[Sources, coverage results, limitations and verification](../../reports/phase-04-target-sentiment/evaluation/issuer-alias-review.md).
