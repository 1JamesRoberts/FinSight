# US-listed issuer lookup

Run from the project root. The existing five-company registry stays available;
pass the SEC snapshot explicitly to enable broader coverage. The selected local
snapshot path and hashes are in `configs/us-listed-directory.json`.

```bash
uv run --locked python -m finsight.data.us_listed_companies lookup \
  --directory data/interim/us-listed-directory/d76a4e6fd5d7cc540b3ef6974ece2ca508b3fd16360e8d9eca7b94ddce9f528b \
  --name Tesla --company-context

uv run --locked python -m finsight.sentiment.organizations \
  --text "Tesla reported higher earnings while Microsoft announced a new product." \
  --us-directory data/interim/us-listed-directory/d76a4e6fd5d7cc540b3ef6974ece2ca508b3fd16360e8d9eca7b94ddce9f528b \
  --company-context --device cpu
```

The second command requires the previously acquired pinned NER checkpoint. Add
`--with-sentiment --device cuda` on the GPU machine after copying the snapshot to
the corresponding ignored data path; that machine also needs the existing
verified deployment weights. No weights or raw directory are downloaded by lookup.
This task verified CPU NER/resolution locally; it did not install this update on
the GPU machine.

`--company-context` asserts the proposed names refer to companies; use it only
when appropriate for the input domain. Missing or ambiguous identities abstain.
All SEC listings are retained; a new issuer has no guessed primary ticker.
Company identity output is an issuer association, not proof of security type.

To acquire a fresh snapshot, set `FINSIGHT_SEC_USER_AGENT` in the process environment
to a truthful project identity and contact email, then run:

```bash
uv run --locked python -m finsight.data.us_listed_companies acquire
```

The CLI reads the process environment; it does not automatically load `.env`.
Keep real contact details in local configuration, never committed files or shell
examples. Acquisition makes one SEC request and records provenance; it does not
change the public selected-snapshot pointer. A newly acquired snapshot has a new
path; review it before selecting it. A clone needs local acquisition or a copy of
the original snapshot to reproduce lookups; re-downloading current source bytes
may produce a different hash. No new dependency is needed.

[Inventory, matching rules, examples and limitations](../../reports/phase-04-target-sentiment/experiments/us-listed-directory.md).
