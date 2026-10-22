# Organization mention annotation policy v1

This policy is fixed before drafting annotations for the 50-headline development
sample. Inspect original headline text without displaying native sentiment targets
or detector predictions. Exhaustively mark every explicit named organization,
including names irrelevant to sentiment and repeated occurrences. Annotate both
companies and other organizations. Keep assistant drafts provisional; a person
must review every headline, including empty cases, before reference-quality claims.

## Entity types and boundaries

- `company`: a named business, bank, brokerage, insurer, commercial media outlet
  or corporate group used as an organization. State-owned commercial businesses
  count too. A parent/group name is annotated as written, without mapping to a
  subsidiary, ticker or legal entity. A clearly organizational website name may
  count; flag unclear organization/product uses for adjudication.
- `other_organization`: a named central bank, regulator, court, government body,
  association, political body or institution. For this operational taxonomy,
  exchanges are other organizations even if their operator is incorporated.
  This is a task definition, not a legal incorporation classification.
- Do not mark people, countries/regions, commodities, currencies, stock indices,
  generic sectors/collectives (for example airlines or banks), or unnamed pronouns.
  A named exchange embedded in an index phrase may have its own organization span;
  the index itself is not an organization. Do not expand a location metonym such
  as Wall Street into an inferred organization.
- Use the complete explicit name, including any written legal suffix or named
  organizational unit. Exclude surrounding instructions, role titles, possessive
  endings, prices, descriptive financial words and trailing sentence punctuation.
  Include punctuation that belongs to the name, such as the period in `Acme Inc.`.
- Use zero-based character offsets: start inclusive, end exclusive. The stored
  `text` must equal `headline[start:end]` exactly. Keep repeated mentions as
  separate spans. Use nonoverlapping maximal names; do not add a nested parent span
  inside a longer organizational name. Do not merge separated abbreviations or
  shared-name ellipses into an invented name.
- Flag an uncertain mention/type using `uncertain: true`, explaining the issue in
  the row's notes. The draft type is provisional. Resolve uncertainty with a human
  reviewer before approved evaluation; do not silently exclude hard headlines.

For the invented text `Sell Acme Inc. after Acme Inc. met the SEC.`, both `Acme Inc.`
occurrences are company spans and `SEC` is another organization. `Sell` is not part
of the company. No sentiment labels or ticker identities are added by this task.

## Review states and evaluation

Each row records `annotation_author: assistant`, `exhaustive: true` (an attempted
complete draft, not a gold guarantee), entities, notes, and review metadata.
Initially `review.status` is `pending` and `review.reviewer` is null. Human reviewers
correct omissions, boundaries and types, resolve uncertain flags, then approve
each row under their own name. Software checks the recorded assertion; it cannot
prove who reviewed. Do not mass-approve unread rows.

Exact-span micro metrics pool occurrences across all sampled headlines, including
empty ones. A partial boundary is one false positive and one false negative; no
overlap credit is given. Evaluate all-organization extraction first. A separate
company-candidate diagnostic treats every raw ORG proposal as a company candidate;
it intentionally penalizes non-company organizations and does not imply that the
NER model itself predicts company types. No downstream sentiment score is measured.

Pending assistant annotations can be evaluated only with explicit `--provisional`;
all results must retain that status. They are unsuitable as independent human gold,
an unbiased test set, or grounds to promote a detector. Revised human annotations
produce a separately hashed evaluation; original drafts and predictions are kept.
