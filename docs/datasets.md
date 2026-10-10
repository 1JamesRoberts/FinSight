# Dataset, model, and API review checklist

PhraseBank and SEntFiN have now been acquired for reviewed local noncommercial
research; all source files remain in ignored storage. No model weights or provider
integration are present. See [dataset-loading.md](dataset-loading.md) for checksums,
loading instructions, and measured integrity results.
Phase 1's public-source candidate comparison is recorded in
[dataset-comparison.md](dataset-comparison.md), reviewed on 2026-10-10.
An independent-evaluation shortlist is recorded in
[evaluation-dataset-shortlist.md](evaluation-dataset-shortlist.md).
SEntFiN's scoped local-research assessment and immutable revision are recorded in
[sentfin-asset-review.md](sentfin-asset-review.md) and
[asset-revisions.json](../configs/asset-revisions.json).
**Other asset approvals, public-text rights, and final-test selection remain pending.** This checklist
is not a legal conclusion or authorization to acquire or redistribute an asset.

Complete one record per exact dataset revision, model checkpoint, or provider
product before downloading, persisting, training on, or republishing content.

## Asset record template

- Name, asset type, canonical source, owner/provider:
- Exact revision/checkpoint/product tier and retrieval date:
- License and API terms URLs; version/date; retained evidence location:
- Intended use (research, training, inference, storage, portfolio demo):
- Permitted and prohibited uses; review rationale:
- Attribution/citation requirements:
- Commercial-use, redistribution, derivative-work, and model-output terms:
- Full-text vs metadata rights; retention/deletion requirements:
- API rate limits, access restrictions, and display/link requirements:
- Provenance of underlying material; third-party rights or privacy constraints:
- Storage location, checksum/version, and reproducible acquisition notes:
- Reviewer, review date, decision (pending/approved/rejected), unresolved questions:

## Before acquisition

- [ ] Verify source authenticity and pin the exact asset revision.
- [ ] Review the license and provider terms for the intended usage.
- [ ] Distinguish dataset packaging rights from rights in underlying news text.
- [ ] Record training, local storage, public display, and redistribution rights separately.
- [ ] Record attribution, retention, rate limits, and other obligations.
- [ ] Decide whether only metadata/source links may be retained or displayed.
- [ ] Resolve missing or conflicting terms; record the review decision.
- [ ] Keep credentials out of Git; choose an ignored artifact location.

## Before experiments or publication (later phases)

- [ ] Document labels, annotation quality, coverage, selection bias, and limitations.
- [ ] Record dataset versions, splits, duplicate checks, seeds, and configurations.
- [ ] Audit overlap with pretrained-model training data. ProsusAI/finbert's
      Financial PhraseBank overlap must be addressed before claims of independent
      generalization; create an independent benchmark or state overlap explicitly.
- [ ] Freeze the final test set; do not train or tune against it.
- [ ] For price research, record news availability/ingestion times, market cutoff,
      exchange calendar, and chronological evaluation design.
- [ ] Recheck redistribution/display rights before publishing samples or artifacts.
- [ ] Publish limitations and reproducibility metadata alongside results.

## Candidate review status

| Candidate | Purpose in later phases | Status |
| --- | --- | --- |
| Financial PhraseBank | Three-class baseline development | Reviewed for local noncommercial research; pinned, acquired, validated |
| FiQA Task 1 | Target-aware/external evaluation candidate | Compared; mirror/original terms conflict unresolved; no acquisition |
| SEntFiN 1.0 | Preferred external/target-aware test candidate | Pinned, acquired, validated locally; public text and full overlap audit unresolved |
| SemEval-2017 Task 5 headlines | Backup target-aware news test | Shortlisted; release agreement unresolved |
| Twitter Financial News Sentiment | Optional social-media robustness test | Shortlisted; annotation provenance/counts/rights pending |
| FNSPID | News/price corpus | Excluded from human gold benchmark; generated sentiment |
| ProsusAI/finbert | Financial transformer candidate | Revision pinned; weight license scope unresolved; no weights acquired |
| News provider | Article metadata/text access | Not selected |
| Market data provider | Daily prices | Not selected |

See ROADMAP.md for candidate references and the linked comparison for evidence.
Final selection, terms verification, and artifact revision pinning remain separate
Phase 1 tasks; no access, training, or redistribution rights are assumed here.
