"""Conservative typing of proposed spans; unknown names remain explicit abstentions."""

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from finsight.data.us_listed_companies import Listing, USListedDirectory
from finsight.sentiment.companies import CompanyRegistry, resolve_company
from finsight.sentiment.issuer_aliases import AliasCatalog
from finsight.sentiment.organizations import OrganizationMention

RESOLVER_VERSION = "organization-resolution-v4"

# These describe complete institutional titles, not company-name dictionaries.
# Do not add benchmark acronyms or trim/split bad NER spans to improve scores.
INSTITUTION_PATTERNS = (
    (
        "government_department",
        r"(?:[\w.'-]+\s+){0,3}(?:Ministry|Department) of [\w .'-]+",
    ),
    (
        "named_court",
        r"(?:[\w.'-]+\s+){0,3}(?:Supreme|High|Constitutional) Court(?: of [\w .'-]+)?",
    ),
    (
        "securities_regulator",
        r"(?:[\w.'-]+\s+){0,3}Securities and (?:Exchange Commission|Markets Authority)",
    ),
    (
        "named_exchange",
        r"(?:[\w.'-]+\s+){0,5}(?:Stock|Commodity) Exchange(?: of [\w .'-]+)?",
    ),
)


@dataclass(frozen=True)
class OrganizationResolution:
    mention: OrganizationMention
    status: Literal["verified_company", "other_organization", "unresolved"]
    reason: str
    company_id: str | None
    primary_ticker: str | None
    matched_ticker: str | None
    registry_sha256: str
    sec_cik: str | None = None
    listings: tuple[Listing, ...] = ()
    directory_sha256: str | None = None
    company_type: Literal["company", "other_organization", "uncertain"] = "uncertain"
    classification_reason: str = "no_company_type_evidence"
    aliases_sha256: str | None = None

    @property
    def identity_status(self) -> Literal["linked", "unresolved"]:
        """A company classification alone never establishes issuer identity."""
        return (
            "linked"
            if self.status == "verified_company" and self.company_id
            else "unresolved"
        )


def resolve_organizations(
    text: str,
    mentions: tuple[OrganizationMention, ...],
    registry: CompanyRegistry,
    *,
    company_context: bool = False,
    allow_parent_rollup: bool = False,
    us_directory: USListedDirectory | None = None,
    confirmed_companies: tuple[str, ...] = (),
    aliases: AliasCatalog | None = None,
) -> tuple[OrganizationResolution, ...]:
    """Use only source spans and reviewed rules, never sentiment or gold labels.

    Verified means an identity match under an explicit source policy, not proof
    of real-world relevance. Company type may be known without an issuer link.
    An ORG label alone cannot supply company context.
    Institutional typing is a lexical rule, not verification of a legal identity.
    """
    if aliases is not None and us_directory is None:
        raise ValueError("reviewed aliases require a pinned US directory")
    if not isinstance(confirmed_companies, tuple) or any(
        not isinstance(name, str) or not name.strip() for name in confirmed_companies
    ):
        raise ValueError("confirmed companies must be a tuple of nonempty names")
    # An exact whole-name caller assertion supplies type evidence only. It is
    # deliberately separate from company_context and cannot resolve an issuer.
    confirmed = {name.strip().casefold() for name in confirmed_companies}
    output = []
    for mention in mentions:
        if (
            not 0 <= mention.start < mention.end <= len(text)
            or text[mention.start : mention.end] != mention.text
        ):
            raise ValueError("organization span differs from source")
        identity = resolve_company(
            mention.text,
            registry,
            company_context=company_context,
            allow_parent_rollup=allow_parent_rollup,
        )
        company = identity.company
        status, reason = "unresolved", identity.reason
        issuer = None
        matched_ticker = identity.matched_ticker
        ambiguous_company_type = False
        if company is not None:
            status = "verified_company"
        elif identity.status == "unsupported":
            if us_directory is not None:
                match = (
                    aliases.match(
                        mention.text,
                        us_directory,
                        company_context=company_context,
                        allow_parent_rollup=allow_parent_rollup,
                    )
                    if aliases
                    else us_directory.match(
                        mention.text, company_context=company_context
                    )
                )
                reason = match.reason
                ambiguous_company_type = match.company_type_known or (
                    company_context
                    and match.status == "ambiguous"
                    and bool(match.issuers)
                    and all(
                        not candidate.type_review_required
                        for candidate in match.issuers
                    )
                )
                if match.status == "matched":
                    issuer = match.issuers[0]
                    status = "verified_company"
                    matched_ticker = match.matched_ticker
                    # Reuse reviewed identities when a directory spelling names
                    # the same issuer; never choose a primary security from SEC rows.
                    tickers = {listing.ticker for listing in issuer.listings}
                    known = [
                        c for c in registry.companies if tickers.intersection(c.tickers)
                    ]
                    if len(known) == 1:
                        company = known[0]
                    elif len(known) > 1:
                        issuer = None
                        status, reason = "unresolved", "registry_directory_conflict"
            # Require a whole-span title match; don't classify unknown companies
            # as institutions merely because they contain 'bank' or 'capital'.
            for rule, pattern in INSTITUTION_PATTERNS:
                if status == "unresolved" and re.fullmatch(
                    pattern, mention.text.strip(), re.IGNORECASE
                ):
                    status, reason = "other_organization", rule
                    break
        asserted_company = mention.text.strip().casefold() in confirmed
        if status == "other_organization":
            company_type = "uncertain" if asserted_company else "other_organization"
            classification_reason = (
                "conflicting_company_type_evidence" if asserted_company else reason
            )
        elif status == "verified_company":
            company_type, classification_reason = "company", "accepted_identity_policy"
        elif asserted_company:
            company_type, classification_reason = (
                "company",
                "caller_confirmed_company_name",
            )
        elif ambiguous_company_type:
            company_type, classification_reason = (
                "company",
                "source_backed_company_type",
            )
        else:
            company_type, classification_reason = (
                "uncertain",
                "no_company_type_evidence",
            )
        output.append(
            OrganizationResolution(
                mention,
                status,
                reason,
                company.company_id
                if company
                else f"sec-cik:{issuer.cik}"
                if issuer
                else None,
                company.primary_ticker if company else None,
                matched_ticker,
                registry.sha256,
                issuer.cik if issuer else None,
                issuer.listings if issuer else (),
                us_directory.snapshot_sha256 if us_directory else None,
                company_type,
                classification_reason,
                aliases.sha256 if aliases else None,
            )
        )
    return tuple(output)


def verified_company_sentiments(
    text: str, resolutions: tuple[OrganizationResolution, ...], predictor: Any
) -> list[dict]:
    """Batch one original surface per verified company; skip abstentions entirely."""
    groups: dict[str, list[OrganizationResolution]] = {}
    for resolution in resolutions:
        if resolution.status == "verified_company":
            if not resolution.company_id:
                raise ValueError("verified company identity is missing")
            groups.setdefault(resolution.company_id, []).append(resolution)
    if not groups:
        return []
    targets = [g[0].mention.text for g in groups.values()]
    predictions = predictor.predict_many([text] * len(targets), targets)
    return [
        {
            "company_id": identifier,
            "primary_ticker": group[0].primary_ticker,
            "target": target,
            "company_identity_verified": True,
            "identity_verification_scope": (
                "sec_issuer_association_not_security_type_verification"
                if group[0].sec_cik
                else "registry_match_under_explicit_policy"
            ),
            "sec_cik": group[0].sec_cik,
            "listings": [asdict(listing) for listing in group[0].listings],
            "directory_sha256": group[0].directory_sha256,
            "mentions": [asdict(r.mention) for r in group],
            "sentiment": asdict(prediction),
        }
        for (identifier, group), target, prediction in zip(
            groups.items(), targets, predictions, strict=True
        )
    ]


def company_mention_sentiments(
    text: str, resolutions: tuple[OrganizationResolution, ...], predictor: Any
) -> list[dict]:
    """Score company-classified occurrences without requiring an issuer link.

    Keep unlinked mentions separate even when their names match. This returns
    per-headline target sentiment only; it neither aggregates nor maps stocks.
    The predictor owns the trained input format. Never inject target masking
    into a checkpoint trained on original target/headline pairs.
    """
    eligible = [r for r in resolutions if r.company_type == "company"]
    for resolution in eligible:
        mention = resolution.mention
        if (
            not 0 <= mention.start < mention.end <= len(text)
            or text[mention.start : mention.end] != mention.text
        ):
            raise ValueError("company mention differs from source")
    if not eligible:
        return []
    predictions = predictor.predict_many(
        [text] * len(eligible), [r.mention.text for r in eligible]
    )
    return [
        {
            "target": resolution.mention.text,
            "company_type": resolution.company_type,
            "classification_reason": resolution.classification_reason,
            "identity_status": resolution.identity_status,
            "company_identity_verified": resolution.identity_status == "linked",
            "identity_verification_scope": (
                "not_verified"
                if resolution.identity_status != "linked"
                else "sec_issuer_association_not_security_type_verification"
                if resolution.sec_cik
                else "registry_match_under_explicit_policy"
            ),
            "company_id": resolution.company_id,
            "sec_cik": resolution.sec_cik,
            "primary_ticker": resolution.primary_ticker,
            "matched_ticker": resolution.matched_ticker,
            "listings": [asdict(listing) for listing in resolution.listings],
            "registry_sha256": resolution.registry_sha256,
            "directory_sha256": resolution.directory_sha256,
            "aliases_sha256": resolution.aliases_sha256,
            "aggregation_eligible": resolution.identity_status == "linked",
            "mentions": [asdict(resolution.mention)],
            "sentiment": asdict(prediction),
        }
        for resolution, prediction in zip(eligible, predictions, strict=True)
    ]
