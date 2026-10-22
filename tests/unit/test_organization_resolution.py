"""Abstain on unknown names; no ORG-confidence-based context or benchmark aliases."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from finsight.sentiment.companies import load_registry
from finsight.sentiment.organization_resolution import (
    resolve_organizations,
    verified_company_sentiments,
)
from finsight.sentiment.organizations import OrganizationMention

ROOT = Path(__file__).resolve().parents[2]


def resolve(name, **kwargs):
    return resolve_organizations(
        name,
        (OrganizationMention(name, 0, len(name), 0.999),),
        load_registry(ROOT / "configs/companies.json"),
        **kwargs,
    )[0]


@pytest.mark.parametrize("name", ["Microsoft", "Apple Inc.", "MSFT"])
def test_registry_matches_verify_whole_names(name):
    result = resolve(name)
    assert result.status == "verified_company"
    assert result.company_id and result.primary_ticker


@pytest.mark.parametrize(
    "name",
    [
        "Tesla",
        "Infosys",
        "IPO",
        "Marico Q2",
        "BSE & NSE",
        "RBI",
        "Central Bank of India",
    ],
)
def test_unknown_names_acronyms_and_bad_boundaries_abstain(name):
    result = resolve(name)
    assert result.status == "unresolved"
    assert result.company_id is result.primary_ticker is None


@pytest.mark.parametrize(
    "name",
    [
        "Securities and Exchange Commission",
        "Ministry of Finance",
        "London Stock Exchange",
        "Supreme Court of India",
    ],
)
def test_explicit_institutional_titles_are_other_organizations(name):
    assert resolve(name).status == "other_organization"


def test_ner_does_not_override_ambiguity_or_parent_policy():
    assert resolve("Apple").status == "unresolved"
    assert resolve("Apple", company_context=True).status == "verified_company"
    assert resolve("Google").status == "unresolved"
    assert resolve("Google", allow_parent_rollup=True).status == "verified_company"


def test_wrong_source_span_rejected():
    with pytest.raises(ValueError, match="source"):
        resolve_organizations(
            "Microsoft",
            (OrganizationMention("Other", 0, 5, 0.9),),
            load_registry(ROOT / "configs/companies.json"),
        )


@dataclass
class Prediction:
    label: str = "positive"


class Predictor:
    def __init__(self):
        self.calls = []

    def predict_many(self, texts, targets):
        self.calls.append((texts, targets))
        return [Prediction() for _ in targets]


def test_verified_only_prediction_groups_identity_and_skips_unknowns():
    text = "Microsoft and MSFT met Tesla"
    mentions = tuple(
        OrganizationMention(text[a:b], a, b, 0.9)
        for a, b in [(0, 9), (14, 18), (23, 28)]
    )
    registry = load_registry(ROOT / "configs/companies.json")
    resolved = resolve_organizations(text, mentions, registry)
    predictor = Predictor()
    result = verified_company_sentiments(text, resolved, predictor)
    assert predictor.calls == [([text], ["Microsoft"])]
    assert result[0]["company_id"] == "microsoft"
    assert len(result[0]["mentions"]) == 2
    assert result[0]["company_identity_verified"] is True
    assert verified_company_sentiments("Tesla", (resolve("Tesla"),), predictor) == []
    assert len(predictor.calls) == 1


def test_us_directory_extends_scope_without_bypassing_reviewed_ambiguity():
    from finsight.data.us_listed_companies import build_directory

    directory = build_directory(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Tesla, Inc.", "TSLA", "Nasdaq"],
                [1, "Tesla, Inc.", "TSLAB", "Nasdaq"],
                [2, "Apple", "AAPL", "Nasdaq"],
                [3, "Microsoft Corporation", "MSFT", "Nasdaq"],
                [4, "Example ETF Trust", "ETF", "CBOE"],
            ],
        },
        snapshot_sha256="test-snapshot",
    )
    assert resolve("Apple", us_directory=directory).status == "unresolved"
    assert resolve("Google", us_directory=directory).status == "unresolved"
    assert resolve("Tesla", us_directory=directory).status == "unresolved"
    result = resolve("Tesla", us_directory=directory, company_context=True)
    assert result.company_id == "sec-cik:0000000001"
    assert result.primary_ticker is None
    assert len(result.listings) == 2
    assert result.directory_sha256 == "test-snapshot"
    assert resolve("$ETF", us_directory=directory).status == "unresolved"
    # Full SEC name and reviewed alias must group as one existing identity.
    assert (
        resolve("Microsoft Corporation", us_directory=directory).company_id
        == resolve("Microsoft").company_id
    )
    predictor = Predictor()
    output = verified_company_sentiments("Tesla", (result,), predictor)
    assert output[0]["sec_cik"] == "0000000001"
    assert len(output[0]["listings"]) == 2
    assert predictor.calls == [(["Tesla"], ["Tesla"])]


def test_confirmed_company_can_have_sentiment_without_identity():
    from finsight.sentiment.organization_resolution import company_mention_sentiments

    result = resolve("Toyota", confirmed_companies=("Toyota",))
    assert result.company_type == "company"
    assert result.identity_status == "unresolved"
    assert result.status == "unresolved"  # Historical field describes identity.
    assert result.company_id is result.sec_cik is result.primary_ticker is None
    predictor = Predictor()
    scored = company_mention_sentiments("Toyota", (result,), predictor)
    assert predictor.calls == [(["Toyota"], ["Toyota"])]
    assert scored[0]["company_identity_verified"] is False
    assert scored[0]["aggregation_eligible"] is False
    assert scored[0]["company_id"] is None
    assert scored[0]["company_type"] == "company"


def test_company_context_or_high_ner_score_does_not_confirm_unknown_company():
    from finsight.sentiment.organization_resolution import company_mention_sentiments

    result = resolve("Unknown", company_context=True)
    assert result.company_type == "uncertain"
    predictor = Predictor()
    assert company_mention_sentiments("Unknown", (result,), predictor) == []
    assert predictor.calls == []


def test_explicit_company_evidence_does_not_override_institution_or_identity_policy():
    university = resolve(
        "Ministry of Finance", confirmed_companies=("Ministry of Finance",)
    )
    assert university.company_type == "uncertain"
    assert university.classification_reason == "conflicting_company_type_evidence"
    apple = resolve("Apple", confirmed_companies=("Apple",))
    assert apple.company_type == "company"
    assert apple.identity_status == "unresolved"
    assert apple.company_id is None  # No silent company-context/parent roll-up.
    assert resolve("Google", confirmed_companies=("Google",)).company_id is None
    assert (
        resolve("Toyota Motors", confirmed_companies=("Toyota",)).company_type
        == "uncertain"
    )


def test_unlinked_same_name_mentions_are_not_grouped_into_an_issuer():
    from finsight.sentiment.organization_resolution import company_mention_sentiments

    text = "Acme met Acme"
    resolutions = resolve_organizations(
        text,
        (
            OrganizationMention("Acme", 0, 4, 0.99),
            OrganizationMention("Acme", 9, 13, 0.99),
        ),
        load_registry(ROOT / "configs/companies.json"),
        confirmed_companies=("Acme",),
    )
    output = company_mention_sentiments(text, resolutions, Predictor())
    assert len(output) == 2
    assert all(
        item["company_id"] is None and not item["aggregation_eligible"]
        for item in output
    )
    assert output[0]["mentions"] != output[1]["mentions"]


def test_linked_identity_is_eligible_but_sentiment_keeps_original_target():
    from finsight.sentiment.organization_resolution import company_mention_sentiments

    result = resolve("Microsoft")
    assert result.company_type == "company" and result.identity_status == "linked"
    output = company_mention_sentiments("Microsoft", (result,), Predictor())
    assert output[0]["target"] == "Microsoft"
    assert output[0]["aggregation_eligible"] is True
    assert output[0]["company_identity_verified"] is True


def test_company_type_can_be_known_while_sec_issuer_collision_remains_unresolved():
    from finsight.data.us_listed_companies import build_directory

    directory = build_directory(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[1, "Acme Inc.", "A", "NYSE"], [2, "Acme Ltd.", "B", "NYSE"]],
        }
    )
    result = resolve("Acme", us_directory=directory, company_context=True)
    assert result.company_type == "company"
    assert result.identity_status == "unresolved"
    assert result.company_id is None


@pytest.mark.parametrize("bad", [(" ",), ("Toyota", 1), "Toyota"])
def test_explicit_company_names_require_nonempty_string_tuple(bad):
    with pytest.raises(ValueError, match="confirmed"):
        resolve("Toyota", confirmed_companies=bad)


def test_source_alias_can_confirm_company_without_silent_parent_link():
    # Use a tiny pinned source rather than requiring downloaded data.
    from finsight.data.us_listed_companies import build_directory
    from finsight.sentiment.issuer_aliases import AliasCatalog, IssuerAlias

    directory = build_directory(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[1, "Acme Technologies Inc.", "ACME", "NYSE"]],
        },
        snapshot_sha256="pinned",
    )
    catalog = AliasCatalog(
        "catalog-sha",
        "pinned",
        (
            IssuerAlias(
                "Acme",
                "0000000001",
                "subsidiary",
                True,
                ("https://example.com/filing",),
            ),
        ),
    )
    held = resolve(
        "Acme", us_directory=directory, aliases=catalog, company_context=True
    )
    assert held.company_type == "company" and held.identity_status == "unresolved"
    assert held.sec_cik is None and held.aliases_sha256 == "catalog-sha"
    linked = resolve(
        "Acme",
        us_directory=directory,
        aliases=catalog,
        company_context=True,
        allow_parent_rollup=True,
    )
    assert linked.sec_cik == "0000000001"
    assert linked.reason == "reviewed_parent_rollup_alias"
    with pytest.raises(ValueError, match="directory"):
        resolve("Acme", aliases=catalog)
