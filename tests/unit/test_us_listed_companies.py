"""Issuer identity, ambiguity and multiple listings use synthetic SEC-like rows."""

import pytest

from finsight.data.us_listed_companies import build_directory


def directory():
    return build_directory(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Acme Inc.", "ACME", "Nasdaq"],
                [1, "Acme Inc.", "ACMEB", "Nasdaq"],
                [2, "Pioneer Corp.", "PNR", "NYSE"],
                [3, "Pioneer Ltd.", "PIO", "NYSE"],
                [4, "Other Inc.", "OTHR", "OTC"],
                [5, "Example ETF Trust", "FUND", "CBOE"],
            ],
        }
    )


def test_exact_name_and_all_listings_share_cik_without_primary_guess():
    d = directory()
    result = d.match("Acme Inc.")
    assert result.status == "matched"
    assert result.issuers[0].cik == "0000000001"
    assert {x.ticker for x in result.issuers[0].listings} == {"ACME", "ACMEB"}
    assert d.match("NASDAQ:ACMEB").issuers == result.issuers
    assert d.match("$ACME").matched_ticker == "ACME"


def test_short_alias_and_bare_ticker_need_explicit_context():
    d = directory()
    assert d.match("Acme").status == "unresolved"
    assert d.match("ACME").status == "unresolved"
    assert d.match("Acme", company_context=True).status == "matched"
    assert d.match("ACMEB", company_context=True).status == "matched"


def test_short_name_collision_is_ambiguous_even_with_context():
    result = directory().match("Pioneer", company_context=True)
    assert result.status == "ambiguous" and len(result.issuers) == 2


def test_otc_unknown_and_obvious_fund_vehicles_not_verified_as_companies():
    d = directory()
    assert d.match("Other Inc.").status == "unresolved"
    assert d.match("$FUND").reason == "entity_type_requires_review"
    assert d.match("Nonexistent").status == "unresolved"


@pytest.mark.parametrize(
    "bad",
    [
        {"fields": ["name"], "data": []},
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[True, "Acme", "A", "NYSE"]],
        },
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[1, "", "A", "NYSE"]],
        },
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[1, "Acme", "bad ticker", "NYSE"]],
        },
    ],
)
def test_invalid_source_schema_rejected(bad):
    with pytest.raises(ValueError):
        build_directory(bad)


def test_same_ticker_for_different_issuers_is_ambiguous_not_last_row_wins():
    d = build_directory(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Acme Inc.", "SAME", "NYSE"],
                [2, "Beta Inc.", "SAME", "Nasdaq"],
            ],
        }
    )
    assert d.match("$SAME").status == "ambiguous"


def test_snapshot_checksums_and_policy_are_enforced(tmp_path):
    import json

    from finsight.data.us_listed_companies import load_directory, save_snapshot

    raw = json.dumps(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Acme Inc.", "ACME", "TXSE"],
                [2, "Beta Inc.", "BETA", "CBOE"],
            ],
        }
    ).encode()
    path = save_snapshot(raw, tmp_path, retrieved_at="2026-10-21T00:00:00+00:00")
    d = load_directory(path)
    assert d.match("TXSE:ACME").status == "matched"
    assert d.match("CBOE:BETA").status == "matched"
    assert d.match("NYSE:ACME").status == "unresolved"
    assert "@" not in (path / "manifest.json").read_text()
    (path / "source.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="source"):
        load_directory(path)
    (path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        load_directory(path)
