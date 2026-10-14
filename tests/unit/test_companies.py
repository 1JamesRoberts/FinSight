"""Explicit target identity is separate from free-text entity recognition."""

import json
from pathlib import Path

import pytest

from finsight.sentiment.companies import load_registry, resolve_company


@pytest.fixture
def registry():
    return load_registry(Path("configs/companies.json"))


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])
def test_primary_tickers_and_explicit_financial_forms(registry, ticker):
    for value in (ticker, ticker.lower(), "$" + ticker, "NASDAQ: " + ticker):
        result = resolve_company(value, registry)
        assert result.status == "resolved"
        assert result.company.primary_ticker == ticker
        assert result.matched_ticker == ticker
        assert result.registry_sha256 == registry.sha256


def test_share_class_is_preserved_in_company_resolution(registry):
    result = resolve_company("$GOOG", registry)
    assert result.company.company_id == "alphabet"
    assert result.company.primary_ticker == "GOOGL"
    assert result.matched_ticker == "GOOG"
    assert resolve_company("$GOOGL", registry).matched_ticker == "GOOGL"


@pytest.mark.parametrize(
    "name,ticker",
    [
        (" Apple, Inc. ", "AAPL"),
        ("ＭＳＦＴ", "MSFT"),
        ("microsoft   corp", "MSFT"),
        ("Nvidia Corporation", "NVDA"),
        ("Amazon.com Inc", "AMZN"),
        ("Alphabet Inc.", "GOOGL"),
    ],
)
def test_controlled_name_normalization(registry, name, ticker):
    result = resolve_company(name, registry)
    assert result.status == "resolved"
    assert result.company.primary_ticker == ticker


@pytest.mark.parametrize("name", ["Apple", "amazon", "Alphabet"])
def test_ambiguous_words_require_explicit_company_context(registry, name):
    result = resolve_company(name, registry)
    assert result.status == "ambiguous"
    assert result.company is None
    assert len(result.candidates) == 1
    assert result.reason == "company_context_required"
    assert resolve_company(name, registry, company_context=True).status == "resolved"


def test_google_grouping_requires_separate_explicit_permission(registry):
    result = resolve_company("Google", registry, company_context=True)
    assert result.status == "ambiguous"
    assert result.reason == "parent_rollup_required"
    accepted = resolve_company("Google LLC", registry, allow_parent_rollup=True)
    assert accepted.company.company_id == "alphabet"
    assert accepted.alias_kind == "subsidiary"
    assert accepted.matched_ticker is None
    assert accepted.input_text == "Google LLC"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "TSLA",
        "$TSLA",
        "AWS",
        "YouTube",
        "iPhone",
        "Apple beats Microsoft",
        "pineapple",
        "Amazon rainforest",
        "GOOGLEISH",
        "NYSE:AAPL",
        "NASDAQ: AAPL extra",
        "AAPL.O",
    ],
)
def test_unsupported_strings_are_not_fuzzy_or_substring_matched(registry, value):
    result = resolve_company(
        value, registry, company_context=True, allow_parent_rollup=True
    )
    assert result.status == "unsupported"
    assert result.company is None
    assert result.candidates == ()


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "duplicate_id",
        "ticker_collision",
        "alias_collision",
        "ambiguous_flag",
        "alias_kind",
        "empty_alias",
        "extra_field",
    ],
)
def test_invalid_registry_is_rejected_before_use(tmp_path, change):
    body = json.loads(Path("configs/companies.json").read_bytes())
    if change == "schema":
        body["schema_version"] = True
    elif change == "duplicate_id":
        body["companies"][1]["company_id"] = "apple"
    elif change == "ticker_collision":
        body["companies"][1]["other_tickers"] = ["AAPL"]
    elif change == "alias_collision":
        body["companies"][1]["aliases"][0]["name"] = "Apple Inc"
    elif change == "ambiguous_flag":
        body["companies"][0]["aliases"][0]["requires_context"] = "false"
    elif change == "alias_kind":
        body["companies"][0]["aliases"][0]["kind"] = "product"
    elif change == "empty_alias":
        body["companies"][0]["aliases"][0]["name"] = " "
    elif change == "extra_field":
        body["companies"][0]["unknown"] = 1
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        load_registry(path)


def test_registry_requires_an_explicit_path_and_preserves_version(registry, tmp_path):
    assert registry.version == "companies-v1"
    assert len(registry.companies) == 5
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "missing.json")
