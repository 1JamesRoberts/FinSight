"""Synthetic-only checks for conservative registry mention extraction."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from finsight.sentiment.companies import load_registry
from finsight.sentiment.mentions import detect_companies

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registry():
    return load_registry(ROOT / "configs/companies.json")


def test_longest_mentions_and_original_offsets(registry):
    text = "Apple Inc. profits rose while Microsoft Corporation reported losses."
    result = detect_companies(text, registry)
    assert [m.text for m in result.resolved] == ["Apple Inc.", "Microsoft Corporation"]
    assert not result.ambiguous
    assert [m.resolution.company.company_id for m in result.resolved] == [
        "apple",
        "microsoft",
    ]
    assert all(text[m.start : m.end] == m.text for m in result.resolved)
    assert result.detector_version == "registry-mentions-v1"
    assert result.registry_sha256 == registry.sha256


@pytest.mark.parametrize(
    "text",
    [
        "pineapple sales rose",
        "MICROSOFTISH",
        "NVIDIACorp",
        "foo_AAPL_bar",
        "Amazonian forest",
        "Appleton expands",
        "AAPL.O falls",
        "NYSE:AAPL gains",
        "NYSE: AAPL gains",
        "https://Microsoft.com/news",
        "person@Microsoft.com",
    ],
)
def test_substrings_and_unsupported_ticker_forms_do_not_match(registry, text):
    result = detect_companies(text, registry)
    assert not result.resolved and not result.ambiguous


@pytest.mark.parametrize(
    "text,target,ticker",
    [
        ("$msft rose.", "$msft", "MSFT"),
        ("NASDAQ: AAPL rose.", "NASDAQ: AAPL", "AAPL"),
        ("nasdaq:goog rose.", "nasdaq:goog", "GOOG"),
        ("GOOGL rose.", "GOOGL", "GOOGL"),
        ("(NVDA) fell.", "NVDA", "NVDA"),
    ],
)
def test_financial_ticker_forms_preserve_original_surface_and_share_class(
    registry, text, target, ticker
):
    result = detect_companies(text, registry)
    assert len(result.resolved) == 1
    mention = result.resolved[0]
    assert mention.text == target and mention.resolution.matched_ticker == ticker
    assert text[mention.start : mention.end] == target


def test_lowercase_bare_tickers_do_not_resolve_ordinary_text(registry):
    assert not detect_companies("goog and aapl", registry).resolved
    assert len(detect_companies("$goog and $aapl", registry).resolved) == 2


def test_ambiguous_aliases_require_explicit_context(registry):
    text = "Apple and Amazon discussed the Alphabet."
    result = detect_companies(text, registry)
    assert not result.resolved
    assert [m.text for m in result.ambiguous] == ["Apple", "Amazon", "Alphabet"]
    assert {m.resolution.reason for m in result.ambiguous} == {
        "company_context_required"
    }
    assert len(detect_companies(text, registry, company_context=True).resolved) == 3


def test_company_elsewhere_does_not_resolve_a_bare_alias(registry):
    result = detect_companies("Microsoft expands near the Amazon river.", registry)
    assert [m.text for m in result.resolved] == ["Microsoft"]
    assert [m.text for m in result.ambiguous] == ["Amazon"]
    assert (
        detect_companies("Apple Inc. sells apple juice.", registry).ambiguous[0].text
        == "apple"
    )


def test_google_rollup_requires_its_own_opt_in(registry):
    text = "Google LLC and Google expanded."
    result = detect_companies(text, registry, company_context=True)
    assert not result.resolved
    assert [m.text for m in result.ambiguous] == ["Google LLC", "Google"]
    assert len(detect_companies(text, registry, allow_parent_rollup=True).resolved) == 2


def test_commas_spaces_case_and_unicode_offsets(registry):
    text = "📈 amazon.com, inc. and MICROSOFT   CORP. grew."
    result = detect_companies(text, registry)
    assert [m.text for m in result.resolved] == [
        "amazon.com, inc.",
        "MICROSOFT   CORP.",
    ]
    assert all(text[m.start : m.end] == m.text for m in result.resolved)


def test_repeated_mentions_are_kept_for_audit(registry):
    result = detect_companies(
        "MSFT beat estimates; Microsoft raised guidance.", registry
    )
    assert [m.text for m in result.resolved] == ["MSFT", "Microsoft"]
    assert result.registry_version == registry.version
    json.dumps(asdict(result))


def test_unknown_company_is_not_implicitly_added(registry):
    result = detect_companies("Tesla reported higher profits.", registry)
    assert not result.resolved and not result.ambiguous


@pytest.mark.parametrize("text", [None, "", "   ", 7])
def test_invalid_text_rejected(registry, text):
    with pytest.raises(ValueError):
        detect_companies(text, registry)


def test_flags_are_validated_even_without_matches(registry):
    with pytest.raises(ValueError):
        detect_companies("No known company.", registry, company_context="yes")


def test_detection_cli_runs_without_model_loading(monkeypatch, capsys):
    from finsight.sentiment.mentions import main

    monkeypatch.setattr(
        "sys.argv", ["mentions", "--text", "Microsoft grew near the Amazon river."]
    )
    main()
    output = json.loads(capsys.readouterr().out)
    assert [m["text"] for m in output["resolved"]] == ["Microsoft"]
    assert [m["text"] for m in output["ambiguous"]] == ["Amazon"]
