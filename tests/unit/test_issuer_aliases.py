"""Reviewed aliases never override name collisions, context or parent policy."""

import json
from copy import deepcopy

import pytest

from finsight.data.us_listed_companies import build_directory
from finsight.sentiment.issuer_aliases import load_aliases


def fixture(tmp_path):
    directory = build_directory(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Acme Technologies Inc.", "ACME", "NYSE"],
                [2, "Beta Motor Corp/", "BETA", "NYSE"],
                [3, "Shared Inc.", "SHAR", "Nasdaq"],
                [4, "Other ETF Trust", "FUND", "CBOE"],
            ],
        },
        snapshot_sha256="pinned",
    )
    config = {
        "schema_version": 1,
        "directory_sha256": "pinned",
        "reviewed_on": "2026-10-21",
        "aliases": [
            {
                "name": "Acme",
                "cik": "0000000001",
                "kind": "subsidiary",
                "requires_context": True,
                "source_urls": ["https://example.com/filing"],
            },
            {
                "name": "Beta",
                "cik": "0000000002",
                "kind": "issuer_name",
                "requires_context": True,
                "source_urls": ["https://example.com/about"],
            },
        ],
    }
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(config))
    return directory, config, path


def test_reviewed_alias_and_parent_policy_are_separate(tmp_path):
    directory, _, path = fixture(tmp_path)
    aliases = load_aliases(path, directory)
    assert aliases.match("Beta", directory).status == "unresolved"
    result = aliases.match("Beta", directory, company_context=True)
    assert result.status == "matched" and result.issuers[0].cik == "0000000002"
    result = aliases.match("Acme", directory, company_context=True)
    assert result.status == "unresolved" and result.company_type_known
    assert result.reason == "alias_parent_rollup_required"
    assert (
        aliases.match(
            "Acme", directory, company_context=True, allow_parent_rollup=True
        ).status
        == "matched"
    )
    assert (
        aliases.match("Beta Motor Sales", directory, company_context=True).status
        == "unresolved"
    )


def test_alias_collision_with_canonical_name_is_never_first_match_wins(tmp_path):
    directory, config, path = fixture(tmp_path)
    config["aliases"].append(
        {**config["aliases"][1], "name": "Shared", "cik": "0000000002"}
    )
    path.write_text(json.dumps(config))
    result = load_aliases(path, directory).match(
        "Shared", directory, company_context=True
    )
    assert result.status == "ambiguous"
    assert {i.cik for i in result.issuers} == {"0000000002", "0000000003"}


def test_two_alias_issuers_stay_ambiguous(tmp_path):
    directory, config, path = fixture(tmp_path)
    config["aliases"].append({**config["aliases"][1], "cik": "0000000001"})
    path.write_text(json.dumps(config))
    assert (
        load_aliases(path, directory)
        .match("Beta", directory, company_context=True)
        .status
        == "ambiguous"
    )


@pytest.mark.parametrize(
    "change",
    ["snapshot", "unknown_cik", "fund", "duplicate", "blank_source", "bad_context"],
)
def test_invalid_or_stale_catalog_rejected(tmp_path, change):
    directory, config, path = fixture(tmp_path)
    config = deepcopy(config)
    if change == "snapshot":
        config["directory_sha256"] = "other"
    if change == "unknown_cik":
        config["aliases"][0]["cik"] = "0000000099"
    if change == "fund":
        config["aliases"][0]["cik"] = "0000000004"
    if change == "duplicate":
        config["aliases"].append(config["aliases"][0])
    if change == "blank_source":
        config["aliases"][0]["source_urls"] = []
    if change == "bad_context":
        config["aliases"][0]["requires_context"] = "yes"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_aliases(path, directory)


def test_company_suffix_normalization_retains_collisions_and_business_words():
    directory = build_directory(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Pioneer Co.", "A", "NYSE"],
                [2, "Pioneer Company", "B", "NYSE"],
                [3, "Acme Motor Corp/", "C", "NYSE"],
            ],
        }
    )
    assert directory.match("Pioneer", company_context=True).status == "ambiguous"
    assert directory.match("Acme", company_context=True).status == "unresolved"


def test_fresh_sample_excludes_old_ids_and_normalized_titles():
    from finsight.data.issuer_alias_review import select_fresh_sample

    pool = {
        "A": [
            {"pageid": 1, "ns": 0, "title": "Old headline"},
            {"pageid": 2, "ns": 0, "title": "OLD HEADLINE!"},
            {"pageid": 3, "ns": 0, "title": "Fresh news"},
        ]
    }
    old = [{"record_id": "wikinews:1", "headline": "Old headline"}]
    selected = select_fresh_sample(pool, old, per_category=1)
    assert selected[0]["record_id"] == "wikinews:3"
    assert selected == select_fresh_sample({"A": pool["A"][::-1]}, old, per_category=1)
    with pytest.raises(ValueError, match="insufficient"):
        select_fresh_sample(pool, old, per_category=2)


def test_alias_does_not_override_held_canonical_fund_name(tmp_path):
    directory, config, path = fixture(tmp_path)
    config["aliases"].append({**config["aliases"][1], "name": "Other ETF Trust"})
    path.write_text(json.dumps(config))
    result = load_aliases(path, directory).match(
        "Other ETF Trust", directory, company_context=True
    )
    assert result.status == "ambiguous" and not result.company_type_known


def test_archived_baseline_source_tampering_is_rejected(tmp_path):
    from finsight.data.target_preparation import json_bytes
    from finsight.sentiment.artifacts import sha
    from finsight.sentiment.experiments.issuer_alias_review import verify_release

    body = b"baseline source"
    manifest = json_bytes({"source/example.py": sha(body)})
    release = tmp_path / sha(manifest)
    (release / "source").mkdir(parents=True)
    (release / "source-manifest.json").write_bytes(manifest)
    (release / "source/example.py").write_bytes(body)
    verify_release(release)
    (release / "source/example.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source"):
        verify_release(release)


def test_pending_fresh_references_rejected_before_model_or_baseline(
    tmp_path, monkeypatch
):
    from finsight.sentiment.experiments import issuer_alias_review as runner

    source = [{"record_id": "x", "headline": "Acme", "partition": "validation"}]
    pending = [
        {
            **source[0],
            "exhaustive": True,
            "entities": [],
            "review": {"status": "pending", "reviewer": None},
        }
    ]
    path = tmp_path / "draft.jsonl"
    path.write_text(json.dumps(pending[0]) + "\n")
    monkeypatch.setattr(runner, "read_packet", lambda *args: ({}, source))

    def forbidden(*args):
        pytest.fail("Pending references must fail before baseline loading")

    monkeypatch.setattr(runner, "verify_release", forbidden)
    with pytest.raises(ValueError, match="review"):
        runner.run(tmp_path, tmp_path, tmp_path, annotations=path)
