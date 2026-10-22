"""Headline counts without treating unannotated matches as errors."""

from pathlib import Path

import pytest

from finsight.sentiment.companies import load_registry
from finsight.sentiment.experiments.company_mentions import audit_partition

ROOT = Path(__file__).resolve().parents[2]


def row(identifier, headline, target, partition="train"):
    return dict(
        record_id=identifier,
        headline=headline,
        target=target,
        label="neutral",
        partition=partition,
    )


def test_counts_once_per_headline_and_preserves_ambiguity():
    registry = load_registry(ROOT / "configs/companies.json")
    rows = [
        row("1", "Apple and Microsoft meet Microsoft", "Apple"),
        row("1", "Apple and Microsoft meet Microsoft", "Microsoft"),
        row("2", "Microsoft signs a deal", "Other"),
        row("3", "The device maker reports results", "Apple"),
        row("4", "Google reports results", "Google"),
    ]
    summary, packet = audit_partition(rows, registry, "train")
    assert summary["totals"]["headlines"] == 4
    assert summary["totals"]["target_annotations"] == 5
    assert summary["totals"]["resolved_spans"] == 3
    assert summary["totals"]["resolved_headlines"] == 2
    assert summary["companies"]["apple"]["native_target_ambiguous_only"] == 1
    assert summary["companies"]["apple"]["native_target_no_candidate"] == 1
    assert summary["companies"]["microsoft"]["resolved_without_native_target"] == 1
    assert summary["ambiguous_span_reasons"]["parent_rollup_required"] == 1
    assert len(packet) == 4
    assert packet[0]["headline"] == rows[0]["headline"]
    assert len(packet[0]["annotations"]) == 2


@pytest.mark.parametrize("partition", ["test", "test_candidate"])
def test_reject_reserved_partition(partition):
    registry = load_registry(ROOT / "configs/companies.json")
    with pytest.raises(ValueError, match="development partitions"):
        audit_partition([], registry, partition)
    with pytest.raises(ValueError, match="unexpected partition"):
        audit_partition([row("1", "Apple", "Apple", partition)], registry, "train")


def test_reject_inconsistent_headline_identity():
    registry = load_registry(ROOT / "configs/companies.json")
    with pytest.raises(ValueError, match="inconsistent text"):
        audit_partition(
            [row("1", "Apple", "Apple"), row("1", "Microsoft", "Microsoft")],
            registry,
            "train",
        )
