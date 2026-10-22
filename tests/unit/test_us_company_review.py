"""Source sampling is prediction-free; pending identity labels cannot be scored."""

from copy import deepcopy

import pytest

from finsight.data.us_company_review import select_sample, validate_review
from finsight.sentiment.evaluation.company_linking import linking_metrics


def source_and_review():
    source = [
        {"record_id": "x", "headline": "Acme met Agency", "partition": "validation"}
    ]
    review = [
        {
            **source[0],
            "exhaustive": True,
            "review": {"status": "approved", "reviewer": "Reviewer"},
            "entities": [
                {
                    "start": 0,
                    "end": 4,
                    "text": "Acme",
                    "type": "company",
                    "uncertain": False,
                    "link_status": "linked",
                    "cik": "0000000001",
                    "identity_evidence": "source URL",
                },
                {
                    "start": 9,
                    "end": 15,
                    "text": "Agency",
                    "type": "other_organization",
                    "uncertain": False,
                    "link_status": "not_applicable",
                    "cik": None,
                    "identity_evidence": "institution title",
                },
            ],
        }
    ]
    return source, review


def test_sampling_is_order_invariant_deduplicated_and_ignores_predictions():
    a = [
        {"pageid": i, "title": f"Headline {i}", "ns": 0, "prediction": "anything"}
        for i in range(1, 8)
    ]
    b = [{"pageid": i, "title": f"Headline {i}", "ns": 0} for i in range(7, 14)]
    x = select_sample({"A": a, "B": b}, per_category=3, seed=42)
    assert x == select_sample({"B": b[::-1], "A": a[::-1]}, per_category=3, seed=42)
    assert len({r["record_id"] for r in x}) == 6
    assert all("prediction" not in r for r in x)
    assert select_sample({"A": a}, per_category=3, seed=43) != x[:3]


def test_quota_failure_does_not_silently_change_sample():
    with pytest.raises(ValueError, match="insufficient"):
        select_sample(
            {"A": [{"pageid": 1, "title": "Briefs: today", "ns": 0}]},
            per_category=1,
            seed=42,
        )


@pytest.mark.parametrize(
    "change", ["pending", "uncertain", "missing_evidence", "bad_cik", "wrong_span"]
)
def test_review_gate_rejects_unreviewed_or_malformed_labels(change):
    source, reviewed = source_and_review()
    row, entity = reviewed[0], reviewed[0]["entities"][0]
    if change == "pending":
        row["review"]["status"] = "pending"
    if change == "uncertain":
        entity["link_status"], entity["cik"] = "uncertain", None
    if change == "missing_evidence":
        entity["identity_evidence"] = ""
    if change == "bad_cik":
        entity["cik"] = "TSLA"
    if change == "wrong_span":
        entity["end"] = 3
    with pytest.raises(ValueError):
        validate_review(source, reviewed)


def test_wrong_issuer_counts_as_extra_and_missed_link_and_abstention_remains_miss():
    source, reviewed = source_and_review()
    prediction = {"x": [{"start": 0, "end": 4, "text": "Acme", "cik": "0000000002"}]}
    metrics = linking_metrics(source, reviewed, prediction)
    assert metrics["company_span_extraction"]["f1"] == 1
    assert metrics["issuer_links"]["false_positives"] == 1
    assert metrics["issuer_links"]["false_negatives"] == 1
    assert metrics["issuer_links"]["true_positives"] == 0
    prediction["x"][0]["cik"] = None
    metrics = linking_metrics(source, reviewed, prediction)
    assert metrics["issuer_links"]["precision"] is None
    assert metrics["issuer_links"]["recall"] == 0


def test_exact_link_and_no_company_rows_are_counted():
    source, reviewed = source_and_review()
    blank = {
        "record_id": "blank",
        "headline": "Markets opened higher",
        "partition": "validation",
    }
    source.append(blank)
    reviewed.append(
        {
            **blank,
            "entities": [],
            "exhaustive": True,
            "review": {"status": "approved", "reviewer": "Reviewer"},
        }
    )
    predictions = {
        "x": [{"start": 0, "end": 4, "text": "Acme", "cik": "0000000001"}],
        "blank": [],
    }
    assert linking_metrics(source, reviewed, predictions)["issuer_links"]["f1"] == 1
    bad = deepcopy(predictions)
    bad["x"].append(bad["x"][0])
    with pytest.raises(ValueError, match="duplicate"):
        linking_metrics(source, reviewed, bad)


def test_packet_tamper_and_review_gate_precede_model_loading(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from finsight.data.target_preparation import json_bytes
    from finsight.sentiment.artifacts import file_sha256, sha
    from finsight.sentiment.experiments.us_company_review import read_packet, run

    root = Path(__file__).resolve().parents[2]
    sources, reviewed = source_and_review()
    body = "".join(json.dumps(row) + "\n" for row in sources).encode()
    manifest = json_bytes(
        {
            "sample_sha256": sha(body),
            "plan_sha256": file_sha256(root / "configs/us-company-review-plan.json"),
            "selection_source_sha256": file_sha256(
                root / "src/finsight/data/us_company_review.py"
            ),
        }
    )
    packet = tmp_path / sha(manifest)
    packet.mkdir()
    (packet / "manifest.json").write_bytes(manifest)
    (packet / "sample.jsonl").write_bytes(body)
    assert read_packet(packet, root) == sources
    reviewed[0]["review"]["status"] = "pending"
    annotations = tmp_path / "draft.jsonl"
    annotations.write_text("".join(json.dumps(row) + "\n" for row in reviewed))

    def forbidden(*args, **kwargs):
        pytest.fail("Directory/model must not load before review gate")

    monkeypatch.setattr(
        "finsight.sentiment.experiments.us_company_review.load_directory", forbidden
    )
    with pytest.raises(ValueError, match="review"):
        run(root, tmp_path, packet, annotations=annotations)
    (packet / "sample.jsonl").write_bytes(body + b" ")
    with pytest.raises(ValueError, match="differs"):
        read_packet(packet, root)
