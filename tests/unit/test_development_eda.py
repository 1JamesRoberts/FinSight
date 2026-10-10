"""Check EDA counting units and refusal to explore held-out records."""

import json
from pathlib import Path

import pytest

from finsight.data.eda import FIGURE_NAMES, plot_development, summarize_development


@pytest.fixture
def development_records() -> list[dict]:
    return [
        {
            "corpus": "phrasebank",
            "partition": "train",
            "native_record": {"sentence": "Private phrase canary", "label": "positive"},
        },
        {
            "corpus": "sentfin",
            "partition": "validation",
            "native_record": {
                "title": "Private headline canary",
                "sentiments": [
                    {"target": "First", "label": "positive"},
                    {"target": "Second", "label": "negative"},
                ],
            },
        },
        {
            "corpus": "fiqa",
            "partition": "train",
            "native_record": {
                "sentence": "Private post canary",
                "source_type": "post",
                "annotations": [{"score": -1.0}, {"score": 0.5}],
            },
            "annotation_eligibility": [
                {"review_required": True},
                {"review_required": False},
            ],
        },
    ]


def test_counting_units_scores_and_privacy(development_records: list[dict]) -> None:
    summary = summarize_development(development_records)
    sentfin = summary["sentfin"]
    assert sentfin["records"] == 1
    assert sentfin["labels"]["total"] == 2
    assert sentfin["lengths"]["records"] == 1
    assert sentfin["target_multiplicity"] == {"1": 0, "2": 1, "3+": 0}
    assert sentfin["multi_target_sentiments"] == {"uniform": 0, "mixed": 1}
    assert summary["phrasebank"]["labels"]["counts"]["neutral"] == 0
    assert summary["fiqa"]["scores"]["total"] == 2
    assert summary["fiqa"]["scores"]["histogram"]["[0.5, 1]"] == 1
    assert summary["fiqa"]["flagged_snippet_annotations"] == 1
    assert "Private" not in json.dumps(summary)
    assert "First" not in json.dumps(summary)


@pytest.mark.parametrize("partition", ["test", "test_candidate", None])
def test_heldout_input_is_rejected(
    development_records: list[dict], partition: str | None
) -> None:
    development_records[0]["partition"] = partition
    with pytest.raises(ValueError, match="only train/validation"):
        summarize_development(development_records)


def test_unscored_inputs_are_rejected(development_records: list[dict]) -> None:
    development_records[2]["native_record"]["annotations"][0]["score"] = None
    with pytest.raises(ValueError, match="unscored"):
        summarize_development(development_records)


def test_empty_groups_are_explicit() -> None:
    summary = summarize_development([])
    assert all(group["records"] == 0 for group in summary.values())
    assert summary["fiqa"]["scores"] is None


def test_plots_export_all_five_figures(
    tmp_path: Path, development_records: list[dict]
) -> None:
    pytest.importorskip("matplotlib")
    from PIL import Image

    summary = {
        "corpora": summarize_development(development_records),
        "phrasebank_agreement_membership": {"50": 1, "66": 1, "75": 1, "all": 1},
    }
    plot_development(summary, tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(FIGURE_NAMES)
    for name in FIGURE_NAMES:
        with Image.open(tmp_path / name) as figure:
            assert figure.format == "PNG"
            assert figure.width >= 1400 and figure.height >= 700
