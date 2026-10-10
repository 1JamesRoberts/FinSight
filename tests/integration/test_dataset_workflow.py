"""Exercise the acquisition/validation workflow with synthetic offline artifacts."""

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from finsight.data.__main__ import main
from finsight.data.acquisition import acquire_datasets
from finsight.data.audit import audit_datasets
from finsight.data.labels import analyze_labels, render_label_report
from finsight.data.lengths import analyze_lengths
from finsight.data.review import review_datasets
from finsight.data.splits import prepare_splits

SPLIT_POLICY = Path(__file__).resolve().parents[2] / "configs/split-policy.json"


def test_offline_workflow_with_verified_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    phrasebank_dir = data / "raw/financial_phrasebank"
    sentfin_dir = data / "raw/sentfin"
    phrasebank_dir.mkdir(parents=True)
    sentfin_dir.mkdir(parents=True)
    fiqa_dir = data / "raw/fiqa"
    fiqa_dir.mkdir(parents=True)
    fiqa_paths = []
    for kind in ("headline", "post"):
        for split in ("train", "test"):
            annotation = {"target": "Example", "snippets": "[nan, 'improves']"}
            if split == "train":
                annotation |= {"aspects": "['Stock/Price']", "sentiment_score": "0.25"}
            path = fiqa_dir / f"task1_{kind}_ABSA_{split}.json"
            path.write_text(
                json.dumps(
                    {"1": {"sentence": "Example improves", "info": [annotation]}}
                )
            )
            fiqa_paths.append(path)
    archive_path = phrasebank_dir / "FinancialPhraseBank-v1.0.zip"
    with ZipFile(archive_path, "w") as archive:
        for agreement in ("50", "66", "75", "All"):
            archive.writestr(
                f"FinancialPhraseBank-v1.0/Sentences_{agreement}Agree.txt",
                "Example improves@positive\nExample improves@neutral\n",
            )
    for filename, columns, rows in [
        (
            "SEntFiN.csv",
            ["S No.", "Title", "Decisions", "Words"],
            [
                [
                    "0",
                    "EXAMPLE improves",
                    "{'Example': 'positive', 'Other': 'negative', 'Third': 'positive'}",
                    "3",
                ],
                ["1", "Another headline", "{'Example': 'neutral'}", "2"],
            ],
        ),
        (
            "entity_list_comprehensive.csv",
            ["", "Symbol", "Entity"],
            [
                ["0", "EX", "Example"],
            ],
        ),
    ]:
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)
        (sentfin_dir / filename).write_text(output.getvalue())
    assets = {}
    for name, paths in {
        "financial_phrasebank": [archive_path],
        "sentfin_1_0": [
            sentfin_dir / "SEntFiN.csv",
            sentfin_dir / "entity_list_comprehensive.csv",
        ],
        "fiqa_task1_original": fiqa_paths,
    }.items():
        assets[name] = {
            "revision": "synthetic-revision",
            "local_research_decision": "supported_under_repository_license",
            "acquisition_files": [
                {
                    "local_path": str(path.relative_to(data)),
                    "source_url": "https://example.invalid/not-contacted",
                    "size_bytes": len(path.read_bytes()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in paths
            ],
        }
    assets["fiqa_task1_original"]["local_research_decision"] = (
        "supported_under_original_noncommercial_terms"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"assets": assets, "local_dataset_assets": list(assets)})
    )
    receipt = acquire_datasets(manifest_path, data)
    assert len(receipt) == 7
    summary = audit_datasets(manifest_path, data)
    assert summary["phrasebank"]["50"]["normalized_duplicate_rows"] == 1
    assert summary["phrasebank_conflicting_label_texts"] == 1
    assert summary["sentfin"]["entity_annotations"] == 4
    assert summary["sentfin"]["conflicting_sentiment_headlines"] == 1
    assert summary["cross_corpus_normalized_exact_text_matches"] == 1
    assert summary["final_test_frozen"] is False
    assert summary["fiqa"]["headline_train"]["scored_annotations"] == 1
    assert summary["fiqa"]["post_test"]["scored_annotations"] == 0
    assert summary["fiqa"]["headline_train"]["missing_snippet_values"] == 1
    assert summary["fiqa_overlap"]["original_train_test_exact_matches"] == 1
    assert summary["fiqa_overlap"]["phrasebank_exact_matches"] == 1
    labels = analyze_labels(manifest_path, data)
    assert labels["phrasebank"]["50"]["percentages"]["neutral"] == 50
    assert labels["sentfin"]["annotation_labels"]["total"] == 4
    assert labels["sentfin"]["mixed_share_of_multi_target_headlines_percent"] == 100
    assert labels["fiqa_training"]["combined"]["scores"]["total"] == 2
    assert labels["fiqa_training"]["combined"]["scores"]["mean"] == 0.25
    assert "fiqa_test" not in labels
    assert labels["class_thresholds_assigned"] is False
    assert labels["new_splits_assigned"] is False
    markdown = render_label_report(labels)
    assert "50.00%" in markdown
    assert "EXAMPLE improves" not in markdown
    assert "synthetic-revision" in markdown
    lengths = analyze_lengths(manifest_path, data)
    assert lengths["phrasebank"]["50"]["overall"]["records"] == 2
    assert lengths["phrasebank"]["50"]["by_label"]["negative"]["records"] == 0
    assert lengths["sentfin"]["overall"]["records"] == 2
    assert lengths["sentfin"]["source_word_count_mismatches"] == 1
    assert lengths["sentfin"]["by_target_label"]["positive"]["records"] == 1
    assert lengths["sentfin"]["by_target_label"]["negative"]["records"] == 1
    assert lengths["fiqa_training"]["combined"]["records"] == 2
    assert lengths["filtering_applied"] is False
    monkeypatch.chdir(tmp_path)
    label_path = tmp_path / "reports/label-analysis.md"
    label_path.parent.mkdir()
    label_path.write_text("Existing label report")
    monkeypatch.setattr(
        "sys.argv",
        [
            "finsight.data",
            "analyze-lengths",
            "--manifest",
            str(manifest_path),
            "--data-dir",
            str(data),
        ],
    )
    main()
    assert label_path.read_text() == "Existing label report"
    policy_path = tmp_path / "split-policy.json"
    policy_path.write_text(SPLIT_POLICY.read_text())
    monkeypatch.setattr(
        "sys.argv",
        [
            "finsight.data",
            "prepare-splits",
            "--manifest",
            str(manifest_path),
            "--data-dir",
            str(data),
            "--split-policy",
            str(policy_path),
        ],
    )
    main()
    splits = json.loads((data / "interim/split-preparation.json").read_text())
    assert splits["corpora"]["phrasebank"]["prepared_records"] == 0
    assert splits["corpora"]["fiqa"]["source_statuses"] == {
        "quarantined": 2,
        "unscored": 2,
    }
    assert splits["corpora"]["sentfin"]["prepared_records"] == 1
    directory = Path(splits["directory"])
    original_bytes = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert prepare_splits(manifest_path, policy_path, data) == splits
    assert original_bytes == {
        path.name: path.read_bytes() for path in directory.iterdir()
    }
    split_report = (tmp_path / "reports/split-preparation.md").read_text()
    assert "EXAMPLE improves" not in split_report
    assert "No final test is frozen" in split_report
    changed_policy = json.loads(policy_path.read_text()) | {"seed": 7}
    policy_path.write_text(json.dumps(changed_policy))
    changed_splits = prepare_splits(manifest_path, policy_path, data)
    assert changed_splits["directory"] != splits["directory"]
    freeze_path = tmp_path / "synthetic-freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "parent_split_manifest_sha256": splits["split_manifest_sha256"],
            }
        )
    )
    with pytest.raises(ValueError, match="resplitting prohibited"):
        prepare_splits(manifest_path, policy_path, data, freeze_path=freeze_path)
    assert original_bytes == {
        path.name: path.read_bytes() for path in directory.iterdir()
    }
    policy_path.write_text(SPLIT_POLICY.read_text())
    (directory / "prepared-records.jsonl").write_text("corrupt cache")
    with pytest.raises(ValueError, match="immutable split artifact differs"):
        prepare_splits(manifest_path, policy_path, data)
    (directory / "prepared-records.jsonl").write_bytes(
        original_bytes["prepared-records.jsonl"]
    )
    # Equivalent regression zeros must not become apparent gold conflicts.
    for specification in assets["fiqa_task1_original"]["acquisition_files"]:
        path = data / specification["local_path"]
        if "train" not in path.name:
            continue
        body = json.loads(path.read_text())
        body["1"]["info"][0]["sentiment_score"] = (
            "-0.0" if "headline" in path.name else "0.0"
        )
        path.write_text(json.dumps(body))
        specification["size_bytes"] = len(path.read_bytes())
        specification["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps({"assets": assets, "local_dataset_assets": list(assets)})
    )
    zero_review = review_datasets(manifest_path, data)
    assert (
        zero_review["summary"]["corpora"]["fiqa"][
            "groups_with_conflicting_gold_contexts"
        ]
        == 0
    )
    length_report = (tmp_path / "reports/length-analysis.md").read_text()
    assert "EXAMPLE improves" not in length_report
    assert "synthetic-revision" in length_report
    assert json.loads((data / "interim/length-analysis.json").read_text()) == lengths
    review = review_datasets(manifest_path, data)
    assert review["summary"]["corpora"]["phrasebank"]["exact_duplicate_groups"] == 1
    assert (
        review["summary"]["corpora"]["phrasebank"][
            "groups_with_conflicting_gold_contexts"
        ]
        == 1
    )
    assert (
        review["summary"]["corpora"]["fiqa"]["groups_with_conflicting_gold_contexts"]
        == 0
    )
    assert review["summary"]["sentfin_mixed_target_label_headlines"] == 1
    assert review["summary"]["records_removed_or_relabeled"] is False
    monkeypatch.setattr(
        "sys.argv",
        [
            "finsight.data",
            "review-annotations",
            "--manifest",
            str(manifest_path),
            "--data-dir",
            str(data),
        ],
    )
    main()
    persisted_review = json.loads((data / "interim/annotation-review.json").read_text())
    assert persisted_review == json.loads(json.dumps(review))
    public_report = (tmp_path / "reports/annotation-review.md").read_text()
    assert "EXAMPLE improves" not in public_report
    assert label_path.read_text() == "Existing label report"
