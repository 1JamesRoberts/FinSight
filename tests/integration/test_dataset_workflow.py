"""Exercise the acquisition/validation workflow with synthetic offline artifacts."""

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from finsight.data.acquisition import acquire_datasets
from finsight.data.audit import audit_datasets


def test_offline_workflow_with_verified_files(tmp_path: Path) -> None:
    data = tmp_path / "data"
    phrasebank_dir = data / "raw/financial_phrasebank"
    sentfin_dir = data / "raw/sentfin"
    phrasebank_dir.mkdir(parents=True)
    sentfin_dir.mkdir(parents=True)
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
                    "{'Example': 'positive', 'Other': 'negative'}",
                    "2",
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
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"assets": assets, "local_dataset_assets": list(assets)})
    )
    receipt = acquire_datasets(manifest_path, data)
    assert len(receipt) == 3
    summary = audit_datasets(manifest_path, data)
    assert summary["phrasebank"]["50"]["normalized_duplicate_rows"] == 1
    assert summary["phrasebank_conflicting_label_texts"] == 1
    assert summary["sentfin"]["entity_annotations"] == 3
    assert summary["sentfin"]["conflicting_sentiment_headlines"] == 1
    assert summary["cross_corpus_normalized_exact_text_matches"] == 1
    assert summary["final_test_frozen"] is False
