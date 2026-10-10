"""Basic integrity checks; no split design or claim of complete contamination audit."""

import json
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from finsight.data.acquisition import artifact_path, verify_artifact
from finsight.data.loaders import load_entity_aliases, load_phrasebank, load_sentfin


def text_identity(text: str) -> str:
    """Normalize Unicode, case, and whitespace only; preserve punctuation/numbers."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def duplicate_count(texts: Iterable[str]) -> int:
    """Count repeated rows beyond the first under the documented normalization."""
    keys = [text_identity(text) for text in texts]
    return len(keys) - len(set(keys))


def audit_datasets(manifest_path: Path, data_dir: Path) -> dict[str, Any]:
    """Verify local bytes and schemas, and summarize counts without source text."""
    manifest = json.loads(manifest_path.read_text())
    assets = manifest["assets"]
    for name in manifest["local_dataset_assets"]:
        for specification in assets[name]["acquisition_files"]:
            verify_artifact(artifact_path(data_dir, specification), specification)
    phrasebank = assets["financial_phrasebank"]
    sentfin = assets["sentfin_1_0"]
    archive = artifact_path(data_dir, phrasebank["acquisition_files"][0])
    variants = {
        agreement: load_phrasebank(
            archive, revision=phrasebank["revision"], agreement=agreement
        )
        for agreement in ("50", "66", "75", "all")
    }
    headlines = load_sentfin(
        data_dir / "raw/sentfin/SEntFiN.csv", revision=sentfin["revision"]
    )
    aliases = load_entity_aliases(
        data_dir / "raw/sentfin/entity_list_comprehensive.csv",
        revision=sentfin["revision"],
    )
    primary = variants["50"]
    primary_keys = {text_identity(record.sentence) for record in primary}
    headline_keys = {text_identity(record.title) for record in headlines}
    labels_per_text: dict[str, set[str]] = {}
    for record in primary:
        labels_per_text.setdefault(text_identity(record.sentence), set()).add(
            record.label
        )
    return {
        "revisions": {
            name: assets[name]["revision"] for name in manifest["local_dataset_assets"]
        },
        "phrasebank": {
            agreement: {
                "records": len(records),
                "labels": dict(
                    sorted(Counter(record.label for record in records).items())
                ),
                "normalized_duplicate_rows": duplicate_count(
                    record.sentence for record in records
                ),
                "normalized_texts_outside_50agree": len(
                    {text_identity(r.sentence) for r in records} - primary_keys
                ),
            }
            for agreement, records in variants.items()
        },
        "phrasebank_conflicting_label_texts": sum(
            len(labels) > 1 for labels in labels_per_text.values()
        ),
        "sentfin": {
            "headlines": len(headlines),
            "entity_annotations": sum(len(record.sentiments) for record in headlines),
            "labels": dict(
                sorted(
                    Counter(
                        sentiment.label
                        for record in headlines
                        for sentiment in record.sentiments
                    ).items()
                )
            ),
            "multi_entity_headlines": sum(
                len(record.sentiments) > 1 for record in headlines
            ),
            "conflicting_sentiment_headlines": sum(
                len({s.label for s in record.sentiments}) > 1 for record in headlines
            ),
            "normalized_duplicate_rows": duplicate_count(
                record.title for record in headlines
            ),
            "entity_database_rows": len(aliases),
        },
        "cross_corpus_normalized_exact_text_matches": len(primary_keys & headline_keys),
        "overlap_scope": (
            "NFKC + casefold + whitespace exact matches only; "
            "no fuzzy matching or pretraining corpus audit"
        ),
        "splits_assigned": False,
        "final_test_frozen": False,
    }
