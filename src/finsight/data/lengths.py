"""Descriptive source text lengths without model tokenization or filtering."""

import hashlib
import json
from collections.abc import Iterable
from math import floor
from pathlib import Path
from statistics import fmean, median
from typing import Any

from finsight.data.acquisition import artifact_path, verify_artifact
from finsight.data.fiqa import load_fiqa
from finsight.data.loaders import SentimentLabel, load_phrasebank, load_sentfin


def _percentile(values: list[int], fraction: float) -> float:
    """Inclusive linear interpolation on sorted observations."""
    position = (len(values) - 1) * fraction
    lower = floor(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _statistics(values: list[int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return dict.fromkeys(("min", "mean", "median", "p90", "p95", "p99", "max"))
    return {
        "min": ordered[0],
        "mean": fmean(ordered),
        "median": median(ordered),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def length_summary(texts: Iterable[str]) -> dict[str, Any]:
    """Count Unicode code points and whitespace-separated words in original text.

    Flags (<=3, >50, >100 words) are descriptive counts, not filtering rules
    or model context limits. Blank texts are rejected; empty groups are explicit.
    """
    rows = list(texts)
    if any(not isinstance(text, str) or not text.strip() for text in rows):
        raise ValueError("text records must be nonempty strings")
    words = [len(text.split()) for text in rows]
    return {
        "records": len(rows),
        "characters": _statistics([len(text) for text in rows]),
        "words": _statistics(words),
        "word_histogram": {
            "1-5": sum(1 <= count <= 5 for count in words),
            "6-10": sum(6 <= count <= 10 for count in words),
            "11-20": sum(11 <= count <= 20 for count in words),
            "21-50": sum(21 <= count <= 50 for count in words),
            "51-100": sum(51 <= count <= 100 for count in words),
            ">100": sum(count > 100 for count in words),
        },
        "word_flags": {
            "at_most_3": sum(count <= 3 for count in words),
            "over_50": sum(count > 50 for count in words),
            "over_100": sum(count > 100 for count in words),
        },
    }


def analyze_lengths(manifest_path: Path, data_dir: Path) -> dict[str, Any]:
    """Verify artifacts and summarize text groups without inspecting FiQA test text."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assets = manifest["assets"]
    for name in manifest["local_dataset_assets"]:
        for specification in assets[name]["acquisition_files"]:
            verify_artifact(artifact_path(data_dir, specification), specification)
    phrasebank = assets["financial_phrasebank"]
    variants = {}
    for agreement in ("50", "66", "75", "all"):
        rows = load_phrasebank(
            artifact_path(data_dir, phrasebank["acquisition_files"][0]),
            revision=phrasebank["revision"],
            agreement=agreement,
        )
        variants[agreement] = {
            "overall": length_summary(row.sentence for row in rows),
            "by_label": {
                label.value: length_summary(
                    row.sentence for row in rows if row.label == label
                )
                for label in SentimentLabel
            },
        }
    headlines = load_sentfin(
        data_dir / "raw/sentfin/SEntFiN.csv", revision=assets["sentfin_1_0"]["revision"]
    )
    report: dict[str, Any] = {
        "analysis_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "revisions": {
            name: assets[name]["revision"] for name in manifest["local_dataset_assets"]
        },
        "counting": {
            "characters": (
                "len(original_text): Unicode code points, including whitespace"
            ),
            "words": "len(original_text.split()): whitespace-separated units",
            "percentiles": "inclusive linear interpolation at (n-1)*p",
            "sentfin_by_label": "each source headline once per distinct target label",
            "fiqa": "training source text records once each, not target annotations",
        },
        "phrasebank": variants,
        "sentfin": {
            "overall": length_summary(row.title for row in headlines),
            "source_word_count_mismatches": sum(
                row.word_count != len(row.title.split()) for row in headlines
            ),
            "by_target_label": {
                label.value: length_summary(
                    row.title
                    for row in headlines
                    if label in {annotation.label for annotation in row.sentiments}
                )
                for label in SentimentLabel
            },
        },
        "filtering_applied": False,
        "model_tokenization_applied": False,
        "new_splits_assigned": False,
    }
    if "fiqa_task1_original" in manifest["local_dataset_assets"]:
        fiqa = assets["fiqa_task1_original"]
        groups = {
            kind: load_fiqa(
                data_dir / f"raw/fiqa/task1_{kind}_ABSA_train.json",
                revision=fiqa["revision"],
                source_type=kind,
                source_split="train",
            )
            for kind in ("headline", "post")
        }
        report["fiqa_training"] = {
            **{
                kind: length_summary(row.sentence for row in rows)
                for kind, rows in groups.items()
            },
            "combined": length_summary(
                row.sentence for rows in groups.values() for row in rows
            ),
        }
    return report


def _table(groups: dict[str, Any]) -> list[str]:
    lines = [
        "| Group | Texts | Words median | Words P95 | Words max | "
        "Chars median | Chars P95 | Chars max | <=3 words | >50 words |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in groups.items():
        cells = [
            "—" if group[unit][stat] is None else f"{group[unit][stat]:.1f}"
            for unit in ("words", "characters")
            for stat in ("median", "p95", "max")
        ]
        lines.append(
            f"| {name} | {group['records']:,} | "
            + " | ".join(cells)
            + f" | {group['word_flags']['at_most_3']:,} "
            + f"| {group['word_flags']['over_50']:,} |"
        )
    return lines


def render_length_report(report: dict[str, Any]) -> str:
    """Render aggregate Markdown without source text or record identifiers."""
    lines = [
        "# FinSight text-length analysis",
        "",
        "Phase 1 descriptive EDA. Character counts measure Unicode code points in",
        "original text, including whitespace; word counts use whitespace splitting.",
        "Words are not FinBERT subword tokens. No cleaning, filtering, tokenization,",
        "FiQA class thresholds, or new partitions are applied. Duplicate source rows",
        "remain counted. FiQA uses original training texts only.",
        "",
        "P95 is the inclusive interpolated 95th percentile; fractional counts arise",
        "from interpolation. Median/P95/max are shown here; JSON also includes mean,",
        "minimum, P90/P99, a word histogram, and >100-word counts. Flags <=3, >50,",
        "and >100 words are fixed descriptive markers, not outlier rejection rules.",
        "",
        "## PhraseBank agreement subsets",
        "",
    ]
    lines += _table(
        {name: item["overall"] for name, item in report["phrasebank"].items()}
    )
    lines += [
        "",
        "Subsets overlap. `all` means unanimous annotator agreement.",
        "",
        "## PhraseBank by source sentiment label",
        "",
    ]
    lines += _table(
        {
            f"{name}/{label}": group
            for name, item in report["phrasebank"].items()
            for label, group in item["by_label"].items()
        }
    )
    lines += [
        "",
        "## SEntFiN headline lengths",
        "",
        "Overall counts each headline record once. Label groups count a headline",
        "once if any target has that label; multiple targets with the same label",
        "do not multiply its weight. Mixed-label headlines appear in several",
        "label groups, so group counts do not sum to the overall count.",
        "",
    ]
    lines += _table(
        {
            "overall": report["sentfin"]["overall"],
            **report["sentfin"]["by_target_label"],
        }
    )
    lines += [
        "",
        f"The supplied `Words` field differs from whitespace counting for "
        f"{report['sentfin']['source_word_count_mismatches']:,} headlines.",
        "This report consistently uses its stated counting rule. Differences may",
        "reflect source counting conventions; neither source field nor text "
        "is changed.",
    ]
    if "fiqa_training" in report:
        lines += [
            "",
            "## FiQA training text lengths by source type",
            "",
            "Each original source text record contributes once, regardless of its",
            "annotation count. Continuous scores are not converted to classes.",
            "",
        ]
        lines += _table(report["fiqa_training"])
    lines += [
        "",
        "## Reproduction and limitations",
        "",
        "Run `uv run --locked python -m finsight.data analyze-lengths`.",
        "JSON output: `data/interim/length-analysis.json` (ignored).",
        "All manifest artifact bytes are verified before reading source records.",
        "This report contains only aggregates; no source text is committed.",
        "",
        f"Manifest SHA-256: `{report['manifest_sha256']}`.",
        "",
    ]
    lines += [
        f"- {name}: `{revision}`" for name, revision in report["revisions"].items()
    ]
    lines += [
        "",
        "These distributions describe the selected copies, not all financial news.",
        "They do not establish a transformer context limit or measure truncation.",
        "Model-specific token length analysis belongs to later model work. No",
        "conclusions about sentiment quality or causal effects follow from length",
        "differences alone. Duplicate/ambiguity review and split design remain open.",
    ]
    return "\n".join(lines) + "\n"
