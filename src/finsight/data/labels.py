"""Descriptive label/score analysis; no class thresholds or split decisions."""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean, median, pstdev, quantiles
from typing import Any

from finsight.data.acquisition import artifact_path, verify_artifact
from finsight.data.fiqa import load_fiqa
from finsight.data.loaders import SentimentLabel, load_phrasebank, load_sentfin


def label_distribution(labels: Iterable[SentimentLabel]) -> dict[str, Any]:
    """Report all three source classes, including absent classes and empty inputs."""
    counts = Counter(labels)
    if set(counts) - set(SentimentLabel):
        raise ValueError("unknown sentiment label")
    total = sum(counts.values())
    class_counts = {label.value: counts[label] for label in SentimentLabel}
    smallest = min(class_counts.values())
    largest = max(class_counts.values())
    return {
        "total": total,
        "counts": class_counts,
        "percentages": {
            label: 100 * count / total if total else 0.0
            for label, count in class_counts.items()
        },
        "majority_share_percent": 100 * largest / total if total else None,
        "largest_to_smallest_class_ratio": largest / smallest if smallest else None,
    }


def score_distribution(scores: Iterable[float]) -> dict[str, Any]:
    """Describe scores with inclusive quartiles and fixed bins, not class labels.

    Histogram bins: [-1, -0.5), [-0.5, 0), [0, 0.5), [0.5, 1].
    Sign counts describe numeric signs; zero does not define a neutral class.
    """
    values = list(scores)
    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in values):
        raise ValueError("scores must be finite and within [-1, 1]")
    if not values:
        raise ValueError("at least one score is required")
    quartiles = (
        quantiles(values, n=4, method="inclusive")
        if len(values) > 1
        else [values[0]] * 3
    )
    signs = {
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive": sum(value > 0 for value in values),
    }
    return {
        "total": len(values),
        "minimum": min(values),
        "q25": quartiles[0],
        "median": median(values),
        "q75": quartiles[2],
        "maximum": max(values),
        "mean": fmean(values),
        "population_stddev": pstdev(values),
        "numeric_sign_counts": signs,
        "numeric_sign_percentages": {
            sign: 100 * count / len(values) for sign, count in signs.items()
        },
        "histogram": {
            "[-1, -0.5)": sum(-1 <= value < -0.5 for value in values),
            "[-0.5, 0)": sum(-0.5 <= value < 0 for value in values),
            "[0, 0.5)": sum(0 <= value < 0.5 for value in values),
            "[0.5, 1]": sum(0.5 <= value <= 1 for value in values),
        },
    }


def analyze_labels(manifest_path: Path, data_dir: Path) -> dict[str, Any]:
    """Verify pinned bytes; analyze candidate class corpora and FiQA train only."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assets = manifest["assets"]
    for name in manifest["local_dataset_assets"]:
        for specification in assets[name]["acquisition_files"]:
            verify_artifact(artifact_path(data_dir, specification), specification)
    phrasebank = assets["financial_phrasebank"]
    variants = {
        agreement: label_distribution(
            record.label
            for record in load_phrasebank(
                artifact_path(data_dir, phrasebank["acquisition_files"][0]),
                revision=phrasebank["revision"],
                agreement=agreement,
            )
        )
        for agreement in ("50", "66", "75", "all")
    }
    sentfin = load_sentfin(
        data_dir / "raw/sentfin/SEntFiN.csv", revision=assets["sentfin_1_0"]["revision"]
    )
    mixed = sum(len({s.label for s in row.sentiments}) > 1 for row in sentfin)
    multi = sum(len(row.sentiments) > 1 for row in sentfin)
    report: dict[str, Any] = {
        "analysis_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "revisions": {
            name: assets[name]["revision"] for name in manifest["local_dataset_assets"]
        },
        "scope": "Full candidate class corpora; FiQA original training scores only",
        "class_order": [label.value for label in SentimentLabel],
        "phrasebank": variants,
        "sentfin": {
            "annotation_labels": label_distribution(
                annotation.label for row in sentfin for annotation in row.sentiments
            ),
            "headlines": len(sentfin),
            "multi_target_headlines": multi,
            "mixed_label_headlines": mixed,
            "mixed_share_of_all_headlines_percent": 100 * mixed / len(sentfin),
            "mixed_share_of_multi_target_headlines_percent": (
                100 * mixed / multi if multi else None
            ),
        },
        "class_thresholds_assigned": False,
        "new_splits_assigned": False,
        "final_test_frozen": False,
    }
    if "fiqa_task1_original" in manifest["local_dataset_assets"]:
        fiqa = assets["fiqa_task1_original"]
        records_by_type = {
            kind: load_fiqa(
                data_dir / f"raw/fiqa/task1_{kind}_ABSA_train.json",
                revision=fiqa["revision"],
                source_type=kind,
                source_split="train",
            )
            for kind in ("headline", "post")
        }
        report["fiqa_training"] = {}
        for name, records in {
            **records_by_type,
            "combined": tuple(row for rows in records_by_type.values() for row in rows),
        }.items():
            report["fiqa_training"][name] = {
                "text_records": len(records),
                "scores": score_distribution(
                    annotation.score
                    for row in records
                    for annotation in row.annotations
                    if annotation.score is not None
                ),
            }
    return report


def render_label_report(report: dict[str, Any]) -> str:
    """Render reproducible aggregate Markdown without source text or IDs."""
    lines = [
        "# FinSight label analysis",
        "",
        "Phase 1 descriptive analysis of pinned local artifacts. Raw source rows",
        "are counted without deduplication. No thresholds or new splits are assigned.",
        "FiQA analysis uses original training annotations only; unscored test inputs",
        "are excluded. PhraseBank and SEntFiN are candidate corpora, not frozen tests.",
        "",
        "## Source class distributions",
        "",
        "PhraseBank counts sentences; SEntFiN counts target annotations. These",
        "represent different supervision and must not be pooled as one task.",
        "",
        "| Source | Total | Negative | Neutral | Positive | Largest/smallest |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    distributions = {
        **{f"PhraseBank {key}": value for key, value in report["phrasebank"].items()},
        "SEntFiN targets": report["sentfin"]["annotation_labels"],
    }
    for name, distribution in distributions.items():
        cells = [
            f"{distribution['counts'][label]:,} "
            f"({distribution['percentages'][label]:.2f}%)"
            for label in report["class_order"]
        ]
        ratio = distribution["largest_to_smallest_class_ratio"]
        ratio_text = f"{ratio:.2f}" if ratio is not None else "undefined"
        lines.append(
            f"| {name} | {distribution['total']:,} | "
            + " | ".join(cells)
            + f" | {ratio_text} |"
        )
    sf = report["sentfin"]
    lines += [
        "",
        "Class percentages use the stated row/annotation denominator. The ratio is",
        "largest class count divided by smallest; absent classes make it undefined.",
        "Majority shares describe this corpus, not measured held-out model accuracy.",
        "PhraseBank variants overlap and must not be concatenated or split apart as",
        "independent corpora. Higher agreement indicates annotator consensus, not",
        "verified truth; subset composition can change alongside agreement.",
        "",
        "## Target-specific label differences",
        "",
        f"SEntFiN contains {sf['headlines']:,} headlines and "
        f"{sf['multi_target_headlines']:,} with multiple targets.",
        f"{sf['mixed_label_headlines']:,} headlines have different labels "
        "across targets:",
        f"{sf['mixed_share_of_all_headlines_percent']:.2f}% of all headlines and "
        f"{sf['mixed_share_of_multi_target_headlines_percent']:.2f}% "
        "of multi-target headlines.",
        "These differences support target-aware evaluation; they do not by themselves",
        "indicate annotation disagreement or errors. Keep each headline group intact.",
    ]
    if "fiqa_training" in report:
        lines += [
            "",
            "## FiQA continuous scores (original training annotations)",
            "",
            "| Source | Texts | Scores | Mean | Stddev | Min | Q25 | "
            "Median | Q75 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, item in report["fiqa_training"].items():
            stats = item["scores"]
            cells = [
                f"{stats[field]:.4f}"
                for field in (
                    "mean",
                    "population_stddev",
                    "minimum",
                    "q25",
                    "median",
                    "q75",
                    "maximum",
                )
            ]
            lines.append(
                f"| {name} | {item['text_records']:,} | {stats['total']:,} | "
                + " | ".join(cells)
                + " |"
            )
        lines += [
            "",
            "| Source | Score < 0 | Score = 0 | Score > 0 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, item in report["fiqa_training"].items():
            stats = item["scores"]
            cells = [
                f"{stats['numeric_sign_counts'][sign]:,} "
                f"({stats['numeric_sign_percentages'][sign]:.2f}%)"
                for sign in ("negative", "zero", "positive")
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Signs describe numbers, not three-class gold labels. Exact zero does not",
            "define a neutral class or justify choosing a neutral band. Any future",
            "classification conversion must be specified using development data only.",
            "Quartiles use inclusive linear interpolation; standard deviation is",
            "population standard deviation. Each target annotation has equal weight:",
            "texts with multiple annotations contribute more than single-target texts.",
            "The JSON report also contains fixed histogram bins [-1, -0.5),",
            "[-0.5, 0), [0, 0.5), [0.5, 1]; these are descriptive bins, not labels.",
        ]
    lines += [
        "",
        "## Reproduction and boundaries",
        "",
        "Run `uv run --locked python -m finsight.data analyze-labels` "
        "after acquisition.",
        "The machine-readable report is `data/interim/label-analysis.json` (ignored).",
        "This Markdown report contains only aggregates. All manifest files are",
        "verified before analysis; byte mismatches fail instead of being accepted.",
        "",
        f"Manifest SHA-256: `{report['manifest_sha256']}`.",
        "",
    ]
    lines += [
        f"- {name}: `{revision}`" for name, revision in report["revisions"].items()
    ]
    lines += [
        "",
        "Lengths, duplicate/ambiguity investigations, per-target coverage, individual",
        "annotator agreement, and split design remain open Phase 1 tasks. No",
        "sentiment model is trained and no independent-test claim is made.",
    ]
    return "\n".join(lines) + "\n"
