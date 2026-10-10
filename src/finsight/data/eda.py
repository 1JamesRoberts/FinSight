"""Aggregate development-only EDA and render optional static research figures."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from finsight.data.acquisition import artifact_path, verify_artifact
from finsight.data.benchmark import load_development_records
from finsight.data.labels import label_distribution, score_distribution
from finsight.data.lengths import length_summary
from finsight.data.loaders import SentimentLabel, load_phrasebank
from finsight.data.review import text_hash

FIGURE_NAMES = (
    "class-distributions.png",
    "word-lengths.png",
    "phrasebank-agreement.png",
    "fiqa-scores.png",
    "sentfin-targets.png",
)
DISPLAY_NAMES = {"phrasebank": "PhraseBank", "sentfin": "SEntFiN", "fiqa": "FiQA"}


def summarize_development(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count headlines once for lengths/multiplicity, annotations for target labels."""
    if any(row["partition"] not in {"train", "validation"} for row in records):
        raise ValueError("EDA accepts only train/validation records")
    if any(row["corpus"] not in {"phrasebank", "sentfin", "fiqa"} for row in records):
        raise ValueError("unknown EDA corpus")
    summary: dict[str, Any] = {}
    for corpus in ("phrasebank", "sentfin", "fiqa"):
        rows = [row for row in records if row["corpus"] == corpus]
        native = [row["native_record"] for row in rows]
        group = {
            "records": len(rows),
            "partitions": dict(
                sorted(Counter(row["partition"] for row in rows).items())
            ),
            "lengths": length_summary(
                row["title"] if corpus == "sentfin" else row["sentence"]
                for row in native
            ),
        }
        if corpus == "phrasebank":
            group["labels"] = label_distribution(
                SentimentLabel(row["label"]) for row in native
            )
        elif corpus == "sentfin":
            group["labels"] = label_distribution(
                SentimentLabel(a["label"]) for row in native for a in row["sentiments"]
            )
            group["target_multiplicity"] = {
                "1": sum(len(row["sentiments"]) == 1 for row in native),
                "2": sum(len(row["sentiments"]) == 2 for row in native),
                "3+": sum(len(row["sentiments"]) >= 3 for row in native),
            }
            mixed = sum(
                len({a["label"] for a in row["sentiments"]}) > 1 for row in native
            )
            multi = sum(len(row["sentiments"]) > 1 for row in native)
            group["multi_target_sentiments"] = {
                "uniform": multi - mixed,
                "mixed": mixed,
            }
        else:
            scores = [a["score"] for row in native for a in row["annotations"]]
            if any(score is None for score in scores):
                raise ValueError("unscored FiQA annotation entered development EDA")
            group["scores"] = score_distribution(scores) if scores else None
            group["source_types"] = dict(
                sorted(Counter(row["source_type"] for row in native).items())
            )
            group["flagged_snippet_annotations"] = sum(
                flag["review_required"]
                for row in rows
                for flag in row["annotation_eligibility"]
            )
        summary[corpus] = group
    return summary


def analyze_development(
    data_dir: Path,
    manifest_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Verify frozen lineage and raw pins, then join agreement metadata to dev only."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records = load_development_records(data_dir, receipt_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != json.loads(
            (
                data_dir
                / "processed/benchmarks"
                / receipt["benchmark_manifest_sha256"]
                / "benchmark-manifest.json"
            ).read_text(encoding="utf-8")
        )["asset_manifest_sha256"]
    ):
        raise ValueError("source manifest differs from frozen benchmark")
    for name in manifest["local_dataset_assets"]:
        for specification in manifest["assets"][name]["acquisition_files"]:
            verify_artifact(artifact_path(data_dir, specification), specification)
    corpora = summarize_development(records)
    phrasebank = manifest["assets"]["financial_phrasebank"]
    development = {
        (row["text_sha256"], row["native_record"]["label"])
        for row in records
        if row["corpus"] == "phrasebank"
    }
    agreement = {}
    for level in ("50", "66", "75", "all"):
        identities = {
            (text_hash(row.sentence), row.label.value)
            for row in load_phrasebank(
                artifact_path(data_dir, phrasebank["acquisition_files"][0]),
                revision=phrasebank["revision"],
                agreement=level,
            )
        }
        agreement[level] = len(development & identities)
    return {
        "analysis_version": 1,
        "scope": "verified train + validation only; all test candidates excluded",
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
        "asset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "development_records_sha256": hashlib.sha256(
            json.dumps(
                records, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "corpora": corpora,
        "phrasebank_agreement_membership": agreement,
        "agreement_counting": (
            "development text/label matches in overlapping source subsets; "
            "not independent samples or individual votes"
        ),
    }


def plot_development(summary: dict[str, Any], destination: Path) -> None:
    """Save five fixed-bin figures with no model tokenization or learned thresholds."""
    if any(summary["corpora"][name]["records"] == 0 for name in DISPLAY_NAMES):
        raise ValueError("all three development corpora must be nonempty for figures")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
        }
    )
    colors = ["#b84b45", "#64748b", "#23856d"]
    corpora = summary["corpora"]

    def save(fig: Any, name: str) -> None:
        fig.savefig(
            destination / name,
            dpi=180,
            facecolor="white",
            metadata={"Software": "FinSight Phase 1 EDA"},
        )
        plt.close(fig)

    def bars(ax: Any, names: list[str], values: list[int], color: Any) -> None:
        ax.bar(names, values, color=color, width=0.65)
        ax.set_ylim(0, max(values, default=0) * 1.18 or 1)
        for i, value in enumerate(values):
            ax.annotate(
                f"{value:,}",
                (i, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, corpus, unit in zip(
        axes,
        ("phrasebank", "sentfin"),
        ("Sentences", "Target annotations"),
        strict=True,
    ):
        labels = corpora[corpus]["labels"]["counts"]
        bars(ax, list(labels), list(labels.values()), colors)
        ax.set(title=f"{DISPLAY_NAMES[corpus]} · development", ylabel=unit)
    save(fig, FIGURE_NAMES[0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, corpus, color in zip(axes, corpora, colors, strict=True):
        hist = corpora[corpus]["lengths"]["word_histogram"]
        total = corpora[corpus]["records"]
        ax.bar(
            list(hist),
            [100 * count / total if total else 0 for count in hist.values()],
            color=color,
        )
        ax.set(
            title=f"{DISPLAY_NAMES[corpus]} development\nn={total:,}",
            xlabel="Whitespace words",
            ylabel="Texts (%)",
        )
        ax.tick_params(axis="x", rotation=35)
    save(fig, FIGURE_NAMES[1])

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    agreement = summary["phrasebank_agreement_membership"]
    bars(ax, ["≥50%", "≥66%", "≥75%", "Unanimous"], list(agreement.values()), "#466aa3")
    ax.set(
        title="PhraseBank agreement membership · development only",
        ylabel="Development text/label matches",
        xlabel="Nested subsets overlap; never concatenate",
    )
    save(fig, FIGURE_NAMES[2])

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    histogram = corpora["fiqa"]["scores"]["histogram"]
    bars(ax, list(histogram), list(histogram.values()), "#7c5e9c")
    ax.set(
        title="FiQA continuous sentiment scores · development only",
        xlabel="Fixed descriptive score intervals (not classes)",
        ylabel="Target annotations",
    )
    save(fig, FIGURE_NAMES[3])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key, title in zip(
        axes,
        ("target_multiplicity", "multi_target_sentiments"),
        ("Targets per headline", "Sentiment within multi-target headlines"),
        strict=True,
    ):
        counts = corpora["sentfin"][key]
        bars(ax, list(counts), list(counts.values()), "#23856d")
        ax.set(title=title, ylabel="SEntFiN development headlines")
    save(fig, FIGURE_NAMES[4])


def generate_eda(
    data_dir: Path,
    manifest_path: Path,
    receipt_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    """Write portable aggregate JSON, figures, and a development EDA report."""
    summary = analyze_development(data_dir, manifest_path, receipt_path)
    plot_development(summary, report_dir / "figures/phase1")
    (report_dir / "eda-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase 1 development EDA",
        "",
        summary["scope"],
        "",
        "| Corpus | Development texts | Words median | Words P95 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, group in summary["corpora"].items():
        words = group["lengths"]["words"]
        lines.append(
            f"| {name} | {group['records']} | {words['median']} | {words['p95']} |"
        )
    lines += [
        "",
        "Charts use retained train/validation examples only. SEntFiN class",
        "counts are targets; length/multiplicity counts are headlines. FiQA",
        "scores stay continuous; histogram edges are descriptive, not thresholds.",
        "Agreement subsets overlap; membership matches are not individual votes.",
        "",
    ]
    lines += [
        f"![{name.removesuffix('.png').replace('-', ' ')}](figures/phase1/{name})\n"
        for name in FIGURE_NAMES
    ]
    lines += [
        "See `eda-summary.json` for exact counts and source/freeze digests.",
        "See `../docs/phase1-findings.md` for interpretation and limitations.",
        "No frozen-test exploration, model training, or new filtering is performed.",
    ]
    (report_dir / "phase1-eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
