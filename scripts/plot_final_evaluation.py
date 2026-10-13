"""Plot exported aggregate test metrics and shared group-bootstrap intervals."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/phase3-final.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/phase3/final-evaluation.png"),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    if report["scope"] != "frozen_test" or report["final_test_evaluated"] is not True:
        raise ValueError("completed final aggregate required")
    names = (
        "training_prior",
        "vader",
        "tfidf_logreg",
        "tfidf_svm",
        "finbert_pretrained",
        "finbert",
        "roberta_base",
        "deberta_base",
        "soft",
        "hard",
    )
    labels = (
        "Training prior",
        "VADER",
        "TF-IDF LR",
        "TF-IDF SVM",
        "Unchanged FinBERT",
        "Fine-tuned FinBERT",
        "RoBERTa Base",
        "DeBERTa Base",
        "Soft voting",
        "Hard voting",
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.5), sharey=True, layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("macro_f1", "nll", "ece"),
        (
            "Macro F1 — higher is better",
            "NLL — lower is better",
            "ECE — lower is better",
        ),
        strict=True,
    ):
        for i, name in enumerate(names):
            key = (
                name + (".raw" if metric == "macro_f1" else ".calibrated")
                if name in ("finbert", "roberta_base", "deberta_base")
                else name
            )
            if metric not in report["statistics"][key]:
                ax.text(0.03, i, "N/A", transform=ax.get_yaxis_transform(), va="center")
                continue
            value = report["statistics"][key][metric]["mean"]
            low, high = report["uncertainty"]["intervals"][key][metric]
            color = (
                "#157f86"
                if name == "soft"
                else "#8653a3"
                if name == "hard"
                else "#5e7285"
            )
            ax.errorbar(
                value,
                i,
                xerr=[[value - low], [high - value]],
                fmt="o",
                capsize=3,
                color=color,
            )
        reference = (
            "deberta_base.raw" if metric == "macro_f1" else "deberta_base.calibrated"
        )
        ax.axvline(
            report["statistics"][reference][metric]["mean"],
            color="#8c99a6",
            linestyle="--",
            linewidth=1,
        )
        ax.set_yticks(range(len(names)), labels)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Locked final evaluation: 766 single-target SEntFiN headlines\n"
        "Points: seed-mean or fixed baseline; bars: 95% group-bootstrap intervals\n"
        "Trained models calibrated in probability panels; dashed line: DeBERTa Base",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
