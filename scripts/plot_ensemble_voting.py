"""Plot public validation voting aggregates and seed variation only."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/experiments/ensemble-voting.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/phase3/ensemble-voting.png"),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    names = ("finbert", "roberta_base", "deberta_base", "soft", "hard")
    labels = ("FinBERT", "RoBERTa Base", "DeBERTa Base", "Soft voting", "Hard voting")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True, layout="constrained")
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
            if metric not in report["statistics"][name]:
                ax.text(
                    0.05,
                    i,
                    "N/A (label votes)",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontsize=9,
                )
                continue
            stat = report["statistics"][name][metric]
            ax.errorbar(
                stat["mean"],
                i,
                xerr=stat["sample_std"],
                fmt="o",
                capsize=4,
                color="#157f86"
                if name == "soft"
                else "#8653a3"
                if name == "hard"
                else "#5e7285",
            )
        ax.axvline(
            report["statistics"]["deberta_base"][metric]["mean"],
            color="#8c99a6",
            linestyle="--",
            linewidth=1,
        )
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(5), labels)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Equal-weight soft and democratic voting on SEntFiN validation\n"
        "Mean ± sample SD across 3 matching seeds; dashed line: DeBERTa Base\n"
        "Exploratory: validation already used for epoch/rate selection",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
