"""Plot public cross-fitted calibration aggregates; never read source data."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/temperature-scaling.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/phase3/temperature-scaling.png"),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    models = ("finbert", "roberta_base", "deberta_base")
    names = ("FinBERT", "RoBERTa Base", "DeBERTa Base")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ("ece", "brier", "nll"),
        ("Top-label ECE (10 bins)", "Multiclass Brier", "Negative log likelihood"),
        strict=True,
    ):
        for stage, offset, color in (
            ("before", -0.18, "#8c99a6"),
            ("after", 0.18, "#157f86"),
        ):
            stats = [report["models"][m]["statistics"][metric][stage] for m in models]
            ax.bar(
                [i + offset for i in range(3)],
                [s["mean"] for s in stats],
                width=0.36,
                yerr=[s["sample_std"] for s in stats],
                capsize=4,
                label="Original" if stage == "before" else "Cross-fitted T",
                color=color,
            )
        ax.set_xticks(range(3), names, rotation=15)
        ax.set_title(label)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle(
        "Temperature scaling on SEntFiN validation\n"
        "Mean ± sample SD across 3 seeds; lower is better\n"
        "Exploratory: validation already used for model selection",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
