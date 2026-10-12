"""Plot aggregate diagnostics only; requires the existing eda dependency group."""

import argparse
import json
from pathlib import Path


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/transformer-analysis.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/figures/phase3/calibration.png")
    )
    args = parser.parse_args()
    summary = json.loads(args.report.read_bytes())
    models = tuple(
        m for m in ("roberta_base", "finbert", "deberta_base") if m in summary["models"]
    )
    names = {
        "roberta_base": "RoBERTa Base",
        "finbert": "FinBERT",
        "deberta_base": "DeBERTa-v3 Base",
    }
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(6 * len(models), 8),
        layout="constrained",
        squeeze=False,
    )
    colors = ("#2674a6", "#b75c16", "#548235")
    for column, model in enumerate(models):
        ax, counts = axes[0, column], axes[1, column]
        ax.plot([0, 1], [0, 1], "--", color="#999999", label="Perfect calibration")
        for j, run in enumerate(summary["models"][model]["runs"]):
            bins = [b for b in run["bins"] if b["count"]]
            ax.plot(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                color=colors[j],
                alpha=0.75,
                linewidth=1,
            )
            ax.scatter(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                s=[12 + b["count"] * 0.12 for b in bins],
                color=colors[j],
                label=f"Seed {run['seed']} · ECE {run['ece']:.3f}",
            )
            counts.bar(
                [b["lower"] + 0.02 + j * 0.027 for b in run["bins"]],
                [b["count"] for b in run["bins"]],
                width=0.026,
                color=colors[j],
                label=f"Seed {run['seed']}",
            )
        ax.set(
            title=names[model],
            xlabel="Mean top-label confidence in bin",
            ylabel="Fraction correct in bin",
            xlim=(0, 1.02),
            ylim=(0, 1.02),
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.2)
        counts.set(
            xlabel="Top-label confidence (10 equal-width bins)",
            ylabel="Headlines per bin",
            xlim=(0, 1.02),
        )
        counts.legend(fontsize=9)
        counts.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "SEntFiN validation · 781 headlines per run\n"
        "Unchanged probabilities · top-label reliability\n"
        "Marker size reflects bin count; sparse bins are noisy",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, metadata={"Software": "FinSight Matplotlib"})
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
