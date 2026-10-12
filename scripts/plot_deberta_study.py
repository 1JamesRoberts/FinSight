"""Plot the bounded DeBERTa rate search and selected-rate seed variation."""

import json
from pathlib import Path


def main() -> None:
    import matplotlib.pyplot as plt

    study = json.loads(
        Path("reports/experiments/deberta-tuning-runs/study.json").read_bytes()
    )
    if not study["complete"]:
        raise ValueError("completed study required")
    model = "deberta_base"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    grid = study["grid"][model]
    axes[0].plot(
        range(3), [r["metrics"]["macro_f1"] for r in grid], "o-", color="#548235"
    )
    axes[0].set_xticks(range(3), ["1e-5", "2e-5", "3e-5"])
    axes[0].set(xlabel="Learning rate (seed 42)", title="DeBERTa-v3 Base rate search")
    rows = study["seed_runs"][model]
    for i, row in enumerate(rows):
        axes[1].scatter(i, row["metrics"]["macro_f1"], color="#548235", s=50)
    stats = study["statistics"][model]["macro_f1"]
    axes[1].errorbar(
        3, stats["mean"], yerr=stats["sample_std"], fmt="D", color="#222222", capsize=5
    )
    axes[1].set_xticks(range(4), ["Seed 42", "Seed 43", "Seed 44", "Mean ± SD"])
    axes[1].set(
        title=f"Selected LR {study['selected'][model]['learning_rate']:.0e}",
        xlim=(-0.4, 3.5),
    )
    values = [r["metrics"]["macro_f1"] for r in grid + rows]
    for ax in axes:
        ax.set(
            ylabel="SEntFiN validation macro F1",
            ylim=(min(values) - 0.01, max(values) + 0.01),
        )
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Validation only · seed 42 selects rate and is reused in mean "
        "· no test scoring",
        fontsize=11,
    )
    output = Path("reports/figures/phase3/deberta-learning-rate-seeds.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
