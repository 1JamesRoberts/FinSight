"""Explicit local research acquisition and validation commands."""

import argparse
import json
from pathlib import Path

from finsight.config import Settings
from finsight.data.acquisition import acquire_datasets
from finsight.data.audit import audit_datasets
from finsight.data.benchmark import freeze_benchmark, render_benchmark_report
from finsight.data.labels import analyze_labels, render_label_report
from finsight.data.lengths import analyze_lengths, render_length_report
from finsight.data.review import render_review_report, review_datasets
from finsight.data.splits import prepare_splits, render_split_report


def main() -> None:
    """Run one bounded dataset operation; network access requires acquire command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "acquire",
            "validate",
            "analyze-labels",
            "analyze-lengths",
            "review-annotations",
            "prepare-splits",
            "freeze-benchmark",
        ),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/asset-revisions.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    parser.add_argument(
        "--split-policy", type=Path, default=Path("configs/split-policy.json")
    )
    parser.add_argument(
        "--benchmark-policy", type=Path, default=Path("configs/benchmark-policy.json")
    )
    parser.add_argument(
        "--freeze-receipt", type=Path, default=Path("configs/benchmark-freeze.json")
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Override the command's default Markdown report destination",
    )
    args = parser.parse_args()
    if args.command == "acquire":
        receipt = acquire_datasets(args.manifest, args.data_dir)
        destination = args.data_dir / "raw/acquisition-receipt.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Verified {len(receipt)} files; receipt saved to {destination}")
    elif args.command == "validate":
        summary = audit_datasets(args.manifest, args.data_dir)
        destination = args.data_dir / "interim/dataset-integrity.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Integrity report saved to {destination}")
    elif args.command == "freeze-benchmark":
        receipt = freeze_benchmark(
            args.manifest,
            args.split_policy,
            args.benchmark_policy,
            args.data_dir,
            args.freeze_receipt,
        )
        report_path = args.report_path or Path("reports/frozen-benchmark.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_benchmark_report(receipt), encoding="utf-8")
        print(f"Frozen benchmark receipt verified at {args.freeze_receipt}")
        print(f"Aggregate report saved to {report_path}")
    elif args.command == "prepare-splits":
        summary = prepare_splits(
            args.manifest,
            args.split_policy,
            args.data_dir,
            freeze_path=args.freeze_receipt,
        )
        destination = args.data_dir / "interim/split-preparation.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        report_path = args.report_path or Path("reports/split-preparation.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_split_report(summary), encoding="utf-8")
        print(f"Prepared split artifacts saved to {summary['directory']}")
        print(f"Reports saved to {destination} and {report_path}")
    elif args.command == "review-annotations":
        review = review_datasets(args.manifest, args.data_dir)
        destination = args.data_dir / "interim/annotation-review.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n")
        report_path = args.report_path or Path("reports/annotation-review.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_review_report(review["summary"]), encoding="utf-8"
        )
        print(f"Review reports saved to {destination} and {report_path}")
    else:
        if args.command == "analyze-labels":
            summary = analyze_labels(args.manifest, args.data_dir)
            markdown = render_label_report(summary)
            kind = "label"
        else:
            summary = analyze_lengths(args.manifest, args.data_dir)
            markdown = render_length_report(summary)
            kind = "length"
        destination = args.data_dir / f"interim/{kind}-analysis.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        report_path = args.report_path or Path(f"reports/{kind}-analysis.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown, encoding="utf-8")
        print(f"Analysis reports saved to {destination} and {report_path}")


if __name__ == "__main__":
    main()
