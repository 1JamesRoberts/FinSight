"""Explicit local research acquisition and validation commands."""

import argparse
import json
from pathlib import Path

from finsight.config import Settings
from finsight.data.acquisition import acquire_datasets
from finsight.data.audit import audit_datasets
from finsight.data.labels import analyze_labels, render_label_report


def main() -> None:
    """Run one bounded dataset operation; network access requires acquire command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "validate", "analyze-labels"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/asset-revisions.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/label-analysis.md"),
        help="Markdown destination for analyze-labels",
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
    else:
        summary = analyze_labels(args.manifest, args.data_dir)
        destination = args.data_dir / "interim/label-analysis.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(render_label_report(summary), encoding="utf-8")
        print(f"Label reports saved to {destination} and {args.report_path}")


if __name__ == "__main__":
    main()
