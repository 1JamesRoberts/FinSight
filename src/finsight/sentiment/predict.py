"""Score explicit targets with the pinned saved RoBERTa model; no acquisition."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from finsight.sentiment.artifacts import project_root
from finsight.sentiment.companies import load_registry
from finsight.sentiment.deployment import CONFIG, check_deployment
from finsight.sentiment.models.target_predictor import TargetSentimentPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text")
    parser.add_argument(
        "--target",
        action="append",
        help="Repeat for several targets in the same headline",
    )
    parser.add_argument(
        "--company",
        action="store_true",
        help="Require registry resolution; default targets are explicit native names",
    )
    parser.add_argument("--company-context", action="store_true")
    parser.add_argument("--allow-parent-rollup", action="store_true")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Verify/print metadata without loading weights or reading datasets",
    )
    args = parser.parse_args()
    config_path = args.config or args.root / CONFIG
    if args.show_config:
        print(json.dumps(check_deployment(args.root, config_path), indent=2))
        return
    if (
        not args.text
        or not args.text.strip()
        or not args.target
        or any(not t.strip() for t in args.target)
    ):
        parser.error("--text and at least one nonempty --target are required")
    if not args.company and (args.company_context or args.allow_parent_rollup):
        parser.error("company policy flags require --company")
    predictor = TargetSentimentPredictor.load(
        root=args.root,
        data_dir=args.data_dir,
        config_path=config_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    if args.company:
        registry = load_registry(args.root / "configs/companies.json")
        results = [
            predictor.predict_company(
                args.text,
                t,
                registry,
                company_context=args.company_context,
                allow_parent_rollup=args.allow_parent_rollup,
            )
            for t in args.target
        ]
    else:
        results = predictor.predict_many([args.text] * len(args.target), args.target)
    print(json.dumps([asdict(r) for r in results], indent=2))


if __name__ == "__main__":
    main()
