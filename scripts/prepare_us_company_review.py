"""Acquire permitted Wikinews category titles or freeze their review sample."""

import argparse
import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from finsight.data.target_preparation import json_bytes
from finsight.data.us_company_review import select_sample
from finsight.sentiment.artifacts import file_sha256, project_root, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "prepare"))
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data")
    args = parser.parse_args()
    root = project_root()
    plan_path = root / "configs/us-company-review-plan.json"
    plan = json.loads(plan_path.read_bytes())
    if args.action == "acquire":
        if args.raw_dir.exists() and any(args.raw_dir.iterdir()):
            parser.error("acquisition requires an empty output directory")
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        log = []
        for category in plan["categories_considered"]:
            url = "https://en.wikinews.org/w/api.php?" + urllib.parse.urlencode(
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": "Category:" + category,
                    "cmnamespace": 0,
                    "cmlimit": 500,
                    "format": "json",
                }
            )
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "FinSight/0.1 (portfolio research; https://github.com/1JamesRoberts/financial-news-intelligence)"
                },
            )
            # Sequential bounded requests; no crawling or retry loops.
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read(2_000_001)
            raw = json.loads(body)
            if len(body) > 2_000_000 or "continue" in raw or "error" in raw:
                raise ValueError("category response exceeds the bounded protocol")
            filename = category.replace(" ", "_") + ".json"
            (args.raw_dir / filename).write_bytes(body)
            log.append(
                {
                    "category": category,
                    "request_url": url,
                    "retrieved_at_utc": datetime.now(UTC).isoformat(),
                    "file": filename,
                    "rows": len(raw["query"]["categorymembers"]),
                }
            )
        (args.raw_dir / "acquisition.json").write_bytes(json_bytes(log))
        print(args.raw_dir)
        return
    log = json.loads((args.raw_dir / "acquisition.json").read_bytes())
    if {entry["category"] for entry in log} != set(plan["categories_considered"]):
        raise ValueError("source category membership differs")
    categories = {}
    for entry in log:
        filename = entry["file"]
        if Path(filename).name != filename or not entry["request_url"].startswith(
            "https://en.wikinews.org/w/api.php?"
        ):
            raise ValueError("source provenance differs")
        raw = json.loads((args.raw_dir / filename).read_bytes())
        if "continue" in raw or "error" in raw:
            raise ValueError("incomplete source response")
        if entry["category"] in plan["categories_selected"]:
            categories[entry["category"]] = raw["query"]["categorymembers"]
    rows = select_sample(
        categories, per_category=plan["per_category"], seed=plan["seed"]
    )
    if len(rows) != plan["sample_size"]:
        raise ValueError("sample size differs")
    body = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    manifest = {
        "experiment": plan["experiment"],
        "plan_sha256": file_sha256(plan_path),
        "sample_sha256": sha(body),
        "selection_source_sha256": file_sha256(
            root / "src/finsight/data/us_company_review.py"
        ),
        "source_responses": [
            {**entry, "sha256": file_sha256(args.raw_dir / entry["file"])}
            for entry in log
        ],
        "headlines": len(rows),
        "sampling_uses_predictions": False,
        "human_review_status": "pending",
        "frozen_records_read": False,
    }
    manifest_body = json_bytes(manifest)
    packet = args.data_dir / "interim/us-company-review" / sha(manifest_body)
    packet.mkdir(parents=True, exist_ok=True)
    for filename, contents in (
        ("sample.jsonl", body),
        ("manifest.json", manifest_body),
    ):
        path = packet / filename
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("immutable packet differs")
        path.write_bytes(contents)
    print(packet)


if __name__ == "__main__":
    main()
