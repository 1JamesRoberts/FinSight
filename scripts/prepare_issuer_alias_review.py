"""Freeze a new source-attributed packet, excluding the previous reviewed titles."""

import json

from finsight.data.issuer_alias_review import select_fresh_sample
from finsight.data.target_preparation import json_bytes
from finsight.sentiment.artifacts import file_sha256, project_root, sha


def main() -> None:
    root = project_root()
    plan_path = root / "configs/issuer-alias-review-plan.json"
    plan = json.loads(plan_path.read_bytes())
    prior = root / plan["previous_packet"]
    prior_manifest = json.loads((prior / "manifest.json").read_bytes())
    if (
        file_sha256(prior / "manifest.json") != prior.name
        or file_sha256(prior / "sample.jsonl") != prior_manifest["sample_sha256"]
    ):
        raise ValueError("previous sample checksum differs")
    previous = [
        json.loads(line) for line in (prior / "sample.jsonl").read_bytes().splitlines()
    ]
    categories = {}
    sources = []
    for item in plan["source_files"]:
        path = root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("source response differs")
        raw = json.loads(path.read_bytes())
        if "continue" in raw or "error" in raw:
            raise ValueError("incomplete category response")
        categories[item["category"]] = raw["query"]["categorymembers"]
        sources.append(item)
    rows = select_fresh_sample(
        categories, previous, per_category=plan["per_category"], seed=plan["seed"]
    )
    if len(rows) != plan["sample_size"]:
        raise ValueError("sample size differs")
    body = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    manifest = {
        "experiment": plan["experiment"],
        "plan_sha256": file_sha256(plan_path),
        "sample_sha256": sha(body),
        "source_files": sources,
        "previous_sample_sha256": prior_manifest["sample_sha256"],
        "sampling_source_sha256": file_sha256(
            root / "src/finsight/data/issuer_alias_review.py"
        ),
        "base_sampler_sha256": file_sha256(
            root / "src/finsight/data/us_company_review.py"
        ),
        "headlines": len(rows),
        "selection_uses_predictions": False,
        "reference_status": "pending",
        "frozen_records_read": False,
    }
    contents = json_bytes(manifest)
    packet = root / "data/interim/issuer-alias-review" / sha(contents)
    packet.mkdir(parents=True, exist_ok=True)
    for name, content in [("manifest.json", contents), ("sample.jsonl", body)]:
        path = packet / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("immutable packet differs")
        path.write_bytes(content)
    print(packet)


if __name__ == "__main__":
    main()
