"""Prediction-free news sampling and explicit span/issuer review gates."""

import re
from collections import defaultdict

from finsight.data.organization_annotations import validate_annotations
from finsight.sentiment.artifacts import sha

EXPERIMENT = "us-company-linking-dev-v1"


def select_sample(
    categories: dict[str, list[dict]], *, per_category: int, seed: int
) -> list[dict]:
    """Balance a convenience sample by category; never consult detector outputs.

    A title shared by categories belongs to the lexicographically first one.
    Namespace-zero titles with colons are excluded (briefs/roundup bundles).
    Category names are sampling metadata, never entity labels.
    """
    if type(per_category) is not int or per_category < 1 or type(seed) is not int:
        raise ValueError("positive category quota and integer seed required")
    owners: dict[int, dict] = {}
    titles: dict[str, int] = {}
    for category, rows in sorted(categories.items()):
        for row in rows:
            page_id, title = row["pageid"], row["title"]
            if (
                type(page_id) is not int
                or page_id <= 0
                or not isinstance(title, str)
                or not title.strip()
            ):
                raise ValueError("invalid article identity")
            if row.get("ns") != 0 or ":" in title:
                continue
            if page_id in owners and owners[page_id]["headline"] != title:
                raise ValueError("conflicting page identity")
            if title in titles and titles[title] != page_id:
                raise ValueError("duplicate title identity")
            titles[title] = page_id
            owners.setdefault(
                page_id,
                {
                    "record_id": f"wikinews:{page_id}",
                    "headline": title,
                    "partition": "validation",
                    "category": category,
                    "source_url": f"https://en.wikinews.org/?curid={page_id}",
                    "attribution": "Wikinews",
                    "license_policy_url": "https://en.wikinews.org/wiki/Wikinews:Copyright",
                },
            )
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in owners.values():
        groups[row["category"]].append(row)
    result = []
    for category in sorted(categories):
        pool = sorted(
            groups[category],
            key=lambda row: (
                sha(f"{EXPERIMENT}|{seed}|{row['record_id']}".encode()),
                row["record_id"],
            ),
        )
        if len(pool) < per_category:
            raise ValueError(f"insufficient unique headlines in {category}")
        result.extend(pool[:per_category])
    return result


def validate_review(
    sources: list[dict], rows: list[dict], *, provisional: bool = False
) -> None:
    """Require exhaustive spans and separately reviewed identity links.

    None is an explicit no-link decision, not an inferred negative company label.
    Human approval is an assertion, not a software-verified reviewer identity.
    """
    validate_annotations(sources, rows, provisional=provisional)
    for row in rows:
        for entity in row["entities"]:
            status, cik = entity.get("link_status"), entity.get("cik")
            if status not in {
                "linked",
                "not_in_directory",
                "not_applicable",
                "uncertain",
            }:
                raise ValueError("explicit link status required")
            if status == "linked":
                if (
                    entity["type"] != "company"
                    or not isinstance(cik, str)
                    or not re.fullmatch(r"[0-9]{10}", cik)
                    or int(cik) == 0
                    or not entity.get("identity_evidence")
                ):
                    raise ValueError(
                        "linked company requires CIK and identity evidence"
                    )
            elif cik is not None:
                raise ValueError("non-linked entity must not carry CIK")
            if entity["type"] == "other_organization" and status != "not_applicable":
                raise ValueError("other organization cannot receive issuer link")
            if not provisional and (
                status == "uncertain" or not entity.get("identity_evidence")
            ):
                raise ValueError("identity decisions require reviewed evidence")
