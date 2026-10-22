"""Exact span plus CIK evaluation; unresolved companies remain in recall totals."""

import re

from finsight.data.us_company_review import validate_review
from finsight.sentiment.evaluation.organization_metrics import exact_span_metrics


def linking_metrics(
    sources: list[dict], reviewed: list[dict], predictions: dict[str, list[dict]]
) -> dict:
    """Score reviewed labels only; a wrong issuer is both an FP and an FN."""
    validate_review(sources, reviewed)
    if set(predictions) != {row["record_id"] for row in sources}:
        raise ValueError("prediction membership differs")
    company = {
        row["record_id"]: [
            (e["start"], e["end"]) for e in row["entities"] if e["type"] == "company"
        ]
        for row in reviewed
    }
    predicted = {}
    tp = fp = fn = accepted = 0
    for row in reviewed:
        identifier = row["record_id"]
        proposals = predictions[identifier]
        text = row["headline"]
        for p in proposals:
            if (
                type(p["start"]) is not int
                or type(p["end"]) is not int
                or not 0 <= p["start"] < p["end"] <= len(text)
                or text[p["start"] : p["end"]] != p["text"]
            ):
                raise ValueError("prediction span differs from source")
            if p.get("cik") is not None and not re.fullmatch(r"[0-9]{10}", p["cik"]):
                raise ValueError("invalid predicted CIK")
        predicted[identifier] = [(p["start"], p["end"]) for p in proposals]
        gold = {
            (e["start"], e["end"], e["cik"])
            for e in row["entities"]
            if e["link_status"] == "linked"
        }
        links = {
            (p["start"], p["end"], p["cik"])
            for p in proposals
            if p.get("cik") is not None
        }
        tp += len(gold & links)
        fp += len(links - gold)
        fn += len(gold - links)
        accepted += len(links)
    return {
        "company_span_extraction": exact_span_metrics(company, predicted),
        "issuer_links": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "reference_links": tp + fn,
            "accepted_links": accepted,
            "precision": tp / accepted if accepted else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        },
        "reference_status": "human_reviewed",
    }
