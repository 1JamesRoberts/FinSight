"""Duplicate candidates and annotation review, without removing or relabeling rows."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finsight.data.acquisition import artifact_path, verify_artifact
from finsight.data.audit import text_identity
from finsight.data.fiqa import load_fiqa
from finsight.data.loaders import load_phrasebank, load_sentfin


@dataclass(frozen=True)
class ReviewRecord:
    corpus: str
    source_id: str
    text: str
    # Context identifies the annotation unit; None denotes absent gold, not neutral.
    annotations: tuple[tuple[str, str | None], ...]


def text_hash(text: str) -> str:
    """Hash the existing exact-duplicate identity, retaining numbers/punctuation."""
    return hashlib.sha256(text_identity(text).encode("utf-8")).hexdigest()


def snippet_mismatches(text: str, snippets: Iterable[str | None]) -> tuple[str, ...]:
    """Flag nonmissing snippets not found as normalized literal spans, not errors."""
    identity = text_identity(text)
    return tuple(
        snippet
        for snippet in snippets
        if snippet is not None and text_identity(snippet) not in identity
    )


def conflicting_contexts(records: Iterable[ReviewRecord]) -> dict[str, list[str]]:
    """Find inconsistent gold for the same corpus/text/annotation context only."""
    values: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for record in records:
        for context, value in record.annotations:
            if value is not None:
                values[(record.corpus, text_identity(record.text), context)].add(value)
    return {
        json.dumps(key, ensure_ascii=False): sorted(labels)
        for key, labels in sorted(values.items())
        if len(labels) > 1
    }


def exact_duplicate_groups(records: Iterable[ReviewRecord]) -> list[dict[str, Any]]:
    """Group within a corpus; equivalent targets use the caller's context identity."""
    groups: dict[tuple[str, str], list[ReviewRecord]] = defaultdict(list)
    for record in records:
        groups[(record.corpus, text_identity(record.text))].append(record)
    result = []
    for (corpus, _), rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        annotation_sets = {
            frozenset((context, value) for context, value in row.annotations)
            for row in rows
        }
        result.append(
            {
                "corpus": corpus,
                "text_sha256": text_hash(rows[0].text),
                "records": [
                    asdict(row) for row in sorted(rows, key=lambda r: r.source_id)
                ],
                "conflicting_contexts": conflicting_contexts(rows),
                "annotation_sets_differ": len(annotation_sets) > 1,
            }
        )
    return result


def near_duplicate_pairs(
    texts: Iterable[str], *, threshold: float = 0.80, min_words: int = 6
) -> list[dict[str, Any]]:
    """Find unique-text candidates by exact Jaccard of whitespace word trigrams.

    An inverted index enumerates pairs sharing a shingle; the length bound skips
    impossible matches. This is deterministic lexical screening, not semantic
    equivalence. Numbers, punctuation, and negation are retained. Exact duplicates
    are collapsed before searching. Short texts (<min_words) are not screened.
    """
    if not 0 < threshold <= 1 or min_words < 3:
        raise ValueError("threshold must be in (0, 1] and min_words at least 3")
    unique = sorted({text_identity(text) for text in texts})
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    shingles: list[set[tuple[str, ...]]] = []
    result = []
    for index, text in enumerate(unique):
        words = text.split()
        current = (
            {
                tuple(words[position : position + 3])
                for position in range(len(words) - 2)
            }
            if len(words) >= min_words
            else set()
        )
        intersections: Counter[int] = Counter()
        for shingle in sorted(current):
            for previous in postings[shingle]:
                size = len(shingles[previous])
                if min(size, len(current)) >= threshold * max(size, len(current)):
                    intersections[previous] += 1
        for previous, intersection in sorted(intersections.items()):
            similarity = intersection / (
                len(current) + len(shingles[previous]) - intersection
            )
            if similarity >= threshold:
                left = unique[previous]
                result.append(
                    {
                        "left_sha256": text_hash(left),
                        "right_sha256": text_hash(text),
                        "jaccard": similarity,
                        "numeric_tokens_differ": (
                            re.findall(r"\d+(?:[.,]\d+)*", left)
                            != re.findall(r"\d+(?:[.,]\d+)*", text)
                        ),
                    }
                )
        shingles.append(current)
        for shingle in sorted(current):
            postings[shingle].append(index)
    return result


def review_datasets(manifest_path: Path, data_dir: Path) -> dict[str, Any]:
    """Produce aggregates and local review candidates from verified source records."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assets = manifest["assets"]
    for name in manifest["local_dataset_assets"]:
        for specification in assets[name]["acquisition_files"]:
            verify_artifact(artifact_path(data_dir, specification), specification)
    phrasebank = assets["financial_phrasebank"]
    variants = {
        agreement: load_phrasebank(
            artifact_path(data_dir, phrasebank["acquisition_files"][0]),
            revision=phrasebank["revision"],
            agreement=agreement,
        )
        for agreement in ("50", "66", "75", "all")
    }
    records = [
        ReviewRecord(
            "phrasebank", row.source_id, row.sentence, (("sentence", row.label.value),)
        )
        for row in variants["50"]
    ]
    strongest_matches = {}
    for agreement in ("50", "66", "75", "all"):
        for row in variants[agreement]:
            strongest_matches[(text_identity(row.sentence), row.label)] = agreement
    agreement_counts = Counter(
        strongest_matches[(text_identity(row.sentence), row.label)]
        for row in variants["50"]
    )
    sentfin = load_sentfin(
        data_dir / "raw/sentfin/SEntFiN.csv", revision=assets["sentfin_1_0"]["revision"]
    )
    sentfin_records = [
        ReviewRecord(
            "sentfin",
            row.source_id,
            row.title,
            tuple(
                (text_identity(annotation.target), annotation.label.value)
                for annotation in row.sentiments
            ),
        )
        for row in sentfin
    ]
    records.extend(sentfin_records)
    fiqa_training = []
    if "fiqa_task1_original" in manifest["local_dataset_assets"]:
        fiqa = assets["fiqa_task1_original"]
        for kind in ("headline", "post"):
            for split in ("train", "test"):
                rows = load_fiqa(
                    data_dir / f"raw/fiqa/task1_{kind}_ABSA_{split}.json",
                    revision=fiqa["revision"],
                    source_type=kind,
                    source_split=split,
                )
                if split == "train":
                    fiqa_training.extend(rows)
                for row in rows:
                    annotations = tuple(
                        (
                            json.dumps(
                                [
                                    text_identity(a.target),
                                    sorted(
                                        text_identity(value)
                                        for value in a.aspects or ()
                                    ),
                                    [
                                        text_identity(value)
                                        if value is not None
                                        else None
                                        for value in a.snippets
                                    ],
                                ],
                                ensure_ascii=False,
                            ),
                            # Signed numeric zeros are the same gold score.
                            str(a.score or 0.0) if a.score is not None else None,
                        )
                        for a in row.annotations
                    )
                    records.append(
                        ReviewRecord(
                            "fiqa",
                            f"{kind}/{split}/{row.source_id}",
                            row.sentence,
                            annotations,
                        )
                    )
    groups = exact_duplicate_groups(records)
    by_hash: dict[str, list[ReviewRecord]] = defaultdict(list)
    for record in records:
        by_hash[text_hash(record.text)].append(record)
    pairs = near_duplicate_pairs(record.text for record in records)
    for pair in pairs:
        pair["left_records"] = [asdict(row) for row in by_hash[pair["left_sha256"]]]
        pair["right_records"] = [asdict(row) for row in by_hash[pair["right_sha256"]]]
        pair["shared_context_gold_differs"] = any(
            left.corpus == right.corpus
            and left_context == right_context
            and left_value is not None
            and right_value is not None
            and left_value != right_value
            for left in by_hash[pair["left_sha256"]]
            for right in by_hash[pair["right_sha256"]]
            for left_context, left_value in left.annotations
            for right_context, right_value in right.annotations
        )
    corpus_counts = {}
    for corpus in sorted({record.corpus for record in records}):
        subset = [group for group in groups if group["corpus"] == corpus]
        corpus_counts[corpus] = {
            "records": sum(record.corpus == corpus for record in records),
            "exact_duplicate_groups": len(subset),
            "extra_duplicate_rows": sum(len(group["records"]) - 1 for group in subset),
            "groups_with_conflicting_gold_contexts": sum(
                bool(group["conflicting_contexts"]) for group in subset
            ),
            "groups_with_different_annotation_sets": sum(
                group["annotation_sets_differ"] for group in subset
            ),
        }
    cross_exact = sum(
        len({row.corpus for row in rows}) > 1 for rows in by_hash.values()
    )
    near_by_corpora: Counter[str] = Counter()
    for pair in pairs:
        relations = {
            "/".join(sorted((left["corpus"], right["corpus"])))
            for left in pair["left_records"]
            for right in pair["right_records"]
        }
        near_by_corpora.update(relations)
    summary = {
        "analysis_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "revisions": {
            name: assets[name]["revision"] for name in manifest["local_dataset_assets"]
        },
        "corpora": corpus_counts,
        "cross_corpus_exact_text_groups": cross_exact,
        "near_duplicate_candidate_pairs": len(pairs),
        "near_pairs_by_corpora": dict(sorted(near_by_corpora.items())),
        "near_pairs_with_different_numeric_tokens": sum(
            p["numeric_tokens_differ"] for p in pairs
        ),
        "near_pairs_with_different_gold_for_shared_context": sum(
            p["shared_context_gold_differs"] for p in pairs
        ),
        "near_method": {
            "word_shingle_size": 3,
            "jaccard_threshold": 0.80,
            "minimum_words": 6,
        },
        "phrasebank_strongest_text_label_subset_match": {
            agreement: agreement_counts[agreement]
            for agreement in ("50", "66", "75", "all")
        },
        "sentfin_mixed_target_label_headlines": sum(
            len({a.label for a in row.sentiments}) > 1 for row in sentfin
        ),
        "fiqa_training_annotations_with_missing_snippets": sum(
            any(value is None for value in a.snippets)
            for row in fiqa_training
            for a in row.annotations
        ),
        "fiqa_training_scores_abs_at_most_0_1": sum(
            a.score is not None and abs(a.score) <= 0.1
            for row in fiqa_training
            for a in row.annotations
        ),
        "fiqa_training_nonmatching_snippet_values": sum(
            len(snippet_mismatches(row.sentence, a.snippets))
            for row in fiqa_training
            for a in row.annotations
        ),
        "fiqa_training_annotations_with_nonmatching_snippets": sum(
            bool(snippet_mismatches(row.sentence, a.snippets))
            for row in fiqa_training
            for a in row.annotations
        ),
        "within_record_conflicting_gold_contexts": sum(
            bool(conflicting_contexts([record])) for record in records
        ),
        "fiqa_test_scope": (
            "Duplicate screening only; absent gold is not a conflicting label"
        ),
        "records_removed_or_relabeled": False,
        "final_test_frozen": False,
    }
    examples = {
        "phrasebank_lowest_subset_match": [
            asdict(record)
            for record in records
            if (
                record.corpus == "phrasebank"
                and strongest_matches[
                    (text_identity(record.text), record.annotations[0][1])
                ]
                == "50"
            )
        ][:5],
        "sentfin_mixed_targets": [
            asdict(record)
            for record in sentfin_records
            if len({value for _, value in record.annotations}) > 1
        ][:5],
    }
    examples["fiqa_missing_or_nonmatching_snippets"] = [
        {
            "source_id": f"{row.source_type}/train/{row.source_id}",
            "text": row.sentence,
            "annotation": asdict(a),
            "nonmatching_snippets": list(snippet_mismatches(row.sentence, a.snippets)),
        }
        for row in fiqa_training
        for a in row.annotations
        if any(value is None for value in a.snippets)
        or snippet_mismatches(row.sentence, a.snippets)
    ]
    examples["fiqa_small_magnitude_training_scores"] = {
        kind: [
            {
                "source_id": f"{kind}/train/{row.source_id}",
                "text": row.sentence,
                "annotation": asdict(a),
            }
            for row in fiqa_training
            if row.source_type == kind
            for a in row.annotations
            if a.score is not None and abs(a.score) <= 0.1
        ][:5]
        for kind in ("headline", "post")
    }
    return {
        "summary": summary,
        "exact_groups": groups,
        "near_pairs": pairs,
        "examples": examples,
    }


def render_review_report(summary: dict[str, Any]) -> str:
    """Public-safe aggregate report; source text stays in ignored local review JSON."""
    lines = [
        "# FinSight duplicate and annotation review",
        "",
        "Exact identity: Unicode NFKC + casefold + whitespace collapse, retaining",
        "numbers and punctuation. PhraseBank uses >=50% only; agreement subsets",
        "are not pooled. FiQA original train/test inputs are screened for duplicates,",
        "with test inputs excluded from score/ambiguity analysis. No rows are removed",
        "or relabeled and no final evaluation split is selected.",
        "",
        "| Corpus | Records | Exact groups | Extra rows | "
        "Conflicting gold contexts | Different annotation sets |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for corpus, item in summary["corpora"].items():
        lines.append(
            f"| {corpus} | "
            + " | ".join(
                str(item[key])
                for key in (
                    "records",
                    "exact_duplicate_groups",
                    "extra_duplicate_rows",
                    "groups_with_conflicting_gold_contexts",
                    "groups_with_different_annotation_sets",
                )
            )
            + " |"
        )
    lines += [
        "",
        f"Cross-corpus exact text groups: {summary['cross_corpus_exact_text_groups']}.",
        "",
        "## Near-duplicate screening",
        "",
        f"Candidate unique-text pairs: {summary['near_duplicate_candidate_pairs']}.",
        "Pairs with different numeric tokens: "
        f"{summary['near_pairs_with_different_numeric_tokens']}.",
        "Pairs with different gold for a shared context: "
        f"{summary['near_pairs_with_different_gold_for_shared_context']}.",
        "Candidates use whitespace-word trigram sets, Jaccard >=0.80, and at least",
        "six words per text. An inverted index enumerates shared-shingle pairs and",
        "an exact Jaccard check accepts candidates. Exact matches are collapsed first.",
        "This lexical method misses short, paraphrased, and lower-similarity overlap.",
        "Candidate counts do not establish semantic equivalence or model independence.",
        "",
    ]
    lines += [
        f"- {relation}: {count} candidate pairs"
        for relation, count in summary["near_pairs_by_corpora"].items()
    ]
    lines += [
        "",
        "## Annotation ambiguity indicators",
        "",
        "Gold conflicts require the same normalized text and annotation context.",
        "For PhraseBank the context is the whole sentence; for SEntFiN it is the",
        "normalized target name. FiQA also requires matching aspects and snippets.",
        "Targets are not alias-resolved. Different target/aspect/snippet annotations",
        "are not automatically inconsistent. Missing test gold is not a neutral label.",
        "",
        "Within-record conflicting contexts: "
        f"{summary['within_record_conflicting_gold_contexts']}.",
        "SEntFiN mixed-target label headlines: "
        f"{summary['sentfin_mixed_target_label_headlines']}.",
        "Mixed sentiments about different targets can be correct; they are examples",
        "for target-aware review rather than evidence of annotator disagreement.",
        "FiQA training annotations with missing snippets: "
        f"{summary['fiqa_training_annotations_with_missing_snippets']}.",
        "FiQA nonmatching snippet values: "
        f"{summary['fiqa_training_nonmatching_snippet_values']} across "
        f"{summary['fiqa_training_annotations_with_nonmatching_snippets']} "
        "annotations.",
        "A nonmatching snippet is not a normalized literal span of its source text.",
        "This includes possible paraphrases, annotation notes, and mismatches; it",
        "is a review flag, not proof that the target score is wrong.",
        "FiQA training scores with abs(score)<=0.1: "
        f"{summary['fiqa_training_scores_abs_at_most_0_1']}.",
        "The fixed small-magnitude marker is descriptive, not a neutral-band decision",
        "or proof of annotation uncertainty. Continuous scores remain unchanged.",
        "",
        "PhraseBank strongest matching text/label subset (proxy, not per-row votes):",
        "",
    ]
    lines += [
        f"- {name}: {count} >=50% source rows"
        for name, count in summary[
            "phrasebank_strongest_text_label_subset_match"
        ].items()
    ]
    lines += [
        "",
        "Repeated text/label pairs inherit the strongest matching subset; this",
        "cannot distinguish duplicate occurrences or recover individual vote counts.",
        "The source release does not support computing per-item agreement percentages",
        "or an inter-annotator statistic such as kappa. Consensus subsets "
        "are preserved.",
        "",
        "## Reproduction",
        "",
        "Run `uv run --locked python -m finsight.data review-annotations`.",
        "Ignored `data/interim/annotation-review.json` contains local source text,",
        "IDs, annotations, exact groups, near candidates, and five deterministic",
        "examples per selected ambiguity indicator. These are candidates, not an",
        "exhaustive manual adjudication. Aggregate-only Markdown is committed.",
        "See `docs/annotation-review-findings.md` for the completed "
        "qualitative review.",
        "",
        f"Manifest SHA-256: `{summary['manifest_sha256']}`.",
        "",
    ]
    lines += [
        f"- {name}: `{revision}`" for name, revision in summary["revisions"].items()
    ]
    return "\n".join(lines) + "\n"
