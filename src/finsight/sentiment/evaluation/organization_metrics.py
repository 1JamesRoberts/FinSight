"""Micro exact-span extraction metrics with explicit undefined denominators."""


def exact_span_metrics(
    reference: dict[str, list[tuple[int, int]]],
    predicted: dict[str, list[tuple[int, int]]],
) -> dict:
    """Wrong boundaries count as both a missed reference and an extra proposal."""
    if not reference or set(reference) != set(predicted):
        raise ValueError("nonempty exact headline membership required")

    def checked(values: list[tuple[int, int]]) -> set[tuple[int, int]]:
        if any(
            len(v) != 2 or any(type(x) is not int for x in v) or not 0 <= v[0] < v[1]
            for v in values
        ):
            raise ValueError("invalid extraction span")
        result = set(values)
        if len(result) != len(values):
            raise ValueError("duplicate extraction span")
        return result

    tp = fp = fn = 0
    for identifier, spans in reference.items():
        gold, proposal = checked(spans), checked(predicted[identifier])
        tp += len(gold & proposal)
        fp += len(proposal - gold)
        fn += len(gold - proposal)
    return {
        "headlines": len(reference),
        "reference_spans": tp + fn,
        "predicted_spans": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }
