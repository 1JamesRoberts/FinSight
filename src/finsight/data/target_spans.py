"""Literal native-target alignment for a development input-format experiment."""

import re
from dataclasses import dataclass
from typing import Literal

OPEN_TARGET = "[TARGET]"
CLOSE_TARGET = "[/TARGET]"
VARIANTS = ("target_pair", "target_markers", "target_replacement")
SPAN_POLICY = "native-literal-boundary-all-mentions-v1"


@dataclass(frozen=True)
class TargetSpan:
    start: int
    end: int


@dataclass(frozen=True)
class TargetAlignment:
    target: str
    spans: tuple[TargetSpan, ...]
    reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.reasons


def literal_spans(headline: str, target: str) -> tuple[TargetSpan, ...]:
    """Match literal strings at word boundaries, retaining original offsets.

    Case-insensitive matching is explicit; no whitespace normalization, aliases,
    fuzzy resolution or company registry is involved. Offsets are end-exclusive.
    """
    if any(not isinstance(s, str) or not s.strip() for s in (headline, target)):
        raise ValueError("nonempty headline and target strings required")
    pattern = r"(?<!\w)" + re.escape(target) + r"(?!\w)"
    return tuple(
        TargetSpan(m.start(), m.end())
        for m in re.finditer(pattern, headline, flags=re.IGNORECASE)
    )


def align_targets(headline: str, targets: list[str]) -> tuple[TargetAlignment, ...]:
    """Reject missing or overlapping native target spans; keep all repetitions.

    This audits lexical alignment, not whether a surface truly denotes an entity.
    An overlap with another annotated target is excluded rather than guessing
    which identity a marker should indicate. Other unannotated entities are not
    detected, modified or assumed absent.
    """
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("nonempty unique native targets required")
    spans = [literal_spans(headline, target) for target in targets]
    marker_collision = any(
        marker.casefold() in text.casefold()
        for text in (headline, *targets)
        for marker in (OPEN_TARGET, CLOSE_TARGET)
    )
    aligned = []
    for index, (target, matches) in enumerate(zip(targets, spans, strict=True)):
        reasons = []
        if not matches:
            reasons.append("no_literal_boundary_match")
        if any(
            a.start < b.end and b.start < a.end
            for a in matches
            for other_index, other in enumerate(spans)
            if other_index != index
            for b in other
        ):
            reasons.append("overlapping_native_targets")
        if marker_collision:
            reasons.append("reserved_marker_in_original_input")
        aligned.append(TargetAlignment(target, matches, tuple(reasons)))
    return tuple(aligned)


def transform_input(
    headline: str,
    alignment: TargetAlignment,
    variant: Literal["target_pair", "target_markers", "target_replacement"],
) -> tuple[str, str]:
    """Return target sequence and headline sequence for paired tokenization.

    These are ordinary text markers, not pretrained MLM masks or newly registered
    special tokens. Replacement removes identity from BOTH sequences. Only the
    selected target's matched spans change; the rest of the headline stays intact.
    """
    if variant not in VARIANTS:
        raise ValueError("unsupported target input variant")
    if not alignment.eligible or not alignment.spans:
        raise ValueError("eligible nonempty target alignment required")
    if alignment.spans != literal_spans(headline, alignment.target):
        raise ValueError("alignment does not match original headline")
    if any(
        marker.casefold() in headline.casefold()
        for marker in (OPEN_TARGET, CLOSE_TARGET)
    ):
        raise ValueError("reserved marker in original input")
    if variant == "target_pair":
        return alignment.target, headline
    # Replace right to left so earlier offsets remain valid after length changes.
    transformed = headline
    for span in reversed(alignment.spans):
        surface = headline[span.start : span.end]
        replacement = (
            f"{OPEN_TARGET} {surface} {CLOSE_TARGET}"
            if variant == "target_markers"
            else OPEN_TARGET
        )
        transformed = transformed[: span.start] + replacement + transformed[span.end :]
    target = OPEN_TARGET if variant == "target_replacement" else alignment.target
    return target, transformed
