"""Audit paired token IDs and full-text preservation without model weights."""

from typing import Any

from finsight.data.target_pairs import TargetText, encode_batch


def audit_pairs(
    slow: Any,
    fast: Any,
    pairs: list[tuple[str, str]],
    *,
    max_length: int,
) -> list[dict]:
    """Reject truncated/unknown inputs; require slow/fast encoded-ID equivalence.

    Checking the complete input is deliberately conservative: if both sequences
    survive unchanged, every literal target and marker also survives. No offsets
    are guessed from subword counts. Native pair construction and only-second
    truncation are checked by the existing encoder, including on overlong text.
    """
    if type(max_length) is not int or not 4 <= max_length <= 512 or not pairs:
        raise ValueError("nonempty pairs and valid maximum length required")
    if any(
        not isinstance(text, str) or not text.strip() for pair in pairs for text in pair
    ) or any(len(pair) != 2 for pair in pairs):
        raise ValueError("nonempty target/headline text pairs required")
    special = slow.num_special_tokens_to_add(pair=True)
    if special != fast.num_special_tokens_to_add(pair=True):
        raise ValueError("slow/fast special-token count differs")
    result = []
    for start in range(0, len(pairs), 32):
        batch = pairs[start : start + 32]
        targets, headlines = map(list, zip(*batch, strict=True))
        tids = slow(targets, add_special_tokens=False, truncation=False)["input_ids"]
        hids = slow(headlines, add_special_tokens=False, truncation=False)["input_ids"]
        if (
            tids
            != fast(targets, add_special_tokens=False, truncation=False)["input_ids"]
            or hids
            != fast(headlines, add_special_tokens=False, truncation=False)["input_ids"]
        ):
            raise ValueError("slow/fast content token IDs differ")
        encodable = []
        batch_result = []
        for i, (target, headline) in enumerate(batch):
            total = len(tids[i]) + len(hids[i]) + special
            reasons = []
            if not tids[i] or not hids[i]:
                reasons.append("empty_token_sequence")
            if len(tids[i]) + special >= max_length:
                reasons.append("target_leaves_no_headline_budget")
            if total > max_length:
                reasons.append("full_pair_exceeds_max_length")
            if slow.unk_token_id in tids[i] or slow.unk_token_id in hids[i]:
                reasons.append("unknown_token")
            if (
                slow.decode(tids[i], clean_up_tokenization_spaces=False) != target
                or slow.decode(hids[i], clean_up_tokenization_spaces=False) != headline
            ):
                reasons.append("decoded_text_differs")
            item = {
                "target_tokens": len(tids[i]),
                "headline_tokens": len(hids[i]),
                "special_tokens": special,
                "full_pair_tokens": total,
                "eligible": not reasons,
                "reasons": reasons,
            }
            batch_result.append(item)
            if tids[i] and hids[i] and len(tids[i]) + special < max_length:
                encodable.append(i)
        if encodable:
            texts = [TargetText(str(i), headlines[i], targets[i]) for i in encodable]
            a = encode_batch(slow, texts, "target_pair", max_length=max_length)
            b = encode_batch(fast, texts, "target_pair", max_length=max_length)
            if a != b:
                raise ValueError("slow/fast paired token IDs or attention masks differ")
            for local, i in enumerate(encodable):
                if batch_result[i]["eligible"]:
                    actual = [
                        token
                        for token, active in zip(
                            a["input_ids"][local],
                            a["attention_mask"][local],
                            strict=True,
                        )
                        if active
                    ]
                    full = slow.build_inputs_with_special_tokens(tids[i], hids[i])
                    if actual != full:
                        raise ValueError("eligible input was not preserved in full")
        result.extend(batch_result)
    return result
