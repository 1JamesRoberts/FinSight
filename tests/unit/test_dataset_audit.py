from finsight.data.audit import duplicate_count, text_identity


def test_duplicate_identity_normalizes_unicode_case_and_whitespace() -> None:
    assert text_identity("  ＡＣＭＥ\n gains  ") == "acme gains"
    assert duplicate_count(["ACME gains", " acme  gains ", "ACME falls"]) == 1


def test_duplicate_identity_does_not_discard_numbers_or_punctuation() -> None:
    assert text_identity("Profit rose 10%") != text_identity("Profit rose 20%")
    assert text_identity("ACME gains!") != text_identity("ACME gains")
    assert duplicate_count([]) == 0
