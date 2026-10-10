"""Verify source counting units, percentile interpolation, and histogram edges."""

import pytest

from finsight.data.lengths import length_summary


def test_original_whitespace_and_unicode_code_points_are_preserved() -> None:
    whitespace = length_summary(["\tA  B\n"])
    assert whitespace["characters"]["min"] == 6
    assert whitespace["words"]["min"] == 2
    unicode = length_summary(["é 中 👩‍💻"])
    assert unicode["characters"]["min"] == 7
    assert unicode["words"]["min"] == 3


def test_inclusive_percentiles_and_generator_input() -> None:
    result = length_summary(" ".join(["x"] * count) for count in range(1, 11))
    assert result["records"] == 10
    words = result["words"]
    assert words["min"] == 1
    assert words["max"] == 10
    assert words["mean"] == words["median"] == 5.5
    assert words["p90"] == pytest.approx(9.1)
    assert words["p95"] == pytest.approx(9.55)
    assert words["p99"] == pytest.approx(9.91)


def test_word_histogram_edges_and_flag_denominators() -> None:
    counts = [3, 5, 6, 10, 11, 20, 21, 50, 51, 100, 101]
    result = length_summary(" ".join(["x"] * count) for count in counts)
    assert result["word_histogram"] == {
        "1-5": 2,
        "6-10": 2,
        "11-20": 2,
        "21-50": 2,
        "51-100": 2,
        ">100": 1,
    }
    assert sum(result["word_histogram"].values()) == 11
    assert result["word_flags"] == {"at_most_3": 1, "over_50": 3, "over_100": 1}


def test_empty_group_and_single_observation_are_explicit() -> None:
    empty = length_summary([])
    assert empty["records"] == 0
    assert all(value is None for value in empty["words"].values())
    assert all(value == 0 for value in empty["word_histogram"].values())
    single = length_summary(["One word-pair"])
    assert all(value == 2 for value in single["words"].values())


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_blank_source_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        length_summary([text])
