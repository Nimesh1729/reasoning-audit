"""Tests for answer matching utilities."""

from src.evaluation.answer_matching import (
    alias_match,
    answer_matches,
    exact_match,
    extract_option_letter,
    normalize_answer,
    numeric_match,
    option_letter_match,
)


def test_normalize_answer() -> None:
    """Test answer normalization."""
    assert normalize_answer(" Star. ") == "star"


def test_exact_match() -> None:
    """Test exact normalized match."""
    assert exact_match(
        prediction="Star.",
        ground_truth="star",
    )


def test_alias_match() -> None:
    """Test alias matching."""
    assert alias_match(
        prediction="Red",
        ground_truth="redder",
    )


def test_numeric_fraction_match() -> None:
    """Test numeric equivalence for fractions."""
    assert numeric_match(
        prediction="1/4",
        ground_truth="0.25",
    )


def test_numeric_percent_match() -> None:
    """Test numeric equivalence for percentages."""
    assert numeric_match(
        prediction="25%",
        ground_truth="0.25",
    )


def test_negative_number_does_not_match_positive() -> None:
    """Test that negative values do not match positive values."""
    assert not answer_matches(
        prediction="-0.5",
        ground_truth="0.5",
    )


def test_answer_matches_exact() -> None:
    """Test answer matching with exact match."""
    assert answer_matches(
        prediction="Milky Way",
        ground_truth="Milky Way",
    )


def test_answer_matches_alias() -> None:
    """Test answer matching with accepted alias."""
    assert answer_matches(
        prediction="Red",
        ground_truth="redder",
    )


def test_answer_does_not_match() -> None:
    """Test incorrect answer."""
    assert not answer_matches(
        prediction="Andromeda",
        ground_truth="Milky Way",
    )


def test_extract_option_letter_direct() -> None:
    """Test direct option-letter extraction."""
    assert extract_option_letter("B") == "B"


def test_extract_option_letter_with_prefix() -> None:
    """Test option-letter extraction with prefix."""
    assert extract_option_letter("The answer is C.") == "C"


def test_option_letter_match() -> None:
    """Test option-letter matching."""
    assert option_letter_match(
        prediction="Final answer: D",
        ground_truth="D",
    )


def test_option_letter_does_not_match_wrong_letter() -> None:
    """Test wrong option-letter prediction."""
    assert not option_letter_match(
        prediction="A",
        ground_truth="B",
    )
