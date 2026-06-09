"""Tests for answer matching utilities."""

from src.evaluation.answer_matching import (
    answer_matches,
    contains_match,
    exact_match,
    normalize_answer,
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


def test_contains_match() -> None:
    """Test contains match."""
    assert contains_match(
        prediction="The answer is gravity.",
        ground_truth="gravity",
    )


def test_answer_matches_exact() -> None:
    """Test answer matching with exact match."""
    assert answer_matches(
        prediction="Milky Way",
        ground_truth="Milky Way",
    )


def test_answer_matches_contains() -> None:
    """Test answer matching with contained answer."""
    assert answer_matches(
        prediction="The Sun is luminous.",
        ground_truth="sun is luminous",
    )


def test_answer_does_not_match() -> None:
    """Test incorrect answer."""
    assert not answer_matches(
        prediction="Andromeda",
        ground_truth="Milky Way",
    )
