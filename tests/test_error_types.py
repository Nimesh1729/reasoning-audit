"""Tests for error taxonomy."""

from src.evaluation.error_types import (
    ErrorType,
    classify_error,
)


def test_correct_prediction() -> None:
    """Correct prediction should have no error."""
    assert (
        classify_error(
            domain="astronomy",
            prompt_type="clean",
            correct=True,
        )
        == ErrorType.NONE
    )


def test_misleading_hint_error() -> None:
    """Misleading prompt error."""
    assert (
        classify_error(
            domain="astronomy",
            prompt_type="misleading",
            correct=False,
        )
        == ErrorType.MISLEADING_HINT
    )


def test_factual_error() -> None:
    """Astronomy errors should be factual."""
    assert (
        classify_error(
            domain="astronomy",
            prompt_type="clean",
            correct=False,
        )
        == ErrorType.FACTUAL_ERROR
    )


def test_logic_error() -> None:
    """Logic errors should be logic errors."""
    assert (
        classify_error(
            domain="logic",
            prompt_type="clean",
            correct=False,
        )
        == ErrorType.LOGIC_ERROR
    )
