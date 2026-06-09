"""Tests for error analysis."""

from src.analysis.error_analysis import (
    compute_error_distribution,
)
from src.evaluation.error_types import ErrorType
from src.evaluation.schemas import EvaluationResult


def test_compute_error_distribution() -> None:
    """Test error counting."""
    results = [
        EvaluationResult(
            question_id=1,
            domain="astronomy",
            prompt_type="clean",
            ground_truth="star",
            prediction="star",
            correct=True,
            error_type=ErrorType.NONE,
        ),
        EvaluationResult(
            question_id=2,
            domain="astronomy",
            prompt_type="misleading",
            ground_truth="star",
            prediction="planet",
            correct=False,
            error_type=ErrorType.MISLEADING_HINT,
        ),
    ]

    distribution = compute_error_distribution(
        results,
    )

    assert distribution["none"] == 1
    assert distribution["misleading_hint"] == 1
