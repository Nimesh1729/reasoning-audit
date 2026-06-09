"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    compute_accuracy,
    compute_accuracy_by_domain,
    compute_accuracy_by_prompt_type,
    compute_helpful_hint_gain,
    compute_misleading_hint_drop,
)
from src.evaluation.schemas import EvaluationResult


def test_compute_accuracy() -> None:
    """Test overall accuracy."""
    results = [
        EvaluationResult(
            question_id=1,
            domain="astronomy",
            prompt_type="clean",
            ground_truth="star",
            prediction="star",
            correct=True,
        ),
        EvaluationResult(
            question_id=2,
            domain="astronomy",
            prompt_type="misleading",
            ground_truth="Milky Way",
            prediction="Andromeda",
            correct=False,
        ),
    ]

    accuracy = compute_accuracy(results)

    assert accuracy == 0.5


def test_compute_accuracy_empty_results() -> None:
    """Test accuracy on empty results."""
    with pytest.raises(ValueError):
        compute_accuracy([])


def test_compute_accuracy_by_prompt_type() -> None:
    """Test accuracy grouped by prompt type."""
    results = [
        EvaluationResult(
            question_id=1,
            domain="astronomy",
            prompt_type="clean",
            ground_truth="star",
            prediction="star",
            correct=True,
        ),
        EvaluationResult(
            question_id=2,
            domain="astronomy",
            prompt_type="clean",
            ground_truth="Milky Way",
            prediction="Andromeda",
            correct=False,
        ),
        EvaluationResult(
            question_id=3,
            domain="logic",
            prompt_type="misleading",
            ground_truth="yes",
            prediction="no",
            correct=False,
        ),
    ]

    accuracy_by_prompt_type = compute_accuracy_by_prompt_type(results)

    assert accuracy_by_prompt_type["clean"] == 0.5
    assert accuracy_by_prompt_type["misleading"] == 0.0


def test_compute_accuracy_by_domain() -> None:
    """Test accuracy grouped by domain."""
    results = [
        EvaluationResult(
            question_id=1,
            domain="astronomy",
            prompt_type="clean",
            ground_truth="star",
            prediction="star",
            correct=True,
        ),
        EvaluationResult(
            question_id=2,
            domain="astronomy",
            prompt_type="misleading",
            ground_truth="Milky Way",
            prediction="Andromeda",
            correct=False,
        ),
        EvaluationResult(
            question_id=3,
            domain="logic",
            prompt_type="clean",
            ground_truth="yes",
            prediction="yes",
            correct=True,
        ),
    ]

    accuracy_by_domain = compute_accuracy_by_domain(results)

    assert accuracy_by_domain["astronomy"] == 0.5
    assert accuracy_by_domain["logic"] == 1.0


def test_compute_helpful_hint_gain() -> None:
    """Test helpful-hint gain."""
    accuracy_by_prompt_type = {
        "clean": 0.7,
        "helpful": 0.9,
        "misleading": 0.4,
    }

    gain = compute_helpful_hint_gain(
        accuracy_by_prompt_type,
    )

    assert gain == pytest.approx(0.2)


def test_compute_misleading_hint_drop() -> None:
    """Test misleading-hint drop."""
    accuracy_by_prompt_type = {
        "clean": 0.7,
        "helpful": 0.9,
        "misleading": 0.4,
    }

    drop = compute_misleading_hint_drop(
        accuracy_by_prompt_type,
    )

    assert drop == pytest.approx(0.3)
