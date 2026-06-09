"""Tests for evaluator utilities."""

import pandas as pd

from src.evaluation.answer_matching import (
    normalize_answer,
)
from src.evaluation.evaluator import (
    evaluate_prompt_cases,
)
from src.models.mock_model import MockModel


def test_normalize_answer() -> None:
    """Test answer normalization."""
    assert normalize_answer(" Star ") == "star"


def test_evaluate_prompt_cases() -> None:
    """Test prompt-case evaluation."""
    prompt_cases = pd.DataFrame(
        {
            "question_id": [1],
            "domain": ["astronomy"],
            "prompt_type": ["clean"],
            "prompt": ["What type of object is the Sun?"],
            "answer": ["star"],
        }
    )

    model = MockModel(response="star")

    results = evaluate_prompt_cases(
        prompt_cases=prompt_cases,
        model=model,
    )

    assert len(results) == 1
    assert results[0].correct is True
    assert results[0].prediction == "star"
