"""Evaluation result schemas."""

from dataclasses import dataclass

from src.evaluation.error_types import ErrorType


@dataclass
class EvaluationResult:
    """Single benchmark evaluation result.

    Attributes:
        question_id: Unique benchmark question ID.
        domain: Question domain.
        prompt_type: Prompt variant.
        ground_truth: Expected answer.
        prediction: Model prediction.
        correct: Whether prediction is correct.
        error_type: Error category.
    """

    question_id: int
    domain: str
    prompt_type: str
    ground_truth: str
    prediction: str
    correct: bool
    error_type: ErrorType = ErrorType.NONE
