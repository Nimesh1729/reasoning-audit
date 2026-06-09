"""Evaluation pipeline utilities."""

import pandas as pd

from src.evaluation.answer_matching import answer_matches
from src.evaluation.error_types import classify_error
from src.evaluation.schemas import EvaluationResult


def evaluate_prompt_cases(
    prompt_cases: pd.DataFrame,
    model,
) -> list[EvaluationResult]:
    """Evaluate prompt cases using a model.

    Args:
        prompt_cases: DataFrame containing prompt cases.
        model: Object with a generate(prompt: str) method.

    Returns:
        List of evaluation results.
    """
    results = []

    for _, row in prompt_cases.iterrows():
        prediction = model.generate(row["prompt"])

        correct = answer_matches(
            prediction=prediction,
            ground_truth=row["answer"],
        )

        error_type = classify_error(
            domain=row["domain"],
            prompt_type=row["prompt_type"],
            correct=correct,
        )

        results.append(
            EvaluationResult(
                question_id=row["question_id"],
                domain=row["domain"],
                prompt_type=row["prompt_type"],
                ground_truth=row["answer"],
                prediction=prediction,
                correct=correct,
                error_type=error_type,
            )
        )

    return results
