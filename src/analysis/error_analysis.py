"""Error analysis utilities."""

from collections import Counter

from src.evaluation.schemas import EvaluationResult


def compute_error_distribution(
    results: list[EvaluationResult],
) -> dict[str, int]:
    """Compute error distribution.

    Args:
        results: Evaluation results.

    Returns:
        Error counts.
    """
    return dict(Counter(result.error_type.value for result in results))
