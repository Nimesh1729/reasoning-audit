"""Evaluation metric utilities."""

from collections import defaultdict

from src.evaluation.schemas import EvaluationResult


def compute_accuracy(
    results: list[EvaluationResult],
) -> float:
    """Compute overall accuracy.

    Args:
        results: Evaluation results.

    Returns:
        Overall accuracy.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("Cannot compute accuracy for empty results.")

    num_correct = sum(result.correct for result in results)

    return num_correct / len(results)


def compute_accuracy_by_prompt_type(
    results: list[EvaluationResult],
) -> dict[str, float]:
    """Compute accuracy grouped by prompt type.

    Args:
        results: Evaluation results.

    Returns:
        Accuracy per prompt type.
    """
    grouped_results: dict[str, list[EvaluationResult]] = defaultdict(list)

    for result in results:
        grouped_results[result.prompt_type].append(result)

    return {
        prompt_type: compute_accuracy(prompt_results)
        for prompt_type, prompt_results in grouped_results.items()
    }


def compute_accuracy_by_domain(
    results: list[EvaluationResult],
) -> dict[str, float]:
    """Compute accuracy grouped by domain.

    Args:
        results: Evaluation results.

    Returns:
        Accuracy per domain.
    """
    grouped_results: dict[str, list[EvaluationResult]] = defaultdict(list)

    for result in results:
        grouped_results[result.domain].append(result)

    return {
        domain: compute_accuracy(domain_results)
        for domain, domain_results in grouped_results.items()
    }


def compute_helpful_hint_gain(
    accuracy_by_prompt_type: dict[str, float],
) -> float:
    """Compute accuracy gain from clean prompts to helpful prompts.

    Args:
        accuracy_by_prompt_type: Accuracy grouped by prompt type.

    Returns:
        Helpful-hint accuracy minus clean accuracy.
    """
    return accuracy_by_prompt_type["helpful"] - accuracy_by_prompt_type["clean"]


def compute_misleading_hint_drop(
    accuracy_by_prompt_type: dict[str, float],
) -> float:
    """Compute accuracy drop from clean prompts to misleading prompts.

    Args:
        accuracy_by_prompt_type: Accuracy grouped by prompt type.

    Returns:
        Clean accuracy minus misleading-hint accuracy.
    """
    return accuracy_by_prompt_type["clean"] - accuracy_by_prompt_type["misleading"]
