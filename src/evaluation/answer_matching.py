"""Answer matching utilities."""


def normalize_answer(
    answer: str,
) -> str:
    """Normalize answer text.

    Args:
        answer: Raw answer.

    Returns:
        Normalized answer.
    """
    return answer.strip().lower().rstrip(".")


def exact_match(
    prediction: str,
    ground_truth: str,
) -> bool:
    """Check exact normalized match.

    Args:
        prediction: Model prediction.
        ground_truth: Expected answer.

    Returns:
        Whether prediction matches ground truth.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_match(
    prediction: str,
    ground_truth: str,
) -> bool:
    """Check whether ground truth appears in prediction.

    Args:
        prediction: Model prediction.
        ground_truth: Expected answer.

    Returns:
        Whether normalized ground truth appears in normalized prediction.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    return normalized_ground_truth in normalized_prediction


def answer_matches(
    prediction: str,
    ground_truth: str,
) -> bool:
    """Check whether prediction should be counted as correct.

    Args:
        prediction: Model prediction.
        ground_truth: Expected answer.

    Returns:
        Whether prediction matches ground truth.
    """
    if exact_match(
        prediction=prediction,
        ground_truth=ground_truth,
    ):
        return True

    if contains_match(
        prediction=prediction,
        ground_truth=ground_truth,
    ):
        return True

    return False
