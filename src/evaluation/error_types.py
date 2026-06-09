"""Error taxonomy utilities."""

from enum import Enum


class ErrorType(str, Enum):
    """Reasoning-audit error categories."""

    NONE = "none"
    MISLEADING_HINT = "misleading_hint"
    FACTUAL_ERROR = "factual_error"
    LOGIC_ERROR = "logic_error"
    UNKNOWN = "unknown"


def classify_error(
    *,
    domain: str,
    prompt_type: str,
    correct: bool,
) -> ErrorType:
    """Classify an evaluation error.

    Args:
        domain: Question domain.
        prompt_type: Prompt variant.
        correct: Whether prediction was correct.

    Returns:
        Error category.
    """
    if correct:
        return ErrorType.NONE

    if prompt_type == "misleading":
        return ErrorType.MISLEADING_HINT

    if domain == "astronomy":
        return ErrorType.FACTUAL_ERROR

    if domain == "logic":
        return ErrorType.LOGIC_ERROR

    return ErrorType.UNKNOWN
