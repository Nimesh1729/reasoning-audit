"""Error taxonomy utilities."""

from enum import Enum


class ErrorType(str, Enum):
    """Reasoning-audit error categories."""

    NONE = "none"
    MISLEADING_HINT = "misleading_hint"
    ASTRONOMY_ERROR = "astronomy_error"
    LOGIC_ERROR = "logic_error"
    PHYSICS_ERROR = "physics_error"
    ARITHMETIC_ERROR = "arithmetic_error"
    UNKNOWN = "unknown"


def classify_error(*, domain: str, prompt_type: str, correct: bool) -> ErrorType:
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
        return ErrorType.ASTRONOMY_ERROR

    if domain == "logic":
        return ErrorType.LOGIC_ERROR

    if domain == "physics":
        return ErrorType.PHYSICS_ERROR

    if domain == "arithmetic":
        return ErrorType.ARITHMETIC_ERROR

    return ErrorType.UNKNOWN
