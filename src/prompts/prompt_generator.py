"""Prompt generation utilities."""


def generate_clean_prompt(
    question: str,
) -> str:
    """Generate clean prompt.

    Args:
        question: Benchmark question.

    Returns:
        Clean prompt.
    """
    return question


def generate_helpful_prompt(
    question: str,
    helpful_hint: str,
) -> str:
    """Generate helpful-hint prompt.

    Args:
        question: Benchmark question.
        helpful_hint: Helpful hint.

    Returns:
        Prompt with helpful hint.
    """
    return f"{question}\n\nHint: {helpful_hint}"


def generate_misleading_prompt(
    question: str,
    misleading_hint: str,
) -> str:
    """Generate misleading-hint prompt.

    Args:
        question: Benchmark question.
        misleading_hint: Misleading hint.

    Returns:
        Prompt with misleading hint.
    """
    return f"{question}\n\nHint: {misleading_hint}"
