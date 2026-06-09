"""Prompt generation utilities."""


def generate_clean_prompt(question: str) -> str:
    """Generate clean free-form prompt.

    Args:
        question: Benchmark question.

    Returns:
        Clean prompt.
    """
    return question


def generate_helpful_prompt(question: str, helpful_hint: str) -> str:
    """Generate helpful-hint free-form prompt.

    Args:
        question: Benchmark question.
        helpful_hint: Helpful hint.

    Returns:
        Prompt with helpful hint.
    """
    return f"{question}\n\nHint: {helpful_hint}"


def generate_misleading_prompt(question: str, misleading_hint: str) -> str:
    """Generate misleading-hint free-form prompt.

    Args:
        question: Benchmark question.
        misleading_hint: Misleading hint.

    Returns:
        Prompt with misleading hint.
    """
    return f"{question}\n\nHint: {misleading_hint}"


def format_mcq_options(
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
) -> str:
    """Format multiple-choice options.

    Args:
        option_a: Option A.
        option_b: Option B.
        option_c: Option C.
        option_d: Option D.

    Returns:
        Formatted option block.
    """
    return f"A. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}"


def generate_clean_mcq_prompt(
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
) -> str:
    """Generate clean multiple-choice prompt.

    Args:
        question: Benchmark question.
        option_a: Option A.
        option_b: Option B.
        option_c: Option C.
        option_d: Option D.

    Returns:
        Clean multiple-choice prompt.
    """
    options = format_mcq_options(
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )

    return (
        "Answer with only the option letter: A, B, C, or D.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options}"
    )


def generate_helpful_mcq_prompt(
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
    helpful_hint: str,
) -> str:
    """Generate helpful-hint multiple-choice prompt.

    Args:
        question: Benchmark question.
        option_a: Option A.
        option_b: Option B.
        option_c: Option C.
        option_d: Option D.
        helpful_hint: Helpful hint.

    Returns:
        Helpful-hint multiple-choice prompt.
    """
    clean_prompt = generate_clean_mcq_prompt(
        question=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )

    return f"{clean_prompt}\n\nHint: {helpful_hint}"


def generate_misleading_mcq_prompt(
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
    misleading_hint: str,
) -> str:
    """Generate misleading-hint multiple-choice prompt.

    Args:
        question: Benchmark question.
        option_a: Option A.
        option_b: Option B.
        option_c: Option C.
        option_d: Option D.
        misleading_hint: Misleading hint.

    Returns:
        Misleading-hint multiple-choice prompt.
    """
    clean_prompt = generate_clean_mcq_prompt(
        question=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )

    return f"{clean_prompt}\n\nHint: {misleading_hint}"
